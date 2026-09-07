"""One real RNNT forward/backward check; never optimizes or saves a model.

Prepare ``batch = next(iter(model.train_dataloader()))`` using the project's
NeMo dataloader and restored HF adapter, retaining its raw texts and locale keys.
The supported batch is exactly (signal, signal_len, transcript, transcript_len,
prompt_indices), as in the pinned EncDecRNNTBPEModelWithPrompt.training_step.
Call run_training_smoke with that batch, matching texts/target_langs and verified
artifact hashes. This module does not invent a JSONL/audio loader.

The verified training_step also reads trainer/optimizer logging fields. Instead
of fabricating them, this check uses its same forward, predictor and RNNT loss
calls directly. Both fused and non-fused RNNT joint paths are supported. A fresh
checkpoint instance is loaded, so training-mode buffer changes are discarded.
Unit tests use fake NeMo objects; no real model execution is implied by them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .checkpoint import _sha256, _torch, inspect_nemo_layout
from .inference import _load_model, _plain, _tokenizer_hash
from .runtime import build_id_map


def run_training_smoke(
    checkpoint_path: str | Path, batch: Sequence[Any], texts: Sequence[str],
    target_langs: Sequence[str], base_tokenizer_json: str | Path, *,
    checkpoint_sha256: str, tokenizer_sha256: str, device: str = "cpu",
) -> dict[str, Any]:
    """Check a caller-prepared raw-audio prompt-RNNT batch and its new-row grads.

    Inputs must originate from the compatible NeMo dataloader. Active labels
    are compared against the restored canonical HF tokenizer; padded positions
    are excluded by transcript_len. At least one newly added row must occur.
    Returns an in-memory report; no optimizer is created and no file is written.
    """
    torch = _torch()
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file() or checkpoint.suffix != ".nemo":
        raise ValueError("A complete local migrated .nemo checkpoint is required")
    if _sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Checkpoint hash differs from the supplied compatibility metadata")
    if not isinstance(batch, (tuple, list)) or len(batch) != 5 or not all(isinstance(t, torch.Tensor) for t in batch):
        raise ValueError("Supply the supported five-tensor raw-audio NeMo batch")
    signal, signal_len, transcript, transcript_len, prompt_indices = batch
    if signal.ndim != 2 or transcript.ndim != 2 or signal.shape[0] != transcript.shape[0] or not signal.shape[0]:
        raise ValueError("Signal and transcript require matching nonempty batch dimensions")
    count = signal.shape[0]
    if not signal.is_floating_point() or not torch.isfinite(signal).all():
        raise ValueError("Raw audio must be finite floating-point samples")
    for tensor in (signal_len, transcript_len, prompt_indices):
        if tensor.shape != (count,) or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("Lengths and prompt indices must be integer batch vectors")
    if transcript.dtype not in (torch.int32, torch.int64):
        raise ValueError("Transcript labels must be integer model IDs")
    if len(texts) != count or len(target_langs) != count or not all(isinstance(t, str) for t in texts):
        raise ValueError("Provide one raw text and target locale per batch item")
    if any(int(n) <= 0 or int(n) > signal.shape[1] for n in signal_len):
        raise ValueError("Signal lengths are outside the supplied audio tensor")
    if any(int(n) <= 0 or int(n) > transcript.shape[1] for n in transcript_len):
        raise ValueError("Transcript lengths are outside the supplied label tensor")
    original_map = build_id_map(base_tokenizer_json)
    model = _load_model(checkpoint, "cpu")
    if _tokenizer_hash(model) != tokenizer_sha256:
        raise ValueError("Restored tokenizer differs from the supplied tokenizer hash")
    mapping = getattr(model.tokenizer, "id_map", None)
    if mapping is None:
        raise ValueError("Training smoke requires the migrated canonical HF tokenizer adapter")
    from tokenizers import Tokenizer
    base_backend = Tokenizer.from_file(str(base_tokenizer_json))
    for old_id in range(len(original_map.canonical_to_model)):
        if base_backend.id_to_token(old_id) != model.tokenizer.backend.id_to_token(old_id):
            raise ValueError("An original canonical token ID changed")
    layout = inspect_nemo_layout(model)
    if layout.blank_id != mapping.model_blank_id:
        raise ValueError("Model blank and tokenizer adapter disagree")
    if transcript.min().item() < 0 or transcript.max().item() >= layout.output_size:
        raise ValueError("Batch contains an out-of-range model token ID")
    registry = _plain(model.cfg).get("model_defaults", {}).get("prompt_dictionary", {})
    exercised = set()
    normalized = []
    for i, (text, locale) in enumerate(zip(texts, target_langs)):
        if locale not in registry or registry[locale] != int(prompt_indices[i]):
            raise ValueError(f"Prompt for batch item {i} does not match its explicit locale")
        active = transcript[i, :int(transcript_len[i])].detach().cpu().tolist()
        if layout.blank_id in active:
            raise ValueError("Active transcript labels contain RNNT blank/padding")
        expected = model.tokenizer.text_to_ids(text)
        if active != expected:
            raise ValueError(f"Batch item {i} labels differ from the restored HF normalization and model-ID mapping")
        if model.tokenizer.unk_id in active:
            raise ValueError("Training smoke transcripts contain unknown tokens")
        normalized.append(model.tokenizer.ids_to_text(active))
        exercised.update(index for index in active
                         if mapping.model_to_canonical[index] >= len(original_map.canonical_to_model))
    if not exercised:
        raise ValueError("Batch must exercise at least one newly added model row")
    model.to(device).float().train()
    model.zero_grad(set_to_none=True)
    signal = signal.to(device=device, dtype=torch.float32)
    signal_len, transcript, transcript_len, prompt_indices = (
        t.to(device=device, dtype=torch.long) for t in (signal_len, transcript, transcript_len, prompt_indices)
    )
    # This sequence follows the inspected prompt RNNT training_step API.
    encoded, encoded_len = model.forward(input_signal=signal, input_signal_length=signal_len, prompt_indices=prompt_indices)
    predicted, target_length, _ = model.decoder(targets=transcript, target_length=transcript_len)
    if model.joint.fuse_loss_wer:
        loss, _, _, _ = model.joint(encoder_outputs=encoded, decoder_outputs=predicted,
                                   encoder_lengths=encoded_len, transcripts=transcript,
                                   transcript_lengths=target_length, compute_wer=False)
        path = "fused_rnnt_joint_loss"
    else:
        logits = model.joint(encoder_outputs=encoded, decoder_outputs=predicted)
        loss = model.loss(log_probs=logits, targets=transcript, input_lengths=encoded_len, target_lengths=target_length)
        path = "rnnt_loss"
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1 or not torch.isfinite(loss).all() or not loss.requires_grad:
        raise ValueError("Actual RNNT loss must be a finite differentiable scalar")
    loss.backward()
    parameters = dict(model.named_parameters())
    row_ids = sorted(exercised)
    checks = {}
    for key in (layout.embedding_key, layout.output_weight_key):
        gradient = parameters[key].grad
        if gradient is None:
            raise ValueError(f"No gradients reached {key}")
        rows = gradient.index_select(0, torch.tensor(row_ids, device=gradient.device))
        if not torch.isfinite(rows).all():
            raise ValueError(f"Non-finite new-row gradients in {key}")
        magnitudes = rows.abs().flatten(1).sum(1)
        if not (magnitudes > 0).all():
            raise ValueError(f"Some exercised new rows received zero gradients in {key}")
        checks[key] = {str(index): float(value) for index, value in zip(row_ids, magnitudes.detach().cpu())}
    report = {"schema_version": 1, "status": "rnnt_training_smoke_passed_on_supplied_batch", "passed": True,
              "checkpoint_sha256": checkpoint_sha256, "tokenizer_sha256": tokenizer_sha256,
              "base_tokenizer_sha256": original_map.tokenizer_sha256,
              "model_class": type(model).__module__ + "." + type(model).__qualname__,
              "batch_size": count, "device": device, "dtype": "float32", "loss_path": path,
              "loss": float(loss.detach().cpu()), "exercised_new_model_rows": row_ids,
              "new_row_gradient_absolute_sums": checks, "normalized_targets": normalized,
              "optimizer_steps": 0, "checkpoint_saved": False, "asr_accuracy_evaluated": False,
              "release_ready": False, "scope": "one actual caller-prepared RNNT batch; not an accuracy evaluation"}
    model.zero_grad(set_to_none=True)
    return report
