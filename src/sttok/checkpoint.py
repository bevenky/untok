"""Fail-closed RNNT checkpoint expansion and independent transfer checks.

Only vocabulary-dependent embedding and joint-output rows may change. The
module never downloads weights, silently changes HF special IDs, or treats a
successful migration as evidence of unchanged ASR accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .runtime import IdMap, build_id_map, get_nemo_model_class


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Checkpoint operations require the optional torch dependency") from exc
    return torch


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RNNTLayout:
    embedding_key: str
    output_weight_key: str
    output_bias_key: str | None
    blank_id: int
    output_size: int

    @property
    def row_keys(self) -> tuple[str, ...]:
        return tuple(k for k in (self.embedding_key, self.output_weight_key, self.output_bias_key) if k)


def inspect_nemo_layout(model: Any) -> RNNTLayout:
    """Recognize the standard blank-as-padding RNNT layout by live modules.

    Rejects CTC/hybrid/TDT heads, extra outputs, shared parameters and unusual
    embedding arrangements instead of guessing based on tensor dimensions.
    """
    torch = _torch()
    if hasattr(model, "ctc_decoder") or hasattr(model, "aux_ctc"):
        raise ValueError("Hybrid/CTC models are not supported by this RNNT migration")
    decoder, joint = getattr(model, "decoder", None), getattr(model, "joint", None)
    if decoder is None or joint is None or not hasattr(decoder, "prediction"):
        raise ValueError("Unrecognized NeMo RNNT model layout")
    if not getattr(decoder, "blank_as_pad", False):
        raise ValueError("Only RNNT blank_as_pad=True is supported")
    if getattr(joint, "_num_extra_outputs", 0) != 0:
        raise ValueError("Extra output classes (for example duration heads) are unsupported")
    embeddings = [(name, module) for name, module in decoder.named_modules() if isinstance(module, torch.nn.Embedding)]
    if len(embeddings) != 1:
        raise ValueError("Expected exactly one prediction-network embedding")
    _, embedding = embeddings[0]
    joint_net = getattr(joint, "joint_net", None)
    if not isinstance(joint_net, torch.nn.Sequential) or not isinstance(joint_net[-1], torch.nn.Linear):
        raise ValueError("Expected a final Linear layer in joint.joint_net")
    head = joint_net[-1]
    blank = int(decoder.blank_idx)
    if embedding.padding_idx != blank or embedding.num_embeddings != blank + 1:
        raise ValueError("Prediction embedding and blank index do not agree")
    if head.out_features != blank + 1:
        raise ValueError("Joint output rows and blank index do not agree")
    if getattr(joint, "num_classes_with_blank", head.out_features) != head.out_features:
        raise ValueError("Joint class count disagrees with output layer")
    parameters = list(model.named_parameters(remove_duplicate=False))

    def name_for(parameter):
        names = [name for name, value in parameters if value is parameter]
        if len(names) != 1:
            raise ValueError("Shared or unregistered vocabulary parameter is unsupported")
        return names[0]

    return RNNTLayout(name_for(embedding.weight), name_for(head.weight),
                      name_for(head.bias) if head.bias is not None else None,
                      blank, head.out_features)


def old_model_row_mapping(old_map: IdMap, new_map: IdMap) -> tuple[int, ...]:
    """Map every old acoustic row, including blank, to its new acoustic row."""
    return tuple(new_map.to_model([i], allow_blank=True)[0] for i in old_map.model_to_canonical)


def transfer_state_dict(
    source: Mapping[str, Any], initialized_target: Mapping[str, Any],
    old_layout: RNNTLayout, new_layout: RNNTLayout, old_to_new: Sequence[int],
) -> dict[str, Any]:
    """Copy all learned state, retaining target initialization only for additions."""
    torch = _torch()
    if set(source) != set(initialized_target):
        raise ValueError(f"State keys differ: missing={sorted(set(source)-set(initialized_target))}, "
                         f"unexpected={sorted(set(initialized_target)-set(source))}")
    if old_layout.row_keys != new_layout.row_keys:
        raise ValueError("Vocabulary-dependent parameter names changed")
    indices = tuple(int(i) for i in old_to_new)
    if len(indices) != old_layout.output_size or len(set(indices)) != len(indices):
        raise ValueError("Old model row mapping must be complete and one-to-one")
    if any(i < 0 or i >= new_layout.output_size for i in indices):
        raise ValueError("Mapped model row outside expanded vocabulary")
    if indices[old_layout.blank_id] != new_layout.blank_id:
        raise ValueError("Old blank row must map to the new blank row")
    result = {}
    for key, value in source.items():
        target = initialized_target[key]
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            raise ValueError(f"Unsupported non-tensor model state: {key}")
        if value.dtype != target.dtype:
            raise ValueError(f"Unexpected dtype change for {key}")
        if key in old_layout.row_keys:
            if value.shape[0] != old_layout.output_size or target.shape[0] != new_layout.output_size:
                raise ValueError(f"Wrong vocabulary row count for {key}")
            if value.shape[1:] != target.shape[1:]:
                raise ValueError(f"Non-vocabulary dimensions changed for {key}")
            copied = target.detach().clone()
            copied.index_copy_(0, torch.tensor(indices, device=copied.device), value.to(copied.device))
            result[key] = copied
        else:
            if value.shape != target.shape:
                raise ValueError(f"Unapproved tensor shape change: {key}")
            result[key] = value.detach().to(target.device).clone()
    return result


def verify_state_transfer(
    source: Mapping[str, Any], migrated: Mapping[str, Any],
    layout: RNNTLayout, old_to_new: Sequence[int],
) -> dict[str, Any]:
    """An independent equality check against the original learned tensors."""
    torch = _torch()
    if set(source) != set(migrated):
        raise ValueError("State keys differ after migration")
    failures, preserved = [], 0
    for key, expected in source.items():
        actual = migrated[key]
        if key in layout.row_keys:
            actual = actual.index_select(0, torch.tensor(old_to_new, device=actual.device))
        if expected.dtype != actual.dtype or expected.shape != actual.shape or not torch.equal(expected.cpu(), actual.cpu()):
            failures.append(key)
        else:
            preserved += expected.numel()
    if failures:
        raise ValueError(f"Learned state changed during migration: {', '.join(failures)}")
    return {"passed": True, "tensors_checked": len(source), "learned_values_preserved": preserved,
            "comparison": "exact_tensor_equality"}


def compare_old_logits(
    original_logits: Any, expanded_logits: Any, old_to_new: Sequence[int],
    *, atol: float = 1e-6, rtol: float = 1e-5,
) -> dict[str, Any]:
    """Compare raw logits from identical features/prefixes, not full softmaxes."""
    torch = _torch()
    if original_logits.shape[-1] != len(old_to_new):
        raise ValueError("Original logit width and row mapping disagree")
    selected = expanded_logits.index_select(-1, torch.tensor(old_to_new, device=expanded_logits.device))
    expected = original_logits.to(selected.device)
    if expected.shape != selected.shape:
        raise ValueError("Logit batch/frame/prefix shapes differ")
    if not torch.isfinite(expected).all() or not torch.isfinite(selected).all():
        raise ValueError("Non-finite old-token logits")
    passed = torch.allclose(expected, selected, atol=atol, rtol=rtol)
    error = (expected - selected).abs().max().item() if expected.numel() else 0.0
    if not passed:
        raise ValueError(f"Mapped old-token/blank logits changed (max absolute error {error})")
    return {"passed": True, "max_absolute_error": error, "atol": atol, "rtol": rtol,
            "scope": "raw_logits_for_supplied_features_and_prefixes"}


def mask_new_outputs_for_test(logits: Any, old_to_new: Sequence[int]):
    """Validation-only old-output mask, applied BEFORE any softmax.

    Do not use this helper as evidence of unchanged production predictions with
    additions enabled. It is not a language lock or a training policy.
    """
    torch = _torch()
    if len(set(old_to_new)) != len(old_to_new) or any(i < 0 or i >= logits.shape[-1] for i in old_to_new):
        raise ValueError("Invalid old-output map")
    result = torch.full_like(logits, float("-inf"))
    index = torch.tensor(old_to_new, device=logits.device)
    result.index_copy_(-1, index, logits.index_select(-1, index))
    return result


def artifact_preflight(tokenizer_json: str | Path, hf_config_json: str | Path | None = None) -> dict[str, Any]:
    """Read-only artifact consistency check; never rewrites a source mismatch."""
    mapping = build_id_map(tokenizer_json)
    report: dict[str, Any] = {"tokenizer_sha256": mapping.tokenizer_sha256,
        "canonical_hf_ids": len(mapping.canonical_to_model), "hf_pad_id": mapping.hf_pad_id,
        "hf_blank_id": mapping.hf_blank_id, "native_model_vocab_size": mapping.model_vocab_size,
        "native_model_blank_id": mapping.model_blank_id, "checkpoint_executed": False}
    if hf_config_json is not None:
        config = json.loads(Path(hf_config_json).read_text())
        conflicts = []
        for name, expected in (("blank_token_id", mapping.hf_blank_id), ("pad_token_id", mapping.hf_pad_id)):
            if config.get(name) != expected:
                conflicts.append({"field": name, "model": config.get(name), "tokenizer": expected})
        if config.get("vocab_size", 0) <= max(i for i in range(len(mapping.canonical_to_model)) if i != mapping.hf_pad_id):
            conflicts.append({"field": "vocab_size", "model": config.get("vocab_size"),
                              "minimum_for_direct_hf_ids": len(mapping.canonical_to_model)})
        report.update(hf_direct_compatible=not conflicts, hf_conflicts=conflicts,
                      config_sha256=_sha256(hf_config_json))
    return report


def _validate_source_tokens(model: Any, old_map: IdMap, base_tokenizer_json: str | Path):
    from tokenizers import Tokenizer

    backend = Tokenizer.from_file(str(base_tokenizer_json))
    tokenizer = model.tokenizer
    for model_id, hf_id in enumerate(old_map.model_to_canonical[:-1]):
        if hasattr(tokenizer, "ids_to_tokens"):
            piece = tokenizer.ids_to_tokens([model_id])[0]
        else:
            raise ValueError("Source model tokenizer does not expose IDs for verification")
        if piece != backend.id_to_token(hf_id):
            raise ValueError(f"Source model token {model_id} disagrees with the canonical base")


def migrate_nemo_checkpoint(
    source: str | Path, tokenizer_json: str | Path, base_tokenizer_json: str | Path,
    output: str | Path, *, seed: int = 0, prompt_dictionary: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Expand a local native .nemo checkpoint, verify all rows and reload it.

    Requires an installed compatible NeMo runtime. No HF safetensors migration
    is attempted because the pinned upstream special-ID mismatch needs an
    independently verified conversion. No files are overwritten.
    """
    torch = _torch()
    source, output = Path(source), Path(output)
    if not source.is_file() or source.suffix != ".nemo":
        raise ValueError("Provide a complete local native .nemo checkpoint")
    if output.exists() or output.suffix != ".nemo" or output.resolve() == source.resolve():
        raise ValueError("Output must be a new .nemo path, leaving the source intact")
    report_path = output.with_suffix(".migration.json")
    if report_path.exists():
        raise ValueError("Migration report already exists; choose a new output path")
    from nemo.collections.asr.models import ASRModel
    from omegaconf import OmegaConf, open_dict

    old_map = build_id_map(base_tokenizer_json)
    new_map = build_id_map(tokenizer_json, base_tokenizer_json)
    old_to_new = old_model_row_mapping(old_map, new_map)
    original = ASRModel.restore_from(str(source), map_location="cpu")
    if original.__class__.__name__ not in {"EncDecRNNTBPEModelWithPrompt", "ExtendedNemotronRNNTModel"}:
        raise ValueError(f"Unsupported source model class: {original.__class__.__name__}")
    old_layout = inspect_nemo_layout(original)
    if old_layout.blank_id != old_map.model_blank_id:
        raise ValueError("Source blank/vocabulary does not match the pinned canonical base")
    _validate_source_tokens(original, old_map, base_tokenizer_json)
    cfg = OmegaConf.create(OmegaConf.to_container(original.cfg, resolve=True))
    with open_dict(cfg):
        cfg.tokenizer = {"type": "sttok_hf_bpe", "hf_tokenizer_json": str(Path(tokenizer_json).resolve())}
        cfg.target = "sttok.runtime.ExtendedNemotronRNNTModel"
        if prompt_dictionary is not None:
            old_prompts = dict(cfg.model_defaults.prompt_dictionary)
            if any(prompt_dictionary.get(k) != v for k, v in old_prompts.items()):
                raise ValueError("An existing language prompt assignment changed")
            limit = int(cfg.model_defaults.num_prompts)
            if any(not isinstance(v, int) or not 0 <= v < limit for v in prompt_dictionary.values()):
                raise ValueError("Prompt IDs must fit the unchanged prompt dimension")
            old_slots = set(old_prompts.values())
            if any(v in old_slots and k not in old_prompts for k, v in prompt_dictionary.items()):
                raise ValueError("New prompt identities must use unused slots; explicit aliases are not inferred")
            cfg.model_defaults.prompt_dictionary = dict(prompt_dictionary)
            for split in ("train_ds", "validation_ds", "test_ds"):
                if split in cfg and cfg[split] is not None and "prompt_dictionary" in cfg[split]:
                    cfg[split].prompt_dictionary = dict(prompt_dictionary)
        # Model construction must not initialize dataloaders on the source's private manifests.
        for split in ("train_ds", "validation_ds", "test_ds"):
            cfg[split] = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        expanded = get_nemo_model_class()(cfg=cfg, trainer=None)
    new_layout = inspect_nemo_layout(expanded)
    if new_layout.blank_id != new_map.model_blank_id:
        raise ValueError("Constructed RNNT layout disagrees with tokenizer adapter")
    transferred = transfer_state_dict(original.state_dict(), expanded.state_dict(), old_layout, new_layout, old_to_new)
    expanded.load_state_dict(transferred, strict=True)
    del transferred
    expanded.eval()
    before_save = verify_state_transfer(original.state_dict(), expanded.state_dict(), old_layout, old_to_new)
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_adapter_map = expanded.tokenizer.id_map.to_dict()
    with tempfile.TemporaryDirectory(prefix=".sttok-migration-", dir=output.parent) as staging:
        staged = Path(staging) / output.name
        expanded.save_to(str(staged))
        del expanded
        restored = get_nemo_model_class().restore_from(str(staged), map_location="cpu")
        after_reload = verify_state_transfer(original.state_dict(), restored.state_dict(), old_layout, old_to_new)
        if restored.tokenizer.id_map.to_dict() != expected_adapter_map:
            raise ValueError("Tokenizer ID mapping changed after checkpoint restoration")
        # Same-filesystem hard linking is atomic and refuses an existing target.
        # A failed restore/verification therefore leaves no claimed output file.
        os.link(staged, output)
    report = {"schema_version": 1, "status": "migrated_weights_verified",
        "source_checkpoint_sha256": _sha256(source), "checkpoint_sha256": _sha256(output),
        "tokenizer_sha256": new_map.tokenizer_sha256, "base_tokenizer_sha256": old_map.tokenizer_sha256,
        "seed": seed, "old_layout": asdict(old_layout), "new_layout": asdict(new_layout),
        "id_mapping": new_map.to_dict(), "old_model_to_new_model": list(old_to_new),
        "before_save": before_save, "after_reload": after_reload,
        "asr_accuracy_evaluated": False, "training_forward_evaluated": False,
        "logit_parity_evaluated": False,
        "remaining_gates": ["training_forward_backward", "old_output_logit_and_audio_parity",
                            "expanded_output_audio_regression", "new_language_fine_tuning_and_evaluation"]}
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
