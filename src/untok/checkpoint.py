"""Fail-closed RNNT checkpoint expansion and independent transfer checks.

Only vocabulary-dependent embedding and joint-output rows may change. The
module never downloads weights, silently changes HF special IDs, or treats a
successful migration as evidence of unchanged ASR accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
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


def initialize_added_rows(
    source: Mapping[str, Any], initialized_target: Mapping[str, Any],
    old_layout: RNNTLayout, new_layout: RNNTLayout, old_to_new: Sequence[int],
    *, max_new_mass_ratio: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Initialize additions below the learned blank with a bounded output prior.

    In exact arithmetic every new logit is old_blank_logit - log(M / epsilon).
    Thus Z_new / Z_old = epsilon * P_old(blank) <= epsilon. Blank provides a
    learned acoustic reference without averaging potentially inactive classes.
    Floating-point arithmetic and actual decoding must still be checked.
    Old rows are copied by transfer_state_dict.
    """
    torch = _torch()
    if not math.isfinite(max_new_mass_ratio) or not 0 < max_new_mass_ratio < 1:
        raise ValueError("New-output mass ratio must be finite and between zero and one")
    if not (0 <= old_layout.blank_id < old_layout.output_size and 0 <= new_layout.blank_id < new_layout.output_size):
        raise ValueError("Invalid blank index for added-row initialization")
    indices = tuple(int(i) for i in old_to_new)
    if (len(indices) != old_layout.output_size or len(set(indices)) != len(indices)
            or any(i < 0 or i >= new_layout.output_size for i in indices)
            or indices[old_layout.blank_id] != new_layout.blank_id):
        raise ValueError("Added-row initialization requires a complete valid old-row mapping")
    additions = sorted(set(range(new_layout.output_size)) - set(indices))
    if not additions:
        return dict(initialized_target), {"policy": "no_added_rows", "new_row_count": 0, "new_model_rows": []}
    if old_layout.output_bias_key is None or new_layout.output_bias_key is None:
        raise ValueError("Bounded added-output initialization requires a joint output bias")
    if old_layout.row_keys != new_layout.row_keys or old_layout.output_size < 2:
        raise ValueError("Unsupported vocabulary layout for blank-anchored initialization")
    for key in old_layout.row_keys:
        if key not in source or key not in initialized_target:
            raise ValueError(f"Missing vocabulary tensor for initialization: {key}")
        old, new = source[key], initialized_target[key]
        if (not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor)
                or not old.is_floating_point() or old.dtype != new.dtype
                or old.ndim < 1 or new.ndim < 1
                or old.shape[0] != old_layout.output_size or new.shape[0] != new_layout.output_size
                or old.shape[1:] != new.shape[1:] or not torch.isfinite(old).all()
                or not torch.isfinite(new).all()):
            raise ValueError(f"Invalid or non-finite vocabulary tensor for initialization: {key}")
    if (source[old_layout.embedding_key].ndim != 2 or source[old_layout.output_weight_key].ndim != 2
            or source[old_layout.output_bias_key].ndim != 1):
        raise ValueError("Blank-anchored initialization requires matrix embeddings/weights and vector bias")
    text_rows = [i for i in range(old_layout.output_size) if i != old_layout.blank_id]
    embedding = source[old_layout.embedding_key].detach().to(device="cpu", dtype=torch.float64)[text_rows].mean(0)
    weight = source[old_layout.output_weight_key][old_layout.blank_id].detach().to(device="cpu")
    blank_bias = source[old_layout.output_bias_key][old_layout.blank_id].detach().to(device="cpu", dtype=torch.float64)
    margin = math.log(len(additions)) - math.log(max_new_mass_ratio)
    desired_bias = blank_bias - margin
    bias = desired_bias.to(dtype=source[old_layout.output_bias_key].dtype)
    # Avoid rounding the negative bias shift toward a less conservative prior.
    if bias.double() > desired_bias:
        bias = torch.nextafter(bias, torch.full_like(bias, -math.inf))
    values = {old_layout.embedding_key: embedding, old_layout.output_weight_key: weight,
              old_layout.output_bias_key: bias}
    result = dict(initialized_target)
    for key, value in values.items():
        target = initialized_target[key].detach().clone()
        converted = value.to(device=target.device, dtype=target.dtype)
        if not torch.isfinite(converted).all():
            raise ValueError(f"Blank-anchored initialization is not finite in target dtype: {key}")
        target[additions] = converted
        result[key] = target
    applied_margin = float(blank_bias - bias.double())
    if applied_margin < margin:
        raise ValueError("Target bias dtype cannot represent the requested output margin")
    ratio_bound = len(additions) * math.exp(-applied_margin)
    return result, {
        "policy": "blank_anchored_bounded_new_output_mass_v1",
        "new_row_count": len(additions), "new_model_rows": additions,
        "old_output_rows_including_blank": old_layout.output_size,
        "embedding_mean_rows_excluding_blank": len(text_rows),
        "reference_old_blank_row": old_layout.blank_id,
        "output_weight_initialization": "exact copy of the old blank output weight row",
        "output_bias_initialization": "old blank output bias minus margin, rounded downward when needed",
        "embedding_initialization": "arithmetic mean of old nonblank embedding rows",
        "statistics_dtype": "float64", "target_dtype": str(source[old_layout.output_weight_key].dtype),
        "max_new_to_old_softmax_mass_ratio": max_new_mass_ratio,
        "requested_logit_margin": margin, "applied_bias_margin": applied_margin,
        "exact_arithmetic_mass_ratio_bound": ratio_bound,
        "exact_arithmetic_new_probability_bound": ratio_bound / (1 + ratio_bound),
        "derivation": "Z_new/Z_old = M*exp(-margin)*P_old(blank) <= M*exp(-margin)",
        "limitation": "Bounds describe exact arithmetic before training; floating-point and real decoding need validation",
        "trainable_independent_rows": True,
    }


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
    # A later extension may use our own saved model as its source. Register
    # the exact trusted class before NeMo validates that restoration target.
    get_nemo_model_class()
    original = ASRModel.restore_from(str(source), map_location="cpu")
    if original.__class__.__name__ not in {"EncDecRNNTBPEModelWithPrompt", "ExtendedNemotronRNNTModel"}:
        raise ValueError(f"Unsupported source model class: {original.__class__.__name__}")
    old_layout = inspect_nemo_layout(original)
    if old_layout.blank_id != old_map.model_blank_id:
        raise ValueError("Source blank/vocabulary does not match the pinned canonical base")
    _validate_source_tokens(original, old_map, base_tokenizer_json)
    native_decoder_proto = getattr(original.tokenizer, "native_decoder_model_proto", None)
    if native_decoder_proto is None:
        processor = getattr(original.tokenizer, "tokenizer", None)
        serialize = getattr(processor, "serialized_model_proto", None)
        if not callable(serialize):
            raise ValueError("Source tokenizer has no native decoder model to preserve")
        native_decoder_proto = serialize()
    if not isinstance(native_decoder_proto, bytes) or not native_decoder_proto:
        raise ValueError("Source native decoder model is empty or invalid")
    native_decoder_sha256 = hashlib.sha256(native_decoder_proto).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".untok-migration-", dir=output.parent) as staging:
        decoder_path = Path(staging) / "native-decoder.model"
        decoder_path.write_bytes(native_decoder_proto)
        cfg = OmegaConf.create(OmegaConf.to_container(original.cfg, resolve=True))
        with open_dict(cfg):
            cfg.tokenizer = {"type": "untok_hf_bpe", "hf_tokenizer_json": str(Path(tokenizer_json).resolve()),
                             "native_decoder_model": str(decoder_path)}
            cfg.target = "untok.runtime.ExtendedNemotronRNNTModel"
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
        if expanded.tokenizer.native_decoder_sha256 != native_decoder_sha256:
            raise ValueError("Native decoder artifact changed during model construction")
        new_layout = inspect_nemo_layout(expanded)
        if new_layout.blank_id != new_map.model_blank_id:
            raise ValueError("Constructed RNNT layout disagrees with tokenizer adapter")
        initialized, initialization = initialize_added_rows(
            original.state_dict(), expanded.state_dict(), old_layout, new_layout, old_to_new)
        added_rows = initialization["new_model_rows"]
        expected_added = {key: initialized[key][added_rows].detach().clone() for key in new_layout.row_keys}
        transferred = transfer_state_dict(original.state_dict(), initialized, old_layout, new_layout, old_to_new)
        del initialized
        expanded.load_state_dict(transferred, strict=True)
        del transferred
        expanded.eval()
        before_save = verify_state_transfer(original.state_dict(), expanded.state_dict(), old_layout, old_to_new)
        if any(not torch.equal(expanded.state_dict()[key][added_rows], value) for key, value in expected_added.items()):
            raise ValueError("New-row initialization changed while transferring model state")
        initialization["verified_before_save"] = True
        expected_adapter_map = expanded.tokenizer.id_map.to_dict()
        staged = Path(staging) / output.name
        expanded.save_to(str(staged))
        del expanded
        restored = get_nemo_model_class().restore_from(str(staged), map_location="cpu")
        if restored.tokenizer.native_decoder_sha256 != native_decoder_sha256:
            raise ValueError("Native decoder artifact changed after checkpoint restoration")
        after_reload = verify_state_transfer(original.state_dict(), restored.state_dict(), old_layout, old_to_new)
        if any(not torch.equal(restored.state_dict()[key][added_rows], value) for key, value in expected_added.items()):
            raise ValueError("New-row initialization changed after checkpoint restoration")
        initialization["verified_after_reload"] = True
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
        "new_row_initialization": initialization,
        "native_decoder_sha256": native_decoder_sha256,
        "native_decoder_verified_before_save_and_after_reload": True,
        "asr_accuracy_evaluated": False, "training_forward_evaluated": False,
        "logit_parity_evaluated": False,
        "remaining_gates": ["training_forward_backward", "old_output_logit_and_audio_parity",
                            "expanded_output_audio_regression", "new_language_fine_tuning_and_evaluation"]}
    with report_path.open("x") as stream:
        stream.write(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report
