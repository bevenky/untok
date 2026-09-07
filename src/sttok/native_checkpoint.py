"""Hash-pinned native Unigram RNNT migration, including explicit vocabulary subsets."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import operator
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

from .checkpoint import RNNTLayout, _sha256, _torch, initialize_added_rows, inspect_nemo_layout
from .native_runtime import get_native_nemo_model_class, native_bundle_config


def retained_row_pairs(old_layout: RNNTLayout, new_layout: RNNTLayout,
                       source_to_target: Sequence[int | None]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Validate a source-sized map; None means an explicitly removed text row."""
    if old_layout.row_keys != new_layout.row_keys:
        raise ValueError("Vocabulary-dependent parameter names changed")
    if (old_layout.output_size < 2 or new_layout.output_size < 2
            or old_layout.blank_id != old_layout.output_size - 1
            or new_layout.blank_id != new_layout.output_size - 1):
        raise ValueError("Expected dense native text rows followed by RNNT blank")
    if len(source_to_target) != old_layout.output_size:
        raise ValueError("Source row mapping must include every source row, including blank")
    source_ids, target_ids = [], []
    for old, new in enumerate(source_to_target):
        if new is None:
            continue
        try:
            if isinstance(new, bool):
                raise TypeError("Boolean row ID")
            new = operator.index(new)
        except TypeError as error:
            raise ValueError("Mapped row IDs must be integers or None") from error
        if not 0 <= new < new_layout.output_size:
            raise ValueError("Mapped row outside target vocabulary")
        source_ids.append(old)
        target_ids.append(new)
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Retained source rows must have one-to-one target IDs")
    if source_to_target[old_layout.blank_id] != new_layout.blank_id:
        raise ValueError("The original blank row must map to the target blank row")
    if len(source_ids) < 2:
        raise ValueError("At least one source text row and blank must be retained")
    return tuple(source_ids), tuple(target_ids)


def _validate_state_shapes(source, target, old_layout, new_layout):
    torch = _torch()
    if set(source) != set(target):
        raise ValueError("State keys differ across native migration")
    if old_layout.row_keys != new_layout.row_keys or not set(old_layout.row_keys) <= set(source):
        raise ValueError("Native vocabulary parameter layout changed or is incomplete")
    for key, old in source.items():
        new = target[key]
        if not isinstance(old, torch.Tensor) or not isinstance(new, torch.Tensor):
            raise ValueError(f"Unsupported non-tensor state: {key}")
        if old.dtype != new.dtype:
            raise ValueError(f"Unexpected dtype change: {key}")
        if key in old_layout.row_keys:
            if (old.ndim < 1 or new.ndim < 1 or old.shape[0] != old_layout.output_size
                    or new.shape[0] != new_layout.output_size or old.shape[1:] != new.shape[1:]):
                raise ValueError(f"Invalid vocabulary tensor shape: {key}")
        elif old.shape != new.shape:
            raise ValueError(f"Unapproved non-vocabulary shape change: {key}")


def initialize_native_added_rows(source, initialized_target, old_layout, new_layout, source_to_target,
                                 *, max_new_mass_ratio=1e-6):
    """Bound additions relative to retained logits, keeping blank as their anchor."""
    torch = _torch()
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, source_to_target)
    _validate_state_shapes(source, initialized_target, old_layout, new_layout)
    compact = dict(source)
    for key in old_layout.row_keys:
        compact[key] = source[key].index_select(0, torch.tensor(old_ids, device=source[key].device))
    compact_layout = replace(old_layout, blank_id=old_ids.index(old_layout.blank_id), output_size=len(old_ids))
    initialized, policy = initialize_added_rows(compact, initialized_target, compact_layout, new_layout,
                                                 new_ids, max_new_mass_ratio=max_new_mass_ratio)
    policy.update(source_output_rows_including_blank=old_layout.output_size,
                  retained_source_model_rows=list(old_ids), removed_source_model_rows=[i for i, v in enumerate(source_to_target) if v is None],
                  reference_source_blank_row=old_layout.blank_id,
                  probability_comparison="new outputs versus retained source outputs, including blank",
                  preservation_limit="Removing source outputs changes the softmax denominator and can change recognition")
    return initialized, policy


def transfer_native_state_dict(source, initialized_target, old_layout, new_layout, source_to_target):
    """Copy all shared state and precisely the retained vocabulary rows."""
    torch = _torch()
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, source_to_target)
    _validate_state_shapes(source, initialized_target, old_layout, new_layout)
    result = {}
    for key, old in source.items():
        target = initialized_target[key]
        if key in old_layout.row_keys:
            selected = old.index_select(0, torch.tensor(old_ids, device=old.device))
            copied = target.detach().clone()
            copied.index_copy_(0, torch.tensor(new_ids, device=copied.device), selected.to(copied.device))
            result[key] = copied
        else:
            result[key] = old.detach().to(target.device).clone()
    return result


def verify_native_state_transfer(source, migrated, old_layout, new_layout, source_to_target):
    """Independently compare retained source tensors, without claiming removed rows survived."""
    torch = _torch()
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, source_to_target)
    _validate_state_shapes(source, migrated, old_layout, new_layout)
    failures, preserved, omitted = [], 0, 0
    for key, source_tensor in source.items():
        expected, actual = source_tensor, migrated[key]
        if key in old_layout.row_keys:
            expected = expected.index_select(0, torch.tensor(old_ids, device=expected.device))
            actual = actual.index_select(0, torch.tensor(new_ids, device=actual.device))
            omitted += source_tensor.numel() - expected.numel()
        if not torch.equal(expected.cpu(), actual.cpu()):
            failures.append(key)
        else:
            preserved += expected.numel()
    if failures:
        raise ValueError("Retained learned state changed: " + ", ".join(failures))
    return {"passed": True, "comparison": "exact_tensor_equality", "tensors_checked": len(source),
            "learned_values_preserved": preserved, "learned_values_omitted": omitted,
            "retained_source_rows_including_blank": len(old_ids),
            "removed_source_text_rows": old_layout.output_size - len(old_ids),
            "all_source_values_preserved": omitted == 0}


def compare_retained_logits(source_logits, target_logits, old_layout, new_layout, source_to_target,
                            *, atol=1e-6, rtol=1e-5):
    torch = _torch()
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, source_to_target)
    if source_logits.shape[-1] != old_layout.output_size or target_logits.shape[-1] != new_layout.output_size:
        raise ValueError("Logit widths disagree with their native layouts")
    left = source_logits.index_select(-1, torch.tensor(old_ids, device=source_logits.device))
    right = target_logits.index_select(-1, torch.tensor(new_ids, device=target_logits.device))
    left = left.to(right.device)
    if left.shape != right.shape or not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("Invalid retained logit shapes or non-finite values")
    error = (left - right).abs().max().item() if left.numel() else 0.0
    if not torch.allclose(left, right, atol=atol, rtol=rtol):
        raise ValueError(f"Retained native logits changed (maximum absolute error {error})")
    return {"passed": True, "max_absolute_error": error, "atol": atol, "rtol": rtol,
            "scope": "retained raw logits for supplied identical features and mapped retained prefixes"}


def _source_native_bytes(model):
    tokenizer = model.tokenizer
    for candidate in (getattr(tokenizer, "backend", None), getattr(tokenizer, "tokenizer", None)):
        serialize = getattr(candidate, "serialized_model_proto", None)
        if callable(serialize):
            data = serialize()
            if isinstance(data, bytes) and data:
                return data
    raise ValueError("Source tokenizer must expose its actual native SentencePiece model")


def validate_source_native_tokenizer(model, base_bytes):
    import sentencepiece as spm

    actual = _source_native_bytes(model)
    if actual != base_bytes:
        raise ValueError("Restored source tokenizer differs from the bundle's pinned native base")
    processor = spm.SentencePieceProcessor(model_proto=base_bytes)
    if int(model.tokenizer.vocab_size) != processor.get_piece_size():
        raise ValueError("Source tokenizer vocabulary size disagrees with its native model")
    for index in range(processor.get_piece_size()):
        if model.tokenizer.ids_to_tokens([index]) != [processor.id_to_piece(index)]:
            raise ValueError(f"Source native token ID changed: {index}")
    probes = ["", "hello", " hello  world ", "नमस्ते", "അവന്‍", "a\u200cb", "Ａ ﬁ", "<unk>"]
    for text in probes:
        if model.tokenizer.text_to_ids(text) != processor.encode(text, out_type=int):
            raise ValueError("Source wrapper encoding disagrees with native SentencePiece")
    return {"native_tokenizer_sha256": hashlib.sha256(actual).hexdigest(),
            "piece_ids_checked": processor.get_piece_size(), "encoding_probes_checked": len(probes)}


def _prompt_registry(model_defaults, supplied=None):
    from .prompts import TARGET_LOCALES, extend_prompt_registry

    source = {"num_prompts": int(model_defaults.num_prompts),
              "prompt_dictionary": dict(model_defaults.prompt_dictionary)}
    targets = [{"language": language} for language in sorted(TARGET_LOCALES)]
    if supplied is None:
        return extend_prompt_registry(source, targets)
    registry = dict(supplied) if isinstance(supplied, Mapping) else json.loads(Path(supplied).read_text())
    # Revalidate the complete manifest using the source's actual prompt slots.
    checked = extend_prompt_registry(source, targets, previous_registry=registry)
    if checked["prompt_dictionary"] != registry.get("prompt_dictionary"):
        raise ValueError("Supplied prompt registry does not cover all target identities")
    return registry


def migrate_native_checkpoint(source, bundle, output, *, expected_source_sha256, seed=0,
                              prompt_registry=None, max_new_mass_ratio=1e-6):
    """Migrate a pinned original .nemo into a full or reduced native bundle.

    Reduced bundles intentionally omit selected original vocabulary rows. The
    original acoustic network and every retained row remain byte-equal, while
    segmentation, softmaxes and speech accuracy still require separate checks.
    """
    from .bundles import load_tokenizer_bundle

    torch = _torch()
    source, bundle, output = Path(source), Path(bundle).resolve(), Path(output)
    if not source.is_file() or source.suffix != ".nemo":
        raise ValueError("Provide the complete local original .nemo checkpoint")
    if (not isinstance(expected_source_sha256, str) or len(expected_source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in expected_source_sha256)
            or _sha256(source) != expected_source_sha256):
        raise ValueError("Source checkpoint SHA256 differs from its explicit pin")
    report_path = output.with_suffix(".migration.json")
    if output.suffix != ".nemo" or output.exists() or output.resolve() == source.resolve() or report_path.exists():
        raise ValueError("Checkpoint and migration report require new output paths")
    adapter = load_tokenizer_bundle(bundle)
    tokenizer_cfg = native_bundle_config(bundle)
    source_to_target = tuple(adapter.source_native_to_target_native)
    target_sha = hashlib.sha256(adapter.model_bytes).hexdigest()
    base_sha = hashlib.sha256(adapter.base_model_bytes).hexdigest()
    mapping = adapter.id_map.to_dict()
    from nemo.collections.asr.models import ASRModel
    from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt
    from omegaconf import OmegaConf, open_dict

    native_class = get_native_nemo_model_class()
    original = ASRModel.restore_from(str(source), map_location="cpu")
    if type(original) not in {EncDecRNNTBPEModelWithPrompt, native_class}:
        raise ValueError("Unsupported source model class for native migration")
    source_tokenizer_check = validate_source_native_tokenizer(original, adapter.base_model_bytes)
    old_layout = inspect_nemo_layout(original)
    if old_layout.output_size != len(source_to_target):
        raise ValueError("Source acoustic vocabulary disagrees with the bundle's original native inventory")
    cfg = OmegaConf.create(OmegaConf.to_container(original.cfg, resolve=True))
    registry = _prompt_registry(cfg.model_defaults, prompt_registry)
    with open_dict(cfg):
        cfg.tokenizer = tokenizer_cfg
        cfg.target = "sttok.native_runtime.NativeNemotronRNNTModel"
        cfg.model_defaults.prompt_dictionary = registry["prompt_dictionary"]
        cfg.sttok_native_migration = {"source_checkpoint_sha256": expected_source_sha256,
                                      "base_tokenizer_sha256": base_sha,
                                      "tokenizer_sha256": target_sha,
                                      "prompt_registry": registry}
        for split in ("train_ds", "validation_ds", "test_ds"):
            cfg[split] = None
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sttok-native-migration-", dir=output.parent) as staging:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            expanded = native_class(cfg=cfg, trainer=None)
        new_layout = inspect_nemo_layout(expanded)
        retained_row_pairs(old_layout, new_layout, source_to_target)
        if expanded.tokenizer.model_bytes != adapter.model_bytes or new_layout.blank_id != adapter.blank_id:
            raise ValueError("Constructed model disagrees with the target native tokenizer")
        initialized, initialization = initialize_native_added_rows(
            original.state_dict(), expanded.state_dict(), old_layout, new_layout, source_to_target,
            max_new_mass_ratio=max_new_mass_ratio)
        added = initialization["new_model_rows"]
        expected_added = {key: initialized[key][added].detach().clone() for key in new_layout.row_keys}
        transferred = transfer_native_state_dict(original.state_dict(), initialized, old_layout, new_layout, source_to_target)
        del initialized
        expanded.load_state_dict(transferred, strict=True)
        del transferred
        expanded.eval()
        before = verify_native_state_transfer(original.state_dict(), expanded.state_dict(), old_layout, new_layout, source_to_target)
        if any(not torch.equal(expanded.state_dict()[key][added], value) for key, value in expected_added.items()):
            raise ValueError("Added native row initialization changed during transfer")
        initialization["verified_before_save"] = True
        staged = Path(staging) / output.name
        expanded.save_to(str(staged))
        del expanded
        restored = native_class.restore_from(str(staged), map_location="cpu")
        if inspect_nemo_layout(restored) != new_layout:
            raise ValueError("Native acoustic layout changed after checkpoint reload")
        after = verify_native_state_transfer(original.state_dict(), restored.state_dict(), old_layout, new_layout, source_to_target)
        if (restored.tokenizer.model_bytes != adapter.model_bytes
                or restored.tokenizer.base_model_bytes != adapter.base_model_bytes
                or restored.tokenizer.id_map.to_dict() != mapping
                or tuple(restored.tokenizer.source_native_to_target_native) != source_to_target
                or restored.native_bundle_manifest_sha256 != tokenizer_cfg["bundle_manifest_sha256"]):
            raise ValueError("Native tokenizer artifacts or row mapping changed after checkpoint reload")
        if dict(restored.cfg.model_defaults.prompt_dictionary) != registry["prompt_dictionary"]:
            raise ValueError("Prompt assignments changed after native checkpoint reload")
        saved_metadata = OmegaConf.to_container(restored.cfg.sttok_native_migration, resolve=True)
        if saved_metadata["prompt_registry"] != registry:
            raise ValueError("Prompt registry provenance changed after checkpoint reload")
        if any(not torch.equal(restored.state_dict()[key][added], value) for key, value in expected_added.items()):
            raise ValueError("Added native rows changed after checkpoint reload")
        initialization["verified_after_reload"] = True
        if _sha256(source) != expected_source_sha256 or native_bundle_config(bundle) != tokenizer_cfg:
            raise ValueError("Pinned migration inputs changed during execution")
        report = {"schema_version": 1, "status": "native_migrated_weights_verified",
                  "source_checkpoint_sha256": expected_source_sha256, "checkpoint_sha256": _sha256(staged),
                  "tokenizer_sha256": target_sha, "base_tokenizer_sha256": base_sha,
                  "bundle_manifest_sha256": tokenizer_cfg["bundle_manifest_sha256"],
                  "seed": seed, "old_layout": asdict(old_layout), "new_layout": asdict(new_layout),
                  "id_mapping": mapping, "old_model_to_new_model": list(source_to_target),
                  "source_tokenizer_check": source_tokenizer_check,
                  "before_save": before, "after_reload": after,
                  "new_row_initialization": initialization, "prompt_registry": registry,
                  "native_tokenizer_verified_before_save_and_after_reload": True,
                  "asr_accuracy_evaluated": False, "training_forward_evaluated": False,
                  "logit_parity_evaluated": False,
                  "remaining_gates": ["training_forward_backward", "retained_logit_and_audio_controls",
                                      "unmasked_audio_regression", "new_language_fine_tuning_and_evaluation"]}
        staged_report = Path(staging) / report_path.name
        staged_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        os.link(staged, output)
        try:
            os.link(staged_report, report_path)
        except OSError:
            # Remove only our just-created hard link if report publication fails.
            if os.path.samestat(output.stat(), staged.stat()):
                output.unlink()
            raise
    return report
