"""Execute bounded migration checks on real local checkpoints and audio.

This is an offline greedy compatibility runner, not a multilingual ASR accuracy
benchmark. Tests inject small fake NeMo objects; production calls restore the
actual supplied checkpoints and never substitute generated predictions.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any

from .checkpoint import (
    _sha256, _torch, _validate_source_tokens, compare_old_logits,
    inspect_nemo_layout, mask_new_outputs_for_test, old_model_row_mapping,
    verify_state_transfer,
)
from .inference import _canonical_hash, _load_model, _plain, _tokenizer_hash
from .runtime import build_id_map


def _manifest(path: Path, device: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    document = json.loads(path.read_text())
    if document.get("schema_version") != 1 or not isinstance(document.get("run_id"), str) or not document["run_id"]:
        raise ValueError("Expected schema_version=1 and a nonempty run_id")
    required_hashes = (
        "source_checkpoint_sha256", "expanded_checkpoint_sha256", "base_tokenizer_sha256",
        "expanded_tokenizer_sha256", "source_runtime_tokenizer_sha256",
    )
    for key in required_hashes:
        value = document.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Missing or invalid compatibility metadata: {key}")
    settings = document.get("settings", {})
    expected = {"mode": "offline", "dtype": "float32", "device": device, "batch_size": 1, "num_workers": 0}
    if set(settings) != set(expected) | {"decoder"}:
        raise ValueError("Settings must explicitly declare only offline/dtype/device/batch_size/num_workers/decoder")
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Checkpoint validation requires settings[{key!r}]={value!r}")
    if settings["decoder"] not in {"greedy", "greedy_batch"}:
        raise ValueError("Only native RNNT greedy decoding is supported")
    requests, seen = [], set()
    for item in document.get("utterances", []):
        if not isinstance(item, dict):
            raise ValueError("Each utterance must be an object")
        identity, audio_name, prompt = item.get("id"), item.get("audio"), item.get("target_lang")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Utterances require unique nonempty IDs")
        if not isinstance(audio_name, str) or not audio_name or not isinstance(prompt, str) or not prompt:
            raise ValueError(f"Utterance {identity} requires audio and an explicit target_lang")
        audio = Path(audio_name)
        if not audio.is_absolute():
            audio = path.parent / audio
        audio = audio.resolve()
        if not audio.is_file() or _sha256(audio) != item.get("audio_sha256"):
            raise ValueError(f"Missing audio or mismatched audio_sha256 for {identity}")
        seen.add(identity)
        requests.append({"id": identity, "audio": audio, "audio_sha256": item["audio_sha256"], "target_lang": prompt})
    if not requests:
        raise ValueError("A nonempty corpus of real local audio is required")
    return document, requests


def _equivalent_inference_config(original: Any, expanded: Any, decoder: str) -> dict[str, Any]:
    old, new = _plain(original.cfg), _plain(expanded.cfg)
    for config in (old, new):
        if config.get("decoding", {}).get("strategy") != decoder:
            raise ValueError("Declared greedy strategy does not match the restored checkpoint")
        # Hooks cannot be assumed to execute during a cached CUDA graph replay.
        for group in (config.get("decoding", {}), config.get("decoding", {}).get("greedy", {})):
            for key, value in group.items():
                if ("cuda_graph" in key or "compile" in key) and value not in (False, None):
                    raise ValueError("Disable graph/compiled decoding explicitly before hook-based migration validation")
    excluded = {"decoder": {"vocab_size"}, "joint": {"num_classes", "vocabulary"},
                "model_defaults": {"prompt_dictionary"}}
    hashes = {}
    for section in ("encoder", "preprocessor", "decoder", "joint", "model_defaults", "decoding"):
        old_value, new_value = old.get(section, {}), new.get(section, {})
        if isinstance(old_value, dict) and isinstance(new_value, dict):
            old_value = {k: v for k, v in old_value.items() if k not in excluded.get(section, set())}
            new_value = {k: v for k, v in new_value.items() if k not in excluded.get(section, set())}
        if old_value != new_value:
            raise ValueError(f"Inference configuration changed outside permitted vocabulary fields: {section}")
        hashes[section] = _canonical_hash(old_value)
    old_prompts = old.get("model_defaults", {}).get("prompt_dictionary", {})
    new_prompts = new.get("model_defaults", {}).get("prompt_dictionary", {})
    if not old_prompts or any(new_prompts.get(k) != v for k, v in old_prompts.items()):
        raise ValueError("Original prompt identities were not preserved")
    return {"config_section_sha256": hashes, "original_prompt_dictionary": old_prompts,
            "expanded_prompt_dictionary": new_prompts}


def _hypothesis(result: Any, mapping) -> dict[str, Any]:
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Expected one RNNT Hypothesis for one audio input")
    item = result[0]
    text, ids = getattr(item, "text", None), getattr(item, "y_sequence", None)
    if not isinstance(text, str) or ids is None:
        raise ValueError("RNNT Hypothesis must expose text and y_sequence; no text-only fallback")
    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().tolist()
    elif hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, (tuple, list)) or any(not isinstance(i, Integral) for i in ids):
        raise ValueError("RNNT y_sequence must be a one-dimensional integer sequence")
    ids = [int(i) for i in ids]
    return {"text": text, "model_ids": ids,
            "canonical_hf_ids": mapping.to_canonical(ids, drop_blank=False)}


@contextmanager
def _head_trace(head, *, old_to_new=None, max_probes: int = 8):
    """Capture actual pre-softmax head inputs; optionally mask before softmax."""
    torch = _torch()
    trace: dict[str, Any] = {"calls": 0, "probes": []}

    def hook(module, args, output):
        if len(args) != 1 or not isinstance(output, torch.Tensor) or not isinstance(args[0], torch.Tensor):
            raise ValueError("Unrecognized joint-head forward signature")
        trace["calls"] += 1
        if len(trace["probes"]) < max_probes:
            # Bound trace memory even if the runtime batches many encoder steps.
            inputs = args[0].detach().reshape(-1, args[0].shape[-1])[:8].cpu().clone()
            logits = output.detach().reshape(-1, output.shape[-1])[:8].cpu().clone()
            if inputs.numel():
                trace["probes"].append((inputs, logits))
        return mask_new_outputs_for_test(output, old_to_new) if old_to_new is not None else None

    handle = head.register_forward_hook(hook)
    try:
        yield trace
    finally:
        handle.remove()


def _probe_checks(source_trace, expanded_trace, expanded_head, row_map, *, atol, rtol):
    torch = _torch()
    if not source_trace["calls"] or not expanded_trace["calls"] or not source_trace["probes"] or not expanded_trace["probes"]:
        raise ValueError("Joint-head hooks did not execute; no migration-logit check was performed")
    errors, input_errors, replay_errors = [], [], []
    shape_match = source_trace["calls"] == expanded_trace["calls"]
    shape_match = shape_match and len(source_trace["probes"]) == len(expanded_trace["probes"])
    if shape_match:
        for (old_inputs, old_logits), (new_inputs, new_logits) in zip(source_trace["probes"], expanded_trace["probes"]):
            if old_inputs.shape != new_inputs.shape:
                shape_match = False
                break
            input_errors.append((old_inputs - new_inputs).abs().max().item())
            if not torch.allclose(old_inputs, new_inputs, atol=atol, rtol=rtol):
                shape_match = False
            try:
                result = compare_old_logits(old_logits, new_logits, row_map, atol=atol, rtol=rtol)
                errors.append(result["max_absolute_error"])
            except ValueError:
                shape_match = False
    # Independently replay captured BASELINE joint inputs into the new final
    # head. This uses real audio-derived states and does not invent decoder
    # prefixes or assume undocumented NeMo encoder.forward interfaces.
    with torch.inference_mode():
        device = expanded_head.weight.device
        for inputs, old_logits in source_trace["probes"]:
            replay = expanded_head(inputs.to(device))
            result = compare_old_logits(old_logits, replay, row_map, atol=atol, rtol=rtol)
            replay_errors.append(result["max_absolute_error"])
    return {"passed": shape_match, "source_joint_calls": source_trace["calls"],
            "masked_joint_calls": expanded_trace["calls"], "captured_probes": len(source_trace["probes"]),
            "max_joint_input_absolute_error": max(input_errors, default=None),
            "max_old_logit_absolute_error": max(errors, default=None),
            "max_fixed_input_replay_absolute_error": max(replay_errors, default=None),
            "atol": atol, "rtol": rtol,
            "scope": "bounded_actual_greedy_joint_states_and_fixed_input_head_replay"}


def validate_checkpoint_pair(
    source_checkpoint: str | Path, expanded_checkpoint: str | Path,
    base_tokenizer_json: str | Path, extended_tokenizer_json: str | Path,
    manifest_path: str | Path, output_path: str | Path, *, device: str = "cpu",
    atol: float = 1e-6, rtol: float = 1e-5,
) -> dict[str, Any]:
    """Run old-state, masked decode and unmasked decode checks on supplied audio.

    Missing metadata/audio or incompatible runtime behavior raises before a
    passing report can be written. Numerical/transcript disagreement produces a
    failed compatibility report. Fine-tuned models normally fail exact state
    checks; use the separate ASR evaluator for post-training quality evaluation.
    """
    torch = _torch()
    source, expanded, manifest_path, output = map(Path, (source_checkpoint, expanded_checkpoint, manifest_path, output_path))
    if output.exists():
        raise ValueError("Validation output already exists; choose a new run path")
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Numerical tolerances must be finite and nonnegative")
    for checkpoint in (source, expanded):
        if not checkpoint.is_file() or checkpoint.suffix != ".nemo":
            raise ValueError("Both checkpoints must be complete local .nemo files")
    manifest, requests = _manifest(manifest_path, device)
    expected_files = {"source_checkpoint_sha256": source, "expanded_checkpoint_sha256": expanded,
                      "base_tokenizer_sha256": Path(base_tokenizer_json), "expanded_tokenizer_sha256": Path(extended_tokenizer_json)}
    for name, path in expected_files.items():
        if not path.is_file() or _sha256(path) != manifest[name]:
            raise ValueError(f"Artifact hash disagrees with compatibility metadata: {name}")
    old_map = build_id_map(base_tokenizer_json)
    new_map = build_id_map(extended_tokenizer_json, base_tokenizer_json)
    row_map = old_model_row_mapping(old_map, new_map)
    original = _load_model(source, "cpu")
    migrated = _load_model(expanded, "cpu")
    source_tokenizer_hash, new_tokenizer_hash = _tokenizer_hash(original), _tokenizer_hash(migrated)
    if source_tokenizer_hash != manifest["source_runtime_tokenizer_sha256"]:
        raise ValueError("Restored source tokenizer differs from declared native tokenizer")
    if new_tokenizer_hash != new_map.tokenizer_sha256:
        raise ValueError("Expanded checkpoint does not contain the supplied canonical tokenizer")
    runtime_map = getattr(getattr(migrated, "tokenizer", None), "id_map", None)
    if runtime_map is None or runtime_map.canonical_to_model != new_map.canonical_to_model:
        raise ValueError("Expanded runtime ID mapping differs from the canonical mapping")
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(migrated)
    if old_layout.blank_id != old_map.model_blank_id or new_layout.blank_id != new_map.model_blank_id:
        raise ValueError("Restored blank/embedding layout differs from tokenizer mappings")
    _validate_source_tokens(original, old_map, base_tokenizer_json)
    state_report = verify_state_transfer(original.state_dict(), migrated.state_dict(), old_layout, row_map)
    config = _equivalent_inference_config(original, migrated, manifest["settings"]["decoder"])
    for item in requests:
        prompt = item["target_lang"]
        if prompt not in config["original_prompt_dictionary"] or prompt not in config["expanded_prompt_dictionary"]:
            raise ValueError(f"Prompt {prompt!r} is unavailable for paired baseline inference; no automatic fallback")
    original.to(device).float().eval()
    migrated.to(device).float().eval()
    old_head, new_head = original.joint.joint_net[-1], migrated.joint.joint_net[-1]
    rows = []
    with torch.inference_mode():
        for item in requests:
            arguments = {"audio": [str(item["audio"])], "batch_size": 1, "num_workers": 0,
                         "return_hypotheses": True, "verbose": False, "target_lang": item["target_lang"]}
            with _head_trace(old_head) as old_trace:
                baseline = _hypothesis(original.transcribe(**arguments), old_map)
            with _head_trace(new_head, old_to_new=row_map) as new_trace:
                masked = _hypothesis(migrated.transcribe(**arguments), new_map)
            probes = _probe_checks(old_trace, new_trace, new_head, row_map, atol=atol, rtol=rtol)
            # Hooks were removed by the context manager. This is a genuinely
            # separate run with all newly added model outputs enabled.
            active = _hypothesis(migrated.transcribe(**arguments), new_map)
            same_tokens = baseline["canonical_hf_ids"] == masked["canonical_hf_ids"]
            same_text = baseline["text"] == masked["text"]
            rows.append({"utterance_id": item["id"], "audio_sha256": item["audio_sha256"],
                         "target_lang": item["target_lang"], "baseline": baseline,
                         "expanded_old_outputs_only": masked, "expanded_all_outputs": active,
                         "old_output_token_parity": same_tokens, "old_output_text_parity": same_text,
                         "joint_probes": probes,
                         "active_text_changed": baseline["text"] != active["text"],
                         "active_token_sequence_changed": baseline["canonical_hf_ids"] != active["canonical_hf_ids"]})
    passed = all(row["old_output_token_parity"] and row["old_output_text_parity"] and row["joint_probes"]["passed"] for row in rows)
    report = {"schema_version": 1, "run_id": manifest["run_id"],
              "status": "migration_compatibility_passed_on_supplied_corpus" if passed else "migration_compatibility_failed",
              "passed": passed, "release_ready": False, "state_transfer": state_report,
              "manifest_sha256": _sha256(manifest_path),
              "artifact_sha256": {name: manifest[name] for name in expected_files},
              "source_runtime_tokenizer_sha256": source_tokenizer_hash,
              "actual_settings": manifest["settings"], "config_section_sha256": config["config_section_sha256"],
              "source_model_class": type(original).__module__ + "." + type(original).__qualname__,
              "expanded_model_class": type(migrated).__module__ + "." + type(migrated).__qualname__,
              "utterance_count": len(rows), "active_text_change_count": sum(r["active_text_changed"] for r in rows),
              "utterances": rows, "streaming_validated": False, "rnnt_training_smoke_executed": False,
              "asr_accuracy_evaluated": False,
              "remaining_gates": ["RNNT forward/backward with new labels", "streaming regression",
                                  "per-language ASR accuracy on held-out speech", "new-language fine-tuning"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return report
