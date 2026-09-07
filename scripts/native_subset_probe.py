#!/usr/bin/env python3
"""Compare real reduced native checkpoints using explicitly restricted source outputs.

The source and reduced controls use identical retained rows and an identical
original prompt. New Indic prompts absent from the source use an explicit auto
control, followed by a separate requested-prompt run. This is a migration and
streaming execution check, not an ASR accuracy or language-support guarantee.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

from native_checkpoint_probe import (
    hypothesis, load, row_provenance, rows, sha, streaming_parity,
    streaming_source_evidence, write,
)


# Published native Nemotron locales using Latin script. Regional prompt paths
# remain separate even when their source evaluation audio uses the same locale.
LATIN_SOURCE_LANGUAGES = frozenset(
    "en es fr it pt nl de tr vi pl sv cs nb da fi hr sk hu ro et lt lv mt sl nn".split()
)


def choose_rows(records, profile, languages=None, max_per_language=1):
    from sttok.prompts import TARGET_LOCALES

    if profile not in {"latin", "latin-indic"}:
        raise ValueError("This probe requires a reduced Latin or Latin+Indic checkpoint")
    if max_per_language < 1:
        raise ValueError("max-per-locale must be positive")
    # Arabic is retained alongside Urdu/Kashmiri in the Latin+Indic bundle.
    # This selection tests its old prompt path without asserting new accuracy.
    allowed = LATIN_SOURCE_LANGUAGES if profile == "latin" else LATIN_SOURCE_LANGUAGES | set(TARGET_LOCALES) | {"ar"}
    requested = set(languages) if languages else {row["language"] for row in records} & allowed
    if not requested or not requested <= allowed:
        raise ValueError("Requested languages are outside this reduced tokenizer's scope")
    present = {row["language"] for row in records}
    missing = requested - present
    if missing:
        raise ValueError(f"Missing evaluation audio for {sorted(missing)}")
    selected = []
    eligible = [row for row in records if row["language"] in requested]
    for locale in sorted({row["target_lang"] for row in eligible}):
        matches = sorted((row for row in eligible if row["target_lang"] == locale),
                         key=lambda row: (row["duration"], row["id"]))
        selected.extend(matches[:max_per_language])
    return selected


def control_prompt(row, original_prompts, target_prompts):
    requested = row["target_lang"]
    if requested not in target_prompts:
        raise ValueError("Requested prompt is missing from the reduced checkpoint")
    control = requested if requested in original_prompts else "auto"
    if control not in original_prompts or target_prompts.get(control) != original_prompts[control]:
        raise ValueError("No identical original prompt is available for a paired subset control")
    return {"requested_target_lang": requested, "control_target_lang": control,
            "control_prompt_id": original_prompts[control], "requested_prompt_id": target_prompts[requested],
            "source_has_requested_prompt": requested in original_prompts,
            "control_policy": "same original requested prompt when present, otherwise explicit auto on both models"}


def mapped_hypothesis_equal(source, target, mapping):
    ids = source["native_ids"]
    if any(i < 0 or i >= len(mapping) or mapping[i] is None for i in ids):
        return False
    return source["text"] == target["text"] and [mapping[i] for i in ids] == target["native_ids"]


def retained_trace_checks(source_trace, target_trace, target_head, old_layout, new_layout, mapping,
                          *, atol=1e-6, rtol=1e-5):
    import torch
    from sttok.native_checkpoint import compare_retained_logits

    if (not source_trace["calls"] or not target_trace["calls"]
            or not source_trace["probes"] or not target_trace["probes"]):
        raise ValueError("Actual source and reduced joint-head hooks must both execute")
    matched = (source_trace["calls"] == target_trace["calls"]
               and len(source_trace["probes"]) == len(target_trace["probes"]))
    input_errors, logit_errors, replay_errors = [], [], []
    for (left, logits), (right, actual) in zip(source_trace["probes"], target_trace["probes"]):
        if left.shape != right.shape:
            matched = False
            continue
        input_errors.append(float((left - right).abs().max()))
        if not torch.allclose(left, right, atol=atol, rtol=rtol):
            matched = False
        try:
            comparison = compare_retained_logits(logits, actual, old_layout, new_layout, mapping, atol=atol, rtol=rtol)
            logit_errors.append(comparison["max_absolute_error"])
        except ValueError:
            matched = False
    # Replay only actual source audio states, with no decoder hook active.
    with torch.inference_mode():
        for states, logits in source_trace["probes"]:
            replay = target_head(states.to(target_head.weight.device))
            comparison = compare_retained_logits(logits, replay, old_layout, new_layout, mapping, atol=atol, rtol=rtol)
            replay_errors.append(comparison["max_absolute_error"])
    return {"passed": matched, "source_joint_calls": source_trace["calls"],
            "reduced_joint_calls": target_trace["calls"], "captured_probes": len(source_trace["probes"]),
            "max_joint_input_absolute_error": max(input_errors, default=None),
            "max_retained_logit_absolute_error": max(logit_errors, default=None),
            "max_fixed_input_replay_absolute_error": max(replay_errors, default=None),
            "atol": atol, "rtol": rtol,
            "scope": "bounded actual greedy states and fixed audio-state replay for retained source rows"}


def offline_row(original, reduced, row, prompt, old_layout, new_layout, mapping):
    from sttok.checkpoint_validation import _head_trace
    from sttok.inference import _transcribe_with_verified_prompt
    from sttok.native_checkpoint import retained_row_pairs

    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, mapping)
    arguments = (Path(row["audio"]), row["audio_sha256"], prompt["control_target_lang"])
    original_unrestricted, original_prompt = _transcribe_with_verified_prompt(original, *arguments)
    with _head_trace(original.joint.joint_net[-1], old_to_new=old_ids) as old_trace:
        value, old_prompt = _transcribe_with_verified_prompt(original, *arguments)
        source_control = hypothesis(value)
    with _head_trace(reduced.joint.joint_net[-1], old_to_new=new_ids) as new_trace:
        value, new_prompt = _transcribe_with_verified_prompt(reduced, *arguments)
        masked = hypothesis(value)
    probes = retained_trace_checks(old_trace, new_trace, reduced.joint.joint_net[-1], old_layout, new_layout, mapping)
    value, active_prompt = _transcribe_with_verified_prompt(reduced, *arguments)
    active_control = hypothesis(value)
    if prompt["requested_target_lang"] == prompt["control_target_lang"]:
        requested, requested_prompt = active_control, active_prompt
    else:
        value, requested_prompt = _transcribe_with_verified_prompt(
            reduced, Path(row["audio"]), row["audio_sha256"], prompt["requested_target_lang"])
        requested = hypothesis(value)
    return {"source_unrestricted": hypothesis(original_unrestricted),
            "source_retained_outputs_only": source_control, "reduced_retained_outputs_only": masked,
            "reduced_unmasked_control_prompt": active_control,
            "reduced_unmasked_requested_prompt": requested,
            "masked_retained_parity": mapped_hypothesis_equal(source_control, masked, mapping),
            "unmasked_same_prompt_parity": mapped_hypothesis_equal(source_control, active_control, mapping),
            "joint_probes": probes,
            "prompt_evidence": {"source_unrestricted": original_prompt, "source_restricted": old_prompt,
                                "reduced_masked": new_prompt, "reduced_active_control": active_prompt,
                                "reduced_active_requested": requested_prompt}}


def prepare_streaming_evaluation(model):
    """Restore every nested module to eval after NeMo transcription teardown.

    The pinned ASR mixin calls submodule.unfreeze(), which calls train() even
    when the parent model remains in eval. inference_mode alone does not disable
    dropout or BatchNorm updates. This boundary changes modes, never tensors.
    """
    def tensor_identity():
        return {name: (value.data_ptr(), value._version, tuple(value.shape), value.dtype)
                for name, value in model.state_dict().items()}

    before = [name for name, module in model.named_modules() if module.training]
    tensors = tensor_identity()
    model.eval()
    after = [name for name, module in model.named_modules() if module.training]
    if after or tensor_identity() != tensors:
        raise ValueError("Streaming eval setup failed or changed model tensors")
    return {"training_module_count_before": len(before), "training_modules_before": before,
            "training_module_count_after": 0, "model_tensor_identity_and_versions_unchanged": True,
            "policy": "explicit recursive eval after offline transcription and before every stream"}


def configure_streaming(models):
    from omegaconf import OmegaConf, open_dict
    from sttok.checkpoint_validation import _eager_decoder_state
    from sttok.inference import _plain

    evidence = []
    for model in models:
        if [56, 13] not in [list(value) for value in model.encoder.att_context_size_all]:
            raise ValueError("Checkpoint does not support the pinned streaming context")
        model.encoder.set_default_att_context_size([56, 13])
        if hasattr(model.encoder, "set_streaming_cuda_graphs"):
            model.encoder.set_streaming_cuda_graphs(enabled=False)
        decoding = OmegaConf.create(_plain(model.cfg)["decoding"])
        with open_dict(decoding):
            decoding.strategy = "greedy_batch"
            decoding.fused_batch_size = -1
            decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding, verbose=False)
        model.decoding.set_strip_lang_tags(False)
        _eager_decoder_state(model)
        evidence.append(prepare_streaming_evaluation(model))
    return evidence


def streaming_row(original, reduced, row, prompt, old_layout, new_layout, mapping):
    from real_streaming_probe import run_stream
    from sttok.inference import _load_audio_tensor
    from sttok.native_checkpoint import retained_row_pairs

    class IdentityMap:
        def to_canonical(self, ids, drop_blank=False):
            return list(ids)

    waveform = _load_audio_tensor(Path(row["audio"]), row["audio_sha256"], 16000)
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, mapping)
    control = prompt["control_target_lang"]
    for model in (original, reduced):
        model.set_inference_prompt(control)
    evaluation_state = {}

    def run(model, label, locale, row_ids=None):
        evaluation_state[label] = prepare_streaming_evaluation(model)
        result = run_stream(model, waveform, IdentityMap(), locale, row_ids)
        if any(module.training for module in model.modules()):
            raise ValueError("A streaming run left a module in training mode")
        return result

    baseline, old_trace = run(original, "source_restricted", control, old_ids)
    masked, new_trace = run(reduced, "reduced_masked", control, new_ids)
    active, _ = run(reduced, "reduced_active_control", control)
    probes = retained_trace_checks(old_trace, new_trace, reduced.joint.joint_net[-1], old_layout, new_layout, mapping)
    if prompt["requested_target_lang"] == control:
        requested = active
    else:
        reduced.set_inference_prompt(prompt["requested_target_lang"])
        requested, _ = run(reduced, "reduced_active_requested", prompt["requested_target_lang"])
    return {"source_retained_outputs_only": baseline, "reduced_retained_outputs_only": masked,
            "reduced_unmasked_control_prompt": active, "reduced_unmasked_requested_prompt": requested,
            "masked_retained_parity": streaming_parity(baseline, masked, mapping),
            "unmasked_same_prompt_parity": streaming_parity(baseline, active, mapping),
            "joint_probes": probes,
            "evaluation_state": evaluation_state,
            "scope": "actual encoder caches and carried RNNT hypotheses; not live frontend latency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "checkpoint", "migration", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--mode", choices=["offline", "streaming", "both"], default="both")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--max-per-locale", "--max-per-language", dest="max_per_locale", type=int, default=1,
                        help="Shortest clips per target locale; regional prompts remain separate")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new subset-probe output path")
    import torch
    import native_checkpoint_probe
    from sttok.checkpoint import inspect_nemo_layout
    from sttok.checkpoint_validation import _configure_eager_decoding, _equivalent_inference_config
    from sttok.native_checkpoint import retained_row_pairs, validate_source_native_tokenizer

    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    migration = json.loads(args.migration.read_text())
    if migration.get("status") != "native_migrated_weights_verified" or not migration.get("after_reload", {}).get("passed"):
        raise ValueError("A completed native migration verification report is required")
    for field, path in (("source_checkpoint_sha256", args.source), ("checkpoint_sha256", args.checkpoint)):
        if migration.get(field) != sha(path):
            raise ValueError("Checkpoint hash differs from its verified native migration")
    report = {"schema_version": 1, "status": "running", "passed": False, "mode": args.mode,
              "release_ready": False, "asr_accuracy_established": False,
              "source_checkpoint_sha256": migration["source_checkpoint_sha256"],
              "checkpoint_sha256": migration["checkpoint_sha256"], "migration_sha256": sha(args.migration),
              "manifest_sha256": sha(args.manifest), "probe_sha256": sha(__file__),
              "native_probe_helper_sha256": sha(native_checkpoint_probe.__file__),
              "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
              "device": args.device, "dtype": "float32", "utterances": []}
    started = time.monotonic()
    write(args.output, report)
    try:
        original, reduced = load(args.source, args.device), load(args.checkpoint, args.device)
        profile = getattr(reduced.tokenizer, "profile", None)
        selected = choose_rows(rows(args.manifest, ("test", "valid")), profile, args.languages, args.max_per_locale)
        mapping = tuple(reduced.tokenizer.source_native_to_target_native)
        if list(mapping) != migration["old_model_to_new_model"]:
            raise ValueError("Restored subset mapping differs from the migration report")
        old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(reduced)
        retained, targets = retained_row_pairs(old_layout, new_layout, mapping)
        if not any(value is None for value in mapping):
            raise ValueError("A reduced checkpoint must explicitly remove original source rows")
        report["source_native_check"] = validate_source_native_tokenizer(original, reduced.tokenizer.base_model_bytes)
        if (sha_bytes(reduced.tokenizer.model_bytes) != migration["tokenizer_sha256"]
                or json.loads(json.dumps(reduced.tokenizer.id_map.to_dict())) != migration["id_mapping"]):
            raise ValueError("Restored native tokenizer differs from migration metadata")
        config = _equivalent_inference_config(original, reduced, "greedy_batch")
        report.update(profile=profile, config_section_sha256=config["config_section_sha256"],
                      retained_source_rows_including_blank=len(retained), removed_source_text_rows=mapping.count(None),
                      added_target_rows=new_layout.output_size-len(targets),
                      native_blank_id=new_layout.blank_id, source_blank_id=old_layout.blank_id)
        for model in (original, reduced):
            _configure_eager_decoding(model)
        if args.mode in {"streaming", "both"}:
            report.update(streaming_source_evidence())
        modes = ("offline", "streaming") if args.mode == "both" else (args.mode,)
        with torch.inference_mode():
            for row in selected:
                prompt = control_prompt(row, config["original_prompt_dictionary"], config["expanded_prompt_dictionary"])
                item = {"id": row["id"], "language": row["language"], "target_lang": row["target_lang"],
                        "audio_sha256": row["audio_sha256"], "duration": row["duration"],
                        "reference": row["text"], **row_provenance(row), "prompt_policy": prompt}
                report["utterances"].append(item)
            # Finish offline first: changing encoder context for streaming must
            # never silently affect a later utterance's offline control.
            for mode in modes:
                if mode == "streaming":
                    report["streaming_setup_evaluation_state"] = configure_streaming((original, reduced))
                runner = offline_row if mode == "offline" else streaming_row
                for row, item in zip(selected, report["utterances"]):
                    item[mode] = runner(original, reduced, row, item["prompt_policy"], old_layout, new_layout, mapping)
                    item["passed"] = all(name in item and item[name]["masked_retained_parity"]
                                         and item[name]["unmasked_same_prompt_parity"] and item[name]["joint_probes"]["passed"]
                                         for name in modes)
                    write(args.output, report)
                    print(json.dumps({"id": row["id"], "language": row["language"], "mode": mode,
                                      "passed": item["passed"]}), flush=True)
        report["languages"] = sorted({item["language"] for item in report["utterances"]})
        report["target_locales"] = sorted({item["target_lang"] for item in report["utterances"]})
        report["selection_policy"] = "Shortest clips per target locale after filtering retained source-audio languages"
        report["passed"] = bool(report["utterances"]) and all(item["passed"] for item in report["utterances"])
        report["status"] = "passed_on_supplied_audio" if report["passed"] else "failed"
        report["scope"] = "Restricted source versus reduced native migration controls; requested-prompt inference is diagnostic only"
    except Exception as error:
        report.update(status="failed", passed=False, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write(args.output, report)
    return 0 if report["passed"] else 2


def sha_bytes(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
