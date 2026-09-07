#!/usr/bin/env python3
"""Bounded native Unigram checkpoint and speech compatibility checks.

Use source transcripts and recorded audio hashes. These development checks do
not establish multilingual accuracy or produce a release-trained checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load(path, device="cuda"):
    from nemo.collections.asr.models import ASRModel
    from sttok.native_runtime import get_native_nemo_model_class

    get_native_nemo_model_class()
    model = ASRModel.restore_from(str(path), map_location="cpu")
    return model.to(device).float().eval()


def rows(path, split=None):
    result = []
    seen = set()
    allowed_splits = None if split is None else {split} if isinstance(split, str) else set(split)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if allowed_splits is not None and row["source_split"] not in allowed_splits:
            continue
        if row["id"] in seen:
            raise ValueError("Duplicate audio record")
        seen.add(row["id"])
        audio = Path(row["audio"])
        if not audio.is_absolute():
            audio = path.parent / audio
        if sha(audio) != row["audio_sha256"]:
            raise ValueError("Audio content disagrees with manifest")
        row["audio"] = str(audio.resolve())
        result.append(row)
    if not result:
        raise ValueError("No real audio records supplied")
    return result


def row_provenance(row):
    keys = ("source_dataset", "source_revision", "source_split", "locale_audio_match", "locale_coverage_limitation")
    return {key: row[key] for key in keys if key in row}


def hypothesis(result):
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Expected one actual RNNT hypothesis")
    item = result[0]
    return {"text": item.text, "native_ids": item.y_sequence.detach().cpu().tolist()}


def streaming_source_evidence():
    import importlib
    import inspect
    from real_streaming_probe import NEMO_REVISION, SOURCE_HASHES, run_stream

    actual = {name: sha(inspect.getsourcefile(importlib.import_module(name))) for name in SOURCE_HASHES}
    if actual != SOURCE_HASHES:
        raise ValueError("Installed NeMo streaming sources differ from the inspected pin")
    return {"nemo_revision": NEMO_REVISION, "native_source_sha256": actual,
            "streaming_helper_sha256": sha(inspect.getsourcefile(run_stream))}


def streaming_parity(baseline, candidate, row_map):
    """Compare recorded encoder caches, chunks and mapped RNNT sequences."""
    headers = ("chunk_count", "initial_cache", "streaming_cfg", "effective_decoding_config", "runtime_decoder", "buffer_exhausted")
    fields = ("step", "chunk", "chunk_lengths", "drop_extra_pre_encoded", "last_chunk",
              "partial_hypotheses_supplied", "cache_input", "cache_output", "cache_lengths")
    if (not baseline["steps"] or len(baseline["steps"]) != len(candidate["steps"])
            or any(baseline[key] != candidate[key] for key in headers)):
        return False
    for left, right in zip(baseline["steps"], candidate["steps"]):
        if any(left[key] != right[key] for key in fields):
            return False
        ids = left["hypothesis"]["model_ids"]
        if any(i < 0 or i >= len(row_map) or row_map[i] is None for i in ids):
            return False
        if (left["hypothesis"]["text"] != right["hypothesis"]["text"]
                or [row_map[i] for i in ids] != right["hypothesis"]["model_ids"]):
            return False
    return True


def offline(args, report):
    import torch
    from sttok.checkpoint_validation import (
        _configure_eager_decoding, _equivalent_inference_config, _head_trace, _probe_checks,
    )
    from sttok.inference import _transcribe_with_verified_prompt

    migration = json.loads(args.migration.read_text())
    row_map = migration["old_model_to_new_model"]
    if any(value is None for value in row_map):
        raise ValueError("Full old-output parity requires a full-vocabulary migration")
    original, extended = load(args.source, args.device), load(args.checkpoint, args.device)
    config = _equivalent_inference_config(original, extended, "greedy_batch")
    for model in (original, extended):
        _configure_eager_decoding(model)
    report["config_section_sha256"] = config["config_section_sha256"]
    old_head, new_head = original.joint.joint_net[-1], extended.joint.joint_net[-1]
    report["utterances"] = []
    with torch.inference_mode():
        for row in rows(args.manifest, ("test", "valid")):
            if args.languages and row["language"] not in args.languages:
                continue
            if row["target_lang"] not in config["original_prompt_dictionary"]:
                raise ValueError("Paired source inference requires an original prompt identity")
            values = (Path(row["audio"]), row["audio_sha256"], row["target_lang"])
            with _head_trace(old_head) as source_trace:
                result, source_prompt = _transcribe_with_verified_prompt(original, *values)
                baseline = hypothesis(result)
            with _head_trace(new_head, old_to_new=row_map) as migrated_trace:
                result, migrated_prompt = _transcribe_with_verified_prompt(extended, *values)
                masked = hypothesis(result)
            probes = _probe_checks(source_trace, migrated_trace, new_head, row_map, atol=1e-6, rtol=1e-5)
            result, active_prompt = _transcribe_with_verified_prompt(extended, *values)
            active = hypothesis(result)
            mapped = [row_map[i] for i in baseline["native_ids"]]
            item = {"id": row["id"], "language": row["language"], "target_lang": row["target_lang"],
                    **row_provenance(row),
                    "audio_sha256": row["audio_sha256"], "reference": row["text"],
                    "duration": row["duration"], "baseline": baseline, "masked": masked, "active": active,
                    "masked_parity": baseline["text"] == masked["text"] and mapped == masked["native_ids"],
                    "active_parity": baseline["text"] == active["text"] and mapped == active["native_ids"],
                    "joint_probes": probes,
                    "prompt_evidence": [source_prompt, migrated_prompt, active_prompt]}
            report["utterances"].append(item)
            write(args.output, report)
            print(json.dumps({k: item[k] for k in ("id", "masked_parity", "active_parity")}), flush=True)
    report["languages"] = sorted({x["language"] for x in report["utterances"]})
    if not report["languages"] or (args.languages and set(args.languages) != set(report["languages"])):
        raise ValueError("Missing requested paired inference-language coverage")
    report["passed"] = all(x["masked_parity"] and x["active_parity"] and x["joint_probes"]["passed"]
                           for x in report["utterances"])


def inference(args, report):
    import torch
    from sttok.checkpoint_validation import _configure_eager_decoding
    from sttok.evaluation import edit_distance, normalize_for_scoring
    from sttok.inference import _transcribe_with_verified_prompt

    model = load(args.checkpoint, args.device)
    _configure_eager_decoding(model)
    report["utterances"] = []
    report["scoring"] = {"normalization": "collapse whitespace only; preserve case, punctuation and Unicode",
                         "purpose": "untrained-checkpoint diagnostic; no accuracy acceptance threshold"}
    with torch.inference_mode():
        for row in rows(args.manifest, ("test", "valid")):
            if args.languages and row["language"] not in args.languages:
                continue
            result, prompt = _transcribe_with_verified_prompt(model, Path(row["audio"]),
                                                              row["audio_sha256"], row["target_lang"])
            prediction = hypothesis(result)
            reference = normalize_for_scoring(row["text"], {})
            predicted = normalize_for_scoring(prediction["text"], {})
            if not reference:
                raise ValueError("Empty source reference")
            item = {"id": row["id"], "language": row["language"], "target_lang": row["target_lang"],
                    **row_provenance(row),
                    "audio_sha256": row["audio_sha256"], "reference": row["text"], "prediction": prediction,
                    "prompt_evidence": prompt, "word_errors": edit_distance(reference.split(), predicted.split()),
                    "reference_words": len(reference.split()), "character_errors": edit_distance(reference, predicted),
                    "reference_characters": len(reference)}
            report["utterances"].append(item)
            write(args.output, report)
            print(json.dumps({"id": row["id"], "inference_executed": True}), flush=True)
    report["languages"] = sorted({x["language"] for x in report["utterances"]})
    if not report["languages"] or (args.languages and set(args.languages) != set(report["languages"])):
        raise ValueError("Missing requested inference-language coverage")
    report["per_language"] = {}
    for language in report["languages"]:
        subset = [x for x in report["utterances"] if x["language"] == language]
        words = sum(x["reference_words"] for x in subset)
        characters = sum(x["reference_characters"] for x in subset)
        report["per_language"][language] = {
            "recordings": len(subset), "wer": sum(x["word_errors"] for x in subset) / words,
            "cer": sum(x["character_errors"] for x in subset) / characters}
    report["passed"] = True
    report["scope"] = "Verified native inference and prompt paths; passing does not mean good transcription accuracy"


def streaming(args, report):
    import torch
    from omegaconf import OmegaConf
    from real_streaming_probe import run_stream
    from sttok.inference import _load_audio_tensor, _plain
    from sttok.checkpoint_validation import _equivalent_inference_config, _probe_checks

    class IdentityMap:
        def to_canonical(self, ids, drop_blank=False):
            return list(ids)

    row_map = json.loads(args.migration.read_text())["old_model_to_new_model"]
    if any(value is None for value in row_map):
        raise ValueError("Full streaming parity requires all source rows")
    original, extended = load(args.source, args.device), load(args.checkpoint, args.device)
    report.update(streaming_source_evidence())
    config = _equivalent_inference_config(original, extended, "greedy_batch")
    report["config_section_sha256"] = config["config_section_sha256"]
    for model in (original, extended):
        model.encoder.set_default_att_context_size([56, 13])
        if hasattr(model.encoder, "set_streaming_cuda_graphs"):
            model.encoder.set_streaming_cuda_graphs(enabled=False)
        decoding = OmegaConf.create(_plain(model.cfg)["decoding"])
        decoding.strategy = "greedy_batch"
        decoding.fused_batch_size = -1
        decoding.greedy.use_cuda_graph_decoder = False
        model.change_decoding_strategy(decoding, verbose=False)
        model.decoding.set_strip_lang_tags(False)
    choices = {}
    for row in rows(args.manifest, ("test", "valid")):
        if args.languages and row["language"] not in args.languages:
            continue
        choices.setdefault(row["target_lang"], row)
    selected_languages = {row["language"] for row in choices.values()}
    if not choices or (args.languages and set(args.languages) != selected_languages):
        raise ValueError("Missing requested streaming-language coverage")
    report["utterances"] = []
    for locale, row in sorted(choices.items()):
        language = row["language"]
        if row["target_lang"] not in config["original_prompt_dictionary"]:
            raise ValueError("Paired full streaming requires an original prompt identity")
        waveform = _load_audio_tensor(Path(row["audio"]), row["audio_sha256"], 16000)
        for model in (original, extended):
            model.set_inference_prompt(row["target_lang"])
        baseline, old_trace = run_stream(original, waveform, IdentityMap(), row["target_lang"])
        masked, masked_trace = run_stream(extended, waveform, IdentityMap(), row["target_lang"], row_map)
        active, _ = run_stream(extended, waveform, IdentityMap(), row["target_lang"])
        probes = _probe_checks(old_trace, masked_trace, extended.joint.joint_net[-1], row_map, atol=1e-6, rtol=1e-5)
        item = {"id": row["id"], "language": language, "audio_sha256": row["audio_sha256"],
                "target_lang": row["target_lang"], **row_provenance(row),
                "baseline": baseline, "masked": masked, "active": active, "joint_probes": probes,
                "masked_parity": streaming_parity(baseline, masked, row_map),
                "active_parity": streaming_parity(baseline, active, row_map)}
        report["utterances"].append(item)
        write(args.output, report)
        print(json.dumps({"language": language, "chunks": baseline["chunk_count"],
                          "masked_parity": item["masked_parity"], "active_parity": item["active_parity"]}), flush=True)
    report["passed"] = all(x["masked_parity"] and x["active_parity"] and x["joint_probes"]["passed"]
                           for x in report["utterances"])
    report["languages"] = sorted(selected_languages)
    report["target_locales"] = sorted(choices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["offline", "streaming", "inference"])
    for name in ("source", "checkpoint", "migration", "manifest", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--languages", nargs="+", help="Require exactly these languages in the selected probe mode")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output path")
    import torch
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    migration = json.loads(args.migration.read_text())
    for key, path in (("source_checkpoint_sha256", args.source), ("checkpoint_sha256", args.checkpoint)):
        if migration.get(key) != sha(path):
            raise ValueError("Checkpoint differs from verified migration")
    report = {"schema_version": 1, "mode": args.mode, "status": "running", "passed": False,
              "release_ready": False, "asr_accuracy_established": False,
              "checkpoint_sha256": migration["checkpoint_sha256"],
              "source_checkpoint_sha256": migration["source_checkpoint_sha256"],
              "manifest_sha256": sha(args.manifest), "probe_sha256": sha(__file__),
              "migration_sha256": sha(args.migration), "torch_version": torch.__version__,
              "cuda_version": torch.version.cuda, "device": args.device, "dtype": "float32"}
    started = time.monotonic()
    write(args.output, report)
    try:
        {"offline": offline, "streaming": streaming, "inference": inference}[args.mode](args, report)
        report["status"] = "passed_on_supplied_audio" if report["passed"] else "failed"
    except Exception as exc:
        report.update(status="failed", passed=False, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        write(args.output, report)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
