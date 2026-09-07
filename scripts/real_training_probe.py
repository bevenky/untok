#!/usr/bin/env python3
"""Run actual NeMo dataloader + RNNT backward probes on a tiny real corpus.

There is no optimizer, synthetic audio, loss replacement, or training update.
The ordinary untok smoke function restores a fresh checkpoint for each clip.
Read-only hooks additionally verify actual language prompt inputs/gradients.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import time
import traceback


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def make_batch(row, audio, adapter, prompt_registry, num_prompts):
    from lhotse import CutSet, Recording, SupervisionSegment
    from nemo.collections.asr.data.audio_to_text_lhotse_prompt_index import LhotseSpeechToTextBpeDatasetWithPromptIndex

    recording = Recording.from_file(str(audio), recording_id=row["id"])
    if recording.sampling_rate != 16000 or recording.num_channels != 1:
        raise ValueError("Real training probe requires original mono 16 kHz audio")
    if recording.num_samples != row["num_samples"]:
        raise ValueError("Lhotse audio sample count differs from frozen corpus metadata")
    cut = recording.to_cut()
    cut.supervisions = [SupervisionSegment(
        id=row["id"], recording_id=recording.id, start=0.0,
        duration=recording.duration, channel=0, text=row["text"], language=row["target_lang"],
    )]
    cut.custom = {"prompt_mode": "langID"}
    dataset = LhotseSpeechToTextBpeDatasetWithPromptIndex(adapter, {
        "prompt_dictionary": prompt_registry, "num_prompts": num_prompts,
        "prompt_mode_field": "prompt_mode", "default_prompt_mode": "langID",
        "unified_auto_ratio": 0.0,
    })
    batch = dataset[CutSet.from_cuts([cut])]
    if len(batch) != 5 or batch[0].shape[0] != 1:
        raise ValueError("The actual NeMo dataloader did not return the five-tensor single-clip batch")
    if int(batch[1][0]) != recording.num_samples:
        raise ValueError("Dataloader dropped or changed the selected original audio")
    if int(batch[4][0]) != prompt_registry[row["target_lang"]]:
        raise ValueError("Dataloader prompt slot differs from requested known-language slot")
    if batch[2][0, :int(batch[3][0])].tolist() != adapter.text_to_ids(row["text"]):
        raise ValueError("Actual NeMo training labels differ from canonical adapter encoding")
    return batch


def run_clip_with_prompt_hooks(smoke, row, batch, args, checkpoint_sha, tokenizer_sha, registry, original_slots):
    import torch
    from untok.inference import _plain

    native_load = smoke._load_model
    handles = []
    capture = {"forward_calls": 0, "gradient_calls": 0, "target_lang": row["target_lang"],
               "prompt_slot": registry[row["target_lang"]],
               "slot_previously_unused": registry[row["target_lang"]] not in original_slots}

    def observed_load(path, device):
        # This wraps the real loader only to install observation hooks. It does
        # not alter weights, audio, loss, activations, gradients or model config.
        model = native_load(path, device)
        cfg = _plain(model.cfg)
        actual_registry = cfg.get("model_defaults", {}).get("prompt_dictionary", {})
        slot = capture["prompt_slot"]
        if actual_registry.get(row["target_lang"]) != slot:
            raise ValueError("Restored checkpoint prompt registry differs from frozen assignments")
        count = int(cfg["model_defaults"]["num_prompts"])
        if not getattr(model, "concat", False) or not hasattr(model, "prompt_kernel"):
            raise ValueError("Expected the native concatenated language prompt kernel")
        kernel = model.prompt_kernel
        # The pinned PromptStreamingMixin uses Linear(D+P, 2D), ReLU,
        # Linear(2D, D). Its FIRST Linear directly receives the one-hot input.
        feature_width = int(cfg["model_defaults"]["enc_hidden"])
        if (not isinstance(kernel, torch.nn.Sequential) or len(kernel) != 3
                or not isinstance(kernel[0], torch.nn.Linear)
                or not isinstance(kernel[1], torch.nn.ReLU)
                or not isinstance(kernel[2], torch.nn.Linear)
                or kernel[0].in_features != feature_width + count
                or kernel[0].out_features != 2 * feature_width
                or kernel[2].in_features != 2 * feature_width
                or kernel[2].out_features != feature_width):
            raise ValueError("Unrecognized prompt kernel layout; cannot infer gradient column")
        projection = kernel[0]
        column = feature_width + slot
        capture.update({"prompt_kernel_type": type(kernel).__qualname__,
                        "observed_parameter": "prompt_kernel.0.weight",
                        "weight_shape": list(projection.weight.shape), "prompt_column": column,
                        "num_prompts": count, "encoded_feature_width": feature_width})

        def inspect_prompt_input(module, inputs):
            value = inputs[0]
            if value.ndim != 3 or value.shape[-1] != projection.in_features:
                raise ValueError("Unexpected actual prompt-kernel input")
            expected = torch.zeros_like(value[..., -count:])
            expected[..., slot] = 1.0
            if not torch.equal(value[..., -count:], expected):
                raise ValueError("Actual prompt vector is not exactly the requested one-hot language slot")
            capture["forward_calls"] += 1

        def inspect_prompt_gradient(gradient):
            prompt_gradient = gradient[:, -count:]
            if not torch.isfinite(prompt_gradient).all():
                raise ValueError("Non-finite gradients reached the real prompt kernel")
            magnitude = float(prompt_gradient[:, slot].abs().sum().detach().cpu())
            if magnitude <= 0.0:
                raise ValueError("The exercised language prompt column received no gradient")
            inactive = torch.cat((prompt_gradient[:, :slot], prompt_gradient[:, slot + 1:]), dim=1)
            inactive_nonzero = int(torch.count_nonzero(inactive).detach().cpu())
            if inactive_nonzero:
                raise ValueError("A different prompt column received gradients for this one-hot single-clip batch")
            capture["gradient_calls"] += 1
            capture["prompt_gradient_absolute_sum"] = magnitude
            capture["inactive_prompt_nonzero_gradients"] = inactive_nonzero
            # Returning None leaves the genuine autograd result untouched.
            return None

        handles.append(projection.register_forward_pre_hook(inspect_prompt_input))
        handles.append(projection.weight.register_hook(inspect_prompt_gradient))
        return model

    smoke._load_model = observed_load
    try:
        result = smoke.run_training_smoke(
            args.checkpoint, batch, [row["text"]], [row["target_lang"]], args.base_tokenizer,
            checkpoint_sha256=checkpoint_sha, tokenizer_sha256=tokenizer_sha, device=args.device,
        )
        if capture["forward_calls"] < 1 or capture["gradient_calls"] < 1:
            raise ValueError("Real prompt observations did not execute during forward/backward")
        capture["passed"] = True
        result["prompt_gradient_check"] = capture
        return result
    finally:
        smoke._load_model = native_load
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--source-processor", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--languages", nargs="+", default=["hi", "ta", "ml", "mr", "kn", "pa", "or"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output report path; previous evidence is never overwritten")
    import torch
    import untok.training_smoke as smoke
    from untok.runtime import HFTokenizerAdapter

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable")
    adapter = HFTokenizerAdapter(args.tokenizer)
    prompt_manifest = json.loads(args.prompts.read_text())
    registry = prompt_manifest["prompt_dictionary"]
    num_prompts = int(prompt_manifest["num_prompts"])
    source_processor = json.loads(args.source_processor.read_text())
    original_slots = set(source_processor["prompt_dictionary"].values())
    candidates = []
    for directory in args.corpus_dir:
        source = directory / "training-smoke-clips.jsonl"
        for line in source.read_text().splitlines():
            row = json.loads(line)
            if row["source_split"] != "train":
                raise ValueError("Only actual training-split records may enter this training probe")
            audio = (directory / row["audio"]).resolve()
            if not audio.is_relative_to(directory.resolve()):
                raise ValueError("Audio path escapes its corpus directory")
            if sha_file(audio) != row["audio_sha256"]:
                raise ValueError("Audio hash disagrees with frozen development corpus")
            encoded = adapter.backend.encode(row["text"], add_special_tokens=False)
            if adapter.backend.token_to_id("<unk>") in encoded.ids:
                raise ValueError("Source transcription contains unknown tokens; no rewriting/filtering permitted")
            row["new_canonical_token_ids"] = sorted({x for x in encoded.ids if x >= 13089})
            candidates.append((row, audio))
    selected = []
    for language in args.languages:
        choices = [(row, audio) for row, audio in candidates if row["language"] == language]
        if not choices:
            raise ValueError(f"No real training record supplied for {language}")
        # One deterministic short clip per requested language; never inspect ASR
        # predictions or loss to select or replace a difficult example.
        row, audio = sorted(choices, key=lambda item: (item[0]["duration"], item[0]["id"]))[0]
        if not row["new_canonical_token_ids"]:
            raise ValueError(f"The selected {language} clip does not exercise tokenizer additions")
        selected.append((row, audio, make_batch(row, audio, adapter, registry, num_prompts)))
    checkpoint_sha, tokenizer_sha = sha_file(args.checkpoint), sha_file(args.tokenizer)
    report = {"schema_version": 1, "status": "running", "passed": False,
        "scope": "actual NeMo real-audio dataloader and RNNT backward development probes",
        "release_ready": False, "accuracy_evaluated": False, "optimizer_steps": 0,
        "checkpoint_saved": False, "checkpoint_sha256": checkpoint_sha,
        "tokenizer_sha256": tokenizer_sha, "base_tokenizer_sha256": sha_file(args.base_tokenizer),
        "prompt_manifest_sha256": sha_file(args.prompts), "device": args.device,
        "dtype": "float32", "seed": args.seed, "batch_size": 1,
        "prompt_mode": "langID", "expected_languages": args.languages,
        "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "dataset_class": "LhotseSpeechToTextBpeDatasetWithPromptIndex",
        "transcripts_modified": False, "probes": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, report)
    try:
        for row, audio, batch in selected:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
            started = time.monotonic()
            result = run_clip_with_prompt_hooks(smoke, row, batch, args, checkpoint_sha, tokenizer_sha, registry, original_slots)
            probe = {"id": row["id"], "language": row["language"], "script": row["script"],
                "target_lang": row["target_lang"], "audio": str(audio),
                "audio_sha256": row["audio_sha256"], "duration": row["duration"],
                "source_dataset": row["source_dataset"], "source_revision": row["source_revision"],
                "source_split": row["source_split"], "source_reference": row["text"],
                "new_canonical_token_ids": row["new_canonical_token_ids"],
                "elapsed_seconds": time.monotonic() - started, "result": result}
            report["probes"].append(probe)
            write_report(args.output, report)
            print(json.dumps({"language": row["language"], "loss": result["loss"],
                "new_rows": len(result["exercised_new_model_rows"]), "prompt_gradient": result["prompt_gradient_check"],
                "elapsed_seconds": probe["elapsed_seconds"]}), flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report["status"] = "real_rnnt_backward_probes_passed"
        report["passed"] = True
        write_report(args.output, report)
    except Exception as error:
        report.update({"status": "failed", "passed": False,
                       "failure": {"type": type(error).__name__, "message": str(error),
                                   "traceback": traceback.format_exc()}})
        write_report(args.output, report)
        raise


if __name__ == "__main__":
    main()
