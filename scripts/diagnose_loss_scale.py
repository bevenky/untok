#!/usr/bin/env python3
"""Read-only two-clip diagnosis of the expanded RNNT's initial loss scale.

Uses real pretrained/expanded models, original real train-split audio, native
Numba RNNT loss and unchanged FastEmit settings. No backward/optimizer/save.
Only the explicitly labelled old-output comparison masks added logits.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_report(path, report):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def summary(value):
    import torch
    value = value.detach().float()
    finite = torch.isfinite(value)
    values = value[finite]
    result = {"shape": list(value.shape), "count": value.numel(), "finite_count": int(finite.sum().item())}
    if values.numel():
        result.update({"minimum": float(values.min()), "maximum": float(values.max()),
                       "mean": float(values.mean()), "std": float(values.std(unbiased=False)),
                       "rms": float(values.square().mean().sqrt()), "absolute_maximum": float(values.abs().max())})
    return result


def compare(a, b, atol, rtol):
    import torch
    if a.shape != b.shape:
        return {"passed": False, "left_shape": list(a.shape), "right_shape": list(b.shape)}
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    return {"passed": finite and torch.allclose(a, b, atol=atol, rtol=rtol),
            "shape": list(a.shape), "maximum_absolute_difference": float((a - b).abs().max()),
            "atol": atol, "rtol": rtol}


def scalar_comparison(a, b, atol, rtol):
    import math
    return {"passed": math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * abs(a),
            "left": a, "right": b, "absolute_difference": abs(a - b), "atol": atol, "rtol": rtol}


def selected_rows(tensor, indices):
    import torch
    return tensor.index_select(0, torch.tensor(indices, device=tensor.device, dtype=torch.long))


def weight_statistics(original, expanded, old_layout, new_layout, old_to_new, new_rows):
    old = original.state_dict()
    new = expanded.state_dict()
    result = {}
    for label, old_key, new_key in (
            ("prediction_embedding", old_layout.embedding_key, new_layout.embedding_key),
            ("output_weight", old_layout.output_weight_key, new_layout.output_weight_key),
            ("output_bias", old_layout.output_bias_key, new_layout.output_bias_key)):
        if old_key is None or new_key is None:
            result[label] = None
            continue
        before = old[old_key]
        mapped = selected_rows(new[new_key], old_to_new)
        additions = selected_rows(new[new_key], new_rows)
        result[label] = {"source_key": old_key, "expanded_key": new_key,
            "source_all_original_rows_including_blank": summary(before),
            "expanded_mapped_original_rows_including_blank": summary(mapped),
            "expanded_new_rows": summary(additions),
            "old_rows_exactly_preserved": bool((before == mapped).all())}
        if before.ndim == 2:
            result[label]["source_row_l2_norms"] = summary(before.float().norm(dim=1))
            result[label]["new_row_l2_norms"] = summary(additions.float().norm(dim=1))
    return result


def loss_settings(model):
    from sttok.inference import _plain
    inner = model.loss._loss
    if type(inner).__name__ != "RNNTLossNumba":
        raise ValueError(f"Expected actual native RNNTLossNumba, received {type(inner).__name__}")
    return {"config": _plain(model.cfg).get("loss"), "outer_class": type(model.loss).__qualname__,
            "inner_class": type(inner).__qualname__, "reduction": model.loss.reduction,
            "blank": int(model.loss._blank), "fastemit_lambda": float(inner.fastemit_lambda),
            "clamp": float(inner.clamp), "fuse_loss_wer": bool(model.joint.fuse_loss_wer),
            "joint_uses_same_loss_object": model.joint.loss is model.loss,
            "joint_log_softmax": model.joint.log_softmax, "temperature": float(model.joint.temperature)}


def predictor(model, ids, device):
    import torch
    targets = torch.tensor([ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(ids)], dtype=torch.long, device=device)
    prediction, actual_lengths, _ = model.decoder(targets=targets, target_length=lengths)
    if actual_lengths.tolist() != lengths.tolist():
        raise ValueError("Prediction network changed transcript lengths")
    return targets, lengths, prediction


def raw_joint(model, encoded, encoded_lengths, predicted, target_lengths):
    # RNNTJoint.forward performs these same transposes/crops before its fused
    # path invokes .joint(). On CUDA, the pinned None/False setting returns raw
    # logits; the unchanged native loss applies its own normalization.
    if not encoded.is_cuda or model.joint.log_softmax not in (None, False):
        raise ValueError("This native CUDA diagnostic expects raw joint logits")
    if float(model.joint.temperature) != 1.0:
        raise ValueError("Non-default joint temperature needs a separately verified comparison")
    length = int(encoded_lengths.max())
    tokens = int(target_lengths.max())
    return model.joint.joint(encoded.transpose(1, 2)[:, :length], predicted.transpose(1, 2)[:, :tokens + 1])


def loss(model, logits, targets, encoded_lengths, target_lengths):
    import torch
    result = model.loss(log_probs=logits, targets=targets,
                        input_lengths=encoded_lengths, target_lengths=target_lengths)
    if result.numel() != 1 or not torch.isfinite(result).all():
        raise ValueError("Native RNNT loss is nonfinite or not scalar")
    return float(result)


def fused_loss(model, encoded, encoded_lengths, predicted, targets, target_lengths, row_map=None):
    from sttok.checkpoint import mask_new_outputs_for_test
    handle = None
    if row_map is not None:
        handle = model.joint.joint_net[-1].register_forward_hook(
            lambda module, arguments, output: mask_new_outputs_for_test(output, row_map))
    try:
        value, _, _, _ = model.joint(encoder_outputs=encoded, decoder_outputs=predicted,
            encoder_lengths=encoded_lengths, transcripts=targets,
            transcript_lengths=target_lengths, compute_wer=False)
        import torch
        if value.numel() != 1 or not torch.isfinite(value).all():
            raise ValueError("Fused native RNNT loss is nonfinite or not scalar")
        return float(value)
    finally:
        if handle is not None:
            handle.remove()


def logits_statistics(logits, row_map, new_rows):
    import torch
    old_index = torch.tensor(row_map, device=logits.device, dtype=torch.long)
    new_index = torch.tensor(new_rows, device=logits.device, dtype=torch.long)
    old = logits.index_select(-1, old_index)
    additions = logits.index_select(-1, new_index)
    all_log_z = logits.logsumexp(-1)
    new_mass = (additions.logsumexp(-1) - all_log_z).exp()
    best_old, best_new = old.max(-1).values, additions.max(-1).values
    return {"all_logits": summary(logits), "old_token_and_blank_logits": summary(old),
            "new_token_logits": summary(additions),
            "new_softmax_mass_over_teacher_forced_time_label_grid": summary(new_mass),
            "new_max_minus_old_max": summary(best_new - best_old),
            "fraction_grid_positions_new_class_beats_every_original_class": float((best_new > best_old).float().mean()),
            "scope": "computed teacher-forced joint grid, not decoded token frequency"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--base-tokenizer", type=Path, required=True)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output report path")
    import numpy as np
    import soundfile as sf
    import torch
    from sttok.inference import _load_model, _plain, _tokenizer_hash
    from sttok.checkpoint import (inspect_nemo_layout, old_model_row_mapping,
                                  mask_new_outputs_for_test, _validate_source_tokens)
    from sttok.runtime import build_id_map
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This diagnostic targets the same native CUDA loss as the real training probes")
    old_map = build_id_map(args.base_tokenizer)
    new_map = build_id_map(args.tokenizer, args.base_tokenizer)
    row_map = old_model_row_mapping(old_map, new_map)
    new_rows = [index for index, canonical in enumerate(new_map.model_to_canonical)
                if canonical >= len(old_map.canonical_to_model)]
    original, expanded = _load_model(args.source, "cpu"), _load_model(args.expanded, "cpu")
    _validate_source_tokens(original, old_map, args.base_tokenizer)
    if _tokenizer_hash(expanded) != new_map.tokenizer_sha256:
        raise ValueError("Expanded checkpoint tokenizer disagrees with the supplied artifact")
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(expanded)
    stats = weight_statistics(original, expanded, old_layout, new_layout, row_map, new_rows)
    old_loss_settings, new_loss_settings = loss_settings(original), loss_settings(expanded)
    for key in ("config", "inner_class", "reduction", "fastemit_lambda", "clamp"):
        if old_loss_settings[key] != new_loss_settings[key]:
            raise ValueError(f"Source/expanded native loss settings disagree: {key}")
    report = {"schema_version": 1, "status": "running", "passed": False,
        "scope": "two real clips diagnosing native RNNT loss and vocabulary additions",
        "release_ready": False, "accuracy_evaluated": False, "optimizer_steps": 0, "backward_calls": 0,
        "checkpoint_saved": False, "device": args.device, "dtype": "float32", "training_mode": False,
        "dropout_and_spec_augmentation_disabled_by_eval": True, "seed": args.seed,
        "source_checkpoint_sha256": sha_file(args.source), "expanded_checkpoint_sha256": sha_file(args.expanded),
        "base_tokenizer_sha256": old_map.tokenizer_sha256, "expanded_tokenizer_sha256": new_map.tokenizer_sha256,
        "source_runtime_tokenizer_sha256": _tokenizer_hash(original),
        "source_native_loss": old_loss_settings, "expanded_native_loss": new_loss_settings,
        "weight_statistics": stats, "torch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "clips": [], "interpretation": "Loss magnitude and gradient finiteness do not establish ASR quality"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_report(args.output, report)
    print(json.dumps({"weight_statistics": stats, "source_native_loss": old_loss_settings,
                      "expanded_native_loss": new_loss_settings}), flush=True)
    original.to(args.device).float().eval()
    expanded.to(args.device).float().eval()
    records = [json.loads(line) for line in (args.corpus_dir / "training-smoke-clips.jsonl").read_text().splitlines()]
    try:
        with torch.inference_mode():
            for language in ("en", "hi"):
                row = next(item for item in records if item["language"] == language)
                if row["source_split"] != "train":
                    raise ValueError("Only declared train-split development audio is permitted")
                audio = (args.corpus_dir / row["audio"]).resolve()
                if not audio.is_relative_to(args.corpus_dir.resolve()) or sha_file(audio) != row["audio_sha256"]:
                    raise ValueError("Development audio path/hash mismatch")
                samples, sample_rate = sf.read(audio, dtype="float32", always_2d=True)
                if sample_rate != 16000 or samples.shape != (row["num_samples"], 1) or not np.isfinite(samples).all():
                    raise ValueError("Original audio shape/rate/finiteness disagrees with the pinned corpus")
                signal = torch.from_numpy(samples[:, 0].copy()).unsqueeze(0).to(args.device)
                signal_lengths = torch.tensor([signal.shape[1]], device=args.device, dtype=torch.long)
                target_lang = row["target_lang"]
                source_registry = _plain(original.cfg)["model_defaults"]["prompt_dictionary"]
                expanded_registry = _plain(expanded.cfg)["model_defaults"]["prompt_dictionary"]
                if source_registry[target_lang] != expanded_registry[target_lang]:
                    raise ValueError("Existing prompt assignment changed")
                prompt = torch.tensor([source_registry[target_lang]], device=args.device, dtype=torch.long)
                torch.manual_seed(args.seed)
                encoded, encoded_lengths = original.forward(input_signal=signal, input_signal_length=signal_lengths, prompt_indices=prompt)
                torch.manual_seed(args.seed)
                new_encoded, new_encoded_lengths = expanded.forward(input_signal=signal, input_signal_length=signal_lengths, prompt_indices=prompt)
                if encoded_lengths.tolist() != new_encoded_lengths.tolist():
                    raise ValueError("Actual encoded lengths changed")
                native_ids = original.tokenizer.text_to_ids(row["text"])
                mapped_ids = [row_map[index] for index in native_ids]
                hf_ids = expanded.tokenizer.text_to_ids(row["text"])
                if old_map.model_blank_id in native_ids or new_map.model_blank_id in hf_ids:
                    raise ValueError("Blank token appeared in actual transcript labels")
                native_targets, native_lengths, predicted = predictor(original, native_ids, args.device)
                mapped_targets, mapped_lengths, new_predicted = predictor(expanded, mapped_ids, args.device)
                hf_targets, hf_lengths, hf_predicted = predictor(expanded, hf_ids, args.device)
                source_logits = raw_joint(original, encoded, encoded_lengths, predicted, native_lengths)
                expanded_old_prefix_logits = raw_joint(expanded, new_encoded, new_encoded_lengths, new_predicted, mapped_lengths)
                mapped_index = torch.tensor(row_map, device=args.device, dtype=torch.long)
                old_logit_check = compare(source_logits, expanded_old_prefix_logits.index_select(-1, mapped_index), args.atol, args.rtol)
                original_cost = loss(original, source_logits, native_targets, encoded_lengths, native_lengths)
                masked = mask_new_outputs_for_test(expanded_old_prefix_logits, row_map)
                masked_cost = loss(expanded, masked, mapped_targets, new_encoded_lengths, mapped_lengths)
                active_old_cost = loss(expanded, expanded_old_prefix_logits, mapped_targets, new_encoded_lengths, mapped_lengths)
                source_fused_cost = fused_loss(original, encoded, encoded_lengths, predicted, native_targets, native_lengths)
                masked_fused_cost = fused_loss(expanded, new_encoded, new_encoded_lengths, new_predicted, mapped_targets, mapped_lengths, row_map)
                active_old_fused_cost = fused_loss(expanded, new_encoded, new_encoded_lengths, new_predicted, mapped_targets, mapped_lengths)
                old_prefix_stats = logits_statistics(expanded_old_prefix_logits, row_map, new_rows)
                del source_logits, expanded_old_prefix_logits, masked
                hf_logits = raw_joint(expanded, new_encoded, new_encoded_lengths, hf_predicted, hf_lengths)
                hf_cost = loss(expanded, hf_logits, hf_targets, new_encoded_lengths, hf_lengths)
                hf_fused_cost = fused_loss(expanded, new_encoded, new_encoded_lengths, hf_predicted, hf_targets, hf_lengths)
                hf_stats = logits_statistics(hf_logits, row_map, new_rows)
                checks = {"encoder_and_existing_prompt_outputs": compare(encoded, new_encoded, args.atol, args.rtol),
                    "predictor_with_same_original_token_prefix": compare(predicted, new_predicted, args.atol, args.rtol),
                    "mapped_original_raw_logits": old_logit_check,
                    "original_vs_expanded_masked_native_label_loss": scalar_comparison(original_cost, masked_cost, args.atol, args.rtol),
                    "source_fused_vs_direct": scalar_comparison(original_cost, source_fused_cost, args.atol, args.rtol),
                    "expanded_masked_fused_vs_direct": scalar_comparison(masked_cost, masked_fused_cost, args.atol, args.rtol),
                    "expanded_active_old_labels_fused_vs_direct": scalar_comparison(active_old_cost, active_old_fused_cost, args.atol, args.rtol),
                    "expanded_hf_labels_fused_vs_direct": scalar_comparison(hf_cost, hf_fused_cost, args.atol, args.rtol)}
                item = {"id": row["id"], "language": language, "audio_sha256": row["audio_sha256"],
                    "source_reference": row["text"], "source_dataset": row["source_dataset"], "source_split": row["source_split"],
                    "target_lang": target_lang, "prompt_slot": int(prompt[0]), "audio_duration": row["duration"],
                    "audio_peak": float(np.abs(samples).max()), "audio_rms": float(np.sqrt(np.square(samples).mean())),
                    "encoded_length": int(encoded_lengths[0]), "native_label_length": len(native_ids),
                    "extended_hf_label_length": len(hf_ids), "native_model_label_ids": native_ids,
                    "mapped_original_model_label_ids": mapped_ids, "extended_hf_model_label_ids": hf_ids,
                    "new_label_occurrences": sum(new_map.model_to_canonical[index] >= len(old_map.canonical_to_model) for index in hf_ids),
                    "loss": {"source_native_labels": original_cost, "expanded_same_native_labels_new_outputs_masked": masked_cost,
                             "expanded_same_native_labels_all_outputs": active_old_cost, "expanded_hf_labels_all_outputs": hf_cost},
                    "expanded_old_prefix_logits": old_prefix_stats, "expanded_hf_prefix_logits": hf_stats,
                    "checks": checks, "passed": all(check["passed"] for check in checks.values())}
                report["clips"].append(item)
                write_report(args.output, report)
                print(json.dumps({"language": language, "loss": item["loss"], "checks": checks,
                    "old_prefix_new_softmax_mass": old_prefix_stats["new_softmax_mass_over_teacher_forced_time_label_grid"]}), flush=True)
                del hf_logits, encoded, new_encoded, predicted, new_predicted, hf_predicted
                gc.collect()
                torch.cuda.empty_cache()
        report["passed"] = all(item["passed"] for item in report["clips"])
        report["status"] = "loss_diagnostic_checks_passed" if report["passed"] else "loss_diagnostic_disagreement"
        write_report(args.output, report)
    except Exception as error:
        report.update({"status": "failed", "passed": False,
            "failure": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})
        write_report(args.output, report)
        raise


if __name__ == "__main__":
    main()
