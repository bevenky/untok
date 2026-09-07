#!/usr/bin/env python3
"""One real cache-aware streaming comparison using NVIDIA's pinned native API.

Follows perform_streaming in speech_to_text_cache_aware_streaming_infer.py at
NeMo ca4daa1470f6c01068c4e6a9a73b19b9a91dc366. This is streaming simulation:
the official buffer precomputes features, then executes actual cached encoder
chunks and carries RNNT hypotheses. It is not a live frontend or latency test.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace


NEMO_REVISION = "ca4daa1470f6c01068c4e6a9a73b19b9a91dc366"
SOURCE_HASHES = {
    "nemo.collections.asr.parts.mixins.mixins": "d4531808440c9f1649064044a21419975c0f2424e2442d4f2ef68a301d576c96",
    "nemo.collections.asr.parts.utils.streaming_utils": "e90bf75cb44106906c001bb52b039d3cbf370e8c33bb5c77ef800bf415864f15",
    "nemo.collections.asr.modules.conformer_encoder": "b0b9c7997d66a654b9d54d976ef154520ce76c65f26395c0a41cdd52432fb26d",
    "nemo.collections.asr.parts.submodules.rnnt_decoding": "7d10acca5d721d2a37a303b70c3c1b70fd5aed5cde9fbe68385e67541242cd5c",
}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def tensor_evidence(tensor):
    import torch
    if not isinstance(tensor, torch.Tensor) or not torch.isfinite(tensor).all():
        raise ValueError("Native streaming returned an invalid cache or chunk tensor")
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "nonzero_values": int(torch.count_nonzero(value))}


@contextmanager
def observe_prompt(model, target_lang):
    import torch
    from sttok.inference import _plain
    defaults = _plain(model.cfg)["model_defaults"]
    width, count = defaults["enc_hidden"], defaults["num_prompts"]
    prompt_id = defaults["prompt_dictionary"][target_lang]
    evidence = {"target_lang": target_lang, "prompt_id": prompt_id, "calls": 0, "frames": 0}
    if not model.concat or model.num_prompts != count:
        raise ValueError("Native streaming prompt conditioning is not active")

    def inspect_input(module, arguments):
        value = arguments[0]
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != width + count:
            raise ValueError("Unexpected actual prompt projection input")
        expected = torch.zeros_like(value[..., width:])
        expected[..., prompt_id] = 1
        if not torch.equal(value[..., width:], expected):
            raise ValueError("Actual streaming prompt differs from the requested language")
        evidence["calls"] += 1
        evidence["frames"] += value.shape[1]

    handle = model.prompt_kernel.register_forward_pre_hook(inspect_input)
    try:
        yield evidence
    finally:
        handle.remove()


def prepare_model(path, target_lang, device):
    from omegaconf import OmegaConf, open_dict
    from sttok.inference import _load_model, _plain
    from sttok.checkpoint_validation import _eager_decoder_state
    model = _load_model(path, "cpu").to(device).float().eval()
    cfg = _plain(model.cfg)
    context = [56, 13]
    supported = [list(item) for item in model.encoder.att_context_size_all]
    if context not in supported:
        raise ValueError(f"Native checkpoint does not support the requested context {context}: {supported}")
    stride = cfg["preprocessor"]["window_stride"]
    subsampling = cfg["encoder"]["subsampling_factor"]
    if abs((context[1] + 1) * subsampling * stride - 1.120) > 1e-9:
        raise ValueError("The selected context does not correspond to a 1120ms chunk")
    if str(cfg["preprocessor"]["normalize"]).lower() not in {"na", "none"}:
        raise ValueError("This probe requires the native non-normalizing frontend")
    model.encoder.set_default_att_context_size(context)
    if hasattr(model.encoder, "set_streaming_cuda_graphs"):
        model.encoder.set_streaming_cuda_graphs(enabled=False)
    decoding = OmegaConf.create(cfg["decoding"])
    with open_dict(decoding):
        decoding.strategy = "greedy_batch"
        decoding.fused_batch_size = -1
        decoding.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(decoding, verbose=False)
    _eager_decoder_state(model)
    model.set_inference_prompt(target_lang)
    model.decoding.set_strip_lang_tags(False)
    return model


def run_stream(model, waveform, mapping, target_lang, row_map=None):
    import torch
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer
    from sttok.checkpoint_validation import _head_trace, _hypothesis, _eager_decoder_state
    from sttok.inference import _plain
    buffer = CacheAwareStreamingAudioBuffer(model, online_normalization=False, pad_and_drop_preencoded=False)
    with torch.inference_mode():
        buffer.append_audio(waveform.numpy(), stream_id=-1)
    if len(buffer.streams_length) != 1:
        raise ValueError("Expected exactly one real audio stream")
    caches = model.encoder.get_initial_cache_state(batch_size=1)
    initial_cache = [tensor_evidence(value) for value in caches]
    if any(item["nonzero_values"] for item in initial_cache):
        raise ValueError("A new stream did not begin with empty caches")
    previous_hypotheses, previous_predictions = None, None
    steps = []
    streaming_cfg = _plain(vars(model.encoder.streaming_cfg))
    with observe_prompt(model, target_lang) as prompt, _head_trace(model.joint.joint_net[-1], old_to_new=row_map) as trace:
        for index, (chunk, lengths) in enumerate(buffer):
            _eager_decoder_state(model)
            if index >= 32:
                raise ValueError("The single-clip streaming probe exceeded its 32-chunk bound")
            before = [tensor_evidence(value) for value in caches]
            if steps and before != steps[-1]["cache_output"]:
                raise ValueError("A returned encoder cache was not carried into the next chunk")
            prompt_before, head_before = prompt["calls"], trace["calls"]
            drop = 0 if index == 0 else model.encoder.streaming_cfg.drop_extra_pre_encoded
            with torch.inference_mode():
                result = model.conformer_stream_step(
                    processed_signal=chunk.to(dtype=torch.float32), processed_signal_length=lengths,
                    cache_last_channel=caches[0], cache_last_time=caches[1], cache_last_channel_len=caches[2],
                    keep_all_outputs=buffer.is_buffer_empty(), previous_hypotheses=previous_hypotheses,
                    previous_pred_out=previous_predictions, drop_extra_pre_encoded=drop,
                    return_transcription=True,
                )
            if not isinstance(result, tuple) or len(result) != 6:
                raise ValueError("Native conformer_stream_step did not return its expected six outputs")
            previous_predictions, hypotheses, *rest = result
            caches = tuple(rest[:3])
            previous_hypotheses = rest[3]
            if prompt["calls"] <= prompt_before or trace["calls"] <= head_before:
                raise ValueError("A streaming chunk skipped the observable prompt or joint-head execution")
            steps.append({"step": index, "chunk": tensor_evidence(chunk), "chunk_lengths": lengths.cpu().tolist(),
                          "drop_extra_pre_encoded": drop, "last_chunk": buffer.is_buffer_empty(),
                          "partial_hypotheses_supplied": index > 0,
                          "cache_input": before, "cache_output": [tensor_evidence(value) for value in caches],
                          "cache_lengths": caches[2].cpu().tolist(),
                          "hypothesis": _hypothesis(hypotheses, mapping),
                          "prompt_calls": prompt["calls"] - prompt_before,
                          "joint_head_calls": trace["calls"] - head_before})
    if len(steps) < 2 or not any(steps[-1]["cache_lengths"]):
        raise ValueError("The recording did not exercise multiple cached streaming chunks")
    result = {"chunk_count": len(steps), "initial_cache": initial_cache, "streaming_cfg": streaming_cfg,
              "effective_decoding_config": _plain(model.cfg)["decoding"], "runtime_decoder": _eager_decoder_state(model),
              "prompt_evidence": prompt, "joint_head_calls": trace["calls"], "steps": steps,
              "final_hypothesis": steps[-1]["hypothesis"], "buffer_exhausted": buffer.is_buffer_empty()}
    return result, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "expanded", "migration-report", "base-tokenizer", "tokenizer", "corpus-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new streaming report path; existing evidence is not overwritten")
    report = {"schema_version": 1, "status": "running", "passed": False, "release_ready": False,
              "nemo_revision": NEMO_REVISION, "probe_script_sha256": sha_file(__file__), "runs": {},
              "scope": "one real English recording; native cached encoder and carried RNNT hypotheses",
              "not_evaluated": ["live audio frontend", "production latency", "all-language streaming accuracy"]}
    started = time.monotonic()
    try:
        import torch
        from sttok.checkpoint import old_model_row_mapping, _validate_source_tokens
        from sttok.checkpoint_validation import _probe_checks, _equivalent_inference_config
        from sttok.inference import _load_audio_tensor, _plain, _runtime_versions, _tokenizer_hash
        from sttok.runtime import build_id_map
        torch.manual_seed(0)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.benchmark = False
        actual_sources = {name: sha_file(inspect.getsourcefile(importlib.import_module(name))) for name in SOURCE_HASHES}
        if actual_sources != SOURCE_HASHES:
            raise ValueError(f"Installed streaming sources differ from inspected pin: {actual_sources}")
        report["native_source_sha256"] = actual_sources
        report["runtime_versions"] = _runtime_versions()
        migration = json.loads(args.migration_report.read_text())
        paths = {"source_checkpoint_sha256": args.source, "checkpoint_sha256": args.expanded,
                 "base_tokenizer_sha256": args.base_tokenizer, "tokenizer_sha256": args.tokenizer}
        hashes = {name: sha_file(path) for name, path in paths.items()}
        if migration.get("status") != "migrated_weights_verified" or any(migration.get(k) != v for k, v in hashes.items()):
            raise ValueError("Streaming inputs differ from the verified checkpoint migration")
        report["artifact_sha256"] = hashes
        report["migration_report_sha256"] = sha_file(args.migration_report)
        rows = [json.loads(line) for line in (args.corpus_dir / "all-clips.jsonl").read_text().splitlines()]
        row = next(r for r in rows if r["language"] == "en" and r["source_split"] == "test" and r["duration"] >= 3)
        audio = args.corpus_dir / row["audio"]
        waveform = _load_audio_tensor(audio, row["audio_sha256"], 16000)
        if waveform.numel() != row["num_samples"]:
            raise ValueError("Audio sample count differs from frozen corpus evidence")
        report["utterance"] = {k: row[k] for k in ("id", "audio_sha256", "target_lang", "num_samples", "text", "source_revision")}
        report["settings"] = {"mode": "native_cache_aware_streaming_simulation", "device": args.device,
                              "dtype": "float32", "amp": False, "batch_size": 1, "att_context_size": [56, 13],
                              "chunk_duration_ms": 1120, "encoder_cuda_graphs": False, "decoder_cuda_graphs": False,
                              "strip_lang_tags": False, "online_normalization": False, "pad_and_drop_preencoded": False,
                              "frontend": "official buffer precomputes non-normalized features with dither=0"}
        old_map, new_map = build_id_map(args.base_tokenizer), build_id_map(args.tokenizer, args.base_tokenizer)
        row_map = old_model_row_mapping(old_map, new_map)
        original = prepare_model(args.source, row["target_lang"], args.device)
        _validate_source_tokens(original, old_map, args.base_tokenizer)
        report["source_runtime_tokenizer_sha256"] = _tokenizer_hash(original)
        original_cfg = _plain(original.cfg)
        baseline, old_trace = run_stream(original, waveform, old_map, row["target_lang"])
        report["runs"]["baseline"] = baseline
        write_report(args.output, report)
        del original
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        expanded = prepare_model(args.expanded, row["target_lang"], args.device)
        if _tokenizer_hash(expanded) != hashes["tokenizer_sha256"]:
            raise ValueError("Expanded checkpoint restored a different tokenizer")
        report["config_comparison"] = _equivalent_inference_config(SimpleNamespace(cfg=original_cfg), expanded, "greedy_batch")
        masked, new_trace = run_stream(expanded, waveform, new_map, row["target_lang"], row_map=row_map)
        report["runs"]["expanded_old_outputs_only"] = masked
        report["old_logit_probes"] = _probe_checks(old_trace, new_trace, expanded.joint.joint_net[-1], row_map, atol=1e-6, rtol=1e-5)
        write_report(args.output, report)
        active, _ = run_stream(expanded, waveform, new_map, row["target_lang"])
        report["runs"]["expanded_all_outputs"] = active
        same_count = baseline["chunk_count"] == masked["chunk_count"]
        parity_fields = ("chunk", "chunk_lengths", "cache_input", "cache_output", "cache_lengths")
        report["cache_and_chunk_parity"] = same_count and all(
            all(a[key] == b[key] for key in parity_fields) for a, b in zip(baseline["steps"], masked["steps"]))
        report["old_output_token_and_text_parity"] = same_count and all(
            a["hypothesis"]["canonical_hf_ids"] == b["hypothesis"]["canonical_hf_ids"]
            and a["hypothesis"]["text"] == b["hypothesis"]["text"] for a, b in zip(baseline["steps"], masked["steps"]))
        report["unmasked_text_changed"] = baseline["final_hypothesis"]["text"] != active["final_hypothesis"]["text"]
        report["unmasked_tokens_changed"] = baseline["final_hypothesis"]["canonical_hf_ids"] != active["final_hypothesis"]["canonical_hf_ids"]
        report["unmasked_chunk_hypotheses_equal"] = baseline["chunk_count"] == active["chunk_count"] and all(
            a["hypothesis"]["text"] == b["hypothesis"]["text"]
            and a["hypothesis"]["canonical_hf_ids"] == b["hypothesis"]["canonical_hf_ids"]
            for a, b in zip(baseline["steps"], active["steps"]))
        report["passed"] = report["cache_and_chunk_parity"] and report["old_output_token_and_text_parity"] and report["old_logit_probes"]["passed"]
        report["initialization_preservation_passed"] = report["passed"] and report["unmasked_chunk_hypotheses_equal"]
        report["status"] = "streaming_smoke_passed_on_one_recording" if report["passed"] else "streaming_smoke_failed"
        report["elapsed_seconds"] = time.monotonic() - started
        write_report(args.output, report)
        return 0 if report["passed"] else 2
    except Exception as exc:
        report.update(status="streaming_smoke_failed", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        write_report(args.output, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
