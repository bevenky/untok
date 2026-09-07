"""Run explicit offline NeMo predictions without fabricating missing evidence.

NeMo imports are lazy. Unit tests inject a fake model; they do not validate the
native integration, audio accuracy, or streaming behavior.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any, Mapping

from .evaluation import PHASES, load_manifest


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    # OmegaConf ListConfig is not a built-in list.
    if type(value).__module__.startswith("omegaconf"):
        from omegaconf import OmegaConf
        return _plain(OmegaConf.to_container(value, resolve=True, enum_to_str=True))
    return value


def _canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(_plain(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _runtime_versions() -> dict[str, str | None]:
    result = {"python": platform.python_version()}
    for package in ("nemo_toolkit", "torch", "tokenizers", "sentencepiece", "sttok"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def _load_model(checkpoint: Path, device: str):
    """Inspect config before selecting the registered custom restoration class."""
    try:
        from nemo.collections.asr.models import ASRModel
    except ImportError as exc:
        raise RuntimeError("Offline inference requires the compatible NVIDIA NeMo runtime; tokenizer-only dependencies are insufficient") from exc
    config = ASRModel.restore_from(str(checkpoint), return_config=True, map_location=device)
    tokenizer_type = config.get("tokenizer", {}).get("type")
    target = config.get("target", "")
    extended = tokenizer_type == "sttok_hf_bpe" or str(target).startswith("sttok.")
    if extended:
        if tokenizer_type != "sttok_hf_bpe":
            raise ValueError("Custom sttok checkpoint has an unsupported tokenizer configuration")
        from .runtime import get_nemo_model_class
        model_class = get_nemo_model_class()
    else:
        model_class = ASRModel
    model = model_class.restore_from(str(checkpoint), map_location=device)
    if extended and getattr(getattr(model, "tokenizer", None), "id_map", None) is None:
        raise ValueError("Custom checkpoint did not restore its canonical HF tokenizer adapter")
    return model


def _tokenizer_hash(model) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    mapping = getattr(tokenizer, "id_map", None)
    if mapping is not None:
        # HFTokenizerAdapter.build_id_map hashes the exact file at construction.
        # NeMo may remove its archive-extraction directory after restoration;
        # the verified construction hash survives with the in-memory adapter.
        path = getattr(tokenizer, "path", None)
        if path is not None and Path(path).exists() and _sha256(path) != mapping.tokenizer_sha256:
            raise ValueError("Restored HF tokenizer file and ID map hashes disagree")
        if not getattr(mapping, "tokenizer_sha256", None):
            raise ValueError("Restored HF tokenizer adapter has no verified construction hash")
        return mapping.tokenizer_sha256
    processor = getattr(tokenizer, "tokenizer", None)
    serialize = getattr(processor, "serialized_model_proto", None)
    if not callable(serialize):
        raise ValueError("Cannot verify the actual runtime tokenizer; expected the HF adapter or a single SentencePiece model")
    payload = serialize()
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Runtime SentencePiece tokenizer returned no serialized model")
    return hashlib.sha256(payload).hexdigest()


def _text(result) -> str:
    # The prompt RNNT transcribe API returns one Hypothesis per input when
    # return_hypotheses=True. Do not silently pick beams or hybrid-head tuples.
    if not isinstance(result, list) or len(result) != 1:
        raise ValueError("Expected exactly one transcription result for one audio file")
    item = result[0]
    text = item if isinstance(item, str) else getattr(item, "text", None)
    if not isinstance(text, str):
        raise ValueError("Transcription returned no text; missing output is not an empty hypothesis")
    return text


def _load_audio_tensor(path: Path, expected_hash: str, sample_rate: int):
    """Decode hash-verified mono audio without resampling or changing its length."""
    import soundfile
    import torch

    if _sha256(path) != expected_hash:
        raise ValueError("Audio hash changed before transcription")
    samples, actual_rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    if _sha256(path) != expected_hash:
        raise ValueError("Audio hash changed while loading transcription input")
    if actual_rate != sample_rate:
        raise ValueError(f"Audio sample rate {actual_rate} differs from model sample rate {sample_rate}; resampling is not implicit")
    if samples.ndim != 2 or samples.shape[1] != 1 or not samples.shape[0]:
        raise ValueError("Transcription requires nonempty mono audio")
    tensor = torch.from_numpy(samples[:, 0].copy())
    if not torch.isfinite(tensor).all():
        raise ValueError("Audio samples must be finite")
    return tensor


def _transcribe_with_verified_prompt(model, audio: Path, audio_sha256: str, target_lang: str):
    """Use tensor audio and verify the one-hot prompt entering the projection.

    The pinned file-path loader can select a random unified prompt. Tensor audio
    instead makes the prompt RNNT create indices from the explicit target_lang.
    Its direct self.forward() call bypasses model hooks, so inspect prompt_kernel.
    """
    import torch

    config = _plain(model.cfg)
    defaults = config.get("model_defaults", {})
    registry = defaults.get("prompt_dictionary", {})
    prompt_id = registry.get(target_lang)
    num_prompts = getattr(model, "num_prompts", None)
    encoder_hidden = defaults.get("enc_hidden")
    sample_rate = getattr(getattr(model, "preprocessor", None), "_sample_rate", None)
    if (not isinstance(prompt_id, int) or not isinstance(num_prompts, int)
            or not 0 <= prompt_id < num_prompts or not isinstance(encoder_hidden, int)
            or encoder_hidden <= 0 or not isinstance(sample_rate, int) or sample_rate <= 0):
        raise ValueError("Model lacks the expected prompt projection or audio configuration")
    kernel = getattr(model, "prompt_kernel", None)
    if not getattr(model, "concat", False) or not isinstance(kernel, torch.nn.Module):
        raise ValueError("Model has no active prompt projection to verify")
    waveform = _load_audio_tensor(Path(audio), audio_sha256, sample_rate)
    trcfg = model.get_transcribe_config()
    for key, value in {"batch_size": 1, "return_hypotheses": True, "num_workers": 0,
                       "verbose": False, "target_lang": target_lang,
                       "pad_min_duration": 0.0, "pad_direction": "right"}.items():
        setattr(trcfg, key, value)
    evidence = {"target_lang": target_lang, "prompt_id": prompt_id,
                "verified_kernel_calls": 0, "verified_frames": 0,
                "verification": "one_hot_input_to_prompt_kernel", "audio_input": "tensor",
                "sample_rate": sample_rate, "audio_samples": waveform.numel(),
                "pad_min_duration": 0.0}

    def verify_prompt(module, arguments):
        if (len(arguments) != 1 or not isinstance(arguments[0], torch.Tensor)
                or arguments[0].ndim != 3 or arguments[0].shape[0] != 1
                or arguments[0].shape[1] == 0
                or arguments[0].shape[-1] != encoder_hidden + num_prompts):
            raise ValueError("Unexpected prompt projection input shape")
        actual = arguments[0][..., encoder_hidden:]
        expected = torch.zeros_like(actual)
        expected[..., prompt_id] = 1
        if not torch.equal(actual, expected):
            raise ValueError(f"Actual conditioning prompt differs from requested {target_lang}")
        evidence["verified_kernel_calls"] += 1
        evidence["verified_frames"] += actual.shape[0] * actual.shape[1]

    handle = kernel.register_forward_pre_hook(verify_prompt)
    try:
        result = model.transcribe(audio=[waveform], batch_size=1, return_hypotheses=True,
                                  num_workers=0, verbose=False, target_lang=target_lang,
                                  override_config=trcfg)
    finally:
        handle.remove()
    if not evidence["verified_kernel_calls"]:
        raise ValueError("Prompt projection hooks did not execute; actual conditioning is unverified")
    return result, evidence


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".inference-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run_nemo_inference(
    manifest_path: str | Path, checkpoint_path: str | Path, phase: str,
    output_path: str | Path, *, device: str = "cpu",
) -> dict[str, Any]:
    """Create actual predictions for explicit offline profile/phase requests.

    All audio and metadata are checked before loading weights. Only a complete
    successful run writes predictions. Existing output is never overwritten.
    Streaming, chunk modes, and arbitrary decoding changes fail explicitly.
    """
    manifest_path, checkpoint, output = Path(manifest_path), Path(checkpoint_path), Path(output_path)
    sidecar = output.with_suffix(output.suffix + ".run.json")
    if output.exists() or sidecar.exists():
        raise ValueError("Prediction output already exists; use a new run/output path")
    if phase not in PHASES:
        raise ValueError(f"Unsupported evaluation phase: {phase}")
    if not checkpoint.is_file() or checkpoint.suffix != ".nemo":
        raise ValueError("Expected an existing local .nemo checkpoint file")
    manifest = load_manifest(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected evaluation manifest schema_version 1")
    run = manifest.get("runs", {}).get(phase, {})
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        raise ValueError("An explicit run_id is required for this phase")
    checkpoint_hash = _sha256(checkpoint)
    if run.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Checkpoint hash does not match the declared phase run")
    settings = run.get("settings", {})
    if settings.get("mode") != "offline":
        raise ValueError("Only explicit offline inference is implemented; streaming requires a separate verified path")
    expected = {"mode": "offline", "device": device, "dtype": "float32", "batch_size": 1, "num_workers": 0}
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"Offline runner requires settings[{key!r}] = {value!r}")
    allowed = set(expected) | {"decoder", "decoding_sha256", "encoder_context", "encoder_config_sha256", "runtime_versions"}
    if set(settings) - allowed or "decoder" not in settings:
        raise ValueError("Unsupported or incomplete decoding settings; the runner does not ignore unknown options")
    profiles = {}
    for profile in manifest.get("profiles", []):
        if not isinstance(profile.get("id"), str) or profile["id"] in profiles:
            raise ValueError("Missing or duplicate profile ID")
        profiles[profile["id"]] = profile
    requests = []
    seen = set()
    for item in manifest.get("utterances", []):
        key = item.get("id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("Missing or duplicate utterance ID")
        seen.add(key)
        profile = profiles.get(item.get("profile_id"))
        if profile is None:
            raise ValueError(f"Unknown profile for utterance {key}")
        condition = item.get("condition_id")
        if not isinstance(condition, str) or condition not in profile.get("required_conditions", []):
            raise ValueError(f"Undeclared condition for utterance {key}")
        if "offline" not in condition.lower() or "stream" in condition.lower():
            raise ValueError(f"Condition {condition!r} is not explicitly offline; no streaming fallback is allowed")
        request = profile.get("inference", {}).get(phase, {}).get(condition)
        if not isinstance(request, dict) or request.get("mode") != "offline":
            raise ValueError(f"Missing explicit offline inference request for {key}/{phase}/{condition}")
        if set(request) - {"mode", "language_mode", "language_prompt"}:
            raise ValueError(f"Unsupported inference options for {key}")
        mode = request.get("language_mode")
        if mode == "automatic":
            if request.get("language_prompt") not in (None, "auto"):
                raise ValueError("Automatic mode cannot also specify a known-language prompt")
            prompt = "auto"
        elif mode == "known":
            prompt = request.get("language_prompt")
            if not isinstance(prompt, str) or not prompt or prompt == "auto":
                raise ValueError("Known-language mode requires an explicit actual prompt key")
        else:
            raise ValueError(f"Unsupported language mode for {key}")
        audio_name = item.get("audio")
        if not isinstance(audio_name, str) or not audio_name:
            raise ValueError(f"Missing local audio path for {key}")
        audio = Path(audio_name)
        if not audio.is_absolute():
            audio = manifest_path.parent / audio
        audio = audio.resolve()
        if not audio.is_file():
            raise ValueError(f"Missing audio for {key}: {audio}")
        digest = _sha256(audio)
        if item.get("audio_sha256") != digest:
            raise ValueError(f"Audio hash mismatch or missing hash for {key}")
        requests.append((item, audio, digest, mode, prompt))
    if not requests:
        raise ValueError("No audio utterances were supplied")
    model = _load_model(checkpoint, device)
    model = model.to(device).float().eval()
    tokenizer_hash = _tokenizer_hash(model)
    if run.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("Declared tokenizer hash does not match the tokenizer actually restored from the checkpoint")
    config = _plain(model.cfg)
    registry = config.get("model_defaults", {}).get("prompt_dictionary", {})
    if not isinstance(registry, dict) or not registry:
        raise ValueError("Restored model has no verified prompt registry")
    for item, _, _, _, prompt in requests:
        if prompt not in registry or not isinstance(registry[prompt], int):
            raise ValueError(f"Prompt {prompt!r} is unavailable in this checkpoint for {item['id']}; no automatic fallback")
    actual = {
        **expected, "decoder": config.get("decoding", {}).get("strategy"),
        "decoding_sha256": _canonical_hash(config.get("decoding", {})),
        "encoder_context": config.get("encoder", {}).get("att_context_size"),
        "encoder_config_sha256": _canonical_hash(config.get("encoder", {})),
        "runtime_versions": _runtime_versions(),
    }
    for key, value in settings.items():
        if actual[key] != value:
            raise ValueError(f"Declared {key} differs from the restored model's actual inference setting")
    rows = []
    for item, audio, audio_hash, language_mode, prompt in requests:
        # One independent transcribe call per utterance deliberately avoids
        # sharing partial RNNT hypotheses or caches across different recordings.
        result, prompt_evidence = _transcribe_with_verified_prompt(model, audio, audio_hash, prompt)
        rows.append({
            "utterance_id": item["id"], "phase": phase, "hypothesis": _text(result),
            "run_id": run["run_id"], "language_mode": language_mode,
            "language_prompt": prompt if language_mode == "known" else None,
            "checkpoint_sha256": checkpoint_hash, "tokenizer_sha256": tokenizer_hash,
            "audio_sha256": audio_hash, "actual_settings": actual,
            "prompt_evidence": prompt_evidence,
        })
    provenance = {
        "schema_version": 1, "phase": phase, "run_id": run["run_id"],
        "checkpoint_sha256": checkpoint_hash, "tokenizer_sha256": tokenizer_hash,
        "evaluation_manifest_file_sha256": _sha256(manifest_path),
        "actual_settings": actual, "utterances": len(rows),
        "model_class": type(model).__module__ + "." + type(model).__qualname__,
        "integration_status": "executed_offline_transcribe", "streaming_validated": False,
    }
    # No output is written before every inference call succeeds. The sidecar
    # precedes the final JSONL commit so a failed output write cannot leave a
    # predictions file that lacks its companion run evidence.
    _write_atomic(sidecar, json.dumps(provenance, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    _write_atomic(output, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))
    return {**provenance, "predictions": str(output), "run_metadata": str(sidecar)}
