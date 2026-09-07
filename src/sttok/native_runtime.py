"""NeMo restoration using the validated native SentencePiece bundle for both directions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import tempfile


def native_bundle_config(directory):
    """Validate a bundle before registering its complete file set with NeMo."""
    from .bundles import load_tokenizer_bundle

    directory = Path(directory).resolve()
    load_tokenizer_bundle(directory)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    names = sorted({"manifest.json", *manifest["files"]})
    if any(Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("Native bundle artifacts must have contained filenames")
    return {
        "type": "sttok_native_unigram",
        "bundle_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "bundle_filenames": {f"file_{i}": name for i, name in enumerate(names)},
        "bundle_files": {f"file_{i}": str(directory / name) for i, name in enumerate(names)},
    }


def _setup_native_tokenizer(model, tokenizer_cfg):
    from .bundles import load_tokenizer_bundle

    if tokenizer_cfg.get("type") != "sttok_native_unigram":
        raise ValueError("NativeNemotronRNNTModel requires a native Unigram bundle")
    filenames = dict(tokenizer_cfg.get("bundle_filenames", {}))
    paths = dict(tokenizer_cfg.get("bundle_files", {}))
    if (not filenames or set(paths) != set(filenames)
            or any(not isinstance(name, str) for name in filenames.values())
            or len(set(filenames.values())) != len(filenames)
            or "manifest.json" not in filenames.values()):
        raise ValueError("Incomplete native bundle artifact configuration")
    for key, name in filenames.items():
        if (not re.fullmatch(r"file_[0-9]+", key) or not isinstance(name, str)
                or Path(name).name != name or name in {".", ".."}):
            raise ValueError("Invalid native bundle artifact name")
    registered = {}
    for key in sorted(paths):
        registered[filenames[key]] = model.register_artifact(f"tokenizer.bundle_files.{key}", paths[key])
    # NeMo renames archived files. Reassemble their logical names only while the
    # bundle verifier loads their bytes; registered originals remain persistent.
    with tempfile.TemporaryDirectory(prefix="sttok-native-bundle-") as temporary:
        directory = Path(temporary)
        for name, path in registered.items():
            (directory / name).write_bytes(Path(path).read_bytes())
        digest = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        if digest != tokenizer_cfg.get("bundle_manifest_sha256"):
            raise ValueError("Registered native bundle manifest hash changed")
        tokenizer = load_tokenizer_bundle(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if set(registered) != {"manifest.json", *manifest["files"]}:
            raise ValueError("Registered native artifact set differs from its bundle manifest")
    model.tokenizer_cfg = tokenizer_cfg
    model.tokenizer_dir = str(Path(registered["manifest.json"]).parent)
    # NeMo calls its SentencePiece category "bpe", including Unigram models.
    # The actual encoder and decoder here both use the pinned Unigram proto.
    model.tokenizer_type = "bpe"
    model.tokenizer = tokenizer
    model.native_bundle_manifest_sha256 = digest
    model.native_tokenizer_sha256 = hashlib.sha256(tokenizer.model_bytes).hexdigest()


_NATIVE_NEMO_CLASS = None


def transcribe_native_file(model, audio, *, target_lang):
    """Transcribe one file through the verified native tensor/prompt path.

    Keep the model in evaluation mode and return the real NeMo hypotheses and
    observed prompt evidence. The pinned
    NeMo file-list loader can choose a unified prompt, so use tensor audio and
    verify the actual requested one-hot conditioning input instead.
    """
    from .inference import _sha256, _transcribe_with_verified_prompt

    if not getattr(model, "native_tokenizer_sha256", None):
        raise ValueError("Restore a native sttok checkpoint before using native inference")
    path = Path(audio)
    model.eval()
    try:
        return _transcribe_with_verified_prompt(model, path, _sha256(path), target_lang)
    finally:
        # NeMo's transcription teardown calls submodule.unfreeze(), which can
        # re-enable training mode after restoring the parent model's mode.
        model.eval()


def _register_native_target(model_class):
    from importlib import import_module

    common = import_module("nemo.core.classes.common")
    original = getattr(common, "_is_target_allowed", None)
    serialization = getattr(common, "Serialization", None)
    if not callable(original) or not isinstance(serialization, type):
        raise RuntimeError("Unsupported NeMo target-validation interface")
    if not isinstance(model_class, type) or not issubclass(model_class, serialization):
        raise ValueError("The native model must be a NeMo Serialization subclass")
    if getattr(original, "_sttok_native_registered_class", None) is model_class:
        return
    target_path = "sttok.native_runtime.NativeNemotronRNNTModel"

    def allow_native_target(target):
        if target == target_path:
            return common.hydra.utils.get_class(target) is model_class
        return original(target)

    allow_native_target._sttok_native_registered_class = model_class
    common._is_target_allowed = allow_native_target


def get_native_nemo_model_class():
    """Register only the exact native model class for save/reload with NeMo."""
    global _NATIVE_NEMO_CLASS
    if _NATIVE_NEMO_CLASS is None:
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

        class NativeNemotronRNNTModel(EncDecRNNTBPEModelWithPrompt):
            def _setup_tokenizer(self, tokenizer_cfg):
                _setup_native_tokenizer(self, tokenizer_cfg)

        NativeNemotronRNNTModel.__module__ = __name__
        NativeNemotronRNNTModel.__qualname__ = "NativeNemotronRNNTModel"
        _NATIVE_NEMO_CLASS = NativeNemotronRNNTModel
        globals()["NativeNemotronRNNTModel"] = _NATIVE_NEMO_CLASS
    _register_native_target(_NATIVE_NEMO_CLASS)
    return _NATIVE_NEMO_CLASS


def __getattr__(name):
    if name == "NativeNemotronRNNTModel":
        return get_native_nemo_model_class()
    raise AttributeError(name)
