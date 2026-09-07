"""Native artifact persistence and exact-class registration without importing NeMo."""
import copy
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from sttok.bundles import load_tokenizer_bundle, package_tokenizer_bundles
from sttok.native_checkpoint import migrate_native_checkpoint
from sttok.native_runtime import _register_native_target, _setup_native_tokenizer, native_bundle_config
from sttok.unigram import build_native_tokenizer


def test_native_file_inference_uses_verified_tensor_prompt_path(tmp_path, monkeypatch):
    import sttok.inference as inference
    from sttok.native_runtime import transcribe_native_file

    audio = tmp_path / "speech.wav"
    audio.write_bytes(b"audio provenance fixture")
    calls = []
    class Model:
        native_tokenizer_sha256 = "verified-native"
        eval_calls = 0
        def eval(self):
            self.eval_calls += 1
            return self
    model = Model()
    def observed(*args):
        calls.append(args)
        return (["actual-return"], {"verified_kernel_calls": 1})
    monkeypatch.setattr(inference, "_transcribe_with_verified_prompt", observed)
    assert transcribe_native_file(model, audio, target_lang="hi-IN") == (["actual-return"], {"verified_kernel_calls": 1})
    assert calls == [(model, audio, hashlib.sha256(audio.read_bytes()).hexdigest(), "hi-IN")]
    assert model.eval_calls == 2
    def fail(*args):
        raise ValueError("transcription failed")
    monkeypatch.setattr(inference, "_transcribe_with_verified_prompt", fail)
    with pytest.raises(ValueError, match="transcription failed"):
        transcribe_native_file(model, audio, target_lang="hi-IN")
    assert model.eval_calls == 4
    with pytest.raises(ValueError, match="native sttok checkpoint"):
        transcribe_native_file(SimpleNamespace(), audio, target_lang="hi-IN")


@pytest.fixture
def native_bundle(tmp_path):
    proto = pb.ModelProto()
    proto.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    proto.trainer_spec.unk_id = 0
    proto.trainer_spec.bos_id = proto.trainer_spec.eos_id = proto.trainer_spec.pad_id = -1
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.remove_extra_whitespaces = False
    for index, (piece, score) in enumerate([("<unk>", 0), ("▁", 0), ("a", -2), ("b", -3), ("z", -253), ("я", -4)]):
        proto.pieces.add(piece=piece, score=score, type=2 if index == 0 else 1)
    proto.trainer_spec.vocab_size = len(proto.pieces)
    base = tmp_path / "base.model"
    base.write_bytes(proto.SerializeToString())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"base_tokenizer_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                                     "additions": [{"piece": "க", "score": -2}, {"piece": "▁க", "score": -1}]}))
    bundle = tmp_path / "bundle"
    build_native_tokenizer(base, selection, bundle)
    return bundle


class ArtifactModel:
    def __init__(self, cfg, directory):
        self.cfg, self.directory = cfg, directory
        self.directory.mkdir()
        self.registered = {}

    def register_artifact(self, key, source):
        # Emulate NeMo archive names instead of relying on original filenames.
        destination = self.directory / f"archive_{len(self.registered)}_{Path(source).name}"
        shutil.copyfile(source, destination)
        self.registered[key] = str(destination)
        self.cfg["bundle_files"][key.rsplit(".", 1)[-1]] = str(destination)
        return str(destination)


def test_native_bundle_survives_renamed_archive_paths_and_source_directory_removal(native_bundle, tmp_path):
    original = load_tokenizer_bundle(native_bundle)
    cfg = native_bundle_config(native_bundle)
    first = ArtifactModel(cfg, tmp_path / "first-archive")
    _setup_native_tokenizer(first, cfg)
    assert len(first.registered) == 6
    assert first.tokenizer.model_bytes == original.model_bytes
    assert first.tokenizer.source_native_to_target_native == (0, 1, 2, 3, 4, 5, 8)
    shutil.rmtree(native_bundle)
    restored_cfg = copy.deepcopy(cfg)
    second = ArtifactModel(restored_cfg, tmp_path / "restored-archive")
    _setup_native_tokenizer(second, restored_cfg)
    assert second.native_bundle_manifest_sha256 == first.native_bundle_manifest_sha256
    assert second.tokenizer.id_map.to_dict() == original.id_map.to_dict()
    assert second.tokenizer.model_bytes == original.model_bytes
    assert second.tokenizer.base_model_bytes == original.base_model_bytes
    for text in ("", "  a  b ", "a\u200cb", "க", "aகb", "🙂a"):
        assert second.tokenizer.text_to_ids(text) == original.text_to_ids(text)
        assert second.tokenizer.ids_to_text(second.tokenizer.text_to_ids(text)) == original.ids_to_text(original.text_to_ids(text))
    assert second.tokenizer.ids_to_text([2, 2, second.tokenizer.blank_id, 3]) == "aab"


@pytest.mark.parametrize("failure", ["wrong_type", "missing_file", "unsafe_name", "manifest_hash", "extra_file"])
def test_native_runtime_rejects_ambiguous_artifacts(native_bundle, tmp_path, failure):
    cfg = native_bundle_config(native_bundle)
    if failure == "wrong_type": cfg["type"] = "sttok_hf_bpe"
    elif failure == "missing_file": cfg["bundle_files"].pop("file_0")
    elif failure == "unsafe_name": cfg["bundle_filenames"]["file_0"] = "../outside.model"
    elif failure == "manifest_hash": cfg["bundle_manifest_sha256"] = "0" * 64
    else:
        cfg["bundle_filenames"]["file_9"] = "extra.model"
        cfg["bundle_files"]["file_9"] = str(native_bundle / "tokenizer.model")
    model = ArtifactModel(cfg, tmp_path / "archive")
    with pytest.raises(ValueError):
        _setup_native_tokenizer(model, cfg)


def test_native_registration_does_not_allow_legacy_or_other_classes(monkeypatch):
    class Serialization:
        pass

    class NativeClass(Serialization):
        pass

    target = "sttok.native_runtime.NativeNemotronRNNTModel"
    resolved = {target: NativeClass}
    common = ModuleType("nemo.core.classes.common")
    common.Serialization = Serialization
    common._is_target_allowed = lambda name: name == "nemo.collections.ExistingModel"
    common.ALLOWED_TARGET_PREFIXES = ["nemo.collections."]
    common.hydra = SimpleNamespace(utils=SimpleNamespace(get_class=lambda name: resolved[name]))
    monkeypatch.setitem(sys.modules, common.__name__, common)
    _register_native_target(NativeClass)
    predicate = common._is_target_allowed
    assert predicate(target) and predicate("nemo.collections.ExistingModel")
    for other in ("sttok.runtime.ExtendedNemotronRNNTModel", "sttok.other.Model", target + "Alias", "os.system"):
        assert not predicate(other)
    assert common.ALLOWED_TARGET_PREFIXES == ["nemo.collections."]
    resolved[target] = type("Other", (Serialization,), {})
    assert not predicate(target)
    _register_native_target(NativeClass)
    assert common._is_target_allowed is predicate
    with pytest.raises(ValueError, match="Serialization"):
        _register_native_target(object)


def test_native_migration_requires_checkpoint_pin_before_nemo_or_output(native_bundle, tmp_path):
    source = tmp_path / "source.nemo"
    source.write_bytes(b"unit-test checkpoint bytes")
    destination = tmp_path / "new.nemo"
    with pytest.raises(ValueError, match="explicit pin"):
        migrate_native_checkpoint(source, native_bundle, destination, expected_source_sha256="0" * 64)
    assert not destination.exists() and not destination.with_suffix(".migration.json").exists()
    correct = hashlib.sha256(source.read_bytes()).hexdigest()
    destination.write_bytes(b"existing output")
    with pytest.raises(ValueError, match="new output paths"):
        migrate_native_checkpoint(source, native_bundle, destination, expected_source_sha256=correct)
    assert destination.read_bytes() == b"existing output"


@pytest.mark.parametrize("corrupt_reload", [False, True])
@pytest.mark.parametrize("profile", ["full", "latin-indic", "latin"])
def test_native_migration_orchestration_saves_verifies_and_rejects_corrupt_reload(
    monkeypatch, native_bundle, tmp_path, corrupt_reload, profile,
):
    """The archive is a unit-test stand-in; actual NeMo remains an integration gate."""
    torch = pytest.importorskip("torch")
    import sentencepiece as spm
    import sttok.native_checkpoint as migration
    from test_native_checkpoint import toy_model

    if profile != "full":
        package_tokenizer_bundles(native_bundle, tmp_path / "variants", profiles=[profile], make_zips=False)
        native_bundle = tmp_path / "variants" / profile

    class Config(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = convert(value)

        def __setitem__(self, name, value):
            super().__setitem__(name, convert(value))

    def convert(value):
        if isinstance(value, dict):
            obj = Config()
            for key, item in value.items():
                dict.__setitem__(obj, key, convert(item))
            return obj
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    def plain(value):
        if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
        if isinstance(value, list): return [plain(item) for item in value]
        return value

    omega = ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace(create=convert, to_container=lambda cfg, resolve: plain(cfg))
    omega.open_dict = lambda cfg: nullcontext()
    monkeypatch.setitem(sys.modules, "omegaconf", omega)

    def copy_modules(model, size):
        toy = toy_model(size)
        for name, module in toy.named_children():
            model.add_module(name, module)
        model.register_buffer("retained_buffer", toy.retained_buffer.clone())

    class SourceModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            backend = spm.SentencePieceProcessor(model_file=str(native_bundle / "base-tokenizer.model"))
            self.tokenizer = SimpleNamespace(tokenizer=backend, vocab_size=backend.get_piece_size(),
                ids_to_tokens=lambda ids: [backend.id_to_piece(i) for i in ids],
                text_to_ids=lambda text: backend.encode(text, out_type=int))
            self.cfg = convert({"model_defaults": {"num_prompts": 64, "prompt_dictionary": {"auto": 0, "en-US": 1}},
                                "tokenizer": {"type": "bpe"}, "train_ds": {"manifest": "private-source"}})
            copy_modules(self, backend.get_piece_size())

    class NativeModel(torch.nn.Module):
        def __init__(self, cfg, trainer):
            super().__init__()
            self.cfg = cfg
            self.registered = {}
            _setup_native_tokenizer(self, cfg.tokenizer)
            copy_modules(self, self.tokenizer.vocab_size)

        def register_artifact(self, key, path):
            self.registered[key.rsplit(".", 1)[-1]] = Path(path).read_bytes()
            return path

        def save_to(self, path):
            payload = {"cfg": plain(self.cfg), "state": self.state_dict(), "artifacts": self.registered}
            torch.save(payload, path)

        @classmethod
        def restore_from(cls, path, map_location):
            payload = torch.load(path, map_location=map_location, weights_only=True)
            cfg = convert(payload["cfg"])
            directory = tmp_path / "restored-payload"
            directory.mkdir()
            for key, data in payload["artifacts"].items():
                artifact = directory / f"renamed-{key}"
                artifact.write_bytes(data)
                cfg.tokenizer.bundle_files[key] = str(artifact)
            restored = cls(cfg, trainer=None)
            restored.load_state_dict(payload["state"])
            if corrupt_reload:
                with torch.no_grad(): restored.encoder.weight[0, 0] += 1
            return restored

    source_model = SourceModel()
    for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.models",
                 "nemo.collections.asr.models.rnnt_bpe_models_prompt"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["nemo.collections.asr.models"].ASRModel = SimpleNamespace(restore_from=lambda *args, **kwargs: source_model)
    sys.modules["nemo.collections.asr.models.rnnt_bpe_models_prompt"].EncDecRNNTBPEModelWithPrompt = SourceModel
    monkeypatch.setattr(migration, "get_native_nemo_model_class", lambda: NativeModel)
    source = tmp_path / "pinned.nemo"
    source.write_bytes(b"pinned synthetic source for orchestration only")
    destination = tmp_path / "migrated.nemo"
    kwargs = {"expected_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if corrupt_reload:
        with pytest.raises(ValueError, match="Retained learned state changed"):
            migration.migrate_native_checkpoint(source, native_bundle, destination, **kwargs)
        assert not destination.exists() and not destination.with_suffix(".migration.json").exists()
    else:
        report = migration.migrate_native_checkpoint(source, native_bundle, destination, **kwargs)
        assert destination.exists() and destination.with_suffix(".migration.json").exists()
        assert report["before_save"]["passed"] and report["after_reload"]["passed"]
        assert report["after_reload"]["all_source_values_preserved"] == (profile == "full")
        assert report["old_model_to_new_model"] == {
            "full": [0, 1, 2, 3, 4, 5, 8],
            "latin-indic": [0, 1, 2, 3, 4, None, 7],
            "latin": [0, 1, 2, 3, 4, None, 5],
        }[profile]
        assert len(report["prompt_registry"]["target_assignments"]) == 22
        assert report["new_row_initialization"]["verified_after_reload"]
        assert not report["asr_accuracy_evaluated"] and not report["training_forward_evaluated"]
        assert source_model.cfg.train_ds == {"manifest": "private-source"}
