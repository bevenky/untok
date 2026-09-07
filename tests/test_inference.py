"""Fake-model unit tests only; these do not establish NeMo integration or WER."""
import hashlib
from enum import Enum
import json
import sys
import types

import pytest

from untok import inference


def test_runtime_configuration_enums_have_stable_json_names():
    class ScoreMode(Enum):
        KEEP = 1

    plain = inference._plain({"decoder": {"score_mode": ScoreMode.KEEP}, "modes": [ScoreMode.KEEP]})
    assert plain == {"decoder": {"score_mode": "KEEP"}, "modes": ["KEEP"]}
    assert json.loads(json.dumps(plain)) == plain


class FakeModel:
    def __init__(self, outputs=None):
        torch = pytest.importorskip("torch")
        self.outputs = outputs if outputs is not None else ["one one two"]
        self.calls = []
        self.cfg = {
            "model_defaults": {"prompt_dictionary": {"en-US": 0, "auto": 1}, "enc_hidden": 2},
            "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}},
            "encoder": {"att_context_size": [70, 13]},
        }
        self.tokenizer = types.SimpleNamespace(tokenizer=types.SimpleNamespace(serialized_model_proto=lambda: b"synthetic-tokenizer-unit-test"))
        self.preprocessor = types.SimpleNamespace(_sample_rate=16000)
        self.num_prompts, self.concat = 2, True
        self.prompt_kernel = torch.nn.Linear(4, 2)
        self.wrong_prompt = False
        self.skip_prompt = False

    def get_transcribe_config(self):
        return types.SimpleNamespace()

    def to(self, device):
        self.device = device
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def transcribe(self, **kwargs):
        torch = pytest.importorskip("torch")
        self.calls.append(kwargs)
        assert isinstance(kwargs["audio"][0], torch.Tensor)
        assert kwargs["override_config"].target_lang == kwargs["target_lang"]
        assert kwargs["override_config"].pad_min_duration == 0
        if not self.skip_prompt:
            conditioning = torch.zeros(1, 3, 4)
            prompt_id = self.cfg["model_defaults"]["prompt_dictionary"][kwargs["target_lang"]]
            if self.wrong_prompt:
                prompt_id = 1 - prompt_id
            conditioning[..., 2 + prompt_id] = 1
            self.prompt_kernel(conditioning)
        output = self.outputs[len(self.calls) - 1]
        if isinstance(output, Exception):
            raise output
        return output if isinstance(output, list) else [types.SimpleNamespace(text=output)]


def fixture(tmp_path, monkeypatch, mode="known", outputs=None):
    audio = tmp_path / "unit-test.wav"
    audio.write_bytes(b"not-real-audio-unit-test-fixture")
    checkpoint = tmp_path / "unit-test.nemo"
    checkpoint.write_bytes(b"not-real-checkpoint-unit-test-fixture")
    model = FakeModel(outputs)
    monkeypatch.setattr(inference, "_load_model", lambda path, device: model)
    torch = pytest.importorskip("torch")
    # Synthetic waveform input; the fake model only exercises the prompt contract.
    monkeypatch.setattr(inference, "_load_audio_tensor", lambda *args: torch.zeros(16000))
    request = {"mode": "offline", "language_mode": mode}
    if mode == "known":
        request["language_prompt"] = "en-US"
    manifest = {
        "schema_version": 1, "purpose": "development",
        "runs": {"baseline": {
            "run_id": "synthetic-unit-test-only", "checkpoint_sha256": inference._sha256(checkpoint),
            "tokenizer_sha256": inference._tokenizer_hash(model),
            "settings": {"mode": "offline", "device": "cpu", "dtype": "float32", "batch_size": 1, "num_workers": 0, "decoder": "greedy_batch"},
        }},
        "profiles": [{"id": "en-US", "language": "en", "script": "Latn", "cohorts": ["existing_asr"], "protected": True,
            "required_conditions": ["offline"], "inference": {"baseline": {"offline": request}}}],
        "utterances": [{"id": "unit-test-1", "profile_id": "en-US", "condition_id": "offline", "reference": "one one two", "audio": audio.name, "audio_sha256": inference._sha256(audio)}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    output = tmp_path / "predictions.jsonl"
    return manifest, manifest_path, checkpoint, output, model


def save(manifest, path):
    path.write_text(json.dumps(manifest))


def test_known_prompt_repeated_text_and_actual_hash_provenance(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, model = fixture(tmp_path, monkeypatch)
    summary = inference.run_nemo_inference(path, checkpoint, "baseline", output)
    prediction = json.loads(output.read_text())
    assert prediction["hypothesis"] == "one one two"
    assert model.calls[0]["target_lang"] == "en-US"
    assert prediction["language_prompt"] == "en-US"
    assert prediction["checkpoint_sha256"] == inference._sha256(checkpoint)
    assert prediction["tokenizer_sha256"] == inference._tokenizer_hash(model)
    assert prediction["audio_sha256"] == manifest["utterances"][0]["audio_sha256"]
    assert prediction["actual_settings"]["mode"] == "offline"
    assert prediction["prompt_evidence"]["prompt_id"] == 0
    assert prediction["prompt_evidence"]["verified_kernel_calls"] == 1
    assert prediction["prompt_evidence"]["verified_frames"] == 3
    assert not model.prompt_kernel._forward_pre_hooks
    assert summary["streaming_validated"] is False
    assert output.with_suffix(".jsonl.run.json").is_file()


def test_automatic_mode_uses_real_auto_prompt(tmp_path, monkeypatch):
    _, path, checkpoint, output, model = fixture(tmp_path, monkeypatch, mode="automatic")
    inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert model.calls[0]["target_lang"] == "auto"
    row = json.loads(output.read_text())
    assert row["language_mode"] == "automatic"
    assert row["language_prompt"] is None
    assert row["prompt_evidence"]["prompt_id"] == 1


@pytest.mark.parametrize("failure", ["wrong_prompt", "skip_prompt", "transcribe_exception"])
def test_actual_prompt_failures_leave_no_evidence_and_remove_hooks(tmp_path, monkeypatch, failure):
    _, path, checkpoint, output, model = fixture(tmp_path, monkeypatch)
    if failure == "transcribe_exception":
        model.outputs = [RuntimeError("transcription failed")]
        error, pattern = RuntimeError, "transcription failed"
    else:
        setattr(model, failure, True)
        error, pattern = ValueError, "conditioning|hooks did not execute"
    with pytest.raises(error, match=pattern):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert not model.prompt_kernel._forward_pre_hooks
    assert not output.exists()
    assert not output.with_suffix(".jsonl.run.json").exists()


@pytest.mark.parametrize("failure", [None, "sample_rate", "stereo", "empty", "nonfinite", "changed_hash"])
def test_audio_loader_checks_decode_shape_rate_and_current_hash(tmp_path, monkeypatch, failure):
    """Mock soundfile decoding; exercise loader checks without adding test dependencies."""
    torch = pytest.importorskip("torch")
    path = tmp_path / "decoder-fixture.wav"
    path.write_bytes(b"explicit mock decoder fixture")
    digest = inference._sha256(path)
    samples = torch.zeros(320, 1)
    rate = 16000
    if failure == "sample_rate":
        rate = 8000
    elif failure == "stereo":
        samples = torch.zeros(320, 2)
    elif failure == "empty":
        samples = torch.zeros(0, 1)
    elif failure == "nonfinite":
        samples[0, 0] = float("nan")

    def read(filename, **kwargs):
        assert filename == str(path)
        assert kwargs == {"dtype": "float32", "always_2d": True}
        if failure == "changed_hash":
            path.write_bytes(b"changed during mocked decoding")
        return samples.numpy(), rate

    monkeypatch.setitem(sys.modules, "soundfile", types.SimpleNamespace(read=read))
    if failure:
        with pytest.raises(ValueError, match="sample rate|mono audio|finite|hash changed"):
            inference._load_audio_tensor(path, digest, 16000)
    else:
        tensor = inference._load_audio_tensor(path, digest, 16000)
        assert tensor.shape == (320,) and tensor.dtype == torch.float32


def test_missing_audio_fails_before_loading_model(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch)
    manifest["utterances"][0]["audio"] = "missing.wav"
    save(manifest, path)
    monkeypatch.setattr(inference, "_load_model", lambda *args: pytest.fail("Should not load weights"))
    with pytest.raises(ValueError, match="Missing audio"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert not output.exists()


def test_unknown_prompt_never_falls_back_to_auto(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, model = fixture(tmp_path, monkeypatch)
    manifest["profiles"][0]["inference"]["baseline"]["offline"]["language_prompt"] = "mr-IN"
    save(manifest, path)
    with pytest.raises(ValueError, match="unavailable in this checkpoint"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert model.calls == []


@pytest.mark.parametrize("result", [None, [], [None], ["first", "second"]])
def test_missing_or_ambiguous_result_never_becomes_empty_hypothesis(tmp_path, monkeypatch, result):
    _, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch, outputs=[result])
    with pytest.raises(ValueError, match="transcription result|no text"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.run.json").exists()


def test_real_empty_model_output_is_preserved(tmp_path, monkeypatch):
    _, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch, outputs=[""])
    inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert json.loads(output.read_text())["hypothesis"] == ""


def test_later_inference_failure_leaves_no_partial_predictions(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch, outputs=["one", RuntimeError("inference failed")])
    manifest["utterances"].append({**manifest["utterances"][0], "id": "unit-test-2"})
    save(manifest, path)
    with pytest.raises(RuntimeError, match="inference failed"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert not output.exists()


def test_streaming_condition_cannot_silently_use_offline_path(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch)
    profile = manifest["profiles"][0]
    profile["required_conditions"] = ["known_language_streaming_1120ms"]
    manifest["utterances"][0]["condition_id"] = profile["required_conditions"][0]
    save(manifest, path)
    monkeypatch.setattr(inference, "_load_model", lambda *args: pytest.fail("Should not load weights"))
    with pytest.raises(ValueError, match="no streaming fallback"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)


@pytest.mark.parametrize("field", ["checkpoint_sha256", "tokenizer_sha256"])
def test_wrong_artifact_hash_fails(tmp_path, monkeypatch, field):
    manifest, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch)
    manifest["runs"]["baseline"][field] = "0" * 64
    save(manifest, path)
    with pytest.raises(ValueError, match="[Hh]ash"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)


def test_declared_decoder_must_match_actual_model(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch)
    manifest["runs"]["baseline"]["settings"]["decoder"] = "beam"
    save(manifest, path)
    with pytest.raises(ValueError, match="actual inference setting"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)


def test_duplicate_or_existing_outputs_are_not_overwritten(tmp_path, monkeypatch):
    _, path, checkpoint, output, model = fixture(tmp_path, monkeypatch)
    output.write_text("existing evidence")
    with pytest.raises(ValueError, match="already exists"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert output.read_text() == "existing evidence"
    assert model.calls == []


def test_loader_selects_registered_extended_class_from_checkpoint_config(tmp_path, monkeypatch):
    calls = []
    config = {"tokenizer": {"type": "untok_hf_bpe"}, "target": "untok.runtime.ExtendedNemotronRNNTModel"}
    class FakeASR:
        @classmethod
        def restore_from(cls, path, **kwargs):
            calls.append(("native", kwargs))
            assert kwargs["return_config"] is True
            return config
    class FakeExtended:
        @classmethod
        def restore_from(cls, path, **kwargs):
            calls.append(("extended", kwargs))
            return types.SimpleNamespace(tokenizer=types.SimpleNamespace(id_map=object()))
    nemo_models = types.ModuleType("nemo.collections.asr.models")
    nemo_models.ASRModel = FakeASR
    monkeypatch.setitem(sys.modules, "nemo.collections.asr.models", nemo_models)
    runtime = types.ModuleType("untok.runtime")
    runtime.get_nemo_model_class = lambda: FakeExtended
    monkeypatch.setitem(sys.modules, "untok.runtime", runtime)
    inference._load_model(tmp_path / "test.nemo", "cpu")
    assert [kind for kind, _ in calls] == ["native", "extended"]


def test_runtime_tokenizer_hash_rejects_unverifiable_tokenizer():
    with pytest.raises(ValueError, match="actual runtime tokenizer"):
        inference._tokenizer_hash(types.SimpleNamespace(tokenizer=object()))


def test_real_hf_adapter_hash_survives_deleted_restore_directory(tmp_path):
    """Actual tokenizer backend, still no NeMo checkpoint or audio inference."""
    from tokenizers import Tokenizer, models
    from untok.runtime import HFTokenizerAdapter
    path = tmp_path / "restored-tokenizer.json"
    backend = Tokenizer(models.BPE(vocab={"<unk>": 0, "a": 1, "<pad>": 2, "<blank>": 3}, merges=[], unk_token="<unk>"))
    backend.save(str(path))
    digest = inference._sha256(path)
    adapter = HFTokenizerAdapter(path)
    path.unlink()
    assert adapter.text_to_ids("a") == [1]
    assert inference._tokenizer_hash(types.SimpleNamespace(tokenizer=adapter)) == digest


def test_actual_native_sentencepiece_hash_survives_deleted_source(tmp_path):
    """Actual serialized SentencePiece backend; no ASR model is exercised."""
    from sentencepiece import SentencePieceProcessor, sentencepiece_model_pb2
    proto = sentencepiece_model_pb2.ModelProto()
    proto.trainer_spec.model_type = sentencepiece_model_pb2.TrainerSpec.BPE
    proto.trainer_spec.unk_id = 0
    proto.trainer_spec.bos_id = -1
    proto.trainer_spec.eos_id = -1
    for index, text in enumerate(("<unk>", "▁", "a")):
        piece = proto.pieces.add()
        piece.piece = text
        piece.score = -index
        piece.type = 2 if index == 0 else 1
    path = tmp_path / "restored-tokenizer.model"
    path.write_bytes(proto.SerializeToString())
    digest = inference._sha256(path)
    processor = SentencePieceProcessor(model_file=str(path))
    path.unlink()
    model = types.SimpleNamespace(tokenizer=types.SimpleNamespace(tokenizer=processor))
    assert inference._tokenizer_hash(model) == digest


def test_surviving_hf_artifact_with_changed_bytes_is_rejected(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_bytes(b"initial contents")
    mapping = types.SimpleNamespace(tokenizer_sha256=inference._sha256(path))
    model = types.SimpleNamespace(tokenizer=types.SimpleNamespace(path=str(path), id_map=mapping))
    path.write_bytes(b"changed contents")
    with pytest.raises(ValueError, match="hashes disagree"):
        inference._tokenizer_hash(model)


def test_actual_settings_capture_versions_and_full_encoder_config(tmp_path, monkeypatch):
    _, path, checkpoint, output, model = fixture(tmp_path, monkeypatch)
    inference.run_nemo_inference(path, checkpoint, "baseline", output)
    actual = json.loads(output.read_text())["actual_settings"]
    assert actual["encoder_config_sha256"] == inference._canonical_hash(model.cfg["encoder"])
    assert actual["runtime_versions"]["python"]
    assert "nemo_toolkit" in actual["runtime_versions"]


def test_declared_runtime_versions_cannot_override_actual_provenance(tmp_path, monkeypatch):
    manifest, path, checkpoint, output, _ = fixture(tmp_path, monkeypatch)
    manifest["runs"]["baseline"]["settings"]["runtime_versions"] = {"python": "invented"}
    save(manifest, path)
    with pytest.raises(ValueError, match="actual inference setting"):
        inference.run_nemo_inference(path, checkpoint, "baseline", output)
