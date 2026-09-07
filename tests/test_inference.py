"""Fake-model unit tests only; these do not establish NeMo integration or WER."""
import hashlib
import json
import sys
import types

import pytest

from sttok import inference


class FakeModel:
    def __init__(self, outputs=None):
        self.outputs = outputs if outputs is not None else ["one one two"]
        self.calls = []
        self.cfg = {
            "model_defaults": {"prompt_dictionary": {"en-US": 0, "auto": 1}},
            "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}},
            "encoder": {"att_context_size": [70, 13]},
        }
        self.tokenizer = types.SimpleNamespace(tokenizer=types.SimpleNamespace(serialized_model_proto=lambda: b"synthetic-tokenizer-unit-test"))

    def to(self, device):
        self.device = device
        return self

    def float(self):
        return self

    def eval(self):
        return self

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
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
    assert summary["streaming_validated"] is False
    assert output.with_suffix(".jsonl.run.json").is_file()


def test_automatic_mode_uses_real_auto_prompt(tmp_path, monkeypatch):
    _, path, checkpoint, output, model = fixture(tmp_path, monkeypatch, mode="automatic")
    inference.run_nemo_inference(path, checkpoint, "baseline", output)
    assert model.calls[0]["target_lang"] == "auto"
    row = json.loads(output.read_text())
    assert row["language_mode"] == "automatic"
    assert row["language_prompt"] is None


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
    config = {"tokenizer": {"type": "sttok_hf_bpe"}, "target": "sttok.runtime.ExtendedNemotronRNNTModel"}
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
    runtime = types.ModuleType("sttok.runtime")
    runtime.get_nemo_model_class = lambda: FakeExtended
    monkeypatch.setitem(sys.modules, "sttok.runtime", runtime)
    inference._load_model(tmp_path / "test.nemo", "cpu")
    assert [kind for kind, _ in calls] == ["native", "extended"]


def test_runtime_tokenizer_hash_rejects_unverifiable_tokenizer():
    with pytest.raises(ValueError, match="actual runtime tokenizer"):
        inference._tokenizer_hash(types.SimpleNamespace(tokenizer=object()))


def test_real_hf_adapter_hash_survives_deleted_restore_directory(tmp_path):
    """Actual tokenizer backend, still no NeMo checkpoint or audio inference."""
    from tokenizers import Tokenizer, models
    from sttok.runtime import HFTokenizerAdapter
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
