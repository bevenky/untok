"""Runner contract tests with fake NeMo models, not real audio/ASR evidence."""
import hashlib
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models

import sttok.checkpoint_validation as runner
import sttok.inference as inference
from sttok.checkpoint import inspect_nemo_layout, old_model_row_mapping, transfer_state_dict
from sttok.runtime import HFTokenizerAdapter, build_id_map


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    backend = Tokenizer(models.BPE(vocab={"<unk>": 0, "▁": 1, "a": 2, "b": 3}, merges=[], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    backend.add_special_tokens(["<unk>", "<pad>", "<blank>"])
    base, extended = tmp_path / "base.json", tmp_path / "extended.json"
    backend.save(str(base))
    data = json.loads(base.read_text())
    data["model"]["vocab"].update({"<pad>": 4, "<blank>": 5, "क": 6})
    extended.write_text(json.dumps(data))

    class FakeDecoder(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.blank_idx, self.blank_as_pad = size, True
            self.prediction = torch.nn.ModuleDict({"embed": torch.nn.Embedding(size + 1, 2, padding_idx=size)})

    class FakeJoint(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self._num_extra_outputs = 0
            self.joint_net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(2, size + 1))

    class FakeNemoModel(torch.nn.Module):
        def __init__(self, tokenizer):
            super().__init__()
            self.tokenizer = HFTokenizerAdapter(tokenizer)
            size = self.tokenizer.vocab_size
            self.decoder = FakeDecoder(size)
            self.joint = FakeJoint(size)
            self.encoder = torch.nn.Linear(2, 2)
            self.prompt_kernel = torch.nn.Linear(130, 2)
            self.preprocessor = SimpleNamespace(_sample_rate=16000)
            self.num_prompts, self.concat = 128, True
            self.cfg = {"encoder": {"att_context_size": [56, 3]}, "preprocessor": {"sample_rate": 16000},
                        "decoder": {"vocab_size": size}, "joint": {"num_classes": size},
                        "model_defaults": {"num_prompts": 128, "enc_hidden": 2, "prompt_dictionary": {"hi-IN": 6, "auto": 101}},
                        "decoding": {"strategy": "greedy_batch", "greedy": {"max_symbols": 10}}}
            self.calls = []
            self.return_text_only = False
            self.wrong_prompt = False
            # Match the pinned NeMo default: omitted config flag enables graphs.
            self.decoding = SimpleNamespace(decoding=SimpleNamespace(
                max_symbols=10, loop_labels=True, use_cuda_graph_decoder=True,
                decoding_computer=SimpleNamespace(allow_cuda_graphs=True, cuda_graphs_mode="FULL_GRAPH")))

        def get_transcribe_config(self):
            return SimpleNamespace()

        def change_decoding_strategy(self, config, verbose=False):
            self.cfg["decoding"] = copy.deepcopy(config)
            graphs = config["greedy"]["use_cuda_graph_decoder"]
            self.decoding = SimpleNamespace(decoding=SimpleNamespace(
                max_symbols=config["greedy"]["max_symbols"], loop_labels=True, use_cuda_graph_decoder=graphs,
                decoding_computer=SimpleNamespace(allow_cuda_graphs=graphs, cuda_graphs_mode="FULL_GRAPH" if graphs else None)))

        def transcribe(self, **kwargs):
            self.calls.append(kwargs)
            assert isinstance(kwargs["audio"][0], torch.Tensor)
            conditioning = self.prompt_kernel.weight.new_zeros((1, 2, 130))
            prompt_id = self.cfg["model_defaults"]["prompt_dictionary"][kwargs["override_config"].target_lang]
            conditioning[..., 2 + (101 if self.wrong_prompt else prompt_id)] = 1
            self.prompt_kernel(conditioning)
            # FakeNeMo performs an actual head forward and softmax so the test
            # verifies masking occurs before normalization, with new rows active
            # on the subsequent separate call. No audio recognition is simulated.
            x = self.joint.joint_net[-1].weight.new_tensor([[1.0, 0.0]])
            probabilities = self.joint.joint_net(x).softmax(-1)
            token = int(probabilities.argmax(-1)[0])
            ids = [token, token] if token != self.decoder.blank_idx else []
            text = self.tokenizer.ids_to_text(ids)
            if self.return_text_only:
                return [text]
            return [SimpleNamespace(text=text, y_sequence=torch.tensor(ids, dtype=torch.long))]

    original, migrated = FakeNemoModel(base), FakeNemoModel(extended)
    with torch.no_grad():
        original.joint.joint_net[-1].weight.zero_()
        original.joint.joint_net[-1].bias.zero_()
        original.joint.joint_net[-1].bias[2] = 2
    row_map = old_model_row_mapping(build_id_map(base), build_id_map(extended, base))
    state = transfer_state_dict(original.state_dict(), migrated.state_dict(),
                                inspect_nemo_layout(original), inspect_nemo_layout(migrated), row_map)
    migrated.load_state_dict(state)
    with torch.no_grad():
        migrated.joint.joint_net[-1].weight[4].zero_()
        migrated.joint.joint_net[-1].bias[4] = 20
    source, output_checkpoint, audio = tmp_path / "original.nemo", tmp_path / "expanded.nemo", tmp_path / "fake.wav"
    source.write_bytes(b"fake original checkpoint fixture")
    output_checkpoint.write_bytes(b"fake expanded checkpoint fixture")
    audio.write_bytes(b"fake audio fixture; never processed by a real ASR model")
    manifest = {"schema_version": 1, "run_id": "fake-nemo-unit-test",
                "source_checkpoint_sha256": _digest(source), "expanded_checkpoint_sha256": _digest(output_checkpoint),
                "base_tokenizer_sha256": _digest(base), "expanded_tokenizer_sha256": _digest(extended),
                "source_runtime_tokenizer_sha256": _digest(base),
                "settings": {"mode": "offline", "dtype": "float32", "device": "cpu", "batch_size": 1,
                             "num_workers": 0, "decoder": "greedy_batch"},
                "utterances": [{"id": "fixture-1", "audio": audio.name, "audio_sha256": _digest(audio), "target_lang": "hi-IN"}]}
    manifest_path, report_path = tmp_path / "manifest.json", tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(runner, "_load_model", lambda path, device: original if Path(path) == source else migrated)
    monkeypatch.setattr(inference, "_load_audio_tensor", lambda *args: torch.zeros(16000))
    return SimpleNamespace(original=original, migrated=migrated, manifest=manifest,
                           manifest_path=manifest_path, report_path=report_path,
                           args=(source, output_checkpoint, base, extended, manifest_path, report_path))


def test_actual_runner_contract_with_fake_nemo_models(setup):
    report = runner.validate_checkpoint_pair(*setup.args)
    assert report["passed"] is True
    assert report["release_ready"] is False and report["asr_accuracy_evaluated"] is False
    assert report["rnnt_training_smoke_executed"] is False
    assert report["actual_settings"]["decoder_execution"] == "eager"
    for evidence in report["decoding_execution"].values():
        assert "use_cuda_graph_decoder" not in evidence["serialized_config"]["greedy"]
        assert evidence["effective_config"]["greedy"]["use_cuda_graph_decoder"] is False
        assert evidence["runtime"]["use_cuda_graph_decoder"] is False
        assert evidence["runtime"]["allow_cuda_graphs"] is False
        assert evidence["runtime"]["max_symbols"] == 10
    row = report["utterances"][0]
    assert row["baseline"]["text"] == "aa"
    assert row["expanded_old_outputs_only"]["text"] == "aa"
    assert row["expanded_all_outputs"]["text"] == "कक"
    assert row["old_output_token_parity"] is True
    assert row["joint_probes"]["passed"] is True
    assert row["joint_probes"]["max_fixed_input_replay_absolute_error"] == 0
    assert report["active_text_change_count"] == 1
    assert report["active_token_change_count"] == 1
    assert report["initialization_preservation_passed"] is False
    assert len(setup.original.calls) == 1 and len(setup.migrated.calls) == 2
    assert not setup.original.joint.joint_net[-1]._forward_hooks
    assert not setup.migrated.joint.joint_net[-1]._forward_hooks
    assert not setup.original.prompt_kernel._forward_pre_hooks
    assert not setup.migrated.prompt_kernel._forward_pre_hooks
    assert all(e["prompt_id"] == 6 and e["verified_kernel_calls"] == 1
               for e in row["prompt_evidence"].values())
    assert "FakeNemoModel" in report["source_model_class"]
    assert json.loads(setup.report_path.read_text()) == report


@pytest.mark.parametrize("corruption", ["empty_corpus", "missing_hash", "wrong_audio_hash", "streaming"])
def test_missing_or_incompatible_evidence_never_passes(setup, corruption, monkeypatch):
    if corruption == "empty_corpus":
        setup.manifest["utterances"] = []
    elif corruption == "missing_hash":
        del setup.manifest["source_runtime_tokenizer_sha256"]
    elif corruption == "wrong_audio_hash":
        setup.manifest["utterances"][0]["audio_sha256"] = "0" * 64
    else:
        setup.manifest["settings"]["mode"] = "streaming"
    setup.manifest_path.write_text(json.dumps(setup.manifest))
    monkeypatch.setattr(runner, "_load_model", lambda *args: pytest.fail("Invalid evidence must fail before loading weights"))
    with pytest.raises(ValueError):
        runner.validate_checkpoint_pair(*setup.args)
    assert not setup.report_path.exists()


def test_rejects_changed_old_weights_and_unavailable_baseline_prompt(setup):
    torch = pytest.importorskip("torch")
    with torch.no_grad():
        setup.migrated.encoder.weight[0, 0] += 1
    with pytest.raises(ValueError, match="Learned state changed"):
        runner.validate_checkpoint_pair(*setup.args)
    assert not setup.report_path.exists()


def test_new_language_prompt_cannot_be_silently_substituted_for_baseline(setup):
    setup.manifest["utterances"][0]["target_lang"] = "or-IN"
    setup.manifest_path.write_text(json.dumps(setup.manifest))
    setup.migrated.cfg["model_defaults"]["prompt_dictionary"]["or-IN"] = 78
    with pytest.raises(ValueError, match="no automatic fallback"):
        runner.validate_checkpoint_pair(*setup.args)
    assert not setup.report_path.exists()


def test_text_only_results_fail_and_hooks_are_removed(setup):
    setup.original.return_text_only = True
    with pytest.raises(ValueError, match="y_sequence"):
        runner.validate_checkpoint_pair(*setup.args)
    assert not setup.report_path.exists()
    assert not setup.original.joint.joint_net[-1]._forward_hooks


def test_transcript_regression_is_reported_as_failure(setup, monkeypatch):
    original_transcribe = setup.migrated.transcribe

    def wrong_text(**kwargs):
        result = original_transcribe(**kwargs)
        result[0].text += " unexpected"
        return result

    monkeypatch.setattr(setup.migrated, "transcribe", wrong_text)
    report = runner.validate_checkpoint_pair(*setup.args)
    assert report["passed"] is False
    assert report["status"] == "migration_compatibility_failed"
    assert report["utterances"][0]["old_output_token_parity"] is True
    assert report["utterances"][0]["old_output_text_parity"] is False


def test_no_forward_hooks_means_no_fabricated_logit_pass(setup, monkeypatch):
    monkeypatch.setattr(setup.original, "transcribe", lambda **kwargs: [SimpleNamespace(text="aa", y_sequence=[2, 2])])
    with pytest.raises(ValueError, match="hooks did not execute"):
        runner.validate_checkpoint_pair(*setup.args)
    assert not setup.report_path.exists()


def test_infinite_tolerance_cannot_disable_numerical_checks(setup):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        runner.validate_checkpoint_pair(*setup.args, atol=float("inf"))


def test_wrong_actual_prompt_cannot_pass_migration_comparison(setup):
    setup.migrated.wrong_prompt = True
    with pytest.raises(ValueError, match="Actual conditioning prompt differs"):
        runner.validate_checkpoint_pair(*setup.args)
    assert not setup.report_path.exists()
    assert not setup.migrated.prompt_kernel._forward_pre_hooks
    assert not setup.migrated.joint.joint_net[-1]._forward_hooks


@pytest.mark.parametrize("failure", ["graph_flag", "computer_flag", "active_graph", "max_symbols", "weights"])
def test_eager_setup_rejects_unapplied_flags_or_unintended_changes(setup, monkeypatch, failure):
    torch = pytest.importorskip("torch")
    real_change = setup.original.change_decoding_strategy

    def broken_change(config, **kwargs):
        real_change(config, **kwargs)
        decoder = setup.original.decoding.decoding
        if failure == "graph_flag":
            decoder.use_cuda_graph_decoder = True
        elif failure == "computer_flag":
            decoder.decoding_computer.allow_cuda_graphs = True
        elif failure == "active_graph":
            decoder.decoding_computer.cuda_graphs_mode = "FULL_GRAPH"
        elif failure == "max_symbols":
            decoder.max_symbols = 20
        else:
            with torch.no_grad():
                setup.original.encoder.weight.add_(1)

    monkeypatch.setattr(setup.original, "change_decoding_strategy", broken_change)
    with pytest.raises(ValueError, match="CUDA graph|algorithm or max_symbols|Model tensors changed"):
        runner.validate_checkpoint_pair(*setup.args)
    assert setup.original.calls == []
    assert not setup.report_path.exists()


def test_serialized_decoder_mismatch_is_rejected_before_eager_setup(setup):
    setup.migrated.cfg["decoding"]["greedy"]["max_symbols"] = 20
    with pytest.raises(ValueError, match="Inference configuration changed"):
        runner.validate_checkpoint_pair(*setup.args)
    assert setup.original.decoding.decoding.use_cuda_graph_decoder is True
    assert setup.original.calls == []
