"""The CLI must not turn incomplete or failed evidence into success."""
import json
from types import SimpleNamespace

import pytest

from sttok.cli import main


@pytest.mark.parametrize("passed,expected", [(True, 0), (False, 2)])
def test_real_checkpoint_gate_exit_code(monkeypatch, capsys, passed, expected):
    import sttok.checkpoint_validation as validation

    def runner(*args, **kwargs):
        return {"passed": passed, "status": "migration_compatibility_passed_on_supplied_corpus" if passed else "migration_compatibility_failed"}

    monkeypatch.setattr(validation, "validate_checkpoint_pair", runner)
    result = main(["legacy-bpe", "verify-checkpoint", "--source", "a.nemo", "--expanded", "b.nemo", "--manifest", "input.json", "--output", "report.json"])
    assert result == expected
    assert "status" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("state,expected", [("incomplete", 2), ("failed", 2), ("passed_on_supplied_text", 0)])
def test_text_evidence_gate_exit_code(monkeypatch, tmp_path, state, expected):
    import sttok.validation as validation

    monkeypatch.setattr(validation, "validate_tokenizer", lambda *args: {
        "status": state, "structural_passed": state != "failed", "errors": []
    })
    output = tmp_path / "report.json"
    assert main(["legacy-bpe", "validate", "--output", str(output)]) == expected
    assert json.loads(output.read_text())["status"] == state


def unexpected_dispatch(*args, **kwargs):
    pytest.fail("The CLI dispatched to the wrong tokenizer implementation")


@pytest.mark.parametrize("command", ["build", "build-unigram"])
def test_build_and_alias_use_native_builder(monkeypatch, capsys, command):
    import sttok.builder as bpe
    import sttok.unigram as native

    calls = []

    def build(*args):
        calls.append(args)
        return {"structural_passed": True, "checkpoint_validated": False, "asr_validated": False}

    monkeypatch.setattr(native, "build_native_tokenizer", build)
    monkeypatch.setattr(bpe, "build_tokenizer", unexpected_dispatch)
    assert main([command, "--base", "base.model", "--selection", "selected.json", "--output", "bundle"]) == 0
    assert calls == [("base.model", "selected.json", "bundle")]
    assert json.loads(capsys.readouterr().out)["structural_passed"]


@pytest.mark.parametrize("command", ["check", "check-unigram"])
def test_check_and_alias_verify_native_bundle(monkeypatch, capsys, command):
    import sttok.unigram as native

    calls = []

    def adapter(bundle):
        calls.append(bundle)
        return SimpleNamespace(vocab_size=20, blank_id=20, id_map=SimpleNamespace(tokenizer_sha256="verified"))

    monkeypatch.setattr(native, "NativeTokenizerAdapter", adapter)
    assert main([command, "--bundle", "candidate"]) == 0
    assert calls == ["candidate"]
    assert json.loads(capsys.readouterr().out) == {
        "structural_passed": True, "native_vocabulary_size": 20, "native_blank_id": 20,
        "tokenizer_sha256": "verified", "checkpoint_validated": False, "asr_validated": False,
    }


@pytest.mark.parametrize("command", ["validate", "validate-unigram"])
@pytest.mark.parametrize("state,expected", [("passed", 0), ("incomplete", 2), ("failed", 2)])
def test_native_validation_dispatch_and_evidence_gate(monkeypatch, tmp_path, capsys, command, state, expected):
    import sttok.unigram_validation as native
    import sttok.validation as bpe

    calls = []
    report = {"passed": state == "passed", "status": state, "structural_passed": state != "failed",
              "corpus_status": state}

    def validate(*args, **kwargs):
        calls.append((args, kwargs))
        return report

    monkeypatch.setattr(native, "validate_native_tokenizer", validate)
    monkeypatch.setattr(bpe, "validate_tokenizer", unexpected_dispatch)
    output = tmp_path / "report.json"
    assert main([command, "--bundle", "candidate", "--policy", "policy.json", "--corpora", "corpora.json",
                 "--phase", "reserve", "--selection-receipt", "receipt.json", "--output", str(output)]) == expected
    assert calls == [(("candidate", "policy.json", "corpora.json"),
                     {"phase": "reserve", "selection_receipt_path": "receipt.json", "max_examples": 0})]
    assert json.loads(output.read_text()) == report
    assert not json.loads(capsys.readouterr().out)["checkpoint_validated"]


@pytest.mark.parametrize("command", ["build", "build-unigram", "check", "check-unigram", "validate", "validate-unigram"])
def test_native_commands_require_explicit_inputs(command):
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["build", "build-unigram"])
def test_native_build_failure_never_falls_back_to_bpe(monkeypatch, capsys, command):
    import sttok.builder as bpe
    import sttok.unigram as native

    def reject(*args):
        raise ValueError("Selection was not fitted against this native base hash")

    monkeypatch.setattr(native, "build_native_tokenizer", reject)
    monkeypatch.setattr(bpe, "build_tokenizer", unexpected_dispatch)
    assert main([command, "--base", "wrong.model", "--selection", "selection.json", "--output", "bundle"]) == 1
    captured = capsys.readouterr()
    assert not captured.out
    assert "native base hash" in captured.err


def test_invalid_native_bundle_returns_error(monkeypatch, capsys):
    import sttok.unigram as native

    def reject(bundle):
        raise ValueError("Native bundle file hash mismatch")

    monkeypatch.setattr(native, "NativeTokenizerAdapter", reject)
    assert main(["check", "--bundle", "changed-bundle"]) == 1
    assert "hash mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["fetch", "fetch-corpora", "scope-corpora", "preflight", "id-map",
                                    "prompts", "migrate", "verify-checkpoint", "infer"])
def test_old_commands_require_explicit_legacy_namespace(monkeypatch, command):
    import sttok.legacy_bpe_cli as legacy

    monkeypatch.setattr(legacy, "main", unexpected_dispatch)
    with pytest.raises(SystemExit) as error:
        main([command])
    assert error.value.code == 2


@pytest.mark.parametrize("command", ["build-unigram", "check-unigram", "validate-unigram"])
def test_native_aliases_are_not_legacy_commands(command):
    with pytest.raises(SystemExit) as error:
        main(["legacy-bpe", command])
    assert error.value.code == 2


def test_legacy_build_keeps_original_defaults_and_previous(monkeypatch, capsys):
    import sttok.builder as bpe
    import sttok.unigram as native

    calls = []

    def build(*args, **kwargs):
        calls.append((args, kwargs))
        return {"build_passed": True, "vocabulary_size": 42, "merge_count": 10,
                "tokenizer_sha256": "legacy", "asr_validated": False}

    monkeypatch.setattr(bpe, "build_tokenizer", build)
    monkeypatch.setattr(native, "build_native_tokenizer", unexpected_dispatch)
    assert main(["legacy-bpe", "build", "--previous", "prior-release"]) == 0
    assert calls == [(("configs/build.json", ".cache/sources", "artifacts/nemotron-indic-v1"),
                     {"previous": "prior-release"})]
    assert json.loads(capsys.readouterr().out)["tokenizer_sha256"] == "legacy"


def test_legacy_migration_preserves_default_bpe_inputs(monkeypatch):
    import sttok.checkpoint as checkpoint

    calls = []

    def migrate(*args, **kwargs):
        calls.append((args, kwargs))
        return {"migration": "test"}

    monkeypatch.setattr(checkpoint, "migrate_nemo_checkpoint", migrate)
    assert main(["legacy-bpe", "migrate", "--source", "source.nemo", "--output", "expanded.nemo"]) == 0
    assert calls == [(("source.nemo", "artifacts/nemotron-indic-v1/tokenizer.json",
                      ".cache/sources/nvidia/tokenizer.json", "expanded.nemo"),
                     {"seed": 0, "prompt_dictionary": None})]


@pytest.mark.parametrize("prefix", [[], ["legacy-bpe"]])
@pytest.mark.parametrize("state,expected", [("passed", 0), ("incomplete", 2), ("failed", 2)])
def test_generic_prediction_evaluation_remains_available(monkeypatch, prefix, state, expected):
    import sttok.evaluation as evaluation

    calls = []
    monkeypatch.setattr(evaluation, "load_manifest", lambda path: ("manifest", path))
    monkeypatch.setattr(evaluation, "load_predictions", lambda path: ("predictions", path))

    def score(*args, **kwargs):
        calls.append((args, kwargs))
        return {"release_status": state}

    monkeypatch.setattr(evaluation, "evaluate_predictions", score)
    monkeypatch.setattr(evaluation, "write_report", lambda report, path: None)
    assert main(prefix + ["evaluate", "--manifest", "speech.json", "--predictions", "predictions.json"]) == expected
    assert calls == [((("manifest", "speech.json"), ("predictions", "predictions.json")),
                     {"bootstrap_samples": 2000, "seed": 0})]


def test_help_explains_native_default_legacy_scope_and_migration(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])
    assert error.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "native SentencePiece Unigram" in text
    assert "Native checkpoint migration is pending" in text
    assert "legacy-bpe" in text
    assert "build-unigram" in text
    with pytest.raises(SystemExit) as error:
        main(["legacy-bpe", "--help"])
    assert error.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert "sttok legacy-bpe" in text
    assert "original arguments and defaults" in text
    assert "migrate" in text
    assert "build-unigram" not in text
