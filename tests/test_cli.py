"""The CLI must not turn incomplete or failed evidence into success."""
import json

import pytest

from sttok.cli import main


@pytest.mark.parametrize("passed,expected", [(True, 0), (False, 2)])
def test_real_checkpoint_gate_exit_code(monkeypatch, capsys, passed, expected):
    import sttok.checkpoint_validation as validation

    def runner(*args, **kwargs):
        return {"passed": passed, "status": "migration_compatibility_passed_on_supplied_corpus" if passed else "migration_compatibility_failed"}

    monkeypatch.setattr(validation, "validate_checkpoint_pair", runner)
    result = main(["verify-checkpoint", "--source", "a.nemo", "--expanded", "b.nemo", "--manifest", "input.json", "--output", "report.json"])
    assert result == expected
    assert "status" in json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("state,expected", [("incomplete", 2), ("failed", 2), ("passed_on_supplied_text", 0)])
def test_text_evidence_gate_exit_code(monkeypatch, tmp_path, state, expected):
    import sttok.validation as validation

    monkeypatch.setattr(validation, "validate_tokenizer", lambda *args: {
        "status": state, "structural_passed": state != "failed", "errors": []
    })
    output = tmp_path / "report.json"
    assert main(["validate", "--output", str(output)]) == expected
    assert json.loads(output.read_text())["status"] == state
