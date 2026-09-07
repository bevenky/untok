import hashlib
import json
from pathlib import Path

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.cli import main
from untok.unigram import build_native_tokenizer
from untok.unigram_validation import validate_native_tokenizer


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def candidate(tmp_path):
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.remove_extra_whitespaces = False
    for piece, score, kind in [("<unk>", 0, 2), ("▁", -1, 1), ("a", -2, 1), ("b", -2, 1), ("z", -8, 1)]:
        model.pieces.add(piece=piece, score=score, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    base = tmp_path / "base.model"
    base.write_bytes(model.SerializeToString())
    selection = tmp_path / "selection.json"
    write(selection, {"base_tokenizer_sha256": sha(base), "additions": [{"piece": "ab", "score": -1}]})
    bundle = tmp_path / "bundle"
    info = build_native_tokenizer(base, selection, bundle)
    policy = tmp_path / "policy.json"
    write(policy, {"schema_version": 1, "base_tokenizer_sha256": sha(base),
                   "normalizer_sha256": info["normalizer_sha256"],
                   "profiles": {"x": {"script": "Latn", "characters": ["a", "b"]}},
                   "protected_piece_groups": {"new": {"pieces": ["ab"]}},
                   "normalizer_probes": ["", "a", "  a  b  ", "\ta\nb", "ﬁ", "क्\u200cष"],
                   "exact_encoding_probes": ["a"]})
    data = tmp_path / "data" / "manifest.json"
    for phase in ("dev", "reserve"):
        path = data.parent / phase / "x.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('\n'.join(json.dumps({"text": text, "source": "toy", "record_id": str(i)}, ensure_ascii=False)
                                  for i, text in enumerate(["ab", "  a  b  ", "xy"])) + '\n')
    write(data, {"native_base_sha256": sha(base), "normalizer_sha256": info["normalizer_sha256"],
                 "files": {f"{phase}/x.jsonl": sha(data.parent / phase / "x.jsonl") for phase in ("dev", "reserve")}})
    return bundle, policy, data


def receipt(candidate):
    bundle, policy, data = candidate
    path = data.parent / "receipt.json"
    write(path, {"tokenizer_sha256": sha(bundle / "tokenizer.model"),
                 "data_manifest_sha256": sha(data), "bundle_manifest_sha256": sha(bundle / "manifest.json"),
                 "selection_sha256": sha(bundle / "selection.json"), "policy_sha256": sha(policy)})
    return path


def change(path, mutate):
    obj = json.loads(path.read_text()); mutate(obj); write(path, obj)


def test_portable_validation_and_known_segmentation_changes(candidate, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = validate_native_tokenizer(*candidate, max_examples=1)
    assert result["passed"] and result["status"] == "passed"
    assert result["structural_passed"] and result["corpus_status"] == "passed"
    assert result["corpora"]["x"]["base_representable_changed"] == 1
    assert result["corpora"]["x"]["unknown_records"] == 1
    assert result["corpora"]["x"]["unknown_codepoint_occurrences"] == 2
    assert result["corpora"]["x"]["roundtrip_failures"] == 0
    assert result["protected_piece_groups"]["new"]["passed"]
    assert result["no_added_match_probes"] > 0
    assert result["old_random_sequence_decode_trials"] == 10000
    assert not result["checkpoint_validated"] and not result["asr_validated"]


def test_missing_or_empty_corpora_are_incomplete(candidate):
    bundle, policy, data = candidate
    assert validate_native_tokenizer(bundle, policy)["status"] == "incomplete"
    change(data, lambda obj: obj["files"].pop("dev/x.jsonl"))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "incomplete" and not result["passed"]
    assert result["missing_profiles"] == ["x"]
    empty = data.parent / "dev/x.jsonl"
    empty.write_text("")
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(empty)}))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "incomplete" and result["empty_profiles"] == ["x"]


@pytest.mark.parametrize("target,field", [(1, "base_tokenizer_sha256"), (1, "normalizer_sha256"),
                                          (2, "native_base_sha256"), (2, "normalizer_sha256")])
def test_wrong_base_and_normalizer_bindings(candidate, target, field):
    change(candidate[target], lambda obj: obj.update({field: "bad"}))
    with pytest.raises(ValueError, match="hash"):
        validate_native_tokenizer(*candidate)


def test_phase_hash_and_escape_checks(candidate):
    _, _, data = candidate
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": "bad"}))
    with pytest.raises(ValueError, match="integrity"):
        validate_native_tokenizer(*candidate)
    change(data, lambda obj: obj["files"].update({"../outside": "bad"}))
    with pytest.raises(ValueError, match="contained"):
        validate_native_tokenizer(*candidate)


def test_dev_does_not_read_or_hash_reserve(candidate, monkeypatch):
    reserve = candidate[2].parent / "reserve/x.jsonl"
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self != reserve, "Dev validation opened sealed reserve"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    assert validate_native_tokenizer(*candidate)["passed"]


def test_reserve_requires_receipt_before_any_corpus_read(candidate, monkeypatch):
    original = Path.open
    def guard(self, *args, **kwargs):
        assert self.suffix != ".jsonl", "Read corpus before rejecting receipt"
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guard)
    with pytest.raises(ValueError, match="requires a selection receipt"):
        validate_native_tokenizer(*candidate, phase="reserve")
    rec = receipt(candidate)
    change(rec, lambda obj: obj.update({"policy_sha256": "bad"}))
    with pytest.raises(ValueError, match="policy_sha256"):
        validate_native_tokenizer(*candidate, phase="reserve", selection_receipt_path=rec)


@pytest.mark.parametrize("field", ["tokenizer_sha256", "data_manifest_sha256", "bundle_manifest_sha256", "selection_sha256", "policy_sha256"])
def test_every_reserve_binding_required(candidate, field):
    rec = receipt(candidate)
    change(rec, lambda obj: obj.pop(field))
    with pytest.raises(ValueError, match=field):
        validate_native_tokenizer(*candidate, phase="reserve", selection_receipt_path=rec)


def test_reserve_omits_examples_even_if_requested(candidate):
    result = validate_native_tokenizer(*candidate, phase="reserve", selection_receipt_path=receipt(candidate), max_examples=5)
    assert result["passed"]
    assert not result["corpora"]["x"]["base_representable_change_examples"]
    assert not result["corpora"]["x"]["roundtrip_failure_examples"]


def test_missing_protected_piece_or_coverage_is_failure(candidate):
    policy = candidate[1]
    change(policy, lambda obj: obj["protected_piece_groups"]["new"].update({"pieces": ["missing"]}))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "failed" and not result["structural_passed"]
    assert result["protected_piece_groups"]["new"]["missing"] == ["missing"]
    change(policy, lambda obj: obj["profiles"]["x"]["characters"].append("क"))
    assert validate_native_tokenizer(*candidate)["alphabet_coverage"]["x"]["missing"] == ["क"]


def test_roundtrip_expected_text_failure_not_hidden(candidate):
    data = candidate[2]
    text = data.parent / "dev/x.jsonl"
    text.write_text(json.dumps({"text": "ab", "expected_text": "wrong"}) + '\n')
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(text)}))
    result = validate_native_tokenizer(*candidate)
    assert result["status"] == "failed"
    assert result["corpora"]["x"]["roundtrip_failures"] == 1
    assert result["corpora"]["x"]["roundtrip_failure_examples"] == []


def test_malformed_record_fails_clearly(candidate):
    data = candidate[2]
    text = data.parent / "dev/x.jsonl"
    text.write_text('{"text": 42}\n')
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(text)}))
    with pytest.raises(ValueError, match="text string"):
        validate_native_tokenizer(*candidate)


def test_cli_writes_report_and_incomplete_is_nonzero(candidate, tmp_path, capsys):
    bundle, policy, data = candidate
    out = tmp_path / "report.json"
    args = ["validate-unigram", "--bundle", str(bundle), "--policy", str(policy), "--output", str(out)]
    assert main(args) == 2
    assert json.loads(out.read_text())["status"] == "incomplete"
    assert main(args + ["--corpora", str(data)]) == 0
    assert json.loads(out.read_text())["passed"]


@pytest.mark.parametrize("add_prefix", [False, True])
@pytest.mark.parametrize("as_suffix", [False, True])
@pytest.mark.parametrize("escape", [False, True])
def test_expected_text_matches_backend_whitespace_contract(add_prefix, as_suffix, escape):
    import sentencepiece as spm
    from untok.unigram_validation import _expected
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.trainer_spec.treat_whitespace_as_suffix = as_suffix
    model.normalizer_spec.name = "identity"
    model.normalizer_spec.remove_extra_whitespaces = False
    model.normalizer_spec.add_dummy_prefix = add_prefix
    model.normalizer_spec.escape_whitespaces = escape
    for piece, kind in [("<unk>", 2), ("▁", 1), (" ", 1), ("a", 1)]:
        model.pieces.add(piece=piece, score=-1, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    proc = spm.SentencePieceProcessor(model_proto=model.SerializeToString())
    for text in ("", " ", "a", " a", "a ", "  a  ", "▁a"):
        assert _expected(proc, model, text) == proc.decode(proc.encode(text))


def test_hash_valid_wrong_language_file_rejected(candidate):
    data = candidate[2]
    text = data.parent / "dev/x.jsonl"
    text.write_text(json.dumps({"text": "ab", "language": "wrong"}) + '\n')
    change(data, lambda obj: obj["files"].update({"dev/x.jsonl": sha(text)}))
    with pytest.raises(ValueError, match="language disagrees with profile x"):
        validate_native_tokenizer(*candidate)
