"""Behavioral tests using real, small tokenizers rather than mocked encodings."""

import hashlib
import json
from pathlib import Path

import pytest
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

from untok.validation import TARGET_SCRIPTS, expected_text, validate_tokenizer


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def artifacts(tmp_path):
    alphabet = sorted(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.!?()[]:%\t\n"))
    vocab = {"<unk>": 0, "▁": 1}
    for char in alphabet:
        if char != " ":
            vocab[char] = len(vocab)
    vocab["of"] = len(vocab)
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[("o", "f")], unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(), normalizers.Replace("\u200c", " "),
        normalizers.Strip(left=False, right=True),
        normalizers.Replace(Regex(" {2,}"), "▁"),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.add_special_tokens([AddedToken("<pad>", special=True), AddedToken("<blank>", special=True)])
    base_path = tmp_path / "base.json"
    tokenizer.save(str(base_path))
    document = json.loads(tokenizer.to_str())
    # HF reserves appended model positions before loading added-token records.
    # Materialize the existing reserved IDs so loading cannot reassign them.
    for row in document["added_tokens"]:
        document["model"]["vocab"][row["content"]] = row["id"]
    next_id = tokenizer.get_vocab_size(with_added_tokens=True)
    entries = []
    for char in dict.fromkeys("தமிழ்"):
        document["model"]["vocab"][char] = next_id
        entries.append({"piece": char, "id": next_id, "witness": char, "reachable": True})
        next_id += 1
    extended_path = dump(tmp_path / "extended.json", document)
    manifest_path = dump(tmp_path / "manifest.json", {
        "entries": entries,
        "expected_targets": [{"language": "ta", "script": "Taml"}],
        "required_characters": [{"character": char, "language": "ta", "script": "Taml"} for char in "தமிழ்"],
        "normalizer_controls": [{"text": "\ufb01 \uff21", "expected": "fi A"}],
    })
    return base_path, extended_path, manifest_path


def corpus(tmp_path, text="தமிழ்", language="ta", script="Taml", **fields):
    path = dump(tmp_path / "corpus.json", [{"text": text, **fields}])
    return [{"path": str(path), "language": language, "script": script,
             "source": "tiny test fixture", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]


def checks(report):
    return {row["name"]: row for row in report["checks"]}


def test_valid_extension_preserves_base_and_roundtrips_tamil(artifacts, tmp_path):
    report = validate_tokenizer(*artifacts, corpus(tmp_path, expected_text="தமிழ்"))
    assert report["status"] == "passed_on_supplied_text"
    assert report["structural_passed"]
    assert not report["errors"]
    assert report["reachability"]["failed"] == 0
    assert report["per_language"][0]["lost_normalized_characters"] == 0
    assert report["per_language"][0]["changed_sequences"] == 1
    assert report["asr_validation"] == "not_run"


def test_no_corpus_is_incomplete_not_language_success(artifacts):
    report = validate_tokenizer(*artifacts)
    assert report["structural_passed"]
    assert report["status"] == "incomplete"
    assert report["per_language"] == [{"language": "ta", "script": "Taml", "status": "missing_corpus", "records": 0}]


def test_all_22_language_script_requirements_are_independent(artifacts, tmp_path):
    required = [{"language": lang, "script": script} for lang, script in TARGET_SCRIPTS.items()]
    report = validate_tokenizer(*artifacts, corpus(tmp_path), required_targets=required)
    assert len(report["per_language"]) == 22
    assert len(report["missing_targets"]) == 21
    assert report["status"] == "incomplete"
    assert {"language": "sd", "script": "Deva"} in report["missing_targets"]


def test_wrong_script_does_not_satisfy_language_requirement(artifacts, tmp_path):
    report = validate_tokenizer(*artifacts, corpus(tmp_path, "hello", "sd", "Arab"), required_targets=[{"language": "sd", "script": "Deva"}])
    assert report["missing_targets"] == [{"language": "sd", "script": "Deva"}]


def test_display_names_match_canonical_language_script_codes(artifacts, tmp_path):
    report = validate_tokenizer(*artifacts, corpus(tmp_path, language="Tamil", script="Tamil"))
    assert report["status"] == "passed_on_supplied_text"
    assert report["per_language"][0]["language"] == "ta"
    assert report["per_language"][0]["script"] == "Taml"


def test_manifest_artifact_hash_is_checked(artifacts):
    manifest = json.loads(artifacts[2].read_text())
    manifest["base_sha256"] = "0" * 64
    dump(artifacts[2], manifest)
    report = validate_tokenizer(*artifacts)
    assert not checks(report)["manifest_base_sha256"]["passed"]


def test_old_id_change_is_detected_even_when_strings_still_encode(artifacts):
    doc = json.loads(artifacts[1].read_text())
    vocab = doc["model"]["vocab"]
    vocab["a"], vocab["b"] = vocab["b"], vocab["a"]
    dump(artifacts[1], doc)
    report = validate_tokenizer(*artifacts)
    assert not checks(report)["original_ids"]["passed"]
    assert report["status"] == "failed"


def test_normalizer_change_cannot_hide_behind_roundtrip(artifacts, tmp_path):
    doc = json.loads(artifacts[1].read_text())
    doc["normalizer"]["normalizers"][1]["content"] = ""
    dump(artifacts[1], doc)
    report = validate_tokenizer(*artifacts, corpus(tmp_path, "of\u200cfice"))
    assert not checks(report)["unchanged_normalizer"]["passed"]
    assert not checks(report)["normalizer_controls"]["passed"]
    assert report["per_language"][0]["roundtrip_failures"] == 1


def test_stored_piece_without_merge_path_fails_reachability(artifacts):
    doc = json.loads(artifacts[1].read_text())
    manifest = json.loads(artifacts[2].read_text())
    index = max(doc["model"]["vocab"].values()) + 1
    doc["model"]["vocab"]["தமிழ்"] = index
    manifest["entries"].append({"piece": "தமிழ்", "id": index, "witness": "தமிழ்", "reachable": True})
    dump(artifacts[1], doc)
    dump(artifacts[2], manifest)
    report = validate_tokenizer(*artifacts)
    assert not checks(report)["new_piece_reachability"]["passed"]
    assert report["reachability"]["failures"][0]["piece"] == "தமிழ்"


def test_appended_latin_merge_is_forbidden_and_changes_english(artifacts):
    doc = json.loads(artifacts[1].read_text())
    manifest = json.loads(artifacts[2].read_text())
    index = max(doc["model"]["vocab"].values()) + 1
    doc["model"]["vocab"]["fi"] = index
    doc["model"]["merges"].append(["f", "i"])
    manifest["entries"].append({"piece": "fi", "id": index, "witness": "fi"})
    dump(artifacts[1], doc)
    dump(artifacts[2], manifest)
    report = validate_tokenizer(*artifacts)
    assert not checks(report)["no_new_latin_merges"]["passed"]
    assert not checks(report)["protected_text_parity"]["passed"]
    assert checks(report)["original_merge_prefix"]["passed"]


def test_fused_unknowns_count_each_lost_unicode_character(artifacts, tmp_path):
    text = "ಕನ್ನಡ"
    report = validate_tokenizer(*artifacts, corpus(tmp_path, text, "kn", "Knda"), required_targets=[{"language": "kn", "script": "Knda"}])
    row = report["per_language"][0]
    assert row["unknown_tokens"] == 1
    assert row["lost_normalized_characters"] == len(text)
    assert sum(item["occurrences"] for item in row["missing_characters"]) == len(text)
    assert row["roundtrip_checked"] == 0
    assert report["status"] == "failed"


@pytest.mark.parametrize(("raw", "wanted"), [
    ("\ufb01 \uff21", "fi A"), ("  office  tomorrow  ", "office tomorrow"),
    ("a\u200cb", "a b"), ("தமிழ்", "தமிழ்"), ("", ""),
])
def test_expected_text_is_defined_without_encoding(artifacts, raw, wanted):
    base = Tokenizer.from_file(str(artifacts[0]))
    document = json.loads(artifacts[0].read_text())
    assert expected_text(base, document, raw) == wanted


def test_explicit_expected_text_fails_even_when_normalized_roundtrip_would_pass(artifacts, tmp_path):
    report = validate_tokenizer(*artifacts, corpus(tmp_path, "தமிழ்", expected_text="different"))
    assert report["per_language"][0]["roundtrip_failures"] == 1
    assert report["examples"][0]["expected"] == "different"


@pytest.mark.parametrize("kind", ["csv", "jsonl", "txt"])
def test_manifest_relative_paths_and_input_formats(artifacts, tmp_path, kind):
    values = {"csv": "text,language,script\nதமிழ்,ta,Taml\n", "jsonl": '{"text":"தமிழ்","language":"ta","script":"Taml"}\n', "txt": "தமிழ்\n"}
    path = tmp_path / f"texts.{kind}"
    path.write_text(values[kind], encoding="utf-8")
    manifest = dump(tmp_path / "corpora.json", {"corpora": [{"path": path.name, "language": "ta", "script": "Taml"}]})
    report = validate_tokenizer(*artifacts, manifest)
    assert report["status"] == "passed_on_supplied_text"
    assert report["per_language"][0]["records"] == 1
    assert not report["corpus_sources"][0]["hash_verified"]


def test_corpus_hash_mismatch_is_a_failure_not_silent_skip(artifacts, tmp_path):
    specs = corpus(tmp_path)
    specs[0]["sha256"] = "0" * 64
    report = validate_tokenizer(*artifacts, specs)
    assert report["status"] == "failed"
    assert any(error["code"] == "corpus_input" for error in report["errors"])


def test_zero_records_do_not_count_as_target_coverage(artifacts, tmp_path):
    path = dump(tmp_path / "empty.json", [])
    report = validate_tokenizer(*artifacts, [{"path": str(path), "language": "ta", "script": "Taml"}])
    assert report["status"] == "incomplete"
    assert len(report["missing_targets"]) == 1


def test_validation_is_repeatable_and_does_not_modify_inputs(artifacts, tmp_path):
    specs = corpus(tmp_path)
    before = [path.read_bytes() for path in artifacts]
    first = validate_tokenizer(*artifacts, specs)
    second = validate_tokenizer(*artifacts, specs)
    assert first == second
    assert before == [path.read_bytes() for path in artifacts]


def test_pinned_actual_base_if_available(tmp_path):
    """Optional local integration control, excluded from portable CI assumptions."""
    source = Path(__file__).parents[1] / ".cache" / "sources" / "nvidia" / "tokenizer.json"
    if not source.exists():
        pytest.skip("Pinned NVIDIA artifact has not been fetched into this checkout")
    manifest = dump(tmp_path / "empty-manifest.json", {"entries": [], "expected_targets": []})
    report = validate_tokenizer(source, source, manifest)
    assert report["structural_passed"]
    assert checks(report)["original_ids"]["detail"]["checked"] == 13089
    assert not report["errors"]
