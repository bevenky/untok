"""A declared script policy must never filter away its own coverage failures."""

import hashlib
import json

import pytest
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers

from untok.scope import allowed_characters, scope_corpora
from untok.sources import sha256, write_json
from untok.validation import validate_tokenizer


@pytest.fixture
def scope_inputs(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    vocab = {"<unk>": 0, "▁": 1, "a": 2, "b": 3, "f": 4, "i": 5}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>", fuse_unk=True))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.save(str(cache / "base.json"))
    # Assigned character Tamil number one thousand is absent from both models.
    tokenizer.save(str(tmp_path / "extended.json"))
    (cache / "UnicodeData.txt").write_text(
        "0B85;TAMIL LETTER A;Lo;0;L;;;;;N;;;;;\n"
        "0BF2;TAMIL NUMBER ONE THOUSAND;No;0;L;;;;;N;;;;;\n"
        "10D0;GEORGIAN LETTER AN;Ll;0;L;;;;;N;;;;;\n",
        encoding="utf-8",
    )
    (tmp_path / "latin.tsv").write_text("0\tð\n", encoding="utf-8")
    config = {
        "base_tokenizer": "base.json", "unicode_data": "UnicodeData.txt",
        "targets": [{"language": "ta", "script": "Tamil", "ranges": [[0x0B80, 0x0BFF]]}],
        "required_characters": [{"character": ":"}], "latin_characters": "latin.tsv",
    }
    write_json(tmp_path / "config.json", config)
    write_json(tmp_path / "build-manifest.json", {"entries": [], "expected_targets": [{"language": "ta", "script": "Taml"}]})
    return tmp_path, cache, config


def make_manifest(root, records, *, text_field="text"):
    source = root / "input.jsonl"
    source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
    manifest = root / "corpora.json"
    write_json(manifest, {"corpora": [{"path": source.name, "format": "jsonl",
               "language": "ta", "script": "Taml", "text_field": text_field,
               "source": "offline scope fixture", "sha256": sha256(source)}]})
    return manifest, source


def test_scope_retains_uncovered_assigned_native_character_and_validation_fails(scope_inputs):
    root, cache, config = scope_inputs
    text = "a௲a"
    assert "௲" not in Tokenizer.from_file(str(cache / "base.json")).get_vocab()
    assert "௲" not in Tokenizer.from_file(str(root / "extended.json")).get_vocab()
    assert "௲" in allowed_characters(config, cache)
    manifest, _ = make_manifest(root, [{"id": "native-gap", "text": text}])

    result = scope_corpora(root / "config.json", cache, manifest, root / "scoped")
    assert result["retained_records"] == 1
    assert result["excluded_count"] == 0
    validation = validate_tokenizer(cache / "base.json", root / "extended.json",
                                    root / "build-manifest.json", root / "scoped/manifest.json")
    assert validation["status"] == "failed"
    assert validation["per_language"][0]["unknown_records"] == 1
    assert validation["per_language"][0]["missing_characters"] == [
        {"character": "௲", "codepoint": "U+0BF2", "occurrences": 1}
    ]


def test_scope_excludes_whole_foreign_script_records_and_preserves_retained_rows(scope_inputs):
    root, cache, _ = scope_inputs
    retained = {"id": "unchanged", "text": "  a\t௲\n  ", "metadata": {"spelling": "original"}}
    foreign = {"id": "foreign", "text": "aაa"}
    manifest, source = make_manifest(root, [retained, foreign])
    before = source.read_bytes()

    result = scope_corpora(root / "config.json", cache, manifest, root / "scoped")
    assert source.read_bytes() == before
    output_rows = [json.loads(line) for line in (root / "scoped/00.jsonl").read_text().splitlines()]
    assert output_rows == [retained]
    assert result["retained_records"] == 1
    assert result["excluded_count"] == 1
    assert result["excluded_records"] == [{
        "source_index": 0, "record": 2, "record_id": "foreign",
        "text_sha256": hashlib.sha256(foreign["text"].encode()).hexdigest(),
        "outside_policy": ["U+10D0"],
    }]
    assert result["source_reports"][0]["outside_policy_characters"] == [
        {"character": "ა", "codepoint": "U+10D0", "records": 1}
    ]


def test_scope_uses_normalized_policy_but_never_normalizes_output(scope_inputs):
    root, cache, _ = scope_inputs
    rows = [{"text": "ﬁ"}, {"text": "ð"}, {"text": ":"}]
    manifest, _ = make_manifest(root, rows)
    result = scope_corpora(root / "config.json", cache, manifest, root / "scoped")
    assert result["retained_records"] == 3
    assert [json.loads(line) for line in (root / "scoped/00.jsonl").read_text().splitlines()] == rows


def test_unassigned_native_block_character_is_outside_declared_policy(scope_inputs):
    root, cache, config = scope_inputs
    # U+0B80 is in the configured block but absent from pinned assigned data.
    assert "\u0b80" not in allowed_characters(config, cache)
    manifest, _ = make_manifest(root, [{"text": "\u0b80"}])
    result = scope_corpora(root / "config.json", cache, manifest, root / "scoped")
    assert result["retained_records"] == 0
    assert result["excluded_records"][0]["outside_policy"] == ["U+0B80"]


@pytest.mark.parametrize("hash_value", [None, "0" * 64])
def test_scope_refuses_unverified_or_changed_corpus(scope_inputs, hash_value):
    root, cache, _ = scope_inputs
    manifest, _ = make_manifest(root, [{"text": "a"}])
    document = json.loads(manifest.read_text())
    if hash_value is None:
        del document["corpora"][0]["sha256"]
    else:
        document["corpora"][0]["sha256"] = hash_value
    write_json(manifest, document)
    with pytest.raises(ValueError, match="verified corpus hash"):
        scope_corpora(root / "config.json", cache, manifest, root / "scoped")


def test_scope_preserves_custom_text_field_and_emits_verified_relative_paths(scope_inputs):
    root, cache, _ = scope_inputs
    manifest, source = make_manifest(root, [{"body": "a௲"}], text_field="body")
    result = scope_corpora(root / "config.json", cache, manifest, root / "scoped")
    spec = result["corpora"][0]
    assert spec["path"] == "00.jsonl"
    assert spec["text_field"] == "body"
    assert spec["format"] == "jsonl"
    assert spec["records"] == 1
    assert spec["sha256"] == sha256(root / "scoped/00.jsonl")
    assert result["source_reports"][0]["input_sha256"] == sha256(source)
    assert result["policy_inputs_sha256"] == {
        "base_tokenizer": sha256(cache / "base.json"),
        "unicode_data": sha256(cache / "UnicodeData.txt"),
        "latin_characters": sha256(root / "latin.tsv"),
    }


def test_scope_exclusion_frequency_counts_records_not_occurrences(scope_inputs):
    root, cache, _ = scope_inputs
    manifest, _ = make_manifest(root, [{"text": "აააა"}, {"text": "ა"}])
    result = scope_corpora(root / "config.json", cache, manifest, root / "scoped")
    assert result["source_reports"][0]["outside_policy_characters"][0]["records"] == 2


@pytest.mark.parametrize("changed_input", ["base.json", "UnicodeData.txt", "latin.tsv"])
def test_scope_rejects_changed_pinned_policy_dependencies(scope_inputs, changed_input):
    root, cache, config = scope_inputs
    write_json(root / "sources.lock.json", {"files": [
        {"path": name, "sha256": sha256(cache / name)}
        for name in ("base.json", "UnicodeData.txt")
    ]})
    config.update(sources_lock="sources.lock.json", latin_sha256=sha256(root / "latin.tsv"))
    write_json(root / "config.json", config)
    manifest, _ = make_manifest(root, [{"text": "a௲"}])
    initial = scope_corpora(root / "config.json", cache, manifest, root / "verified")
    assert initial["retained_records"] == 1

    changed_path = root / changed_input if changed_input == "latin.tsv" else cache / changed_input
    with changed_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    message = "Latin character inventory hash changed" if changed_input == "latin.tsv" else "Missing or changed pinned source"
    with pytest.raises(ValueError, match=message):
        scope_corpora(root / "config.json", cache, manifest, root / "must-not-publish")
    assert not (root / "must-not-publish").exists()
