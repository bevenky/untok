"""Exercise corpus preparation with a pinned local ZIP and no network access."""

import json
import zipfile

import pytest

from untok.corpora import prepare_bhasha
from untok.sources import sha256, write_json


def row(identifier, script, text, source="independent source label"):
    return {"unique_identifier": identifier, "script": script,
            "native sentence": text, "source": source}


@pytest.fixture
def prepared_inputs(tmp_path, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline corpus test attempted a network download")
    monkeypatch.setattr("urllib.request.urlopen", no_network)
    output = tmp_path / "prepared"
    (output / "raw").mkdir(parents=True)
    config = tmp_path / "build.json"
    write_json(config, {"targets": [
        {"language": "doi", "script": "Devanagari"},
        {"language": "kok", "script": "Devanagari"},
        {"language": "ks", "script": "Arabic"},
        {"language": "mni", "script": "Meetei Mayek"},
        {"language": "sat", "script": "Ol Chiki"},
        {"language": "or", "script": "Odia"},
        {"language": "sd", "script": "Devanagari"},
        {"language": "kn", "script": "Kannada"},
    ]})

    def write_archive(rows, extra_member=False):
        archive = output / "raw/bhasha.zip"
        with zipfile.ZipFile(archive, "w") as zipped:
            zipped.writestr("bhasha-abhijnaanam.json", json.dumps({"data": rows}, ensure_ascii=False))
            if extra_member:
                zipped.writestr("../must-not-extract.txt", "unrelated archive member")
        lock = tmp_path / "corpus.lock.json"
        write_json(lock, {"files": [{"path": "bhasha.zip", "url": "https://offline.invalid/bhasha.zip", "sha256": sha256(archive)}]})
        return lock
    return output, config, write_archive


def test_language_and_script_aliases_preserve_original_spelling(prepared_inputs):
    output, config, write_archive = prepared_inputs
    original = "  क्\u200cष\t\n  "
    rows = [row("dg_1", "Devanagari", original),
            row("gom_2", "Devanagari", "कोंकणी"),
            row("ks_3", "Perso-Arabic", "کشمیری"),
            row("mni_4", "Meetei-Mayek", "ꯃꯤ"),
            row("sat_5", "Ol-Chiki", "ᱥᱟ"),
            row("or_6", "Oriya", "ଓଡ଼ିଆ"),
            row("sd_7", "Devanagari", "सिन्धी")]
    lock = write_archive(rows)
    result = prepare_bhasha(lock, config, output)
    assert result["missing_targets"] == ["kn"]
    assert not result["excluded_records"]
    specs = {spec["language"]: spec for spec in result["corpora"]}
    assert specs["doi"]["script"] == "Devanagari"
    assert specs["kok"]["script"] == "Devanagari"
    assert specs["ks"]["script"] == "Arabic"
    assert specs["mni"]["script"] == "Meetei Mayek"
    assert specs["sat"]["script"] == "Ol Chiki"
    assert specs["or"]["script"] == "Odia"
    actual = json.loads((output / "doi.jsonl").read_text())
    assert actual == {"id": "dg_1", "source_row": 0, "text": original,
                      "language": "doi", "script": "Devanagari", "source": rows[0]["source"]}


def test_out_of_scope_scripts_and_empty_text_have_explicit_exclusion_counts(prepared_inputs):
    output, config, write_archive = prepared_inputs
    rows = [row("sd_1", "Perso-Arabic", "سنڌي"), row("en_1", "Latin", "hello"),
            row("kn_1", "Kannada", ""), row("kn_2", "Kannada", None),
            row("dg_1", "Devanagari", "देवनागरी")]
    result = prepare_bhasha(write_archive(rows), config, output)
    assert result["excluded_records"] == {
        "empty_native_text": 2, "outside_declared_script:en:Latin": 1,
        "outside_declared_script:sd:Arabic": 1,
    }
    assert "sd" in result["missing_targets"]
    assert "kn" in result["missing_targets"]
    assert [spec["language"] for spec in result["corpora"]] == ["doi"]


def test_whitespace_only_text_does_not_satisfy_a_language_target(prepared_inputs):
    output, config, write_archive = prepared_inputs
    result = prepare_bhasha(write_archive([row("kn_1", "Kannada", " \t\n ")]), config, output)
    assert result["corpora"] == []
    assert "kn" in result["missing_targets"]
    assert result["excluded_records"]["empty_native_text"] == 1


def test_manifest_pins_archive_and_generated_files_with_relative_paths(prepared_inputs):
    output, config, write_archive = prepared_inputs
    lock = write_archive([row("dg_1", "Devanagari", "क")], extra_member=True)
    archive_before = (output / "raw/bhasha.zip").read_bytes()
    result = prepare_bhasha(lock, config, output)
    assert result["archive_sha256"] == sha256(output / "raw/bhasha.zip")
    assert (output / "raw/bhasha.zip").read_bytes() == archive_before
    assert result["corpora"][0]["path"] == "doi.jsonl"
    assert result["corpora"][0]["sha256"] == sha256(output / "doi.jsonl")
    assert result["corpora"][0]["records"] == 1
    assert not (output / "must-not-extract.txt").exists()
    assert not (output.parent / "must-not-extract.txt").exists()
    assert json.loads((output / "manifest.json").read_text()) == result


def test_duplicates_remain_and_preparation_is_repeatable(prepared_inputs):
    output, config, write_archive = prepared_inputs
    same_text = "क्\u200cष"
    lock = write_archive([row("dg_1", "Devanagari", same_text), row("doi_2", "Devanagari", same_text)])
    first = prepare_bhasha(lock, config, output)
    bytes_before = (output / "doi.jsonl").read_bytes()
    second = prepare_bhasha(lock, config, output)
    assert first == second
    assert bytes_before == (output / "doi.jsonl").read_bytes()
    rows = [json.loads(line) for line in bytes_before.splitlines()]
    assert [item["text"] for item in rows] == [same_text, same_text]
    assert [item["id"] for item in rows] == ["dg_1", "doi_2"]


def test_changed_cached_archive_is_rejected_before_processing(prepared_inputs):
    output, config, write_archive = prepared_inputs
    lock = write_archive([row("dg_1", "Devanagari", "क")])
    with (output / "raw/bhasha.zip").open("ab") as stream:
        stream.write(b"changed after pinning")
    with pytest.raises(ValueError, match="Cached source hash mismatch"):
        prepare_bhasha(lock, config, output)
    assert not (output / "manifest.json").exists()


def test_archive_download_path_cannot_escape_cache(prepared_inputs):
    output, config, write_archive = prepared_inputs
    lock = write_archive([])
    document = json.loads(lock.read_text())
    document["files"][0]["path"] = "../bhasha.zip"
    write_json(lock, document)
    with pytest.raises(ValueError, match="Source path escapes cache"):
        prepare_bhasha(lock, config, output)
