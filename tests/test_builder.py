import hashlib
import json
from pathlib import Path

import pytest
from sentencepiece import sentencepiece_model_pb2 as pb
from tokenizers import AddedToken, Tokenizer, decoders, models, pre_tokenizers

from sttok.builder import all_ids, build_tokenizer
from sttok.sources import sha256, source_path, write_json


@pytest.fixture
def project(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    vocab = {p: i for i, p in enumerate(["<unk>", "▁", "a", "b", "x", "क", "ा", "▁a"])}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[("▁", "a")], unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always", split=True)
    tok.add_special_tokens([AddedToken("<pad>", special=True), AddedToken("<blank>", special=True)])
    tok.save(str(cache / "base.json"))
    targets = []
    for lang, pieces in (("hi", ["▁", "क", "ा", "▁क", "का"]), ("ta", ["▁", "க", "ல", "கல", "▁கல"])):
        proto = pb.ModelProto()
        proto.trainer_spec.model_type = pb.TrainerSpec.BPE
        for rank, piece in enumerate(pieces):
            proto.pieces.add(piece=piece, score=-rank, type=pb.ModelProto.SentencePiece.NORMAL)
        (cache / f"{lang}.model").write_bytes(proto.SerializeToString())
        targets.append({"language": lang, "script": "Devanagari" if lang == "hi" else "Tamil", "donor": f"{lang}.model"})
    (tmp_path / "hindi.tsv").write_text("▁क\t0\nका\t-1\n")
    (tmp_path / "latin.tsv").write_text("U+00F0\tð\tLATIN SMALL LETTER ETH\n")
    chars = set("▁काகலð")
    (cache / "unicode.txt").write_text("\n".join(f"{ord(c):04X};TEST;Lo" for c in sorted(chars)))
    (cache / "scripts.txt").write_text("0041..005A ; Latin\n0061..007A ; Latin\n00C0..00FF ; Latin\n")
    files = [{"path": p.name, "sha256": sha256(p), "url": "https://example.invalid/" + p.name} for p in cache.iterdir()]
    write_json(tmp_path / "sources.json", {"files": files})
    cfg = {"sources_lock": "sources.json", "base_tokenizer": "base.json", "normalizer": "1A", "first_new_id": 10,
           "reserved_ids": {"<pad>": 8, "<blank>": 9}, "base_merge_count": 1,
           "unicode_data": "unicode.txt", "unicode_scripts": "scripts.txt", "targets": targets,
           "hindi_pieces": "hindi.tsv", "hindi_sha256": sha256(tmp_path / "hindi.tsv"), "hindi_count": 2,
           "latin_characters": "latin.tsv", "latin_sha256": sha256(tmp_path / "latin.tsv"), "latin_count": 1,
           "required_characters": []}
    write_json(tmp_path / "build.json", cfg)
    return tmp_path, cache, cfg


def test_build_preserves_loaded_ids_and_reaches_new_pieces(project):
    root, cache, _ = project
    m = build_tokenizer(root / "build.json", cache, root / "out")
    base = Tokenizer.from_file(str(cache / "base.json"))
    extended = Tokenizer.from_file(str(root / "out/tokenizer.json"))
    for piece, index in base.get_vocab().items():
        assert extended.token_to_id(piece) == index
    assert m["build_passed"] and all(e["reachable"] for e in m["entries"])
    assert extended.encode("ab ax bb").ids == base.encode("ab ax bb").ids
    for text in ("கல", "का", "ð"):
        ids = extended.encode(text).ids
        assert 0 not in ids
        assert extended.decode(ids, skip_special_tokens=False) == text
    assert extended.token_to_id("<pad>") == 8
    assert extended.token_to_id("<blank>") == 9
    assert json.loads((root / "out/tokenizer.json").read_text())["model"]["vocab"]["<pad>"] == 8


def test_build_deterministic_and_upgrade_stable(project):
    root, cache, cfg = project
    build_tokenizer(root / "build.json", cache, root / "first")
    build_tokenizer(root / "build.json", cache, root / "second")
    assert (root / "first/tokenizer.json").read_bytes() == (root / "second/tokenizer.json").read_bytes()
    assert (root / "first/manifest.json").read_bytes() == (root / "second/manifest.json").read_bytes()
    cfg["targets"].reverse()
    write_json(root / "build.json", cfg)
    build_tokenizer(root / "build.json", cache, root / "upgrade", previous=root / "first")
    assert (root / "first/tokenizer.json").read_bytes() == (root / "upgrade/tokenizer.json").read_bytes()


def test_tampered_source_fails(project):
    root, cache, _ = project
    (cache / "ta.model").write_bytes(b"modified")
    with pytest.raises(ValueError, match="changed pinned source"):
        build_tokenizer(root / "build.json", cache, root / "out")


def test_unlocked_input_is_rejected(project):
    root, cache, cfg = project
    (cache / "unlocked.model").write_bytes((cache / "ta.model").read_bytes())
    cfg["targets"][1]["donor"] = "unlocked.model"
    write_json(root / "build.json", cfg)
    with pytest.raises(ValueError, match="Every consumed source"):
        build_tokenizer(root / "build.json", cache, root / "out")


def test_source_paths_cannot_escape_cache(tmp_path):
    with pytest.raises(ValueError, match="escapes cache"):
        source_path(tmp_path, "../private")


def test_unknown_normalizer_cannot_silently_replace_a(project):
    root, cache, cfg = project
    cfg["normalizer"] = "1C"
    write_json(root / "build.json", cfg)
    with pytest.raises(ValueError, match="Only the approved 1A"):
        build_tokenizer(root / "build.json", cache, root / "out")


def test_reserved_id_collision_is_rejected(project):
    root, cache, cfg = project
    cfg["first_new_id"] = 9
    write_json(root / "build.json", cfg)
    with pytest.raises(ValueError, match="First new ID"):
        build_tokenizer(root / "build.json", cache, root / "out")


def test_tampered_previous_release_fails(project):
    root, cache, _ = project
    build_tokenizer(root / "build.json", cache, root / "first")
    (root / "first/tokenizer.json").write_text((root / "first/tokenizer.json").read_text() + " ")
    with pytest.raises(ValueError, match="Previous release hash"):
        build_tokenizer(root / "build.json", cache, root / "out", previous=root / "first")


def test_required_inventory_cannot_add_unapproved_latin(project):
    root, cache, cfg = project
    unicode_path = cache / "unicode.txt"
    unicode_path.write_text(unicode_path.read_text() + "\n00F1;LATIN SMALL LETTER N WITH TILDE;Ll\n")
    lock = json.loads((root / "sources.json").read_text())
    for item in lock["files"]:
        if item["path"] == "unicode.txt":
            item["sha256"] = sha256(unicode_path)
    write_json(root / "sources.json", lock)
    cfg["required_characters"] = [{"character": "ñ"}]
    write_json(root / "build.json", cfg)
    with pytest.raises(ValueError, match="Unapproved Latin character"):
        build_tokenizer(root / "build.json", cache, root / "out")


def test_standard_exemplar_gaps_are_covered_in_real_candidate():
    path = Path("artifacts/nemotron-indic-v1/tokenizer.json")
    if not path.exists():
        pytest.skip("Run sttok fetch and sttok build for the actual artifact check")
    tok = Tokenizer.from_file(str(path))
    for character in "ৡৢৣೡౕౡ":
        assert tok.normalizer.normalize_str(character) == character
        for text in (character, "a" + character + "a"):
            encoded = tok.encode(text)
            assert tok.token_to_id("<unk>") not in encoded.ids
            assert tok.decode(encoded.ids, skip_special_tokens=False) == text


def test_all_target_standard_exemplars_in_real_candidate():
    candidate = Path("artifacts/nemotron-indic-v1/tokenizer.json")
    if not candidate.exists():
        pytest.skip("Run sttok fetch and sttok build for the actual artifact check")
    fixture = json.loads(Path("configs/standard-exemplars.json").read_text())
    targets = json.loads(Path("configs/build.json").read_text())["targets"]
    assert {(p["language"], p["script"]) for p in fixture["profiles"]} == {(p["language"], p["script"]) for p in targets}
    base = Tokenizer.from_file(".cache/sources/nvidia/tokenizer.json")
    extended = Tokenizer.from_file(str(candidate))
    for profile in fixture["profiles"]:
        assert profile["elements"], profile["language"]
        for element in profile["elements"]:
            for text in (element, "a" + element + "a"):
                encoded = extended.encode(text)
                assert extended.token_to_id("<unk>") not in encoded.ids, (profile["language"], element)
                assert extended.decode(encoded.ids, skip_special_tokens=False) == base.normalizer.normalize_str(text), (profile["language"], element)
