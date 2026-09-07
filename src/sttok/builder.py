"""Append ordinary BPE pieces without retraining or replacing the base pipeline."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2
from tokenizers import Tokenizer, models

from .sources import read_json, sha256, source_path, verify_sources, write_json


def all_ids(data: dict) -> dict[str, int]:
    result = dict(data["model"]["vocab"])
    for token in data.get("added_tokens", []):
        if token["content"] in result and result[token["content"]] != token["id"]:
            raise ValueError("Conflicting added-token ID")
        result[token["content"]] = token["id"]
    if len(set(result.values())) != len(result):
        raise ValueError("Duplicate token ID")
    return result


def assigned_codepoints(path: Path) -> set[int]:
    """Use the pinned Unicode release, not the host Python Unicode version."""
    result = set()
    first = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(";")
        cp = int(fields[0], 16)
        if fields[1].endswith(", First>"):
            first = cp
        elif fields[1].endswith(", Last>"):
            if first is None:
                raise ValueError("Invalid Unicode range")
            result.update(range(first, cp + 1))
            first = None
        else:
            result.add(cp)
    return result


def script_codepoints(path: Path, script: str) -> set[int]:
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        code, value = [v.strip() for v in line.split(";", 1)]
        if value == script:
            bounds = code.split("..")
            result.update(range(int(bounds[0], 16), int(bounds[-1], 16) + 1))
    return result


def _bpe(data):
    settings = {k: v for k, v in data["model"].items() if k not in {"type", "vocab", "merges"} and v is not None}
    return models.BPE(vocab=data["model"]["vocab"], merges=[tuple(m) for m in data["model"]["merges"]], **settings)


def _witness(tokenizer, piece, token_id):
    if piece.startswith("▁"):
        samples = [piece[1:], "a " + piece[1:]]
    else:
        samples = ["x" + piece, piece, "x" + piece + "x"]
    for sample in samples:
        if token_id in tokenizer.encode(sample, add_special_tokens=False).ids:
            return sample
    return None


def build_tokenizer(
    config_path: str | Path,
    cache: str | Path,
    output: str | Path,
    *,
    previous: str | Path | None = None,
) -> dict:
    config_path = Path(config_path)
    config = read_json(config_path)
    config_dir = config_path.parent
    cache = Path(cache)
    lock_path = config_dir / config["sources_lock"]
    inputs = verify_sources(lock_path, cache)
    locked = {item["path"] for item in inputs}
    consumed = [config["base_tokenizer"], config["unicode_data"], config["unicode_scripts"]] + [t["donor"] for t in config["targets"]]
    if not set(consumed) <= locked:
        raise ValueError("Every consumed source must appear in the verified lock file")
    for relative in consumed:
        source_path(cache, relative)
    base_path = source_path(cache, config["base_tokenizer"])
    base = read_json(base_path)
    if base["model"]["type"] != "BPE":
        raise ValueError("The selected canonical base must be BPE")
    if config["normalizer"] != "1A":
        raise ValueError("Only the approved 1A build is enabled; 1C needs a separate validated release contract")
    base_ids = all_ids(base)
    if max(base_ids.values()) + 1 != config["first_new_id"]:
        raise ValueError("First new ID disagrees with the complete base inventory")
    if any(base_ids.get(k) != v for k, v in config["reserved_ids"].items()):
        raise ValueError("Reserved base IDs changed")
    if len(base["model"]["merges"]) != config["base_merge_count"]:
        raise ValueError("Base merge count changed")
    data = copy.deepcopy(base)
    previous_manifest = None
    if previous is not None:
        previous = Path(previous)
        data = read_json(previous / "tokenizer.json")
        previous_manifest = read_json(previous / "manifest.json")
        if sha256(previous / "tokenizer.json") != previous_manifest["tokenizer_sha256"]:
            raise ValueError("Previous release hash mismatch")
        if previous_manifest["base_sha256"] != sha256(base_path):
            raise ValueError("Previous release uses a different base")
        for k in base:
            if k != "model" and data[k] != base[k]:
                raise ValueError(f"Previous release changed pipeline field {k}")
        for k, value in base["model"].items():
            if k not in {"vocab", "merges"} and data["model"][k] != value:
                raise ValueError(f"Previous release changed BPE setting {k}")
        if any(all_ids(data).get(k) != v for k, v in base_ids.items()):
            raise ValueError("Previous release moved base IDs")
        if data["model"]["merges"][:len(base["model"]["merges"])] != base["model"]["merges"]:
            raise ValueError("Previous release moved base merges")

    initial_ids = all_ids(data)
    # Fill the two already-reserved IDs in the BPE table as well as retaining
    # their added_tokens descriptors. Sparse BPE IDs let the HF loader assign
    # added tokens over appended ordinary entries. This adds no new ID.
    for token in data.get("added_tokens", []):
        data["model"]["vocab"].setdefault(token["content"], token["id"])
    next_id = max(initial_ids.values()) + 1
    records = {}
    assigned = assigned_codepoints(cache / config["unicode_data"])
    latin_codepoints = script_codepoints(cache / config["unicode_scripts"], "Latin")
    base_chars = {c for piece in base["model"]["vocab"] for c in piece}
    # Existing single-character coverage, not arbitrary characters inside pieces,
    # controls whether a genuinely new character can seed a merge.
    base_singletons = {p for p in base["model"]["vocab"] if len(p) == 1}
    hindi_path = config_dir / config["hindi_pieces"]
    latin_path = config_dir / config["latin_characters"]
    for path, expected in ((hindi_path, config["hindi_sha256"]), (latin_path, config["latin_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"Approved inventory hash mismatch: {path.name}")
    hindi = {line.split("\t")[0] for line in hindi_path.read_text().splitlines() if line}
    latin = [line.split("\t")[1] for line in latin_path.read_text().splitlines() if line]
    if len(hindi) != config["hindi_count"] or len(set(latin)) != config["latin_count"]:
        raise ValueError("Approved inventory count mismatch")

    for donor in config["targets"]:
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString((cache / donor["donor"]).read_bytes())
        if proto.trainer_spec.model_type != sentencepiece_model_pb2.TrainerSpec.BPE:
            raise ValueError(f"Donor is not BPE: {donor['language']}")
        for p in proto.pieces:
            if p.type != sentencepiece_model_pb2.ModelProto.SentencePiece.NORMAL:
                continue
            record = records.setdefault(p.piece, {"piece": p.piece, "donors": [], "rank": []})
            record["donors"].append(donor["language"])
            record["rank"].append([donor["language"], -p.score])
    if not hindi <= set(records):
        raise ValueError("Approved Hindi inventory is not contained in the pinned donors")

    accepted = set()
    for piece, record in records.items():
        if piece in base_ids:
            record["status"] = "reused"
        elif any(ord(c) not in assigned for c in piece):
            record["status"] = "rejected_unassigned"
        elif "▁" in piece.lstrip("▁") or piece.startswith("▁▁"):
            record["status"] = "rejected_cross_word"
        elif piece in hindi or any(c not in base_chars for c in piece):
            record["status"] = "candidate"
            accepted.add(piece)
        else:
            record["status"] = "excluded_old_character_subword"

    required = config.get("required_characters", [])
    for piece in latin + [r["character"] for r in required]:
        if len(piece) != 1 or ord(piece) not in assigned:
            raise ValueError(f"Invalid required character: {piece!r}")
        if piece not in base_ids:
            accepted.add(piece)
            records.setdefault(piece, {"piece": piece, "donors": [], "rank": []})["status"] = "candidate"
    for piece in list(accepted):
        for char in piece:
            if char not in initial_ids:
                accepted.add(char)
                records.setdefault(char, {"piece": char, "donors": [], "rank": []})["status"] = "dependency"
    for piece in accepted:
        if len(piece) == 1 and ord(piece) in latin_codepoints and piece not in base_ids and piece not in latin:
            raise ValueError(f"Unapproved Latin character addition: {piece!r}")

    additions = {}
    if previous_manifest:
        additions.update({e["piece"]: dict(e) for e in previous_manifest["entries"]})

    def add_piece(piece, reason):
        nonlocal next_id
        if piece in all_ids(data):
            return
        data["model"]["vocab"][piece] = next_id
        additions[piece] = {"piece": piece, "id": next_id, "reason": reason, "donors": sorted(records.get(piece, {}).get("donors", []))}
        next_id += 1

    # New character IDs precede subwords, with stable codepoint ordering.
    for piece in sorted(p for p in accepted if len(p) == 1):
        add_piece(piece, "latin_character" if piece in latin else "required_character")
    merges = data["model"]["merges"]
    seen_merges = {tuple(m) for m in merges}
    pending = sorted((p for p in accepted if len(p) > 1), key=lambda p: (len(p), sorted(records[p]["rank"]), p))
    for piece in pending:
        tokens = [t.value for t in _bpe(data).tokenize(piece)]
        if "".join(tokens) != piece:
            raise ValueError(f"Character closure failed: {piece!r}")
        # Build from the actual current segmentation. Prefer joins that include
        # new characters so old-only Arabic/punctuation merges are unnecessary.
        while len(tokens) > 1:
            indices = list(range(len(tokens) - 1))
            if piece not in hindi:
                indices = [i for i in indices if any(c not in base_singletons for c in tokens[i] + tokens[i+1])]
            if not indices:
                raise ValueError(f"No policy-compatible merge for {piece!r}")
            i = indices[0]
            left, right = tokens[i:i+2]
            merged = left + right
            if any(ord(c) in latin_codepoints for c in merged):
                raise ValueError(f"Latin subword merge prohibited: {merged!r}")
            add_piece(merged, "donor_subword" if merged == piece else "merge_dependency")
            if (left, right) not in seen_merges:
                merges.append([left, right]); seen_merges.add((left, right))
            tokens = [t.value for t in _bpe(data).tokenize(piece)]
            if len(tokens) == 1 and tokens[0] != piece:
                raise ValueError(f"Unexpected merged piece: {piece!r}")
        add_piece(piece, "donor_subword")

    final = Tokenizer.from_str(json.dumps(data, ensure_ascii=False))
    if final.get_vocab() != all_ids(data):
        raise ValueError("HF loader reassigned reserved or ordinary IDs")
    errors = []
    for piece, entry in additions.items():
        entry["witness"] = _witness(final, piece, entry["id"])
        entry["reachable"] = entry["witness"] is not None
        if not entry["reachable"]:
            errors.append(f"No encoding witness for {piece!r}")
    for piece, record in records.items():
        if record["status"] in {"candidate", "dependency"}:
            record["status"] = "added" if piece in additions else "reused_previous"
        if piece in all_ids(data):
            record["id"] = all_ids(data)[piece]
    if not hindi <= set(all_ids(data)):
        raise ValueError("Some approved Hindi entries were not built")
    if any(all_ids(data).get(k) != v for k, v in initial_ids.items()):
        raise ValueError("Build changed an existing ID")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "tokenizer.json", data)
    reloaded = Tokenizer.from_file(str(output / "tokenizer.json"))
    if reloaded.get_vocab() != all_ids(data):
        raise ValueError("Save/reload changed IDs")
    manifest = {
        "schema_version": 1, "normalizer": "1A", "base_sha256": sha256(base_path),
        "tokenizer_sha256": sha256(output / "tokenizer.json"),
        "config_sha256": sha256(config_path), "sources": inputs,
        "reserved_ids": config["reserved_ids"], "first_new_id": config["first_new_id"],
        "base_id_count": len(base_ids), "vocabulary_size": len(all_ids(data)),
        "base_merge_count": len(base["model"]["merges"]), "merge_count": len(merges),
        "entries": sorted(additions.values(), key=lambda e: e["id"]),
        "candidates": sorted(records.values(), key=lambda r: r["piece"]),
        "expected_targets": [{k: t[k] for k in ("language", "script")} for t in config["targets"]],
        "required_characters": required,
        "hindi_candidates": sorted(hindi), "latin_characters": latin,
        "build_errors": errors, "build_passed": not errors,
        "asr_validated": False, "checkpoint_validated": False,
    }
    if previous_manifest:
        manifest["previous_tokenizer_sha256"] = previous_manifest["tokenizer_sha256"]
    write_json(output / "manifest.json", manifest)
    if errors:
        raise ValueError(f"Tokenizer candidate written, but build failed: {len(errors)} unreachable pieces; see manifest.json")
    return manifest
