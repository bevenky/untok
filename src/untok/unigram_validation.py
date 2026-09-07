"""Portable CPU checks for a native bundle and explicitly pinned text inputs.

This module neither fits a tokenizer nor validates an acoustic checkpoint. The
policy and corpus manifest are inputs; no user paths or research files are used.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata
from typing import Any

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from .unigram import NativeTokenizerAdapter, validate_native_prefix


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _strings(value: Any, field: str, *, nonempty: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(v, str) or (nonempty and not v) for v in value):
        raise ValueError(f"Policy {field} must be a list of strings")
    return value


def _policy(path: Path) -> dict:
    policy = _json(path)
    if policy.get("schema_version") != 1 or not isinstance(policy.get("base_tokenizer_sha256"), str):
        raise ValueError("Policy requires schema_version=1 and base_tokenizer_sha256")
    profiles = policy.get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("Policy requires nonempty profiles")
    for lang, entry in profiles.items():
        if not lang or Path(lang).name != lang or lang in {".", ".."} or not isinstance(entry, dict):
            raise ValueError("Invalid policy language profile")
        chars = _strings(entry.get("characters"), f"profiles.{lang}.characters", nonempty=True)
        if not chars or any(len(c) != 1 for c in chars):
            raise ValueError("Profile alphabets require individual Unicode characters")
    for field in ("normalizer_probes", "exact_encoding_probes"):
        _strings(policy.get(field, []), field)
    groups = policy.get("protected_piece_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("Policy protected_piece_groups must be an object")
    for name, entry in groups.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid protected group: {name}")
        pieces = _strings(entry.get("pieces"), f"protected_piece_groups.{name}", nonempty=True)
        if len(set(pieces)) != len(pieces):
            raise ValueError(f"Duplicate protected pieces: {name}")
    if "approved_new_latin_pieces" in policy:
        _strings(policy["approved_new_latin_pieces"], "approved_new_latin_pieces", nonempty=True)
    return policy


def _witness(proc: spm.SentencePieceProcessor, piece: str) -> dict | None:
    index = proc.piece_to_id(piece)
    if proc.id_to_piece(index) != piece:
        return None
    body = piece.removeprefix("▁")
    for text in (body, " " + body, body + " ", "a " + body + " b", "a" + body + "a", "अ" + body + "अ"):
        if index in proc.encode(text):
            return {"input": text, "pieces": proc.encode(text, out_type=str)}
    return None


def _matcher(pieces: list[str]):
    trie: dict = {}
    for piece in pieces:
        node = trie
        for c in piece:
            node = node.setdefault(c, {})
        node[None] = True

    def matches(text: str) -> bool:
        for i in range(len(text)):
            node = trie
            for j in range(i, len(text)):
                node = node.get(text[j])
                if node is None:
                    break
                if None in node:
                    return True
        return False
    return matches


def _structure(bundle: Path, policy: dict) -> tuple[dict, Any, Any, Any]:
    adapter = NativeTokenizerAdapter(bundle)
    raw_base = (bundle / "base-tokenizer.model").read_bytes()
    raw_new = (bundle / "tokenizer.model").read_bytes()
    report = validate_native_prefix(raw_base, raw_new)
    if report["base_tokenizer_sha256"] != policy["base_tokenizer_sha256"]:
        raise ValueError("Policy native base hash disagrees with bundle")
    if policy.get("normalizer_sha256", report["normalizer_sha256"]) != report["normalizer_sha256"]:
        raise ValueError("Policy normalizer hash disagrees with bundle")
    base = spm.SentencePieceProcessor(model_proto=raw_base)
    proc = adapter.backend
    added = [proc.id_to_piece(i) for i in range(base.get_piece_size(), proc.get_piece_size())]
    coverage = {}
    for lang, entry in policy["profiles"].items():
        chars = entry["characters"]
        coverage[lang] = {"script": entry.get("script"), "required_characters": len(chars),
                          "missing": [c for c in chars if proc.unk_id() in proc.encode(c)]}
    groups = {}
    for name, entry in policy.get("protected_piece_groups", {}).items():
        available = set(added) if entry.get("require_additions", True) else set(adapter.vocab)
        missing = sorted(set(entry["pieces"]) - available)
        witnesses = {p: _witness(proc, p) for p in entry["pieces"] if p in available}
        groups[name] = {"pieces": len(entry["pieces"]), "missing": missing, "witnesses": witnesses,
                        "unwitnessed": sorted(p for p, value in witnesses.items() if value is None),
                        "passed": not missing and all(witnesses.values())}
    normalization = [{"input": text, "native": base.normalize(text), "candidate": proc.normalize(text),
                      "passed": base.normalize(text) == proc.normalize(text)}
                     for text in policy.get("normalizer_probes", [])]
    matches = _matcher(added)
    probes = policy.get("normalizer_probes", []) + [base.id_to_piece(i).replace("▁", " ") for i in range(base.get_piece_size())]
    no_match_count = overlaps = 0
    no_match_fail = []
    for text in probes:
        if matches(base.normalize(text)):
            overlaps += 1
        else:
            no_match_count += 1
            if base.encode(text) != proc.encode(text):
                no_match_fail.append(text)
    decode_fail = [i for i in range(base.get_piece_size()) if base.decode([i]) != proc.decode([i])]
    rng = random.Random(175)
    random_decode_fail = 0
    for _ in range(10000):
        ids = [rng.randrange(base.get_piece_size()) for _ in range(rng.randrange(1, 25))]
        random_decode_fail += base.decode(ids) != proc.decode(ids)
    exact = {text: {"base": base.encode(text, out_type=str), "candidate": proc.encode(text, out_type=str),
                    "passed": base.encode(text) == proc.encode(text)}
             for text in policy.get("exact_encoding_probes", [])}
    gates = {"required_alphabets": not any(v["missing"] for v in coverage.values()),
             "normalizer_preserved": all(v["passed"] for v in normalization),
             "all_old_ID_decoding": not decode_fail and not random_decode_fail,
             "no_match_encoding_parity": not no_match_fail,
             "protected_piece_witnesses": all(v["passed"] for v in groups.values()),
             "exact_encoding_probes": all(v["passed"] for v in exact.values()),
             "public_layout": adapter.id_map.hf_pad_id == base.get_piece_size()
             and adapter.id_map.hf_blank_id == base.get_piece_size() + 1}
    if "approved_new_latin_pieces" in policy:
        approved = set(policy["approved_new_latin_pieces"])
        new_latin = {p for p in added if any("LATIN" in unicodedata.name(c, "") for c in p)}
        report["new_latin"] = {"count": len(new_latin), "unapproved": sorted(new_latin - approved),
                               "missing_approved": sorted(approved - new_latin)}
        gates["only_approved_latin"] = new_latin == approved
    report.update({"sentencepiece_version": spm.__version__, "alphabet_coverage": coverage,
                   "protected_piece_groups": groups, "normalization_controls": normalization,
                   "old_single_ID_decode_failures": decode_fail, "old_random_sequence_decode_trials": 10000,
                   "old_random_sequence_decode_failures": random_decode_fail,
                   "no_added_match_probes": no_match_count, "no_added_match_failures": no_match_fail,
                   "probes_with_added_match_excluded_from_parity": overlaps, "exact_encoding_probes": exact,
                   "public_pad_id": adapter.id_map.hf_pad_id, "public_blank_id": adapter.id_map.hf_blank_id,
                   "first_new_public_id": adapter.id_map.hf_blank_id + 1, "native_blank_id": adapter.blank_id,
                   "public_vocabulary_size": len(adapter.id_map.canonical_to_model),
                   "gates": gates, "structural_passed": all(gates.values())})
    proto = pb.ModelProto()
    proto.ParseFromString(raw_base)
    return report, base, proc, proto


def _expected(base, proto, text: str) -> str:
    if proto.denormalizer_spec.precompiled_charsmap or proto.denormalizer_spec.normalization_rule_tsv:
        raise ValueError("Supply expected_text for a model with a custom denormalizer")
    value = base.normalize(text)
    # SentencePiece decoding consumes one leading metaspace when dummy-prefix
    # mode is enabled. It does not consume an ordinary space or move that rule
    # to the end for a trainer configured with whitespace-as-suffix.
    if proto.normalizer_spec.add_dummy_prefix:
        value = value.removeprefix("▁")
    return value.replace("▁", " ")


def _percentile(values: list, q: float):
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)] if values else None


def _corpus_metrics(path: Path, base, proc, proto, max_examples: int, expected_language: str) -> dict:
    total = Counter({key: 0 for key in ("records", "nonempty_normalized_records", "normalized_characters", "tokens",
                    "normalizer_failures", "unknown_tokens", "unknown_records", "unknown_codepoint_occurrences",
                    "native_unknown_tokens", "native_unknown_records", "roundtrip_failures",
                    "base_representable_records", "base_representable_changed")})
    lengths, ratios, failures, changed = [], [], [], []
    unknown, sources = Counter(), Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"Expected a text string in {path.name}:{line_number}")
            if "language" in row and row["language"] != expected_language:
                raise ValueError(f"Corpus language disagrees with profile {expected_language}: {path.name}:{line_number}")
            text = row["text"]
            if "expected_text" in row and not isinstance(row["expected_text"], str):
                raise ValueError(f"Expected an expected_text string in {path.name}:{line_number}")
            normal = base.normalize(text)
            ids, old = proc.encode(text), base.encode(text)
            total["records"] += 1
            total["normalized_characters"] += len(normal)
            total["nonempty_normalized_records"] += bool(normal.replace("▁", "").strip())
            old_unknown, num = old.count(base.unk_id()), ids.count(proc.unk_id())
            total["native_unknown_tokens"] += old_unknown
            total["native_unknown_records"] += bool(old_unknown)
            total["tokens"] += len(ids)
            total["normalizer_failures"] += normal != proc.normalize(text)
            total["unknown_tokens"] += num
            total["unknown_records"] += bool(num)
            lengths.append(len(ids)); ratios.append(len(ids) / max(1, len(normal)))
            source = row.get("source", "unspecified")
            if not isinstance(source, str):
                raise ValueError(f"Expected a source string in {path.name}:{line_number}")
            sources[source] += 1
            if num:
                for piece in proc.encode_as_immutable_proto(text).pieces:
                    if piece.id == proc.unk_id():
                        unknown.update(piece.piece)
                        total["unknown_codepoint_occurrences"] += len(piece.piece)
            else:
                expected = row["expected_text"] if "expected_text" in row else _expected(base, proto, text)
                if proc.decode(ids) != expected:
                    total["roundtrip_failures"] += 1
                    if len(failures) < max_examples:
                        failures.append({"record_id": row.get("record_id"), "input": text,
                                         "decoded": proc.decode(ids), "expected": expected})
            if not old_unknown:
                total["base_representable_records"] += 1
                total["base_representable_changed"] += old != ids
                if old != ids and len(changed) < max_examples:
                    changed.append({"record_id": row.get("record_id"), "input": text,
                                    "native_pieces": base.encode(text, out_type=str),
                                    "candidate_pieces": proc.encode(text, out_type=str)})
    return {**total, "mean_tokens": sum(lengths) / len(lengths) if lengths else None,
            "tokens_p95": _percentile(lengths, .95), "tokens_p99": _percentile(lengths, .99),
            "mean_tokens_per_normalized_character": sum(ratios) / len(ratios) if ratios else None,
            "sources": dict(sources), "unknown_codepoints": {f"U+{ord(c):04X}": n for c, n in unknown.most_common()},
            "unknown_counting": "Emitted UNK IDs and normalized unknown-piece codepoints are distinct counts.",
            "roundtrip_failure_examples": failures, "base_representable_change_examples": changed}


def _contained(root: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not (root / relative).resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Corpus path must stay contained and relative: {name}")
    return root / relative


def validate_native_tokenizer(
    bundle_path: str | Path,
    policy_path: str | Path,
    corpus_manifest_path: str | Path | None = None,
    *,
    phase: str = "dev",
    selection_receipt_path: str | Path | None = None,
    max_examples: int = 0,
) -> dict[str, Any]:
    """Return structural and text evidence without changing inputs or fitting.

    A corpus manifest declares ``files`` as relative-path-to-SHA256 mappings,
    including ``{phase}/{language}.jsonl`` for each policy profile. Only the
    requested phase is opened. Reserved text requires a hash-bound selection
    receipt before any corpus file is read. Reserve reports never include text
    examples. Missing corpus targets are incomplete, not passed.
    """
    if phase not in {"dev", "reserve"} or isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 0:
        raise ValueError("Use phase dev/reserve and a nonnegative integer max_examples")
    bundle, policy_file = Path(bundle_path), Path(policy_path)
    policy = _policy(policy_file)
    report, base, proc, proto = _structure(bundle, policy)
    report.update({"policy_sha256": _sha(policy_file), "phase": phase, "corpus_status": "incomplete",
                   "corpus_gates": {}, "corpora": {}, "missing_profiles": sorted(policy["profiles"])})
    if corpus_manifest_path is not None:
        manifest_path = Path(corpus_manifest_path)
        manifest = _json(manifest_path)
        if manifest.get("native_base_sha256") != report["base_tokenizer_sha256"]:
            raise ValueError("Corpus manifest native base hash disagrees with bundle")
        if manifest.get("normalizer_sha256") != report["normalizer_sha256"]:
            raise ValueError("Corpus manifest normalizer hash disagrees with bundle")
        files = manifest.get("files")
        if not isinstance(files, dict):
            raise ValueError("Corpus manifest must declare relative file SHA256 digests")
        report["data_manifest_sha256"] = _sha(manifest_path)
        if phase == "reserve":
            if selection_receipt_path is None:
                raise ValueError("Reserve evaluation requires a selection receipt before reading data")
            receipt_path = Path(selection_receipt_path)
            receipt = _json(receipt_path)
            bindings = {"tokenizer_sha256": report["tokenizer_sha256"],
                        "data_manifest_sha256": report["data_manifest_sha256"],
                        "bundle_manifest_sha256": _sha(bundle / "manifest.json"),
                        "selection_sha256": _sha(bundle / "selection.json"),
                        "policy_sha256": report["policy_sha256"]}
            for field, expected in bindings.items():
                if receipt.get(field) != expected:
                    raise ValueError(f"Selection receipt does not bind current artifact: {field}")
            report["selection_receipt_sha256"] = _sha(receipt_path)
            report["receipt_scope"] = "Binds artifact identity; does not independently prove creation chronology."
        # Validate every declared path without reading other phases. Hash all
        # current-phase targets before starting corpus metrics.
        for name in files:
            if not isinstance(name, str):
                raise ValueError("Corpus paths must be strings")
            _contained(manifest_path.parent, name)
        inputs, missing = {}, []
        for lang in sorted(policy["profiles"]):
            name = f"{phase}/{lang}.jsonl"
            path = _contained(manifest_path.parent, name)
            if name not in files or not path.is_file():
                missing.append(lang)
                continue
            if not isinstance(files[name], str) or _sha(path) != files[name]:
                raise ValueError(f"Corpus integrity failure: {name}")
            inputs[lang] = path
        report["corpora"] = {lang: _corpus_metrics(path, base, proc, proto, 0 if phase == "reserve" else max_examples, lang)
                             for lang, path in inputs.items()}
        empty = [lang for lang, metrics in report["corpora"].items()
                 if not metrics["records"] or not metrics["nonempty_normalized_records"]]
        report["missing_profiles"] = missing
        report["empty_profiles"] = empty
        gates = {"all_profiles_have_text": not missing and not empty,
                 "all_representable_roundtrips_pass": not any(v["roundtrip_failures"] for v in report["corpora"].values()),
                 "all_corpus_normalizers_match": not any(v["normalizer_failures"] for v in report["corpora"].values())}
        report["corpus_gates"] = gates
        report["corpus_status"] = "incomplete" if missing or empty else ("passed" if all(gates.values()) else "failed")
    report["passed"] = report["structural_passed"] and report["corpus_status"] == "passed"
    report["status"] = "failed" if not report["structural_passed"] or report["corpus_status"] == "failed" else report["corpus_status"]
    report["checkpoint_validated"] = report["asr_validated"] = False
    report["scope"] = "CPU tokenizer evidence; not acoustic checkpoint migration or speech accuracy. Additions may change segmentation."
    report["corpus_unknown_policy"] = "Mixed-corpus unknowns are reported, not a hard zero gate; pinned standard alphabets are a hard gate."
    report["validator_sha256"] = _sha(Path(__file__))
    return report
