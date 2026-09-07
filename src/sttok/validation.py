"""Evidence-based checks for an append-only Hugging Face BPE extension.

The validator returns data and never writes reports or changes its inputs. Text
coverage is deliberately separate from acoustic-model compatibility/accuracy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tokenizers import Tokenizer


TARGET_SCRIPTS = {
    "as": "Beng", "bn": "Beng", "brx": "Deva", "doi": "Deva",
    "gu": "Gujr", "hi": "Deva", "kn": "Knda", "kok": "Deva",
    "ks": "Arab", "mai": "Deva", "ml": "Mlym", "mni": "Mtei",
    "mr": "Deva", "ne": "Deva", "or": "Orya", "pa": "Guru",
    "sa": "Deva", "sat": "Olck", "sd": "Deva", "ta": "Taml",
    "te": "Telu", "ur": "Arab",
}

_SCRIPT_ALIASES = {
    "bengali": "Beng", "devanagari": "Deva", "gujarati": "Gujr",
    "kannada": "Knda", "arabic": "Arab", "perso-arabic": "Arab",
    "malayalam": "Mlym", "meetei mayek": "Mtei", "meitei mayek": "Mtei",
    "meetei-mayek": "Mtei", "meitei-mayek": "Mtei", "odia": "Orya",
    "oriya": "Orya", "gurmukhi": "Guru", "ol chiki": "Olck",
    "ol-chiki": "Olck", "tamil": "Taml", "telugu": "Telu", "latin": "Latn",
}
_LANGUAGE_ALIASES = dict(zip(
    ("assamese", "bengali", "bodo", "dogri", "gujarati", "hindi", "kannada", "konkani", "kashmiri", "maithili", "malayalam", "manipuri", "marathi", "nepali", "odia", "punjabi", "sanskrit", "santali", "sindhi", "tamil", "telugu", "urdu"),
    TARGET_SCRIPTS,
))
_LANGUAGE_ALIASES.update({"oriya": "or", "english": "en"})


def _target(language: str, script: str) -> tuple[str, str]:
    language = _LANGUAGE_ALIASES.get(language.casefold(), language)
    canonical_scripts = {value.casefold(): value for value in (*TARGET_SCRIPTS.values(), "Latn")}
    script = _SCRIPT_ALIASES.get(script.casefold(), canonical_scripts.get(script.casefold(), script))
    return language, script

PROTECTED_TEXT = (
    "hello world", "I am going to the office.", "office", "kaise ho",
    "vanakkam", "ABC abc 0123456789", "a, b! c? (d) [e]: 12.50%",
    "  office  tomorrow  ", "one\ttwo\nthree", "of\u200cfice",
    "a\u200bb", "a\ufeffb", "caf\u00e9", "\ufb01 \uff21",
)

UNICODE_PROBES = (
    "", " ", "  a  b  ", "a\tb\nc", "a\u200cb", "a\u200db",
    "a\u200bb", "a\ufeffb", "a\u00a0b", "\ufb01", "\uff21",
    "e\u0301", "\u0915\u093c", "\u0d05\u0d35\u0d28\u0d4d\u200d",
    "\u0964\u0965", "\u0be6\u0be7\u0be8",
)


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _full_vocab(document: Mapping[str, Any]) -> dict[str, int]:
    result = dict(document["model"]["vocab"])
    for item in document.get("added_tokens", []):
        if item["content"] in result and result[item["content"]] != item["id"]:
            raise ValueError(f"Conflicting model/added ID for {item['content']!r}")
        result[item["content"]] = item["id"]
    if len(set(result.values())) != len(result):
        raise ValueError("Distinct vocabulary strings share an ID")
    return result


def _merges(document: Mapping[str, Any]) -> list[tuple[str, str]]:
    result = []
    for merge in document["model"].get("merges", []):
        pair = merge.split(" ") if isinstance(merge, str) else merge
        if len(pair) != 2:
            raise ValueError(f"Invalid merge pair: {merge!r}")
        result.append(tuple(pair))
    return result


def _normalize(tokenizer: Tokenizer, text: str) -> str:
    return tokenizer.normalizer.normalize_str(text) if tokenizer.normalizer else text


def expected_text(
    tokenizer: Tokenizer, document: Mapping[str, Any], text: str
) -> str:
    """Render the declared normalization/Metaspace contract without encoding.

This intentionally does not call encode(), decode(), or the tokenizer decoder.
For a different pipeline, supply an explicit expected_text in corpus records.
Special-token literals should be covered by the separate parity checks.
"""
    normalized = _normalize(tokenizer, text)
    pre = document.get("pre_tokenizer")
    decoder = document.get("decoder")
    if pre is None and decoder is None:
        return normalized
    if not pre or not decoder or pre.get("type") != "Metaspace" or decoder.get("type") != "Metaspace":
        raise ValueError("Automatic expected text supports only the declared Metaspace pipeline")
    marker = pre.get("replacement", "▁")
    if decoder.get("replacement", "▁") != marker:
        raise ValueError("Pre-tokenizer and decoder Metaspace markers differ")
    value = normalized.replace(" ", marker)
    scheme = pre.get("prepend_scheme", "always")
    if scheme != "never" and value and not value.startswith(marker):
        value = marker + value
    decoder_scheme = decoder.get("prepend_scheme", "always")
    if decoder_scheme != "never" and value.startswith(marker):
        value = value[len(marker):]
    return value.replace(marker, " ")


def _normalized_unknowns(tokenizer: Tokenizer, text: str, unk_id: int | None) -> dict[str, Any]:
    """Count unknown Unicode scalars, independent of fused-unknown token count.

The low-level BPE model exposes UTF-8 byte offsets into each normalized
pre-token. These offsets are NOT the original-text offsets of encode().
"""
    normalized = _normalize(tokenizer, text)
    chunks = tokenizer.pre_tokenizer.pre_tokenize_str(normalized) if tokenizer.pre_tokenizer else [(normalized, (0, len(normalized)))]
    missing: Counter[str] = Counter()
    spans = 0
    for chunk, _ in chunks:
        encoded = chunk.encode("utf-8")
        for token in tokenizer.model.tokenize(chunk):
            if token.id == unk_id:
                start, end = token.offsets
                missing.update(encoded[start:end].decode("utf-8"))
                spans += 1
    return {"characters": sum(missing.values()), "spans": spans, "inventory": missing}


def _quantiles(values: Sequence[float | int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": ordered[max(0, math.ceil(len(ordered) * .50) - 1)],
        "p95": ordered[max(0, math.ceil(len(ordered) * .95) - 1)],
        "p99": ordered[max(0, math.ceil(len(ordered) * .99) - 1)],
        "max": ordered[-1],
    }


def _specs(corpora: Any) -> list[dict[str, Any]]:
    if corpora is None:
        return []
    root = Path.cwd()
    if isinstance(corpora, (str, Path)):
        manifest_path = Path(corpora)
        root = manifest_path.resolve().parent
        corpora = _read_json(manifest_path)
    if isinstance(corpora, dict):
        corpora = corpora.get("corpora", [corpora])
    result = []
    for item in corpora:
        spec = dict(item)
        path = Path(spec["path"])
        spec["path"] = str(path if path.is_absolute() else root / path)
        result.append(spec)
    return result


def _records(spec: Mapping[str, Any]) -> Iterator[tuple[int, Mapping[str, Any]]]:
    path = Path(spec["path"])
    kind = spec.get("format", path.suffix.lstrip(".").lower())
    with path.open(encoding="utf-8", newline="") as stream:
        if kind in {"txt", "text"}:
            for number, line in enumerate(stream, 1):
                yield number, {spec.get("text_field", "text"): line.rstrip("\r\n")}
        elif kind == "jsonl":
            for number, line in enumerate(stream, 1):
                if line.strip():
                    yield number, json.loads(line)
        elif kind == "csv":
            for number, row in enumerate(csv.DictReader(stream), 2):
                yield number, row
        elif kind == "json":
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = rows[spec.get("records_field", "records")]
            for number, row in enumerate(rows, 1):
                yield number, row
        else:
            raise ValueError(f"Unsupported corpus format {kind!r}")


def validate_tokenizer(
    base_path: str | Path,
    extended_path: str | Path,
    manifest_path: str | Path,
    corpora: Any = None,
    *,
    required_targets: Iterable[Mapping[str, str]] | None = None,
    max_examples: int = 20,
) -> dict[str, Any]:
    """Return structural, text-parity, reachability and corpus evidence.

Corpus specs accept path, format, language, script, source, sha256, text_field,
expected_field and expect_unchanged. JSONL/CSV/JSON rows may supply language,
script and explicit expected_text. Missing target data makes status incomplete.
"""
    if max_examples < 0:
        raise ValueError("max_examples must be non-negative")
    base_doc, ext_doc, manifest = map(_read_json, (base_path, extended_path, manifest_path))
    base, extended = Tokenizer.from_file(str(base_path)), Tokenizer.from_file(str(extended_path))
    errors: list[dict[str, Any]] = []
    warnings: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})
        if not passed:
            errors.append({"code": name, "detail": detail})

    base_vocab, ext_vocab = _full_vocab(base_doc), _full_vocab(ext_doc)
    base_merges, ext_merges = _merges(base_doc), _merges(ext_doc)
    for key, path in (("base_sha256", base_path), ("tokenizer_sha256", extended_path)):
        if key in manifest:
            check(f"manifest_{key}", manifest[key] == _sha256(path))
    check("bpe_model", base_doc["model"]["type"] == ext_doc["model"]["type"] == "BPE")
    mismatched_ids = [piece for piece, index in base_vocab.items() if ext_vocab.get(piece) != index or extended.id_to_token(index) != piece]
    check("original_ids", not mismatched_ids, {"checked": len(base_vocab), "mismatches": mismatched_ids[:max_examples]})
    for field in ("version", "normalizer", "pre_tokenizer", "decoder", "post_processor", "padding", "truncation"):
        check(f"unchanged_{field}", base_doc.get(field) == ext_doc.get(field))
    check("original_added_tokens", base_doc.get("added_tokens", []) == ext_doc.get("added_tokens", []))
    settings = lambda d: {key: value for key, value in d["model"].items() if key not in {"vocab", "merges"}}
    check("unchanged_model_settings", settings(base_doc) == settings(ext_doc))
    check("original_merge_prefix", ext_merges[:len(base_merges)] == base_merges, {"original_count": len(base_merges)})
    check("unique_merge_pairs", len(set(ext_merges)) == len(ext_merges))
    model_vocab = ext_doc["model"]["vocab"]
    invalid_merges = [pair for pair in ext_merges if any(piece not in model_vocab for piece in (*pair, "".join(pair)))]
    check("valid_merge_references", not invalid_merges, invalid_merges[:max_examples])
    new_merges = ext_merges[len(base_merges):]
    latin_merges = [pair for pair in new_merges if any("LATIN" in unicodedata.name(char, "") for char in "".join(pair))]
    shared_only_merges = [pair for pair in new_merges if not any(unicodedata.category(char)[0] in {"L", "M"} for char in "".join(pair))]
    check("no_new_latin_merges", not latin_merges, latin_merges[:max_examples])
    check("no_new_shared_only_merges", not shared_only_merges, shared_only_merges[:max_examples])
    new_vocab = {piece: index for piece, index in ext_vocab.items() if piece not in base_vocab}
    first_new = max(base_vocab.values()) + 1
    check("append_only_ids", all(index >= first_new for index in new_vocab.values()), {"first_new_id": first_new, "new_count": len(new_vocab)})
    check("contiguous_ids", set(ext_vocab.values()) == set(range(max(ext_vocab.values()) + 1)))
    for key, actual in (("base_id_count", len(base_vocab)), ("base_merge_count", len(base_merges)), ("first_new_id", first_new), ("vocabulary_size", len(ext_vocab)), ("merge_count", len(ext_merges))):
        if key in manifest:
            check(f"manifest_{key}", manifest[key] == actual, {"declared": manifest[key], "actual": actual})
    for piece in ("<pad>", "<blank>"):
        if piece in base_vocab:
            check(f"reserved_{piece}", ext_vocab.get(piece) == base_vocab[piece], base_vocab[piece])
    # Serialization is checked in memory: no temporary artifact can alter inputs.
    reloaded = Tokenizer.from_str(extended.to_str())
    check("serialization_id_preservation", reloaded.get_vocab(with_added_tokens=True) == ext_vocab)
    check("serialization_stability", json.loads(reloaded.to_str()) == json.loads(extended.to_str()))
    structural_passed = not errors

    unk_id = extended.token_to_id(ext_doc["model"].get("unk_token", "<unk>"))
    base_unk_id = base.token_to_id(base_doc["model"].get("unk_token", "<unk>"))
    if "hindi_candidates" in manifest:
        hindi = manifest["hindi_candidates"]
        check("declared_hindi_103", len(hindi) == len(set(hindi)) == 103 and all(piece in new_vocab for piece in hindi))
    if "latin_characters" in manifest:
        latin = manifest["latin_characters"]
        latin_failures = [char for char in latin if len(char) != 1 or char not in new_vocab or _normalize(base, char) != char or unk_id in extended.encode(char, add_special_tokens=False).ids or extended.decode(extended.encode(char, add_special_tokens=False).ids, skip_special_tokens=False) != char]
        check("declared_latin_190", len(latin) == len(set(latin)) == 190 and not latin_failures, latin_failures[:max_examples])
    protected = []
    probes = list(manifest.get("protected_text", PROTECTED_TEXT))
    probes += [item["content"] for item in base_doc.get("added_tokens", [])]
    for text in dict.fromkeys(probes):
        old_ids, new_ids = base.encode(text).ids, extended.encode(text).ids
        if base_unk_id in old_ids:
            continue  # Newly covered text is intentionally outside exact parity.
        protected.append({"text": text, "passed": old_ids == new_ids, "base_ids": old_ids, "extended_ids": new_ids})
    check("protected_text_parity", all(row["passed"] for row in protected), {"checked": len(protected), "failures": [row for row in protected if not row["passed"]][:max_examples]})
    if not protected:
        warnings.append("No protected-text probes were fully covered by the base; provide a protected corpus.")

    normalizer_controls = []
    for text in UNICODE_PROBES:
        old_text, new_text = _normalize(base, text), _normalize(extended, text)
        normalizer_controls.append({"input": text, "base": old_text, "extended": new_text, "passed": old_text == new_text})
    for row in manifest.get("normalizer_controls", []):
        normalized = _normalize(extended, row["text"])
        normalizer_controls.append({"input": row["text"], "expected": row["expected"], "extended": normalized, "passed": normalized == row["expected"]})
    check("normalizer_controls", all(row["passed"] for row in normalizer_controls), [row for row in normalizer_controls if not row["passed"]][:max_examples])

    entries = manifest.get("entries", manifest.get("additions", manifest.get("new_tokens", [])))
    entry_map = {row["piece"]: row for row in entries}
    check("unique_manifest_entries", len(entry_map) == len(entries))
    check("manifest_covers_new_entries", set(new_vocab) <= set(entry_map), sorted(set(new_vocab) - set(entry_map))[:max_examples])
    witnesses = []
    for piece, index in new_vocab.items():
        row = entry_map.get(piece, {})
        text = row.get("witness")
        if isinstance(text, dict):
            text = text.get("text")
        actual_ids = extended.encode(text, add_special_tokens=False).ids if isinstance(text, str) else []
        witnesses.append({"piece": piece, "id": index, "witness": text, "passed": row.get("id") == index and index in actual_ids and row.get("reachable", True), "actual_ids": actual_ids})
    check("new_piece_reachability", all(row["passed"] for row in witnesses), {"checked": len(witnesses), "failures": [row for row in witnesses if not row["passed"]][:max_examples]})
    reload_probes = list(dict.fromkeys([row["text"] for row in protected] + [row["witness"] for row in witnesses if isinstance(row["witness"], str)]))
    reload_failures = [text for text in reload_probes if reloaded.encode(text, add_special_tokens=False).ids != extended.encode(text, add_special_tokens=False).ids]
    check("serialization_encode_parity", not reload_failures, {"checked": len(reload_probes), "failures": reload_failures[:max_examples]})

    character_checks = []
    for row in manifest.get("required_characters", []):
        row = {"character": row} if isinstance(row, str) else row
        character = row.get("character", row.get("piece"))
        ids = extended.encode(character, add_special_tokens=False).ids
        wanted = row["expected"] if "expected" in row else expected_text(base, base_doc, character)
        actual = extended.decode(ids, skip_special_tokens=False)
        character_checks.append({**row, "passed": unk_id not in ids and actual == wanted, "actual": actual, "expected": wanted})
    check("required_character_coverage", all(row["passed"] for row in character_checks), {"checked": len(character_checks), "failures": [row for row in character_checks if not row["passed"]][:max_examples]})

    if required_targets is None:
        required_targets = manifest.get("expected_targets", [{"language": language, "script": script} for language, script in TARGET_SCRIPTS.items()])
    target_keys = {_target(row["language"], row["script"]) for row in required_targets}
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    source_reports = []
    examples = []
    for spec in _specs(corpora):
        source_report = {"path": spec["path"], "source": spec.get("source"), "records": 0}
        source_reports.append(source_report)
        try:
            actual_hash = _sha256(spec["path"])
            source_report["sha256"] = actual_hash
            source_report["hash_verified"] = bool(spec.get("sha256")) and actual_hash == spec["sha256"]
            if spec.get("sha256") and actual_hash != spec["sha256"]:
                raise ValueError("Corpus hash does not match pinned manifest")
            for number, record in _records(spec):
                text = record[spec.get("text_field", "text")]
                if not isinstance(text, str):
                    raise ValueError(f"Record {number}: text must be a string")
                language = record.get(spec.get("language_field", "language"), spec.get("language"))
                script = record.get(spec.get("script_field", "script"), spec.get("script"))
                if not isinstance(language, str) or not isinstance(script, str) or not language or not script:
                    raise ValueError(f"Record {number}: language and script are required")
                language, script = _target(language, script)
                key = (language, script)
                if key not in aggregates:
                    aggregates[key] = {
                        "records": 0, "unknown_records": 0, "unknown_tokens": 0,
                        "lost_normalized_characters": 0, "normalized_unknown_spans": 0,
                        "roundtrip_failures": 0, "roundtrip_checked": 0,
                        "changed_sequences": 0, "protected_failures": 0,
                        "token_lengths": [], "base_token_lengths": [],
                        "tokens_per_word": [], "tokens_per_character": [],
                        "missing": Counter(), "sources": set(),
                    }
                stats = aggregates[key]
                source_report["records"] += 1
                stats["records"] += 1
                source_label = spec.get("source") or spec["path"]
                stats["sources"].add(source_label if isinstance(source_label, str) else json.dumps(source_label, sort_keys=True))
                encoded = extended.encode(text, add_special_tokens=False)
                old_ids = base.encode(text, add_special_tokens=False).ids
                n_unknown = encoded.ids.count(unk_id)
                loss = _normalized_unknowns(extended, text, unk_id)
                desired = record.get(spec.get("expected_field", "expected_text"))
                if desired is None:
                    desired = expected_text(base, base_doc, text)
                if not isinstance(desired, str):
                    raise ValueError(f"Record {number}: expected text must be a string")
                decoded = extended.decode(encoded.ids, skip_special_tokens=False)
                mismatch = not n_unknown and decoded != desired
                changed = old_ids != encoded.ids
                protected_failure = bool(spec.get("expect_unchanged", False)) and changed
                stats["unknown_records"] += bool(n_unknown)
                stats["unknown_tokens"] += n_unknown
                stats["lost_normalized_characters"] += loss["characters"]
                stats["normalized_unknown_spans"] += loss["spans"]
                stats["missing"].update(loss["inventory"])
                stats["roundtrip_checked"] += not n_unknown
                stats["roundtrip_failures"] += bool(mismatch)
                stats["changed_sequences"] += changed
                stats["protected_failures"] += protected_failure
                stats["token_lengths"].append(len(encoded.ids))
                stats["base_token_lengths"].append(len(old_ids))
                if desired.split():
                    stats["tokens_per_word"].append(len(encoded.ids) / len(desired.split()))
                if desired:
                    stats["tokens_per_character"].append(len(encoded.ids) / len(desired))
                if (n_unknown or mismatch or changed) and len(examples) < max_examples:
                    examples.append({"language": language, "script": script, "source": spec["path"], "record": number, "text": text, "expected": desired, "decoded": decoded, "base_ids": old_ids, "extended_ids": encoded.ids, "unknown_tokens": n_unknown, "lost_normalized_characters": loss["characters"], "roundtrip_failure": bool(mismatch), "protected_failure": protected_failure})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            source_report["error"] = str(exc)
            errors.append({"code": "corpus_input", "path": spec["path"], "detail": str(exc)})

    language_reports = []
    for language, script in sorted(set(aggregates) | target_keys):
        stats = aggregates.get((language, script))
        if stats is None:
            language_reports.append({"language": language, "script": script, "status": "missing_corpus", "records": 0})
            continue
        failed = any(stats[field] for field in ("unknown_records", "lost_normalized_characters", "roundtrip_failures", "protected_failures"))
        report = {"language": language, "script": script, "status": "failed" if failed else "passed_on_supplied_text"}
        for key, value in stats.items():
            if key == "missing":
                report["missing_characters"] = [{"character": char, "codepoint": f"U+{ord(char):04X}", "occurrences": count} for char, count in sorted(value.items())]
            elif key == "sources":
                report[key] = sorted(value)
            elif isinstance(value, list):
                report[key] = _quantiles(value)
            else:
                report[key] = value
        report["changed_sequence_fraction"] = stats["changed_sequences"] / stats["records"]
        language_reports.append(report)
        if failed:
            errors.append({"code": "corpus_text_failure", "language": language, "script": script})
    missing_targets = [{"language": lang, "script": script} for lang, script in sorted(target_keys - set(aggregates))]
    if missing_targets:
        warnings.append("Required language/script corpora are missing; all-target validation is incomplete.")
    if not source_reports:
        warnings.append("No corpora supplied; structural and synthetic checks do not establish text coverage.")
    if source_reports and any(not row.get("hash_verified") for row in source_reports):
        warnings.append("Some corpus sources are unpinned; returned SHA256 values record this run but do not prove source identity or independence.")
    status = "failed" if errors else "incomplete" if missing_targets or not source_reports else "passed_on_supplied_text"
    return {
        "schema_version": 1, "status": status,
        "artifacts": {"base_sha256": _sha256(base_path), "extended_sha256": _sha256(extended_path), "manifest_sha256": _sha256(manifest_path)},
        "structural_passed": structural_passed,
        "checks": checks, "errors": errors, "warnings": warnings,
        "protected_text": protected, "normalizer_controls": normalizer_controls,
        "reachability": {"checked": len(witnesses), "failed": sum(not row["passed"] for row in witnesses), "failures": [row for row in witnesses if not row["passed"]][:max_examples]},
        "required_characters": character_checks,
        "corpus_sources": source_reports, "per_language": language_reports,
        "missing_targets": missing_targets, "examples": examples,
        "asr_validation": "not_run",
        "limitations": [
            "Results apply only to the pinned artifacts and supplied text; they do not prove arbitrary Unicode coverage.",
            "Lost normalized characters count low-level BPE unknown spans before special-token interception; use ordinary transcript text for this metric.",
            "Expected text follows the base normalization/Metaspace contract or explicit references; ASR labels may require additional independently reviewed spelling conventions.",
            "This report does not validate checkpoint tensors, model runtime mappings, speech accuracy, or independent corpus provenance.",
        ],
    }
