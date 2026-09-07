"""Report corpus scope explicitly, without using the extended vocabulary to filter.

Always retain the unfiltered audit as well. This only separates whole records
containing undeclared scripts/notation; it never strips or repairs their text.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

from .builder import assigned_codepoints
from .sources import read_json, sha256, verify_sources, write_json
from .validation import _records, _specs


def allowed_characters(config: dict, cache: Path) -> set[str]:
    base = read_json(cache / config["base_tokenizer"])
    allowed = {c for piece in base["model"]["vocab"] for c in piece}
    assigned = assigned_codepoints(cache / config["unicode_data"])
    # Whole declared script ranges are allowed, even when an individual
    # character is missing from the built vocabulary and will fail validation.
    for target in config["targets"]:
        for first, last in target["ranges"]:
            allowed.update(chr(cp) for cp in range(first, last + 1) if cp in assigned)
    allowed.update(row["character"] for row in config["required_characters"])
    allowed.update(" \t\r\n")
    return allowed


def scope_corpora(config_path, cache, corpus_manifest, output):
    config_path, cache, output = Path(config_path), Path(cache), Path(output).resolve()
    config = read_json(config_path)
    if "sources_lock" in config:
        verify_sources(config_path.parent / config["sources_lock"], cache)
    latin_path = config_path.parent / config["latin_characters"]
    if config.get("latin_sha256") and sha256(latin_path) != config["latin_sha256"]:
        raise ValueError("Scope policy Latin character inventory hash changed")
    allowed = allowed_characters(config, cache)
    allowed.update(line.split("\t")[1] for line in latin_path.read_text().splitlines() if line)
    base = Tokenizer.from_file(str(cache / config["base_tokenizer"]))
    output.mkdir(parents=True, exist_ok=True)
    specs, sources, exclusions = [], [], []
    for index, spec in enumerate(_specs(corpus_manifest)):
        if spec.get("sha256") != sha256(spec["path"]):
            raise ValueError(f"Scope filtering requires a verified corpus hash: {spec['path']}")
        retained, rejected, chars = [], 0, Counter()
        for number, row in _records(spec):
            text = row[spec.get("text_field", "text")]
            unsupported = sorted(set(base.normalizer.normalize_str(text)) - allowed)
            if unsupported:
                rejected += 1
                chars.update(unsupported)
                exclusions.append({"source_index": index, "record": number,
                                   "record_id": row.get("id"),
                                   "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                                   "outside_policy": [f"U+{ord(c):04X}" for c in unsupported]})
                continue
            retained.append(dict(row))
        name = f"{index:02d}.jsonl"
        path = output / name
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8")
        copied = {k: v for k, v in spec.items() if k not in {"path", "format", "sha256", "records"}}
        copied.update(path=name, format="jsonl", sha256=sha256(path), records=len(retained))
        specs.append(copied)
        sources.append({"input_sha256": spec["sha256"], "source": spec.get("source"),
                        "retained": len(retained), "excluded": rejected,
                        "outside_policy_characters": [{"character": c, "codepoint": f"U+{ord(c):04X}", "records": n} for c, n in sorted(chars.items())]})
    report = {"schema_version": 1, "config_sha256": sha256(config_path), "corpora": specs,
              "policy_inputs_sha256": {"base_tokenizer": sha256(cache / config["base_tokenizer"]),
                                       "unicode_data": sha256(cache / config["unicode_data"]),
                                       "latin_characters": sha256(latin_path)},
              "policy": "Pinned base character inventory + complete assigned declared script ranges + selected Latin190 + explicit shared character inventory. Never derived from the extended tokenizer. Whole records only; original text unchanged.",
              "source_reports": sources, "excluded_records": exclusions,
              "retained_records": sum(x["retained"] for x in sources),
              "excluded_count": sum(x["excluded"] for x in sources),
              "limitation": "This scope report must accompany the unfiltered audit; it does not establish arbitrary Unicode coverage or ASR accuracy."}
    write_json(output / "manifest.json", report)
    return report
