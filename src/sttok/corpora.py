"""Prepare reproducible public text checks without changing source spellings."""
from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path

from .sources import fetch_sources, read_json, sha256, write_json


def prepare_bhasha(lock: str | Path, build_config: str | Path, output: str | Path) -> dict:
    output = Path(output).resolve()
    fetch_sources(lock, output / "raw")
    with zipfile.ZipFile(output / "raw/bhasha.zip") as archive:
        rows = json.loads(archive.read("bhasha-abhijnaanam.json"))["data"]
    targets = {t["language"]: t for t in read_json(build_config)["targets"]}
    aliases = {"Perso-Arabic": "Arabic", "Meetei-Mayek": "Meetei Mayek", "Ol-Chiki": "Ol Chiki", "Oriya": "Odia"}
    grouped = {lang: [] for lang in targets}
    excluded = Counter()
    for number, row in enumerate(rows):
        lang = row["unique_identifier"].split("_", 1)[0]
        lang = {"dg": "doi", "gom": "kok"}.get(lang, lang)
        script = aliases.get(row["script"], row["script"])
        if lang not in targets or script != targets[lang]["script"]:
            excluded[f"outside_declared_script:{lang}:{script}"] += 1
            continue
        text = row.get("native sentence")
        if not isinstance(text, str) or not text.strip():
            excluded["empty_native_text"] += 1
            continue
        grouped[lang].append({"id": row["unique_identifier"], "source_row": number,
                              "text": text, "language": lang, "script": script,
                              "source": row.get("source")})
    specs = []
    for lang, records in grouped.items():
        if not records:
            continue
        path = output / f"{lang}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        specs.append({"path": path.name, "language": lang, "script": targets[lang]["script"],
                      "format": "jsonl", "sha256": sha256(path), "records": len(records),
                      "source": "Bhasha-Abhijnaanam v1.0 (source-specific provenance retained per record)"})
    manifest = {"schema_version": 1, "corpora": specs, "excluded_records": dict(sorted(excluded.items())),
                "missing_targets": sorted(set(targets) - {s["language"] for s in specs}),
                "archive_sha256": sha256(output / "raw/bhasha.zip"),
                "scope": "All nonempty native text in the declared scripts. No Unicode normalization, vocabulary-based filtering, or deduplication.",
                "limitation": "Text coverage only. This is public donor-related material, not independent held-out speech."}
    write_json(output / "manifest.json", manifest)
    return manifest
