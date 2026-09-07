#!/usr/bin/env python3
"""Compare packaged Latin+Indic tokens with the frozen full candidate on text.

This is a packaging regression, not another tokenizer-selection experiment.
No scores or pieces are fitted here, and previously evaluated corpora are not
described as new independent holdouts.
"""
import argparse
import hashlib
import json
from pathlib import Path

from untok.bundles import load_tokenizer_bundle


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--latin-indic", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=["dev", "reserve"], default="dev")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new output path")
    full, subset = load_tokenizer_bundle(args.full), load_tokenizer_bundle(args.latin_indic)
    mapping = subset.full_native_to_subset_native
    manifest = json.loads(args.corpus_manifest.read_text())
    selected_files = {name: digest for name, digest in manifest["files"].items()
                      if name.startswith(args.phase + "/") and name.endswith(".jsonl")}
    if not selected_files:
        raise ValueError("The corpus manifest contains no files for the requested phase")
    report = {"schema_version": 1, "scope": "packaging regression on previously evaluated frozen text",
              "full_manifest_sha256": sha(args.full / "manifest.json"),
              "subset_manifest_sha256": sha(args.latin_indic / "manifest.json"),
              "corpus_manifest_sha256": sha(args.corpus_manifest), "phase": args.phase,
              "refitting_performed": False, "asr_accuracy_evaluated": False, "profiles": {}}
    for name, digest in sorted(selected_files.items()):
        path = args.corpus_manifest.parent / name
        if sha(path) != digest:
            raise ValueError(f"Frozen corpus hash changed: {name}")
        stats = {"records": 0, "full_unknown_records": 0, "subset_unknown_records": 0,
                 "retained_sequence_comparisons": 0, "retained_sequence_mismatches": 0,
                 "retained_decode_mismatches": 0, "records_using_removed_pieces": 0}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            text = row["text"]
            old, new = full.text_to_ids(text), subset.text_to_ids(text)
            stats["records"] += 1
            stats["full_unknown_records"] += full.unk_id in old
            stats["subset_unknown_records"] += subset.unk_id in new
            mapped = [mapping[i] for i in old]
            if None in mapped:
                stats["records_using_removed_pieces"] += 1
                continue
            if full.unk_id not in old:
                stats["retained_sequence_comparisons"] += 1
                stats["retained_sequence_mismatches"] += mapped != new
                stats["retained_decode_mismatches"] += full.ids_to_text(old) != subset.ids_to_text(new)
        report["profiles"][Path(name).stem] = stats
    report["passed"] = all(not x["retained_sequence_mismatches"] and not x["retained_decode_mismatches"]
                            for x in report["profiles"].values())
    if args.phase == "reserve":
        report["passed"] &= all(not x["subset_unknown_records"] for x in report["profiles"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "profiles": len(report["profiles"]),
                      "records": sum(x["records"] for x in report["profiles"].values())}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
