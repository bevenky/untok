"""Native Unigram commands with an explicit namespace for the earlier BPE work."""
from __future__ import annotations

import argparse
import json
import sys

from .sources import write_json


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "legacy-bpe":
        from .legacy_bpe_cli import main as legacy_main

        return legacy_main(argv[1:])

    parser = argparse.ArgumentParser(
        prog="sttok",
        description="Build and validate native SentencePiece Unigram tokenizer bundles for Nemotron.",
        epilog=("Native checkpoint migration is pending. Use 'sttok legacy-bpe --help' "
                "to reproduce the earlier BPE experiment with its original arguments and defaults."),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "build", aliases=["build-unigram"],
        help="Package fitted additions against a hash-pinned native Unigram model",
    )
    build.set_defaults(operation="build")
    build.add_argument("--base", required=True, help="Original SentencePiece model from the native checkpoint")
    build.add_argument("--selection", required=True, help="Ordered, scored additions and native base SHA256")
    build.add_argument("--output", required=True, help="Empty directory for a separate native candidate bundle")
    check = commands.add_parser(
        "check", aliases=["check-unigram"],
        help="Verify native bundle hashes, prefix and ID mapping on CPU",
    )
    check.set_defaults(operation="check")
    check.add_argument("--bundle", required=True)
    validate = commands.add_parser(
        "validate", aliases=["validate-unigram"],
        help="Validate native structure and pinned text corpora on CPU",
    )
    validate.set_defaults(operation="validate")
    validate.add_argument("--bundle", required=True)
    validate.add_argument("--policy", required=True, help="Explicit native validation policy JSON")
    validate.add_argument("--corpora", help="Frozen corpus manifest; omitted inputs mean incomplete")
    validate.add_argument("--phase", choices=["dev", "reserve"], default="dev")
    validate.add_argument("--selection-receipt", help="Required artifact receipt for reserve evaluation")
    validate.add_argument("--max-examples", type=int, default=0, help="Dev text examples only; reserve always omits text")
    validate.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate", help="Score supplied ASR predictions independently of tokenizer format")
    evaluate.set_defaults(operation="evaluate")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", default="reports/speech.json")
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=0)
    commands.add_parser("legacy-bpe", help="Reproduce earlier BPE commands with their original flags and defaults")
    args = parser.parse_args(argv)
    try:
        status = 0
        if args.operation == "build":
            from .unigram import build_native_tokenizer

            result = build_native_tokenizer(args.base, args.selection, args.output)
        elif args.operation == "check":
            from .unigram import NativeTokenizerAdapter

            adapter = NativeTokenizerAdapter(args.bundle)
            result = {"structural_passed": True, "native_vocabulary_size": adapter.vocab_size,
                      "native_blank_id": adapter.blank_id, "tokenizer_sha256": adapter.id_map.tokenizer_sha256,
                      "checkpoint_validated": False, "asr_validated": False}
        elif args.operation == "validate":
            from .unigram_validation import validate_native_tokenizer

            report = validate_native_tokenizer(args.bundle, args.policy, args.corpora, phase=args.phase,
                                               selection_receipt_path=args.selection_receipt,
                                               max_examples=args.max_examples)
            write_json(args.output, report)
            status = 0 if report["passed"] else 2
            result = {"status": report["status"], "structural_passed": report["structural_passed"],
                      "corpus_status": report["corpus_status"], "report": args.output,
                      "checkpoint_validated": False, "asr_validated": False}
        elif args.operation == "evaluate":
            from .evaluation import evaluate_predictions, load_manifest, load_predictions, write_report

            result = evaluate_predictions(load_manifest(args.manifest), load_predictions(args.predictions),
                                          bootstrap_samples=args.bootstrap_samples, seed=args.seed)
            write_report(result, args.output)
            status = 0 if result["release_status"] == "passed" else 2
            result = {"release_status": result["release_status"], "report": args.output}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return status
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"sttok: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
