"""Command line operations with nonzero exit codes for unpassed evidence gates."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .sources import fetch_sources, read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sttok")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="Download hash-pinned tokenizer inputs")
    fetch.add_argument("--lock", default="configs/sources.lock.json")
    fetch.add_argument("--cache", default=".cache/sources")
    corpora = commands.add_parser("fetch-corpora", help="Prepare pinned public native text for the declared scripts")
    corpora.add_argument("--lock", default="configs/corpora.lock.json")
    corpora.add_argument("--config", default="configs/build.json")
    corpora.add_argument("--output", default=".cache/bhasha")
    scope = commands.add_parser("scope-corpora", help="Explicitly separate undeclared-script records; retain raw audit separately")
    scope.add_argument("--config", default="configs/build.json")
    scope.add_argument("--cache", default=".cache/sources")
    scope.add_argument("--corpora", required=True)
    scope.add_argument("--output", required=True)
    build = commands.add_parser("build", help="Build canonical HF BPE and a candidate manifest")
    build.add_argument("--config", default="configs/build.json")
    build.add_argument("--cache", default=".cache/sources")
    build.add_argument("--output", default="artifacts/nemotron-indic-v1")
    build.add_argument("--previous", help="Previously released artifact directory for append-only upgrades")
    validate = commands.add_parser("validate", help="Validate structure, character coverage and supplied corpora")
    validate.add_argument("--base", default=".cache/sources/nvidia/tokenizer.json")
    validate.add_argument("--tokenizer", default="artifacts/nemotron-indic-v1/tokenizer.json")
    validate.add_argument("--manifest", default="artifacts/nemotron-indic-v1/manifest.json")
    validate.add_argument("--corpora", help="JSON corpus manifest; absent corpora means incomplete")
    validate.add_argument("--output", default="reports/tokenizer.json")
    preflight = commands.add_parser("preflight", help="Check source HF tokenizer/model ID agreement")
    preflight.add_argument("--tokenizer", default=".cache/sources/nvidia/tokenizer.json")
    preflight.add_argument("--model-config", default=".cache/sources/nvidia/config.json")
    preflight.add_argument("--output", default="reports/preflight.json")
    mapping = commands.add_parser("id-map", help="Export explicit HF to NeMo RNNT ID mapping")
    mapping.add_argument("--tokenizer", default="artifacts/nemotron-indic-v1/tokenizer.json")
    mapping.add_argument("--base", default=".cache/sources/nvidia/tokenizer.json")
    mapping.add_argument("--output", default="artifacts/nemotron-indic-v1/nemo-id-map.json")
    prompts = commands.add_parser("prompts", help="Allocate new language identities without moving old prompt slots")
    prompts.add_argument("--processor", default=".cache/sources/nvidia/processor_config.json")
    prompts.add_argument("--config", default="configs/build.json")
    prompts.add_argument("--output", default="artifacts/nemotron-indic-v1/prompts.json")
    prompts.add_argument("--previous")
    migrate = commands.add_parser("migrate", help="Expand and reload a complete local NeMo checkpoint")
    migrate.add_argument("--source", required=True)
    migrate.add_argument("--tokenizer", default="artifacts/nemotron-indic-v1/tokenizer.json")
    migrate.add_argument("--base", default=".cache/sources/nvidia/tokenizer.json")
    migrate.add_argument("--output", required=True)
    migrate.add_argument("--prompts", help="Verified complete prompt dictionary JSON")
    migrate.add_argument("--seed", type=int, default=0)
    compatibility = commands.add_parser("verify-checkpoint", help="Check real migrated weights, old logits and paired audio decoding")
    compatibility.add_argument("--source", required=True)
    compatibility.add_argument("--expanded", required=True)
    compatibility.add_argument("--base", default=".cache/sources/nvidia/tokenizer.json")
    compatibility.add_argument("--tokenizer", default="artifacts/nemotron-indic-v1/tokenizer.json")
    compatibility.add_argument("--manifest", required=True)
    compatibility.add_argument("--output", required=True)
    compatibility.add_argument("--device", default="cpu")
    compatibility.add_argument("--atol", type=float, default=1e-6)
    compatibility.add_argument("--rtol", type=float, default=1e-5)
    evaluate = commands.add_parser("evaluate", help="Score real ASR prediction manifests; absent evidence blocks release")
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", default="reports/speech.json")
    evaluate.add_argument("--bootstrap-samples", type=int, default=2000)
    evaluate.add_argument("--seed", type=int, default=0)
    infer = commands.add_parser("infer", help="Run actual offline NeMo inference with recorded provenance")
    infer.add_argument("--manifest", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--phase", required=True, choices=["baseline", "expanded_untrained", "fine_tuned"])
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    try:
        status = 0
        if args.command == "fetch":
            result = {"verified_files": len(fetch_sources(args.lock, args.cache))}
        elif args.command == "fetch-corpora":
            from .corpora import prepare_bhasha
            manifest = prepare_bhasha(args.lock, args.config, args.output)
            result = {"corpora": len(manifest["corpora"]), "missing_targets": manifest["missing_targets"], "manifest": str(Path(args.output) / "manifest.json")}
        elif args.command == "scope-corpora":
            from .scope import scope_corpora
            report = scope_corpora(args.config, args.cache, args.corpora, args.output)
            result = {"retained": report["retained_records"], "excluded": report["excluded_count"], "manifest": str(Path(args.output) / "manifest.json")}
        elif args.command == "build":
            from .builder import build_tokenizer
            manifest = build_tokenizer(args.config, args.cache, args.output, previous=args.previous)
            result = {key: manifest[key] for key in ("build_passed", "vocabulary_size", "merge_count", "tokenizer_sha256", "asr_validated")}
        elif args.command == "validate":
            from .validation import validate_tokenizer
            result = validate_tokenizer(args.base, args.tokenizer, args.manifest, args.corpora)
            write_json(args.output, result)
            status = 0 if result["status"] == "passed_on_supplied_text" else 2
            result = {"status": result["status"], "structural_passed": result["structural_passed"], "errors": len(result["errors"]), "report": args.output}
        elif args.command == "preflight":
            from .checkpoint import artifact_preflight
            result = artifact_preflight(args.tokenizer, args.model_config)
            write_json(args.output, result)
            status = 0 if result["hf_direct_compatible"] else 2
        elif args.command == "id-map":
            from .runtime import build_id_map
            mapping = build_id_map(args.tokenizer, args.base)
            write_json(args.output, mapping.to_dict())
            result = {"output": args.output, "model_blank_id": mapping.model_blank_id, "model_output_size": mapping.model_output_size}
        elif args.command == "prompts":
            from .prompts import build_prompt_registry
            registry = build_prompt_registry(args.processor, args.config, args.output, previous_registry=read_json(args.previous) if args.previous else None)
            result = {"output": args.output, "prompt_identities": len(registry["prompt_dictionary"])}
        elif args.command == "migrate":
            from .checkpoint import migrate_nemo_checkpoint
            prompts = read_json(args.prompts) if args.prompts else None
            if prompts and "prompt_dictionary" in prompts:
                if prompts.get("explicit_aliases"):
                    raise ValueError("Migration of new prompt aliases is not enabled; use the default registry")
                prompts = prompts["prompt_dictionary"]
            result = migrate_nemo_checkpoint(args.source, args.tokenizer, args.base, args.output, seed=args.seed, prompt_dictionary=prompts)
        elif args.command == "verify-checkpoint":
            from .checkpoint_validation import validate_checkpoint_pair
            report = validate_checkpoint_pair(args.source, args.expanded, args.base, args.tokenizer, args.manifest, args.output, device=args.device, atol=args.atol, rtol=args.rtol)
            status = 0 if report["passed"] else 2
            result = {"status": report["status"], "report": args.output}
        elif args.command == "infer":
            from .inference import run_nemo_inference
            result = run_nemo_inference(args.manifest, args.checkpoint, args.phase, args.output, device=args.device)
        else:
            from .evaluation import evaluate_predictions, load_manifest, load_predictions, write_report
            result = evaluate_predictions(load_manifest(args.manifest), load_predictions(args.predictions), bootstrap_samples=args.bootstrap_samples, seed=args.seed)
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
