"""Score real saved ASR predictions against immutable evaluation references.

This module performs no inference. A passing development fixture never becomes
evidence of model accuracy. Release coverage and run provenance are explicit.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PHASES = ("baseline", "expanded_untrained", "fine_tuned")
BASE_ASR_LOCALES = tuple("en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA pl-PL sv-SE cs-CZ nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE".split())
ADAPTATION_LOCALES = tuple("el-GR lt-LT lv-LV mt-MT sl-SI he-IL th-TH nn-NO".split())
INDIC_SCRIPTS = {
    "as": "Beng", "bn": "Beng", "brx": "Deva", "doi": "Deva",
    "gu": "Gujr", "hi": "Deva", "kn": "Knda", "ks": "Arab",
    "kok": "Deva", "mai": "Deva", "ml": "Mlym", "mni": "Mtei",
    "mr": "Deva", "ne": "Deva", "or": "Orya", "pa": "Guru",
    "sa": "Deva", "sat": "Olck", "sd": "Deva", "ta": "Taml",
    "te": "Telu", "ur": "Arab",
}
DEFAULT_NORMALIZATION = {
    "unicode_form": "none", "casefold": False, "punctuation": "preserve",
    "cer_remove_spaces": False,
}


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("Manifest must be a JSON object")
    return value


def load_predictions(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid predictions JSON on line {number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Prediction line {number} must be an object")
            rows.append(row)
    return rows


def _normalization(config: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(config) - set(DEFAULT_NORMALIZATION)
    if unknown:
        raise ValueError(f"Unknown normalization settings: {sorted(unknown)}")
    result = {**DEFAULT_NORMALIZATION, **config}
    if result["unicode_form"] not in ("none", "NFC", "NFKC"):
        raise ValueError("unicode_form must be none, NFC, or NFKC")
    if result["punctuation"] not in ("preserve", "remove"):
        raise ValueError("punctuation must be preserve or remove")
    if not isinstance(result["casefold"], bool) or not isinstance(result["cer_remove_spaces"], bool):
        raise ValueError("casefold and cer_remove_spaces must be booleans")
    return result


def normalize_for_scoring(text: str, config: Mapping[str, Any]) -> str:
    """Apply only the manifest's scoring policy, never tokenizer normalization.

    Whitespace is collapsed to single spaces and stripped in every mode. Joiners
    are never removed. The separately reported raw score retains case, Unicode
    spelling, and punctuation, but uses this same whitespace convention.
    """
    config = _normalization(config)
    if config["unicode_form"] != "none":
        text = unicodedata.normalize(config["unicode_form"], text)
    if config["casefold"]:
        text = text.casefold()
    if config["punctuation"] == "remove":
        text = "".join(c for c in text if not unicodedata.category(c).startswith("P"))
    return " ".join(text.split())


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Levenshtein distance with linear auxiliary storage; repeated tokens count."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(hypothesis, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def score_pair(reference: str, hypothesis: str, normalization: Mapping[str, Any] | None = None) -> dict[str, Any]:
    config = _normalization(normalization or {})
    silent_reference = not reference.strip()
    reference = normalize_for_scoring(reference, config)
    hypothesis = normalize_for_scoring(hypothesis, config)
    words = reference.split()
    hypothesis_words = hypothesis.split()
    chars = reference.replace(" ", "") if config["cer_remove_spaces"] else reference
    hypothesis_chars = hypothesis.replace(" ", "") if config["cer_remove_spaces"] else hypothesis
    return {
        "word_errors": edit_distance(words, hypothesis_words), "reference_words": len(words),
        "character_errors": edit_distance(chars, hypothesis_chars), "reference_characters": len(chars),
        "silent_utterances": int(silent_reference),
        "silent_word_insertions": len(hypothesis_words) if silent_reference else 0,
        "silent_character_insertions": len(hypothesis_chars) if silent_reference else 0,
        "exact_match": reference == hypothesis,
    }


def _aggregate(scores: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    scores = list(scores)
    keys = ("word_errors", "reference_words", "character_errors", "reference_characters", "silent_utterances", "silent_word_insertions", "silent_character_insertions")
    result = {key: sum(score[key] for score in scores) for key in keys}
    result.update({
        "utterances": len(scores), "exact_matches": sum(score["exact_match"] for score in scores),
        "wer": result["word_errors"] / result["reference_words"] if result["reference_words"] else None,
        "cer": result["character_errors"] / result["reference_characters"] if result["reference_characters"] else None,
    })
    return result


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_bootstrap_delta(
    baseline: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
    clusters: Sequence[str], *, samples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """95% percentile intervals for candidate-minus-baseline WER/CER.

    Resample identical speaker/session clusters in both systems; an utterance
    may act as its own cluster in development only. Intervals describe sampled
    material and do not prove population-wide non-regression.
    """
    if not (len(baseline) == len(candidate) == len(clusters)):
        raise ValueError("Paired scores and cluster IDs must have equal lengths")
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, cluster in enumerate(clusters):
        groups[cluster].append(index)
    keys = sorted(groups)
    rng = random.Random(seed)
    deltas: dict[str, list[float]] = {"wer": [], "cer": []}
    for _ in range(samples):
        indices = [i for _ in keys for i in groups[rng.choice(keys)]] if keys else []
        old = _aggregate(baseline[i] for i in indices)
        new = _aggregate(candidate[i] for i in indices)
        for metric in deltas:
            if old[metric] is not None and new[metric] is not None:
                deltas[metric].append(new[metric] - old[metric])
    return {
        "method": "paired_cluster_percentile", "confidence": 0.95,
        "samples": samples, "seed": seed, "clusters": len(keys),
        "intervals": {metric: {"low": _quantile(values, .025), "high": _quantile(values, .975), "valid_samples": len(values)} for metric, values in deltas.items()},
    }


def _required_text(row: Mapping[str, Any], name: str) -> str:
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _release_inventory(profiles: Mapping[str, Mapping[str, Any]]) -> list[str]:
    blockers = []
    for cohort, expected in (("existing_asr", set(BASE_ASR_LOCALES)), ("adaptation", set(ADAPTATION_LOCALES))):
        actual = {profile.get("locale") for profile in profiles.values() if cohort in profile["cohorts"]}
        if actual != expected:
            blockers.append(f"{cohort} inventory mismatch: missing={sorted(expected - actual)}, unexpected={sorted(str(x) for x in actual - expected)}")
    actual_indic = {(p["language"], p["script"]) for p in profiles.values() if "indic" in p["cohorts"]}
    expected_indic = set(INDIC_SCRIPTS.items())
    if actual_indic != expected_indic:
        blockers.append(f"Indic inventory mismatch: missing={sorted(expected_indic - actual_indic)}, unexpected={sorted(actual_indic - expected_indic)}")
    for key, profile in profiles.items():
        if "existing_asr" in profile["cohorts"] and not profile["protected"]:
            blockers.append(f"Existing ASR profile {key} must be protected")
    return blockers


def evaluate_predictions(
    manifest: Mapping[str, Any], predictions: Iterable[Mapping[str, Any]], *,
    bootstrap_samples: int = 2000, seed: int = 0,
) -> dict[str, Any]:
    """Return coverage, per-profile/condition scores, CIs, and conservative gates.

    Bad identifiers/duplicate predictions raise ValueError. Absent evidence is
    reported as blocked, never as a passing zero-error result. Development mode
    permits small fixtures but unconditionally blocks the release gate.
    """
    if manifest.get("schema_version") != 1:
        raise ValueError("Expected evaluation manifest schema_version 1")
    purpose = manifest.get("purpose", "development")
    if purpose not in ("release", "development"):
        raise ValueError("purpose must be release or development")
    normalization = _normalization(manifest.get("normalization", {}))
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    profiles = {}
    for profile in manifest.get("profiles", []):
        key = _required_text(profile, "id")
        if key in profiles:
            raise ValueError(f"Duplicate profile {key}")
        for name in ("language", "script"):
            _required_text(profile, name)
        if not isinstance(profile.get("protected"), bool):
            raise ValueError(f"Profile {key} requires an explicit protected boolean")
        cohorts = profile.get("cohorts", [])
        if not isinstance(cohorts, list) or not cohorts or set(cohorts) - {"existing_asr", "adaptation", "indic"}:
            raise ValueError(f"Invalid cohorts for {key}")
        conditions = profile.get("required_conditions", ["known_language"])
        if not isinstance(conditions, list) or not conditions or any(not isinstance(c, str) or not c for c in conditions) or len(set(conditions)) != len(conditions):
            raise ValueError(f"Invalid required_conditions for {key}")
        profiles[key] = {**profile, "required_conditions": conditions}
    if not profiles:
        raise ValueError("At least one evaluation profile is required")
    utterances = {}
    slices: dict[tuple[str, str], list[str]] = defaultdict(list)
    for item in manifest.get("utterances", []):
        key = _required_text(item, "id")
        if key in utterances:
            raise ValueError(f"Duplicate utterance {key}")
        profile = _required_text(item, "profile_id")
        condition = _required_text(item, "condition_id")
        if profile not in profiles or condition not in profiles[profile]["required_conditions"]:
            raise ValueError(f"Undeclared profile/condition on utterance {key}")
        if not isinstance(item.get("reference"), str):
            raise ValueError(f"Reference for {key} must be a string; use empty text for silence")
        utterances[key] = item
        slices[(profile, condition)].append(key)
    records = {}
    prediction_values = list(predictions)
    for row in prediction_values:
        utterance_id = _required_text(row, "utterance_id")
        phase = row.get("phase")
        if utterance_id not in utterances or phase not in PHASES:
            raise ValueError(f"Unknown utterance or phase: {utterance_id}, {phase}")
        if not isinstance(row.get("hypothesis"), str):
            raise ValueError(f"Hypothesis for {utterance_id} must be a string")
        key = (utterance_id, phase)
        if key in records:
            raise ValueError(f"Duplicate prediction {key}")
        records[key] = row
    blockers = []
    if purpose != "release":
        blockers.append("Development evaluation cannot establish release readiness")
    if purpose == "release":
        blockers.extend(_release_inventory(profiles))
    if not utterances:
        blockers.append("No held-out utterances supplied")
    runs = manifest.get("runs", {})
    if purpose == "release":
        if "normalization" not in manifest:
            blockers.append("Release requires an explicit fixed normalization policy")
        for phase in PHASES:
            run = runs.get(phase, {})
            if not run.get("run_id") or not isinstance(run.get("settings"), dict):
                blockers.append(f"Missing run identity or decoding settings for {phase}")
            for artifact in ("checkpoint_sha256", "tokenizer_sha256"):
                if not re.fullmatch(r"[0-9a-f]{64}", str(run.get(artifact, ""))):
                    blockers.append(f"Missing SHA-256 {artifact} for {phase}")
        for key, item in utterances.items():
            if not item.get("cluster_id") or not item.get("dataset") or not item.get("split") or not item.get("audio_sha256"):
                blockers.append(f"Missing held-out provenance/cluster for utterance {key}")
            elif not re.fullmatch(r"[0-9a-f]{64}", str(item["audio_sha256"])):
                blockers.append(f"Invalid audio SHA-256 for utterance {key}")
            if item.get("split") not in ("test", "validation", "held_out"):
                blockers.append(f"Utterance {key} must use a held-out split")
        for (key, phase), row in records.items():
            if row.get("run_id") != runs.get(phase, {}).get("run_id") or not row.get("run_id"):
                blockers.append(f"Run identity mismatch for {key}/{phase}")
            if row.get("language_mode") not in ("automatic", "known"):
                blockers.append(f"Missing actual language mode for {key}/{phase}")
            if row.get("language_mode") == "known" and not row.get("language_prompt"):
                blockers.append(f"Missing actual language prompt for {key}/{phase}")
            for artifact in ("checkpoint_sha256", "tokenizer_sha256"):
                if artifact in row and row[artifact] != runs.get(phase, {}).get(artifact):
                    blockers.append(f"Prediction {artifact} mismatch for {key}/{phase}")
            if "audio_sha256" in row and row["audio_sha256"] != utterances[key].get("audio_sha256"):
                blockers.append(f"Prediction audio hash mismatch for {key}/{phase}")
    missing = []
    regressions = []
    reports = []
    for profile_id, profile in sorted(profiles.items()):
        for condition in profile["required_conditions"]:
            ids = sorted(slices[(profile_id, condition)])
            if not ids:
                blockers.append(f"No evaluation data for {profile_id}/{condition}")
            row_scores: dict[str, list[dict[str, Any]]] = {}
            raw_scores: dict[str, list[dict[str, Any]]] = {}
            phase_reports = {}
            for phase in PHASES:
                available = [key for key in ids if (key, phase) in records]
                absent = [key for key in ids if (key, phase) not in records]
                missing.extend({"utterance_id": key, "phase": phase} for key in absent)
                row_scores[phase] = [score_pair(utterances[key]["reference"], records[(key, phase)]["hypothesis"], normalization) for key in available]
                raw_scores[phase] = [score_pair(utterances[key]["reference"], records[(key, phase)]["hypothesis"]) for key in available]
                phase_reports[phase] = {
                    "status": "complete" if ids and not absent else "blocked",
                    "expected_utterances": len(ids), "scored_utterances": len(available),
                    "score": _aggregate(row_scores[phase]) if available else None,
                    "raw_score": _aggregate(raw_scores[phase]) if available else None,
                }
            comparisons = {}
            for candidate in PHASES[1:]:
                complete = bool(ids) and all((key, phase) in records for key in ids for phase in ("baseline", candidate))
                if not complete:
                    comparisons[candidate] = {"status": "blocked", "reason": "Incomplete paired predictions"}
                    continue
                same_modes = all((records[(key, "baseline")].get("language_mode"), records[(key, "baseline")].get("language_prompt")) == (records[(key, candidate)].get("language_mode"), records[(key, candidate)].get("language_prompt")) for key in ids)
                same_settings = runs.get("baseline", {}).get("settings") == runs.get(candidate, {}).get("settings")
                same_settings = same_settings and all(records[(key, "baseline")].get("actual_settings") == records[(key, candidate)].get("actual_settings") for key in ids)
                comparable = same_modes and same_settings
                if profile["protected"] and not comparable:
                    blockers.append(f"Changed language mode/prompt or decoding settings for protected {profile_id}/{condition}/{candidate}")
                old = phase_reports["baseline"]["score"]
                new = phase_reports[candidate]["score"]
                worse = [metric for metric in ("wer", "cer") if old[metric] is not None and new[metric] is not None and new[metric] > old[metric]]
                worse += [metric for metric in ("silent_word_insertions", "silent_character_insertions") if new[metric] > old[metric]]
                delta = {metric: new[metric] - old[metric] if new[metric] is not None and old[metric] is not None else None for metric in ("wer", "cer")}
                cluster_ids = [str(utterances[key].get("cluster_id", key)) for key in ids]
                comparisons[candidate] = {
                    "status": "regression" if worse else "no_measured_regression",
                    "same_language_and_decoding_protocol": comparable, "delta": delta,
                    "worse_metrics": worse,
                    "confidence_interval": paired_bootstrap_delta(row_scores["baseline"], row_scores[candidate], cluster_ids, samples=bootstrap_samples, seed=seed),
                }
                if profile["protected"] and candidate == "fine_tuned" and worse:
                    regressions.append({"profile_id": profile_id, "condition_id": condition, "metrics": worse})
            reports.append({"profile_id": profile_id, "condition_id": condition, "language": profile["language"], "script": profile["script"], "cohorts": profile["cohorts"], "protected": profile["protected"], "phases": phase_reports, "comparisons": comparisons})
    if missing:
        blockers.append(f"Missing {len(missing)} required predictions")
    # Silence-only recordings do not establish recognition accuracy for a locale.
    for report in reports:
        baseline_score = report["phases"]["baseline"]["score"]
        if baseline_score and not baseline_score["reference_words"]:
            blockers.append(f"Only empty references for {report['profile_id']}/{report['condition_id']}")
    # Quality floors for new languages are declared before evaluation, never
    # invented from a passing token-coverage check or from observed scores.
    targets = manifest.get("indic_accuracy_targets", {})
    for report in reports:
        if "indic" not in report["cohorts"]:
            continue
        target = targets.get(report["profile_id"])
        if purpose == "release" and (not isinstance(target, dict) or set(target) != {"max_wer", "max_cer"}):
            blockers.append(f"Missing predeclared Indic accuracy targets for {report['profile_id']}")
            continue
        if target is not None:
            for metric in ("wer", "cer"):
                limit = target.get(f"max_{metric}")
                if isinstance(limit, bool) or not isinstance(limit, (int, float)) or not math.isfinite(limit) or limit < 0:
                    raise ValueError("Accuracy limits must be finite nonnegative numbers")
                score = report["phases"]["fine_tuned"]["score"]
                if score and score[metric] is not None and score[metric] > limit:
                    regressions.append({"profile_id": report["profile_id"], "condition_id": report["condition_id"], "metrics": [metric], "reason": "Indic accuracy target exceeded", "limit": limit})
    blockers = sorted(set(blockers))
    canonical = json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return {
        "schema_version": 1, "purpose": purpose,
        "manifest_sha256": hashlib.sha256(canonical).hexdigest(),
        "normalization": normalization, "rates_are_fractions": True,
        "cer_unit": "Unicode code point, not grapheme cluster",
        "release_status": "blocked" if blockers else ("failed" if regressions else "passed"),
        "blockers": blockers, "regressions": regressions, "missing_predictions": missing,
        "profiles": reports,
        "limitations": [
            "Only saved predictions are scored; this report does not execute inference or verify checkpoint tensors.",
            "Accuracy preservation applies to the declared held-out material, not every possible recording.",
            "Paired confidence intervals are descriptive; the no-worsening gate uses point WER/CER and silence insertions.",
            "Raw scores preserve case, punctuation, and Unicode spelling but collapse whitespace.",
        ],
    }


def write_report(report: Mapping[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return path
