import copy
import json

import pytest

from sttok.evaluation import (
    ADAPTATION_LOCALES, BASE_ASR_LOCALES, INDIC_SCRIPTS, PHASES,
    evaluate_predictions, load_manifest, load_predictions,
    paired_bootstrap_delta, score_pair, write_report,
)


def fixture(profile_ids=("en-US",), references=("one two",)):
    profiles = [{"id": key, "locale": key, "language": key.split("-")[0], "script": "Latn", "cohorts": ["existing_asr"], "protected": True, "required_conditions": ["known_language"]} for key in profile_ids]
    utterances = [{"id": f"{key}-{index}", "profile_id": key, "condition_id": "known_language", "reference": reference} for key in profile_ids for index, reference in enumerate(references)]
    manifest = {"schema_version": 1, "purpose": "development", "profiles": profiles, "utterances": utterances}
    predictions = [{"utterance_id": item["id"], "phase": phase, "hypothesis": item["reference"]} for item in utterances for phase in PHASES]
    return manifest, predictions


def evaluate(manifest, predictions):
    return evaluate_predictions(manifest, predictions, bootstrap_samples=40, seed=19)


def test_perfect_predictions_are_measured_but_development_never_releases():
    manifest, predictions = fixture(references=("one two", "one one two"))
    report = evaluate(manifest, predictions)
    assert report["release_status"] == "blocked"
    assert report["regressions"] == []
    comparison = report["profiles"][0]["comparisons"]["fine_tuned"]
    assert comparison["delta"] == {"wer": 0, "cer": 0}
    assert comparison["confidence_interval"]["intervals"]["wer"]["low"] == 0
    assert comparison["confidence_interval"]["intervals"]["wer"]["high"] == 0


def test_repeated_tokens_are_not_deduplicated():
    score = score_pair("one one two", "one two")
    assert score["word_errors"] == 1
    assert score["reference_words"] == 3


def test_punctuation_only_reference_is_not_relabelled_as_silence():
    score = score_pair("!!!", "speech", {"punctuation": "remove"})
    assert score["reference_words"] == 0
    assert score["word_errors"] == 1
    assert score["silent_utterances"] == 0


def test_empty_reference_insertions_are_counted_without_infinite_json(tmp_path):
    manifest, predictions = fixture(references=("",))
    for prediction in predictions:
        if prediction["phase"] == "fine_tuned":
            prediction["hypothesis"] = "hallucinated speech"
    report = evaluate(manifest, predictions)
    score = report["profiles"][0]["phases"]["fine_tuned"]["score"]
    assert score["wer"] is None and score["cer"] is None
    assert score["silent_word_insertions"] == 2
    assert report["regressions"][0]["metrics"] == ["silent_word_insertions", "silent_character_insertions"]
    assert any("Only empty references" in reason for reason in report["blockers"])
    result = write_report(report, tmp_path / "report.json")
    assert "Infinity" not in result.read_text()
    assert json.loads(result.read_text())["release_status"] == "blocked"


def test_silence_hallucination_cannot_hide_behind_improved_speech():
    manifest, predictions = fixture(references=("one two three four", ""))
    for prediction in predictions:
        if prediction["utterance_id"] == "en-US-0" and prediction["phase"] == "baseline":
            prediction["hypothesis"] = "wrong words"
        if prediction["utterance_id"] == "en-US-1" and prediction["phase"] == "fine_tuned":
            prediction["hypothesis"] = "hallucination"
    report = evaluate(manifest, predictions)
    assert report["profiles"][0]["comparisons"]["fine_tuned"]["delta"]["wer"] < 0
    assert "silent_word_insertions" in report["regressions"][0]["metrics"]


def test_locale_regression_cannot_hide_in_an_average_improvement():
    manifest, predictions = fixture(profile_ids=("en-US", "fr-FR"), references=("one two three four",))
    for prediction in predictions:
        if prediction["utterance_id"].startswith("en") and prediction["phase"] == "baseline":
            prediction["hypothesis"] = "zero"
        if prediction["utterance_id"].startswith("fr") and prediction["phase"] == "fine_tuned":
            prediction["hypothesis"] = "one two three"
    report = evaluate(manifest, predictions)
    assert len(report["regressions"]) == 1
    assert report["regressions"][0]["profile_id"] == "fr-FR"


def test_missing_predictions_and_entire_profile_are_blockers():
    manifest, predictions = fixture()
    manifest["profiles"].append({"id": "mr-Deva", "language": "mr", "script": "Deva", "cohorts": ["indic"], "protected": False})
    predictions = [p for p in predictions if p["phase"] != "fine_tuned"]
    report = evaluate(manifest, predictions)
    assert report["missing_predictions"] == [{"utterance_id": "en-US-0", "phase": "fine_tuned"}]
    assert "No evaluation data for mr-Deva/known_language" in report["blockers"]
    assert report["profiles"][0]["comparisons"]["fine_tuned"]["status"] == "blocked"


def test_paired_bootstrap_is_deterministic_and_uses_same_clusters():
    baseline = [score_pair("a b", "a b"), score_pair("c d", "c d"), score_pair("e f", "e f")]
    candidate = [score_pair("a b", "a"), score_pair("c d", "c"), score_pair("e f", "e")]
    first = paired_bootstrap_delta(baseline, candidate, ["speaker1", "speaker1", "speaker2"], samples=100, seed=7)
    assert first == paired_bootstrap_delta(baseline, candidate, ["speaker1", "speaker1", "speaker2"], samples=100, seed=7)
    assert first["clusters"] == 2
    assert first["intervals"]["wer"]["low"] == .5
    assert first["intervals"]["wer"]["high"] == .5


def test_scoring_policy_does_not_hide_raw_spelling_or_remove_joiners():
    manifest, predictions = fixture(references=("Office!", "उद्‌गार"))
    manifest["normalization"] = {"casefold": True, "punctuation": "remove"}
    for prediction in predictions:
        if prediction["phase"] == "fine_tuned":
            prediction["hypothesis"] = "office" if prediction["utterance_id"].endswith("0") else "उद्गार"
    report = evaluate(manifest, predictions)
    score = report["profiles"][0]["phases"]["fine_tuned"]
    assert score["score"]["word_errors"] == 1
    assert score["raw_score"]["word_errors"] == 2
    assert score_pair("e\u0301", "é")["character_errors"] == 2
    assert score_pair("e\u0301", "é", {"unicode_form": "NFC"})["character_errors"] == 0


def test_condition_specific_regression_is_preserved():
    manifest, predictions = fixture()
    manifest["profiles"][0]["required_conditions"].append("automatic_streaming_112ms")
    manifest["utterances"].append({"id": "auto", "profile_id": "en-US", "condition_id": "automatic_streaming_112ms", "reference": "one two"})
    predictions.extend({"utterance_id": "auto", "phase": phase, "hypothesis": "one" if phase == "fine_tuned" else "one two"} for phase in PHASES)
    report = evaluate(manifest, predictions)
    assert report["regressions"][0]["condition_id"] == "automatic_streaming_112ms"


def test_changed_prompt_protocol_blocks_protected_comparison():
    manifest, predictions = fixture()
    for row in predictions:
        row["language_mode"] = "automatic" if row["phase"] == "baseline" else "known"
        row["language_prompt"] = None if row["phase"] == "baseline" else "en-US"
    report = evaluate(manifest, predictions)
    assert any("Changed language mode/prompt" in reason for reason in report["blockers"])
    assert not report["profiles"][0]["comparisons"]["fine_tuned"]["same_language_and_decoding_protocol"]


def test_actual_decoder_changes_block_even_if_run_labels_match():
    manifest, predictions = fixture()
    for row in predictions:
        row["actual_settings"] = {"decoding_sha256": "old" if row["phase"] == "baseline" else "different"}
    report = evaluate(manifest, predictions)
    assert any("Changed language mode/prompt or decoding settings" in reason for reason in report["blockers"])


def test_release_inventory_has_exact_40_base_and_22_indic_pairs():
    manifest, predictions = fixture()
    manifest["purpose"] = "release"
    report = evaluate(manifest, predictions)
    assert len(BASE_ASR_LOCALES) == 32
    assert len(ADAPTATION_LOCALES) == 8
    assert len(INDIC_SCRIPTS) == 22
    assert any("existing_asr inventory mismatch" in reason for reason in report["blockers"])
    assert any("Indic inventory mismatch" in reason for reason in report["blockers"])
    assert any("Missing SHA-256" in reason for reason in report["blockers"])
    assert any("Missing held-out provenance" in reason for reason in report["blockers"])


def test_new_indic_accuracy_target_is_separate_from_old_language_preservation():
    manifest, predictions = fixture()
    manifest["profiles"][0].update({"cohorts": ["indic"], "language": "mr", "script": "Deva", "protected": False})
    manifest["indic_accuracy_targets"] = {"en-US": {"max_wer": .1, "max_cer": .1}}
    for row in predictions:
        row["hypothesis"] = "wrong"
    report = evaluate(manifest, predictions)
    assert report["profiles"][0]["comparisons"]["fine_tuned"]["status"] == "no_measured_regression"
    assert report["regressions"][0]["reason"] == "Indic accuracy target exceeded"


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "missing_hypothesis"])
def test_invalid_prediction_evidence_is_rejected(mutation):
    manifest, predictions = fixture()
    if mutation == "duplicate":
        predictions.append(copy.deepcopy(predictions[0]))
    elif mutation == "unknown":
        predictions[0]["utterance_id"] = "not_in_manifest"
    else:
        del predictions[0]["hypothesis"]
    with pytest.raises(ValueError):
        evaluate(manifest, predictions)


def test_json_loaders_and_duplicate_manifest_reference(tmp_path):
    manifest, predictions = fixture()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text("\n".join(json.dumps(row) for row in predictions))
    assert load_manifest(manifest_path) == manifest
    assert load_predictions(predictions_path) == predictions
    manifest["utterances"].append(copy.deepcopy(manifest["utterances"][0]))
    with pytest.raises(ValueError, match="Duplicate utterance"):
        evaluate(manifest, predictions)


def test_incomplete_pair_never_reports_a_zero_error_comparison():
    manifest, predictions = fixture(references=("one", "two"))
    predictions = [row for row in predictions if not (row["phase"] == "fine_tuned" and row["utterance_id"].endswith("1"))]
    report = evaluate(manifest, predictions)
    assert report["profiles"][0]["phases"]["fine_tuned"]["status"] == "blocked"
    assert report["profiles"][0]["comparisons"]["fine_tuned"] == {"status": "blocked", "reason": "Incomplete paired predictions"}


def complete_release_fixture():
    """Synthetic evidence for gate unit tests, not actual project evaluation."""
    manifest = {"schema_version": 1, "purpose": "release", "normalization": {}, "profiles": [], "utterances": [], "runs": {}, "indic_accuracy_targets": {}}
    for cohort, locales in (("existing_asr", BASE_ASR_LOCALES), ("adaptation", ADAPTATION_LOCALES)):
        for locale in locales:
            manifest["profiles"].append({"id": locale, "locale": locale, "language": locale.split("-")[0], "script": "Latn", "cohorts": [cohort], "protected": cohort == "existing_asr"})
    for language, script in INDIC_SCRIPTS.items():
        profile_id = f"{language}-{script}"
        manifest["profiles"].append({"id": profile_id, "language": language, "script": script, "cohorts": ["indic"], "protected": False})
        manifest["indic_accuracy_targets"][profile_id] = {"max_wer": .1, "max_cer": .1}
    predictions = []
    for phase in PHASES:
        manifest["runs"][phase] = {"run_id": phase, "checkpoint_sha256": "a" * 64, "tokenizer_sha256": "b" * 64, "settings": {"decoder": "greedy"}}
    for profile in manifest["profiles"]:
        utterance_id = profile["id"] + "-test"
        manifest["utterances"].append({"id": utterance_id, "profile_id": profile["id"], "condition_id": "known_language", "reference": "a b", "cluster_id": "synthetic-test-speaker", "dataset": "synthetic-unit-test-only", "split": "test", "audio_sha256": "c" * 64})
        predictions.extend({"utterance_id": utterance_id, "phase": phase, "hypothesis": "a b", "run_id": phase, "language_mode": "known", "language_prompt": profile["id"]} for phase in PHASES)
    return manifest, predictions


def test_complete_synthetic_evidence_exercises_release_pass_and_fail():
    manifest, predictions = complete_release_fixture()
    report = evaluate(manifest, predictions)
    assert report["release_status"] == "passed"
    assert report["blockers"] == []
    for row in predictions:
        if row["utterance_id"] == "en-US-test" and row["phase"] == "fine_tuned":
            row["hypothesis"] = "a"
    report = evaluate(manifest, predictions)
    assert report["release_status"] == "failed"
    assert report["regressions"][0]["profile_id"] == "en-US"


def test_release_cannot_ignore_mismatched_run_or_missing_indic_target():
    manifest, predictions = complete_release_fixture()
    predictions[0]["run_id"] = "different-checkpoint-run"
    del manifest["indic_accuracy_targets"]["mr-Deva"]
    report = evaluate(manifest, predictions)
    assert report["release_status"] == "blocked"
    assert any("Run identity mismatch" in reason for reason in report["blockers"])
    assert any("Missing predeclared Indic accuracy targets for mr-Deva" in reason for reason in report["blockers"])
