"""CPU controls for the read-only native audio-probe instrumentation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


@pytest.fixture
def probes(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    loaded = []
    for name in ("native_checkpoint_probe", "native_subset_probe"):
        spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def test_evaluation_reads_test_and_valid_audio_without_training(probes, tmp_path):
    full, _ = probes
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"hashed fixture; audio decode is outside this test")
    manifest = tmp_path / "audio.jsonl"
    entries = [{"id": split, "source_split": split, "audio": audio.name,
                "audio_sha256": full.sha(audio)} for split in ("test", "valid", "train")]
    manifest.write_text("\n".join(map(json.dumps, entries)))
    assert [row["id"] for row in full.rows(manifest, ("test", "valid"))] == ["test", "valid"]
    assert [row["id"] for row in full.rows(manifest, "train")] == ["train"]
    audio.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Audio content"):
        full.rows(manifest, ("test", "valid"))


def test_source_and_locale_proxy_evidence_is_preserved(probes):
    full, _ = probes
    evidence = {"source_dataset": "fixture", "source_revision": "pinned",
                "source_split": "valid", "locale_audio_match": False,
                "locale_coverage_limitation": "regional locale uses another region's audio"}
    assert full.row_provenance({**evidence, "text": "excluded"}) == evidence


def stream_pair():
    source = {"chunk_count": 2, "initial_cache": [0], "streaming_cfg": {"chunk": 56},
              "effective_decoding_config": {"strategy": "greedy_batch"},
              "runtime_decoder": {"use_cuda_graph_decoder": False}, "buffer_exhausted": True,
              "steps": []}
    for index in range(2):
        source["steps"].append({"step": index, "chunk": {"sha256": str(index)}, "chunk_lengths": [12],
                                "drop_extra_pre_encoded": index, "last_chunk": index == 1,
                                "partial_hypotheses_supplied": index > 0, "cache_input": [index],
                                "cache_output": [index + 1], "cache_lengths": [index + 1],
                                "hypothesis": {"text": "ab", "model_ids": [0, 2]}})
    target = copy.deepcopy(source)
    for step in target["steps"]:
        step["hypothesis"]["model_ids"] = [0, 1]
    return source, target


@pytest.mark.parametrize("field", ["chunk_count", "initial_cache", "streaming_cfg",
                                    "effective_decoding_config", "runtime_decoder", "buffer_exhausted"])
def test_streaming_header_mismatch_fails(probes, field):
    full, _ = probes
    source, target = stream_pair()
    assert full.streaming_parity(source, target, (0, None, 1, 3))
    target[field] = "different"
    assert not full.streaming_parity(source, target, (0, None, 1, 3))


@pytest.mark.parametrize("field", ["step", "chunk", "chunk_lengths", "drop_extra_pre_encoded", "last_chunk",
                                    "partial_hypotheses_supplied", "cache_input", "cache_output", "cache_lengths"])
def test_streaming_cache_and_chunk_mismatch_fails(probes, field):
    full, _ = probes
    source, target = stream_pair()
    target["steps"][1][field] = "different"
    assert not full.streaming_parity(source, target, (0, None, 1, 3))


def test_streaming_rejects_removed_ids_and_empty_execution(probes):
    full, _ = probes
    source, target = stream_pair()
    source["steps"][0]["hypothesis"]["model_ids"] = [1]
    assert not full.streaming_parity(source, target, (0, None, 1, 3))
    source["steps"] = target["steps"] = []
    assert not full.streaming_parity(source, target, (0, None, 1, 3))


def test_subset_scope_selection_and_missing_audio_failures(probes):
    _, subset = probes
    records = [{"id": f"{lang}-{seconds}", "language": lang, "duration": seconds, "target_lang": lang + "-XX"}
               for lang in ("en", "fr", "hi", "ta", "ar", "ru") for seconds in (12, 8)]
    assert [row["id"] for row in subset.choose_rows(records, "latin")] == ["en-8", "fr-8"]
    assert [row["id"] for row in subset.choose_rows(records, "latin-indic")] == ["ar-8", "en-8", "fr-8", "hi-8", "ta-8"]
    with pytest.raises(ValueError, match="outside"):
        subset.choose_rows(records, "latin", ["hi"])
    with pytest.raises(ValueError, match="Missing"):
        subset.choose_rows(records, "latin-indic", ["kn"])
    with pytest.raises(ValueError, match="positive"):
        subset.choose_rows(records, "latin", max_per_language=0)


def test_subset_preserves_all_retained_source_regional_prompt_paths(probes):
    from untok.evaluation import ADAPTATION_LOCALES, BASE_ASR_LOCALES
    from untok.prompts import TARGET_LOCALES

    _, subset = probes
    assert len(subset.LATIN_SOURCE_LANGUAGES) == 25
    locales = sorted(set(BASE_ASR_LOCALES) | set(ADAPTATION_LOCALES) | set(TARGET_LOCALES.values()))
    records = [{"id": locale + "-" + str(seconds), "target_lang": locale,
                # Nynorsk is a Bokmal-audio proxy, with a separate prompt.
                "language": "nb" if locale == "nn-NO" else locale.split("-")[0], "duration": seconds}
               for locale in locales for seconds in (9, 6)]
    latin = subset.choose_rows(records, "latin")
    indic = subset.choose_rows(records, "latin-indic")
    assert len(latin) == 29 and len(indic) == 52
    assert {row["target_lang"] for row in latin if row["language"] == "en"} == {"en-US", "en-GB"}
    assert {row["target_lang"] for row in latin if row["language"] == "nb"} == {"nb-NO", "nn-NO"}
    assert all(row["duration"] == 6 for row in latin + indic)
    assert "ar-AR" in {row["target_lang"] for row in indic}
    assert {row["target_lang"] for row in subset.choose_rows(records, "latin", ["en"])} == {"en-US", "en-GB"}


def test_subset_control_uses_same_original_prompt_or_explicit_auto(probes):
    _, subset = probes
    old = {"auto": 0, "en": 1, "hi": 2}
    new = {**old, "ta": 40}
    known = subset.control_prompt({"target_lang": "hi"}, old, new)
    assert known["control_target_lang"] == "hi" and known["source_has_requested_prompt"]
    added = subset.control_prompt({"target_lang": "ta"}, old, new)
    assert added["control_target_lang"] == "auto" and not added["source_has_requested_prompt"]
    assert added["requested_prompt_id"] == 40
    with pytest.raises(ValueError, match="identical"):
        subset.control_prompt({"target_lang": "ta"}, old, {**new, "auto": 9})
    with pytest.raises(ValueError, match="missing"):
        subset.control_prompt({"target_lang": "kn"}, old, new)


def test_subset_hypothesis_mapping_rejects_pruned_source_rows(probes):
    _, subset = probes
    old, new = {"text": "ab", "native_ids": [0, 2]}, {"text": "ab", "native_ids": [0, 1]}
    assert subset.mapped_hypothesis_equal(old, new, (0, None, 1, 3))
    old["native_ids"] = [1]
    assert not subset.mapped_hypothesis_equal(old, new, (0, None, 1, 3))


def test_streaming_eval_boundary_restores_nested_modes_without_changing_tensors(probes):
    torch = pytest.importorskip("torch")
    _, subset = probes
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(0.8), torch.nn.BatchNorm1d(3)).eval()
    # Match NeMo teardown: parent mode is false, child unfreeze sets train true.
    model[1].train()
    model[2].train()
    assert not model.training and model[1].training
    tensors = {name: value.clone() for name, value in model.state_dict().items()}
    evidence = subset.prepare_streaming_evaluation(model)
    assert evidence["training_module_count_before"] == 2
    assert evidence["training_module_count_after"] == 0
    assert not any(module.training for module in model.modules())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in tensors.items())
    with torch.inference_mode():
        values = torch.ones(4, 3)
        assert torch.equal(model(values), model(values))


def test_subset_actual_head_capture_and_retained_state_replay(probes):
    torch = pytest.importorskip("torch")
    from untok.checkpoint import RNNTLayout
    from untok.checkpoint_validation import _head_trace

    _, subset = probes
    source, target = torch.nn.Linear(3, 5), torch.nn.Linear(3, 4)
    old = RNNTLayout("embed", "weight", "bias", 4, 5)
    new = RNNTLayout("embed", "weight", "bias", 3, 4)
    mapping = (0, None, 1, None, 3)
    with torch.inference_mode():
        target.weight[[0, 1, 3]] = source.weight[[0, 2, 4]]
        target.bias[[0, 1, 3]] = source.bias[[0, 2, 4]]
        states = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
        with _head_trace(source, old_to_new=[0, 2, 4]) as left:
            masked_left = source(states)
        with _head_trace(target, old_to_new=[0, 1, 3]) as right:
            masked_right = target(states)
        assert torch.isneginf(masked_left[:, [1, 3]]).all()
        assert torch.isneginf(masked_right[:, 2]).all()
        assert subset.retained_trace_checks(left, right, target, old, new, mapping)["passed"]
        changed = copy.deepcopy(right)
        changed["calls"] += 1
        assert not subset.retained_trace_checks(left, changed, target, old, new, mapping)["passed"]
        changed = copy.deepcopy(right)
        changed["probes"][0][0][0, 0] += 1
        assert not subset.retained_trace_checks(left, changed, target, old, new, mapping)["passed"]
        with pytest.raises(ValueError, match="must both execute"):
            subset.retained_trace_checks({"calls": 0, "probes": []}, right, target, old, new, mapping)
        target.bias[0] += 1
        with pytest.raises(ValueError, match="logits changed"):
            subset.retained_trace_checks(left, right, target, old, new, mapping)
