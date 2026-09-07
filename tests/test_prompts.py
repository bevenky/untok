import copy
import json
from pathlib import Path

import pytest

from untok.prompts import TARGET_LOCALES, build_prompt_registry, extend_prompt_registry


@pytest.fixture
def processor():
    # The important upstream identity/alias relationships, without requiring
    # network or a populated artifact cache for the independent unit tests.
    return {"num_prompts": 128, "prompt_dictionary": {
        "en-US": 0, "en": 0, "en-GB": 1, "hi-IN": 6, "hi": 6,
        "bn-IN": 36, "ur-PK": 37, "ta-IN": 39, "te-IN": 40,
        "mr-IN": 41, "gu-IN": 42, "kn-IN": 43, "ml-IN": 44,
        "ne-NP": 46, "or-KE": 59, "auto": 101,
    }}


@pytest.fixture
def targets():
    return [{"language": language, "prompt_locale": locale} for language, locale in TARGET_LOCALES.items()]


def test_all_22_identities_preserve_existing_ten_without_inferred_aliases(processor, targets):
    report = extend_prompt_registry(processor, targets)
    assert report["existing_target_count"] == 10
    assert report["new_target_count"] == 12
    assert len(report["allocated_this_build"]) == 12
    assert all(report["prompt_dictionary"][name] == index for name, index in processor["prompt_dictionary"].items())
    assert report["prompt_dictionary"]["hi-IN"] == 6
    assert report["prompt_dictionary"]["mr-IN"] == 41
    assert report["prompt_dictionary"]["or-KE"] == 59
    assert report["prompt_dictionary"]["or-IN"] != 59
    assert "mr" not in report["prompt_dictionary"]
    assert len(set(report["identity_assignments"].values())) == 22
    assert report["output_language_tags_added"] is False
    assert report == extend_prompt_registry(processor, list(reversed(targets)))


def test_explicit_aliases_only_and_no_identity_slot_reuse(processor, targets):
    report = extend_prompt_registry(processor, targets, aliases={"marathi": "mr-IN", "odia": "or-IN"})
    assert report["prompt_dictionary"]["marathi"] == 41
    assert report["prompt_dictionary"]["odia"] == report["prompt_dictionary"]["or-IN"]
    assert report["explicit_aliases"] == {"marathi": "mr-IN", "odia": "or-IN"}
    with pytest.raises(ValueError, match="upstream identity"):
        extend_prompt_registry(processor, targets, aliases={"or-KE": "or-IN"})
    with pytest.raises(ValueError, match="Canonical identity"):
        extend_prompt_registry(processor, targets, aliases={"mr-IN": "hi-IN"})
    with pytest.raises(ValueError, match="canonical locale"):
        extend_prompt_registry(processor, targets, aliases={"odia": "or-KE"})


def test_previous_registry_preserves_slots_when_a_sorted_earlier_language_is_added(processor, targets):
    initial = extend_prompt_registry(processor, targets)
    added = targets + [{"language": "aa", "prompt_locale": "aa-IN"}]
    later = extend_prompt_registry(processor, added, previous_registry=initial)
    assert all(later["prompt_dictionary"][key] == index for key, index in initial["prompt_dictionary"].items())
    assert set(later["allocated_this_build"]) == {"aa-IN"}
    assert later["prompt_dictionary"]["aa-IN"] in initial["unused_prompt_slots"]
    # Removing a target from the selected list does not erase its released ID.
    subset = extend_prompt_registry(processor, targets[:5], previous_registry=later)
    assert subset["identity_assignments"] == later["identity_assignments"]
    assert subset["allocated_this_build"] == {}


def test_rejects_changed_prior_assignments_or_source(processor, targets):
    report = extend_prompt_registry(processor, targets)
    corrupt = copy.deepcopy(report)
    corrupt["prompt_dictionary"]["hi-IN"] = 7
    with pytest.raises(ValueError, match="changed an upstream"):
        extend_prompt_registry(processor, targets, previous_registry=corrupt)
    corrupt = copy.deepcopy(report)
    corrupt["prompt_dictionary"]["or-IN"] = 59
    corrupt["identity_assignments"]["or-IN"] = 59
    with pytest.raises(ValueError, match="reused an upstream"):
        extend_prompt_registry(processor, targets, previous_registry=corrupt)
    other = copy.deepcopy(processor)
    other["prompt_dictionary"]["auto"] = 102
    with pytest.raises(ValueError, match="different pinned processor"):
        extend_prompt_registry(other, targets, previous_registry=report)


def test_exhaustion_and_ambiguous_locale_fail(processor, targets):
    full = {"num_prompts": 2, "prompt_dictionary": {"en-US": 0, "auto": 1}}
    with pytest.raises(ValueError, match="Insufficient unused"):
        extend_prompt_registry(full, [{"language": "as"}])
    with pytest.raises(ValueError, match="must retain explicit identity"):
        extend_prompt_registry(processor, [{"language": "or", "prompt_locale": "or-KE"}])
    with pytest.raises(ValueError, match="explicit prompt_locale"):
        extend_prompt_registry(processor, [{"language": "xyz"}])
    with pytest.raises(ValueError, match="duplicate target"):
        extend_prompt_registry(processor, targets + [targets[0]])


def test_file_builder_writes_migration_ready_dictionary(processor, targets, tmp_path):
    source, config, output = (tmp_path / p for p in ("processor.json", "build.json", "prompts.json"))
    source.write_text(json.dumps(processor))
    config.write_text(json.dumps({"targets": targets}))
    report = build_prompt_registry(source, config, output)
    assert json.loads(output.read_text()) == report
    assert report["source_processor_sha256"] and report["build_config_sha256"]
    assert all(type(index) is int for index in report["prompt_dictionary"].values())
    with pytest.raises(FileExistsError):
        build_prompt_registry(source, config, output)


def test_pinned_nvidia_all_40_locales_unchanged_when_cache_present(targets):
    root = Path(__file__).resolve().parents[1]
    source = root / ".cache/sources/nvidia/processor_config.json"
    if not source.exists():
        pytest.skip("Pinned processor cache is not populated")
    original = json.loads(source.read_text())
    report = extend_prompt_registry(source, targets)
    locales = """en-US en-GB es-US es-ES fr-FR fr-CA it-IT pt-BR pt-PT nl-NL de-DE tr-TR ru-RU ar-AR hi-IN ja-JP ko-KR vi-VN uk-UA
    pl-PL sv-SE cs-CZ nb-NO da-DK bg-BG fi-FI hr-HR sk-SK zh-CN hu-HU ro-RO et-EE
    el-GR lt-LT lv-LV mt-MT sl-SI he-IL th-TH nn-NO""".split()
    assert len(locales) == 40
    for locale in locales:
        assert report["prompt_dictionary"][locale] == original["prompt_dictionary"][locale]
    assert all(report["prompt_dictionary"][key] == value for key, value in original["prompt_dictionary"].items())
    assert report["upstream_slots_preserved"] == 84
    assert report["num_prompts"] == 128
    assert report["existing_target_count"] == 10 and report["new_target_count"] == 12
    assert set(report["allocated_this_build"].values()).isdisjoint(original["prompt_dictionary"].values())
    assert len(report["unused_prompt_slots"]) == 32
