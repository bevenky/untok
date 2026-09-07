"""Deterministic language-prompt registry extension without renumbering slots.

Prompt identities condition the acoustic model. They do not add tokenizer
language-tag tokens, hard output masks, or learned recognition capability.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


# Explicit project decisions, not prefix matching against upstream aliases.
# In particular, upstream's or-KE must never be mistaken for Odia or-IN.
TARGET_LOCALES = {
    "as": "as-IN", "bn": "bn-IN", "brx": "brx-IN", "doi": "doi-IN",
    "gu": "gu-IN", "hi": "hi-IN", "kn": "kn-IN", "kok": "kok-IN",
    "ks": "ks-IN", "mai": "mai-IN", "ml": "ml-IN", "mni": "mni-IN",
    "mr": "mr-IN", "ne": "ne-NP", "or": "or-IN", "pa": "pa-IN",
    "sa": "sa-IN", "sat": "sat-IN", "sd": "sd-IN", "ta": "ta-IN",
    "te": "te-IN", "ur": "ur-PK",
}


def _read(value: str | Path | Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    if isinstance(value, Mapping):
        data = dict(value)
        raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    else:
        raw = Path(value).read_bytes()
        data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Prompt source must be a JSON object")
    return data, hashlib.sha256(raw).hexdigest()


def _validate_dictionary(dictionary: Any, size: int) -> dict[str, int]:
    if not isinstance(dictionary, Mapping) or not dictionary:
        raise ValueError("A nonempty prompt_dictionary is required")
    for identity, index in dictionary.items():
        if not isinstance(identity, str) or not identity or type(index) is not int or not 0 <= index < size:
            raise ValueError(f"Invalid prompt entry {identity!r}: {index!r}")
    return dict(dictionary)


def extend_prompt_registry(
    processor_config: str | Path | Mapping[str, Any], targets: Sequence[Mapping[str, Any]],
    *, aliases: Mapping[str, str] | None = None,
    previous_registry: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a full dictionary plus an auditable identity/allocation manifest.

    Each target names ``language`` and may repeat its explicit ``prompt_locale``.
    Unknown languages require an explicit locale. Known project languages may
    not silently switch identities. Optional aliases name their canonical locale
    explicitly; no bare-code or same-script aliases are inferred. Future builds
    must pass the prior registry to retain already allocated slots.
    """
    source, source_hash = _read(processor_config)
    size = source.get("num_prompts")
    if type(size) is not int or size <= 0:
        raise ValueError("Source num_prompts must be a positive integer")
    original = _validate_dictionary(source.get("prompt_dictionary"), size)
    dictionary = dict(original)
    identity_assignments: dict[str, int] = {}
    explicit_aliases: dict[str, str] = {}
    previous_hash = None
    if previous_registry is not None:
        previous, previous_hash = _read(previous_registry)
        if previous.get("schema_version") != 1 or previous.get("num_prompts") != size:
            raise ValueError("Previous registry schema or prompt dimension changed")
        if previous.get("source_processor_sha256") != source_hash:
            raise ValueError("Previous registry used a different pinned processor artifact")
        dictionary = _validate_dictionary(previous.get("prompt_dictionary"), size)
        if any(dictionary.get(key) != index for key, index in original.items()):
            raise ValueError("Previous registry changed an upstream prompt assignment")
        identity_assignments = dict(previous.get("identity_assignments", {}))
        explicit_aliases = dict(previous.get("explicit_aliases", {}))
        if any(dictionary.get(locale) != index for locale, index in identity_assignments.items()):
            raise ValueError("Previous identity assignments disagree with its dictionary")
        if any(locale not in original and index in set(original.values())
               for locale, index in identity_assignments.items()):
            raise ValueError("A previous new identity reused an upstream prompt slot")
        if len(set(identity_assignments.values())) != len(identity_assignments):
            raise ValueError("Distinct language identities share a slot in the previous registry")
        for alias, locale in explicit_aliases.items():
            if locale not in identity_assignments or dictionary.get(alias) != dictionary.get(locale):
                raise ValueError("Previous alias does not name its explicit language identity")
        declared = set(original) | set(identity_assignments) | set(explicit_aliases)
        if set(dictionary) != declared:
            raise ValueError("Previous registry contains unexplained prompt identities")

    selections: dict[str, Mapping[str, Any]] = {}
    locales: dict[str, str] = {}
    for target in targets:
        language = target.get("language")
        if not isinstance(language, str) or not language or language in selections:
            raise ValueError(f"Missing or duplicate target language: {language!r}")
        expected = TARGET_LOCALES.get(language)
        locale = target.get("prompt_locale", expected)
        if not isinstance(locale, str) or not locale or locale == "auto":
            raise ValueError(f"An explicit prompt_locale is required for {language!r}")
        if expected is not None and locale != expected:
            raise ValueError(f"{language!r} must retain explicit identity {expected!r}, not {locale!r}")
        if expected is None and not locale.startswith(language + "-"):
            raise ValueError(f"Explicit locale {locale!r} does not identify {language!r}")
        if locale in locales.values():
            raise ValueError(f"Two target identities claim the same locale: {locale}")
        selections[language], locales[language] = target, locale
    if not selections:
        raise ValueError("At least one target language is required")

    used_slots = set(dictionary.values())
    free_slots = [i for i in range(size) if i not in used_slots]
    missing = sorted(set(locales.values()) - set(dictionary))
    if len(missing) > len(free_slots):
        raise ValueError(f"Insufficient unused prompt slots: need {len(missing)}, available {len(free_slots)}")
    allocated = {}
    for locale, index in zip(missing, free_slots):
        dictionary[locale] = index
        allocated[locale] = index
    for locale in sorted(locales.values()):
        if locale in explicit_aliases:
            raise ValueError(f"A new target identity cannot reuse a previous alias: {locale}")
        identity_assignments[locale] = dictionary[locale]
    if len(set(identity_assignments.values())) != len(identity_assignments):
        raise ValueError("Distinct target language identities share a prompt slot")

    for alias, locale in sorted((aliases or {}).items()):
        if not isinstance(alias, str) or not alias or not isinstance(locale, str) or locale not in identity_assignments:
            raise ValueError(f"Alias {alias!r} must explicitly name a target canonical locale")
        if alias in identity_assignments:
            raise ValueError(f"Canonical identity {alias!r} cannot be declared as an alias")
        if alias in original and original[alias] != dictionary[locale]:
            raise ValueError(f"Alias would overwrite upstream identity {alias!r}")
        if alias in dictionary and dictionary[alias] != dictionary[locale]:
            raise ValueError(f"Alias would overwrite an existing identity {alias!r}")
        if alias in explicit_aliases and explicit_aliases[alias] != locale:
            raise ValueError(f"Alias {alias!r} was assigned to another identity")
        dictionary[alias] = dictionary[locale]
        explicit_aliases[alias] = locale

    assignments = []
    for language in sorted(selections):
        target, locale = selections[language], locales[language]
        assignments.append({"language": language, "name": target.get("name", language),
                            "script": target.get("script"), "locale": locale,
                            "prompt_id": dictionary[locale],
                            "status": "existing" if locale in original else "new",
                            "pretrained_asr_capability_asserted": False})
    return {"schema_version": 1, "num_prompts": size,
            "source_processor_sha256": source_hash, "previous_registry_sha256": previous_hash,
            "prompt_dictionary": dict(sorted(dictionary.items())),
            "identity_assignments": dict(sorted(identity_assignments.items())),
            "explicit_aliases": dict(sorted(explicit_aliases.items())),
            "target_assignments": assignments, "allocated_this_build": allocated,
            "upstream_entries_preserved": len(original),
            "upstream_slots_preserved": len(set(original.values())),
            "existing_target_count": sum(a["status"] == "existing" for a in assignments),
            "new_target_count": sum(a["status"] == "new" for a in assignments),
            "unused_prompt_slots": [i for i in range(size) if i not in set(dictionary.values())],
            "output_language_tags_added": False,
            "validation_boundary": "Prompt configuration only; speech training, output tags and ASR evaluation remain separate."}


def build_prompt_registry(
    processor_config: str | Path, build_config: str | Path, output: str | Path,
    *, previous_registry: str | Path | None = None,
) -> dict[str, Any]:
    """Build a manifest; migration consumes manifest['prompt_dictionary']."""
    config, config_hash = _read(build_config)
    if not isinstance(config.get("targets"), list):
        raise ValueError("Build config must contain a targets list")
    report = extend_prompt_registry(processor_config, config["targets"],
                                    aliases=config.get("prompt_aliases"), previous_registry=previous_registry)
    report["build_config_sha256"] = config_hash
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return report
