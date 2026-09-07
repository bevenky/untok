"""Reproducible script subsets with explicit checkpoint row remapping."""
from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass
from importlib import resources
import json
from pathlib import Path
import tempfile
from typing import Any, Sequence
import zipfile

import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from ._bundle_script_ranges import INDIC_SCRIPTS, RANGES, SOURCES, TABLE_SHA256, UNICODE_VERSION
from .runtime import IdMap
from .unigram import NativeTokenizerAdapter, _digest, _load, _vocabulary, validate_native_prefix


def _load_tokenizer_directory(directory: str | Path):
    """Load a verified full native bundle or a separately validated subset."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("algorithm") == "native_sentencepiece_unigram_subset":
        return NativeSubsetTokenizerAdapter(directory)
    adapter = NativeTokenizerAdapter(directory)
    adapter.base_model_bytes = (directory / "base-tokenizer.model").read_bytes()
    adapter.model_bytes = (directory / "tokenizer.model").read_bytes()
    base_size = spm.SentencePieceProcessor(model_proto=adapter.base_model_bytes).get_piece_size()
    adapter.source_native_to_target_native = tuple(range(base_size)) + (adapter.blank_id,)
    adapter.full_native_to_subset_native = tuple(range(adapter.blank_id + 1))
    adapter.subset_native_to_full_native = tuple(range(adapter.blank_id + 1))
    return adapter


PROFILES = ("latin", "latin-indic", "full")


def load_tokenizer_bundle(directory: str | Path):
    """Load a bundled profile name or an explicit local bundle directory.

    The strings ``latin``, ``latin-indic`` and ``full`` select installed package
    data. Use ``Path("full")`` or ``"./full"`` for a same-named local directory.
    Every path uses the same strict artifact and metadata checks.
    """
    if not isinstance(directory, str) or directory not in PROFILES:
        return _load_tokenizer_directory(directory)
    resource = resources.files("untok").joinpath("data", directory)
    if not resource.is_dir():
        raise ValueError(f"Packaged tokenizer {directory!r} is missing; reinstall untok with its bundle data")
    # Ordinary wheel and editable installations expose real paths. Avoid an
    # unnecessary copy of these immutable files on each load.
    if isinstance(resource, Path):
        return _load_tokenizer_directory(resource)
    # Python 3.11's resources.as_file() cannot extract resource directories.
    # Materialize just this profile, supporting zip-imported packages as well.
    manifest = json.loads(resource.joinpath("manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if (not isinstance(files, dict) or not files
            or any(not isinstance(name, str) or Path(name).name != name
                   or name in {".", ".."} or "\\" in name for name in files)):
        raise ValueError("Invalid packaged tokenizer artifact filenames")
    with tempfile.TemporaryDirectory(prefix="untok-bundled-tokenizer-") as temporary:
        local = Path(temporary)
        for name in sorted({"manifest.json", *files}):
            (local / name).write_bytes(resource.joinpath(name).read_bytes())
        # Adapters retain model bytes and parsed mappings, not these file paths.
        return _load_tokenizer_directory(local)


load_tokenizer = load_tokenizer_bundle
_STARTS = {name: tuple(lo for lo, _ in ranges) for name, ranges in RANGES.items()}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def character_allowed(character: str, profile: str) -> bool:
    """Use pinned Unicode data, independent of the host Python Unicode version.

    Shared Common punctuation is retained. Other Common/Inherited characters
    with explicit Script_Extensions need an allowed script in that set.
    Unassigned characters are excluded. This is a vocabulary scope, not a
    language detector, transliterator, normalizer or inference language lock.
    """
    if profile not in PROFILES:
        raise ValueError(f"Unknown tokenizer profile: {profile}")
    if len(character) != 1:
        raise ValueError("Expected one Unicode character")
    if profile == "full":
        return True
    codepoint = ord(character)
    index = bisect.bisect_right(_STARTS[profile], codepoint) - 1
    return index >= 0 and codepoint <= RANGES[profile][index][1]


def script_policy(profile: str) -> dict[str, Any]:
    if profile not in RANGES:
        raise ValueError(f"Not a reduced tokenizer profile: {profile}")
    return {
        "schema_version": 1, "profile": profile,
        "policy": "unicode-script-and-extensions-common-punctuation-v1",
        "unicode_version": UNICODE_VERSION,
        "unicode_sources": {name: {"sha256": digest,
            "url": f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd/{name}"}
            for name, digest in SOURCES.items()},
        "script_table_sha256": TABLE_SHA256,
        "allowed_scripts": ["Latin"] + (list(INDIC_SCRIPTS) if profile == "latin-indic" else []),
        "shared_characters": "Common punctuation; Common/Inherited with no restricted Script_Extensions or an allowed extension",
        "mixed_script_pieces": "Keep a NORMAL piece only when every character is allowed",
        "non_normal_pieces": "Retain all special piece messages without filtering their spellings",
        "normalizer": "Exact copy of the source; no text cleaning added",
        "scope": "Finite source vocabulary subset; no assertion of ASR accuracy or universal script coverage",
    }


@dataclass(frozen=True)
class NativeRowMap:
    """Checkpoint output/prediction rows, including each layout's final blank."""

    source_native_to_target_native: tuple[int | None, ...]
    full_native_to_subset_native: tuple[int | None, ...]
    subset_native_to_full_native: tuple[int, ...]
    source_blank_id: int
    full_blank_id: int
    target_blank_id: int
    base_tokenizer_sha256: str
    full_tokenizer_sha256: str
    tokenizer_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "layout": "dense_native_subset_blank_last", **asdict(self)}

    def from_source(self, ids: Sequence[int], *, allow_blank: bool = False) -> list[int]:
        return self._remap(ids, self.source_native_to_target_native, allow_blank)

    def from_full(self, ids: Sequence[int], *, allow_blank: bool = False) -> list[int]:
        return self._remap(ids, self.full_native_to_subset_native, allow_blank)

    def _remap(self, ids, mapping, allow_blank):
        result = []
        for index in NativeTokenizerAdapter._integer_ids(ids):
            if not 0 <= index < len(mapping):
                raise ValueError("Source native token ID out of range")
            target = mapping[index]
            if target is None:
                raise ValueError(f"Source token ID {index} was removed from this subset")
            if target == self.target_blank_id and not allow_blank:
                raise ValueError("RNNT blank is not a transcript label")
            result.append(target)
        return result


def _subset_artifacts(base_bytes: bytes, full_bytes: bytes, profile: str):
    # The full source always retains the original append-only contract.
    source_report = validate_native_prefix(base_bytes, full_bytes)
    full = _load(full_bytes)
    policy = script_policy(profile)
    selected = [i for i, piece in enumerate(full.pieces)
                if piece.type != pb.ModelProto.SentencePiece.NORMAL
                or all(character_allowed(c, profile) for c in piece.piece)]
    kept_normal_scores = [full.pieces[i].score for i in selected
                          if full.pieces[i].type == pb.ModelProto.SentencePiece.NORMAL]
    if not kept_normal_scores or [min(kept_normal_scores), max(kept_normal_scores)] != source_report["normal_score_range"]:
        # SentencePiece derives its unknown penalty from the NORMAL minimum,
        # and USER_DEFINED lattice scores use the maximum. Do not alter scores
        # or sneak disallowed pieces into a scope to compensate for pruning.
        raise ValueError("Subset would change native NORMAL score extrema and unknown behavior")
    size, full_size = len(selected), len(full.pieces)
    old_size = source_report["native_entries_preserved"]
    full_forward: list[int | None] = [None] * (full_size + 1)
    for target, original in enumerate(selected):
        full_forward[original] = target
    full_forward[full_size] = size
    target = pb.ModelProto()
    target.CopyFrom(full)
    target.ClearField("pieces")
    for i in selected:
        target.pieces.add().CopyFrom(full.pieces[i])
    target.trainer_spec.vocab_size = size
    for field in ("unk_id", "bos_id", "eos_id", "pad_id"):
        original = getattr(full.trainer_spec, field)
        if original >= 0:
            mapped = full_forward[original]
            if mapped is None:
                raise ValueError(f"Subset removed required trainer special ID: {field}")
            if mapped != original:
                setattr(target.trainer_spec, field, mapped)
    model_bytes = target.SerializeToString()
    _load(model_bytes)
    mapping = NativeRowMap(
        tuple(full_forward[:old_size]) + (size,), tuple(full_forward), tuple(selected) + (full_size,),
        old_size, full_size, size, _digest(base_bytes), _digest(full_bytes), _digest(model_bytes),
    )
    # Subset public IDs are dense text, then public padding and public blank.
    # They do not use the full bundle's old-size public-ID gap.
    public_map = IdMap(tuple(range(size)) + (None, size), tuple(range(size)) + (size + 1,),
                       size, size + 1, size, _digest(model_bytes), _digest(base_bytes))
    files = {
        "tokenizer.model": model_bytes,
        "base-tokenizer.model": base_bytes,
        "full-tokenizer.model": full_bytes,
        "native-row-map.json": _json_bytes(mapping.to_dict()),
        "nemo-id-map.json": _json_bytes(public_map.to_dict()),
        "vocabulary.json": _json_bytes(_vocabulary(target, public_map)),
        "script-policy.json": _json_bytes(policy),
    }
    original_retained = sum(i < old_size for i in selected)
    manifest = {
        "schema_version": 1, "algorithm": "native_sentencepiece_unigram_subset",
        "profile": profile, "status": "tokenizer_subset_candidate",
        "structural_passed": True, "checkpoint_validated": False, "asr_validated": False,
        "checkpoint_compatibility": "requires_explicit_row_remapping_and_acoustic_validation",
        "native_prefix_preserved": False,
        "native_vocabulary_size": size, "native_blank_id": size, "acoustic_output_size": size + 1,
        "public_vocabulary_size": size + 2, "public_pad_id": size, "public_blank_id": size + 1,
        "original_native_vocabulary_size": old_size, "full_native_vocabulary_size": full_size,
        "original_native_pieces_retained": original_retained,
        "appended_pieces_retained": size - original_retained,
        "full_pieces_removed": full_size - size,
        "special_pieces_retained": sum(full.pieces[i].type != pb.ModelProto.SentencePiece.NORMAL for i in selected),
        "normal_score_range": source_report["normal_score_range"],
        "normalizer_sha256": source_report["normalizer_sha256"],
        "base_tokenizer_sha256": _digest(base_bytes), "full_tokenizer_sha256": _digest(full_bytes),
        "tokenizer_sha256": _digest(model_bytes),
        "files": {name: _digest(data) for name, data in files.items()},
    }
    return files, manifest, mapping, public_map


class NativeSubsetTokenizerAdapter(NativeTokenizerAdapter):
    """A reduced native inventory, validated against its source models.

    This separate loader never relaxes NativeTokenizerAdapter's prefix check.
    Checkpoint rows must be gathered/remapped before using reduced token IDs.
    """

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("algorithm") != "native_sentencepiece_unigram_subset":
            raise ValueError("This adapter requires a reduced native bundle")
        required = {"tokenizer.model", "base-tokenizer.model", "full-tokenizer.model",
                    "native-row-map.json", "nemo-id-map.json", "vocabulary.json", "script-policy.json"}
        hashes = manifest.get("files")
        if not isinstance(hashes, dict) or set(hashes) != required:
            raise ValueError("Incomplete reduced bundle integrity manifest")
        actual = {name: (directory / name).read_bytes() for name in required}
        for name, data in actual.items():
            if _digest(data) != hashes[name]:
                raise ValueError(f"Reduced bundle file hash mismatch: {name}")
        expected, expected_manifest, row_map, public_map = _subset_artifacts(
            actual["base-tokenizer.model"], actual["full-tokenizer.model"], manifest.get("profile"))
        if actual != expected or manifest != expected_manifest:
            raise ValueError("Reduced bundle disagrees with its source models, script policy or ID maps")
        self.base_model_bytes = actual["base-tokenizer.model"]
        self.full_model_bytes = actual["full-tokenizer.model"]
        self.model_bytes = actual["tokenizer.model"]
        self.row_map = row_map
        self.source_native_to_target_native = row_map.source_native_to_target_native
        self.full_native_to_subset_native = row_map.full_native_to_subset_native
        self.subset_native_to_full_native = row_map.subset_native_to_full_native
        self.id_map = public_map
        self.profile = manifest["profile"]
        self.backend = spm.SentencePieceProcessor(model_proto=self.model_bytes)
        self.tokenizer = self
        self.vocab_size = self.backend.get_piece_size()
        self.blank_id = self.id_map.model_blank_id
        self.pad_id = self.blank_id
        self.unk_id, self.bos_id, self.eos_id = self.backend.unk_id(), self.backend.bos_id(), self.backend.eos_id()
        self.vocab = self.get_vocab()


def deterministic_bundle_zip(directory: str | Path, archive: str | Path) -> str:
    """Write identical bytes across hosts: fixed metadata, sorted uncompressed ZIP.

    ZIP_STORED intentionally avoids compressor-version-dependent output. The
    directory's name is not included, so output paths do not affect the hash.
    """
    directory, archive = Path(directory), Path(archive)
    if archive.exists():
        raise ValueError(f"Refusing to overwrite bundle archive: {archive}")
    paths = sorted(directory.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Bundle archives cannot include symlinks")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as out:
        for path in paths:
            if path.is_file():
                info = zipfile.ZipInfo(path.relative_to(directory).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                out.writestr(info, path.read_bytes())
    return _digest(archive.read_bytes())


def package_tokenizer_bundles(bundle: str | Path, output: str | Path,
                              profiles: Sequence[str] = PROFILES, *, make_zips: bool = True) -> dict[str, Any]:
    """Package full, Latin, and Latin+Indic bundles without fitting any scores.

    Existing destinations must be empty. Every candidate and ID map is checked
    before publishing the output directory. Original files are never modified.
    """
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    profiles = tuple(profiles)
    if not profiles or len(profiles) != len(set(profiles)) or any(p not in PROFILES for p in profiles):
        raise ValueError("Choose unique profiles from latin, latin-indic and full")
    if output == bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
        raise ValueError("Output and source bundle directories must not overlap")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Expected an empty output directory")
    NativeTokenizerAdapter(bundle)
    source_manifest_bytes = (bundle / "manifest.json").read_bytes()
    source_manifest = json.loads(source_manifest_bytes)
    names = sorted(set(source_manifest["files"]) | {"manifest.json"})
    if any((bundle / name).is_symlink() for name in names):
        raise ValueError("Source bundle files must not be symlinks")
    source_files = {name: (bundle / name).read_bytes() for name in names}
    base_bytes, full_bytes = source_files["base-tokenizer.model"], source_files["tokenizer.model"]
    prepared = {}
    for profile in profiles:
        if profile == "full":
            prepared[profile] = (source_files, source_manifest)
        else:
            files, manifest, _, _ = _subset_artifacts(base_bytes, full_bytes, profile)
            prepared[profile] = ({**files, "manifest.json": _json_bytes(manifest)}, manifest)
    receipt = {
        "schema_version": 1, "source_manifest_sha256": _digest(source_manifest_bytes),
        "full_tokenizer_sha256": _digest(full_bytes), "base_tokenizer_sha256": _digest(base_bytes),
        "zip_format": "stored, sorted filenames, fixed Unix permissions and 1980 timestamp" if make_zips else None,
        "bundles": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".untok-bundles-", dir=output.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        for profile in sorted(profiles):
            files, manifest = prepared[profile]
            directory = stage / profile
            directory.mkdir()
            for name, data in files.items():
                (directory / name).write_bytes(data)
            adapter = load_tokenizer_bundle(directory)
            item = {"directory": profile, "native_vocabulary_size": adapter.vocab_size,
                    "native_blank_id": adapter.blank_id, "acoustic_output_size": adapter.blank_id + 1,
                    "public_vocabulary_size": len(adapter.id_map.canonical_to_model),
                    "tokenizer_sha256": _digest(adapter.model_bytes),
                    "files": {name: _digest(data) for name, data in sorted(files.items())}}
            if make_zips:
                name = f"{profile}.zip"
                item["zip"] = name
                item["zip_sha256"] = deterministic_bundle_zip(directory, stage / name)
            receipt["bundles"][profile] = item
        (stage / "bundles.json").write_bytes(_json_bytes(receipt))
        if output.exists():
            output.rmdir()
        stage.rename(output)
    # Read-back checks use no source writes; callers can bind this receipt hash.
    return receipt
