"""Append-only native SentencePiece Unigram artifacts and explicit public IDs.

This module packages fitted additions. It does not train scores, convert BPE
merge ranks, or establish acoustic checkpoint compatibility.
"""
from __future__ import annotations

import hashlib
import json
import math
import operator
from pathlib import Path
from typing import Any, Sequence

import sentencepiece as spm
from google.protobuf.message import DecodeError
from sentencepiece import sentencepiece_model_pb2 as pb

from .runtime import IdMap
from .sources import write_json


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(data: bytes) -> pb.ModelProto:
    model = pb.ModelProto()
    try:
        model.ParseFromString(data)
    except DecodeError as error:
        raise ValueError("Expected a binary native SentencePiece model, not tokenizer.json") from error
    if model.trainer_spec.model_type != pb.TrainerSpec.UNIGRAM:
        raise ValueError("Expected a native UNIGRAM SentencePiece model")
    if not model.pieces or len({p.piece for p in model.pieces}) != len(model.pieces):
        raise ValueError("The native inventory must be nonempty and unique")
    if any(p.piece in ("<pad>", "<blank>") for p in model.pieces):
        raise ValueError("Public padding and RNNT blank must not be text pieces")
    if not any(p.type == pb.ModelProto.SentencePiece.NORMAL for p in model.pieces):
        raise ValueError("No native NORMAL scores available")
    if any(not math.isfinite(p.score) for p in model.pieces):
        raise ValueError("Nonfinite native piece score")
    spm.SentencePieceProcessor(model_proto=data)
    return model


def _selection(data: bytes, base_sha256: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selection = json.loads(data)
    if not isinstance(selection, dict) or selection.get("base_tokenizer_sha256") != base_sha256:
        raise ValueError("Selection was not fitted against this native base hash")
    additions = selection.get("additions")
    if not isinstance(additions, list) or not additions:
        raise ValueError("Selection must contain fitted additions")
    for item in additions:
        if not isinstance(item, dict) or "piece" not in item or "score" not in item:
            raise ValueError("Each selected addition must declare a piece and score")
        if not isinstance(item["piece"], str) or not item["piece"]:
            raise ValueError("Selected pieces must be nonempty strings")
        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError(f"Invalid fitted score for {item['piece']!r}")
    return selection, additions


def _vocabulary(model: pb.ModelProto, mapping: IdMap) -> list[dict[str, Any]]:
    return [
        {"native_id": i, "public_id": mapping.model_to_canonical[i], "piece": p.piece,
         "score": p.score, "type": p.type}
        for i, p in enumerate(model.pieces)
    ]


def validate_native_prefix(base_bytes: bytes, expanded_bytes: bytes) -> dict[str, Any]:
    """Check all native metadata, not only piece strings or normalizer name."""
    base, expanded = _load(base_bytes), _load(expanded_bytes)
    if len(expanded.pieces) < len(base.pieces):
        raise ValueError("The expanded model removed native pieces")
    for index, original in enumerate(base.pieces):
        if original.SerializeToString() != expanded.pieces[index].SerializeToString():
            raise ValueError(f"Native piece message changed at ID {index}")
    # Only the piece list and declared vocabulary length may change.
    left, right = pb.ModelProto(), pb.ModelProto()
    left.CopyFrom(base)
    right.CopyFrom(expanded)
    left.ClearField("pieces")
    right.ClearField("pieces")
    left.trainer_spec.ClearField("vocab_size")
    right.trainer_spec.ClearField("vocab_size")
    if left.SerializeToString() != right.SerializeToString():
        raise ValueError("Native metadata or normalization behavior changed")
    if expanded.trainer_spec.vocab_size != len(expanded.pieces):
        raise ValueError("Expanded vocabulary length disagrees with its trainer spec")
    scores = [p.score for p in base.pieces if p.type == pb.ModelProto.SentencePiece.NORMAL]
    low, high = min(scores), max(scores)
    for piece in expanded.pieces[len(base.pieces):]:
        if piece.type != pb.ModelProto.SentencePiece.NORMAL:
            raise ValueError("Appended pieces must be NORMAL")
        if not low <= piece.score <= high:
            raise ValueError("Appended scores changed the native score extrema")
    return {
        "native_entries_preserved": len(base.pieces),
        "native_vocabulary_size": len(expanded.pieces),
        "new_pieces": len(expanded.pieces) - len(base.pieces),
        "normalizer_sha256": _digest(base.normalizer_spec.SerializeToString()),
        "normal_score_range": [low, high],
        "base_tokenizer_sha256": _digest(base_bytes),
        "tokenizer_sha256": _digest(expanded_bytes),
    }


def native_id_map(base_bytes: bytes, expanded_bytes: bytes) -> IdMap:
    report = validate_native_prefix(base_bytes, expanded_bytes)
    old_size = report["native_entries_preserved"]
    new_size = report["native_vocabulary_size"]
    # Public IDs reserve old_size for padding and old_size + 1 for blank.
    # SentencePiece itself remains a dense native text inventory.
    public_blank = old_size + 1
    reverse = tuple(range(old_size)) + tuple(range(old_size + 2, new_size + 2)) + (public_blank,)
    forward: list[int | None] = [None] * (new_size + 2)
    for native_index, public_index in enumerate(reverse):
        forward[public_index] = native_index
    return IdMap(
        tuple(forward), reverse, old_size, public_blank, new_size,
        report["tokenizer_sha256"], report["base_tokenizer_sha256"],
    )


def build_native_tokenizer(base_path: str | Path, selection_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Package an ordered, scored selection with a hash-pinned native base.

    The selection JSON contains ``base_tokenizer_sha256`` and ``additions``.
    Each addition supplies ``piece`` and ``score``; additional provenance fields
    are retained verbatim in the selection artifact.
    """
    base_bytes = Path(base_path).read_bytes()
    selection_bytes = Path(selection_path).read_bytes()
    _, additions = _selection(selection_bytes, _digest(base_bytes))
    base = _load(base_bytes)
    expanded = pb.ModelProto()
    expanded.CopyFrom(base)
    seen = {piece.piece for piece in base.pieces}
    for item in additions:
        piece, score = item["piece"], item["score"]
        if not isinstance(piece, str) or not piece or piece in seen:
            raise ValueError(f"Empty or duplicate appended piece: {piece!r}")
        appended = expanded.pieces.add()
        appended.piece = piece
        appended.score = score
        appended.type = pb.ModelProto.SentencePiece.NORMAL
        seen.add(piece)
    expanded.trainer_spec.vocab_size = len(expanded.pieces)
    expanded_bytes = expanded.SerializeToString()
    report = validate_native_prefix(base_bytes, expanded_bytes)
    mapping = native_id_map(base_bytes, expanded_bytes)
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Use an empty output directory to preserve existing candidates")
    (destination / "base-tokenizer.model").write_bytes(base_bytes)
    (destination / "tokenizer.model").write_bytes(expanded_bytes)
    (destination / "selection.json").write_bytes(selection_bytes)
    write_json(destination / "nemo-id-map.json", mapping.to_dict())
    write_json(destination / "vocabulary.json", _vocabulary(expanded, mapping))
    files = {p.name: _digest(p.read_bytes()) for p in sorted(destination.iterdir())}
    manifest = {
        "schema_version": 1, "algorithm": "native_sentencepiece_unigram",
        "status": "tokenizer_candidate", "structural_passed": True,
        "checkpoint_validated": False, "asr_validated": False,
        **report,
        "public_vocabulary_size": len(mapping.canonical_to_model),
        "public_pad_id": mapping.hf_pad_id, "public_blank_id": mapping.hf_blank_id,
        "first_new_public_id": mapping.hf_blank_id + 1,
        "native_blank_id": mapping.model_blank_id,
        "acoustic_output_size": mapping.model_output_size,
        "selection_sha256": _digest(selection_bytes), "files": files,
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


class NativeTokenizerAdapter:
    """Native IDs for NeMo-shaped callers; public IDs are explicit conversions."""

    def __init__(self, directory: str | Path):
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("algorithm") != "native_sentencepiece_unigram":
            raise ValueError("This adapter requires a native Unigram bundle")
        required = {"tokenizer.model", "base-tokenizer.model", "selection.json", "nemo-id-map.json", "vocabulary.json"}
        files = manifest.get("files", {})
        if not isinstance(files, dict) or set(files) != required:
            raise ValueError("Incomplete native bundle integrity manifest")
        for name, digest in files.items():
            if _digest((directory / name).read_bytes()) != digest:
                raise ValueError(f"Native bundle file hash mismatch: {name}")
        base = (directory / "base-tokenizer.model").read_bytes()
        model = (directory / "tokenizer.model").read_bytes()
        self.id_map = native_id_map(base, model)
        report = validate_native_prefix(base, model)
        selection_bytes = (directory / "selection.json").read_bytes()
        _, additions = _selection(selection_bytes, report["base_tokenizer_sha256"])
        expanded = _load(model)
        native_size = report["native_entries_preserved"]
        if len(additions) != len(expanded.pieces) - native_size:
            raise ValueError("Stored selection length disagrees with tokenizer artifacts")
        for item, actual in zip(additions, expanded.pieces[native_size:]):
            declared = pb.ModelProto.SentencePiece(piece=item["piece"], score=item["score"],
                                                 type=pb.ModelProto.SentencePiece.NORMAL)
            if declared.SerializeToString() != actual.SerializeToString():
                raise ValueError("Stored selection piece/score order disagrees with tokenizer artifacts")
        if json.loads((directory / "vocabulary.json").read_text()) != _vocabulary(expanded, self.id_map):
            raise ValueError("Stored vocabulary disagrees with tokenizer artifacts")
        expected_manifest = {
            "schema_version": 1, "structural_passed": True, **report,
            "public_vocabulary_size": len(self.id_map.canonical_to_model),
            "public_pad_id": self.id_map.hf_pad_id,
            "public_blank_id": self.id_map.hf_blank_id,
            "first_new_public_id": self.id_map.hf_blank_id + 1,
            "native_blank_id": self.id_map.model_blank_id,
            "acoustic_output_size": self.id_map.model_output_size,
            "selection_sha256": _digest(selection_bytes),
        }
        for key, expected in expected_manifest.items():
            if manifest.get(key) != expected:
                raise ValueError(f"Native bundle manifest disagrees with tokenizer artifacts: {key}")
        if json.loads((directory / "nemo-id-map.json").read_text()) != json.loads(json.dumps(self.id_map.to_dict())):
            raise ValueError("Stored native ID map disagrees with tokenizer artifacts")
        self.backend = spm.SentencePieceProcessor(model_proto=model)
        self.tokenizer = self
        self.vocab_size = self.backend.get_piece_size()
        self.blank_id = self.id_map.model_blank_id
        self.pad_id = self.blank_id
        self.unk_id = self.backend.unk_id()
        self.bos_id = self.backend.bos_id()
        self.eos_id = self.backend.eos_id()
        self.vocab = self.get_vocab()

    def get_vocab(self) -> dict[str, int]:
        return {self.backend.id_to_piece(i): i for i in range(self.vocab_size)}

    def text_to_ids(self, text: str) -> list[int]:
        return self.backend.encode(text, out_type=int)

    def __call__(self, text: str) -> list[int]:
        return self.text_to_ids(text)

    @staticmethod
    def _integer_ids(ids: Sequence[int]) -> list[int]:
        try:
            result = []
            for index in ids:
                if isinstance(index, bool):
                    raise TypeError("Boolean token ID")
                result.append(operator.index(index))
            return result
        except TypeError as error:
            raise ValueError("Token IDs must be integers") from error

    def _text_ids(self, ids: Sequence[int]) -> list[int]:
        ids = self._integer_ids(ids)
        if any(i < 0 or i > self.blank_id for i in ids):
            raise ValueError("Native token ID out of range")
        return [i for i in ids if i != self.blank_id]

    def ids_to_text(self, ids: Sequence[int]) -> str:
        return self.backend.decode(self._text_ids(ids))

    def text_to_tokens(self, text: str) -> list[str]:
        return self.backend.encode(text, out_type=str)

    def ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.backend.id_to_piece(i) for i in self._text_ids(ids)]

    def token_to_id(self, token: str) -> int:
        if token not in self.vocab:
            raise ValueError(f"Unknown token string: {token!r}")
        return self.vocab[token]

    def id_to_token(self, index: int) -> str:
        index = self._integer_ids([index])[0]
        if index == self.blank_id:
            return "<blank>"
        return self.ids_to_tokens([index])[0]

    def tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        return [self.token_to_id(t) for t in tokens]

    def tokens_to_text(self, tokens: Sequence[str]) -> str:
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def text_to_public_ids(self, text: str) -> list[int]:
        return self.id_map.to_canonical(self.text_to_ids(text))

    def public_ids_to_text(self, ids: Sequence[int]) -> str:
        return self.ids_to_text(self.id_map.to_model(self._integer_ids(ids), allow_blank=True))
