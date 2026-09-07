"""Canonical HF BPE IDs and NeMo's dense, blank-last RNNT ID space.

The canonical tokenizer is never rewritten for NeMo. ``<pad>`` has no acoustic
output row, while ``<blank>`` maps to the final RNNT row. A model consuming the
adapter therefore sees different IDs for additions; persist the mapping with it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _artifact(path: str | Path) -> tuple[dict[str, Any], str]:
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def _vocabulary(data: Mapping[str, Any]) -> dict[str, int]:
    if data.get("model", {}).get("type") != "BPE":
        raise ValueError("Only the approved HF BPE tokenizer is supported")
    vocab = dict(data["model"]["vocab"])
    for token in data.get("added_tokens", []):
        text, index = token["content"], token["id"]
        if text in vocab and vocab[text] != index:
            raise ValueError(f"Conflicting ID for {text!r}")
        vocab[text] = index
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("Multiple token strings share an ID")
    if set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("Canonical IDs must form a dense zero-based inventory")
    for name in ("<pad>", "<blank>", "<unk>"):
        if name not in vocab:
            raise ValueError(f"Missing required reserved token {name}")
    return vocab


@dataclass(frozen=True)
class IdMap:
    """A complete, reversible mapping except for non-acoustic HF padding."""

    canonical_to_model: tuple[int | None, ...]
    model_to_canonical: tuple[int, ...]
    hf_pad_id: int
    hf_blank_id: int
    model_blank_id: int
    tokenizer_sha256: str
    base_tokenizer_sha256: str | None = None

    @property
    def model_vocab_size(self) -> int:
        """Non-blank vocabulary size, as expected by NeMo RNNT modules."""
        return self.model_blank_id

    @property
    def model_output_size(self) -> int:
        return len(self.model_to_canonical)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "layout": "nemo_dense_blank_last", **asdict(self)}

    def to_model(self, ids: Sequence[int], *, allow_blank: bool = False) -> list[int]:
        result = []
        for index in ids:
            index = int(index)
            if not 0 <= index < len(self.canonical_to_model):
                raise ValueError(f"Canonical token ID out of range: {index}")
            mapped = self.canonical_to_model[index]
            if mapped is None:
                raise ValueError("HF padding is not an acoustic token or training label")
            if mapped == self.model_blank_id and not allow_blank:
                raise ValueError("RNNT blank is not a transcript label")
            result.append(mapped)
        return result

    def to_canonical(self, ids: Sequence[int], *, drop_blank: bool = True) -> list[int]:
        result = []
        for index in ids:
            index = int(index)
            if not 0 <= index < len(self.model_to_canonical):
                raise ValueError(f"Model token ID out of range: {index}")
            if drop_blank and index == self.model_blank_id:
                continue
            result.append(self.model_to_canonical[index])
        return result


def build_id_map(
    tokenizer_json: str | Path, base_tokenizer_json: str | Path | None = None
) -> IdMap:
    data, digest = _artifact(tokenizer_json)
    vocab = _vocabulary(data)
    from tokenizers import Tokenizer

    loaded_vocab = Tokenizer.from_file(str(tokenizer_json)).get_vocab(with_added_tokens=True)
    if loaded_vocab != vocab:
        raise ValueError("HF loader changed declared token IDs; materialize reserved IDs before appending")
    base_digest = None
    if base_tokenizer_json is not None:
        base, base_digest = _artifact(base_tokenizer_json)
        base_vocab = _vocabulary(base)
        for piece, old_id in base_vocab.items():
            if vocab.get(piece) != old_id:
                raise ValueError(f"Original HF ID changed: {piece!r} was {old_id}")
        old_added = {item["content"]: item for item in base.get("added_tokens", [])}
        new_added = {item["content"]: item for item in data.get("added_tokens", [])}
        for piece, descriptor in old_added.items():
            if new_added.get(piece) != descriptor:
                raise ValueError(f"Original added/special token behavior changed: {piece!r}")
    pad, blank = vocab["<pad>"], vocab["<blank>"]
    ordinary = tuple(i for i in range(len(vocab)) if i not in (pad, blank))
    reverse = ordinary + (blank,)
    forward: list[int | None] = [None] * len(vocab)
    for model_id, hf_id in enumerate(reverse):
        forward[hf_id] = model_id
    return IdMap(tuple(forward), reverse, pad, blank, len(ordinary), digest, base_digest)


class HFTokenizerAdapter:
    """TokenizerSpec-shaped adapter, with native RNNT IDs at every public edge.

    This intentionally does not inherit a NeMo class, allowing CPU tokenizer
    checks without importing its training stack. Its callable interface supports
    NeMo's TokenizerWrapper fallback for non-TokenizerSpec objects. Language tags
    remain available when decoding, and consecutive repeated RNNT tokens are retained.
    """

    def __init__(self, tokenizer_json: str | Path, native_decoder_model: str | Path | None = None):
        from tokenizers import Tokenizer

        self.path = str(Path(tokenizer_json).resolve())
        self.id_map = build_id_map(self.path)
        self.backend = Tokenizer.from_file(self.path)
        # NeMo BPE model construction expects tokenizer.tokenizer.get_vocab().
        self.tokenizer = self
        self.vocab = self.get_vocab()
        self.vocab_size = self.id_map.model_vocab_size
        self.blank_id = self.id_map.model_blank_id
        self.pad_id = self.blank_id  # RNNT's padded labels are excluded by length.
        self.unk_id = self.id_map.to_model([self.backend.token_to_id("<unk>")])[0]
        self.bos_id = self.eos_id = -1
        self.native_decoder_model_path = None
        self.native_decoder_sha256 = None
        self.native_decoder_model_proto = None
        self._native_decoder = None
        if native_decoder_model is not None:
            self._setup_native_decoder(native_decoder_model)

    def _setup_native_decoder(self, path: str | Path) -> None:
        """Keep native piece-to-text semantics while HF alone encodes text.

        Native SentencePiece renders unknowns and boundary spaces differently
        from HF. DecodePieces also emits an unrecognized piece literally,
        including its metaspace marker. Copy the original proto and append
        missing ordinary pieces as NORMAL solely for decoding. Added scores
        are unused: this processor never encodes or segments text. The source
        artifact, HF BPE JSON, original piece types and normalizer are unchanged.
        """
        import sentencepiece as spm
        from sentencepiece import sentencepiece_model_pb2

        path = Path(path).resolve()
        source = path.read_bytes()
        proto = sentencepiece_model_pb2.ModelProto()
        proto.ParseFromString(source)
        # The selected base puts the HF-only padding/blank entries immediately
        # after every original native piece. A shorter or unrelated decoder
        # cannot establish native decoding compatibility for that old prefix.
        expected = min(self.id_map.hf_pad_id, self.id_map.hf_blank_id)
        if len(proto.pieces) != expected:
            raise ValueError("Native decoder does not cover the complete original HF piece prefix")
        for index, piece in enumerate(proto.pieces):
            if self.backend.id_to_token(index) != piece.piece:
                raise ValueError(f"Native decoder piece {index} disagrees with its original HF ID")
        for model_id, canonical_id in enumerate(self.id_map.model_to_canonical[:-1]):
            if model_id < expected:
                continue
            piece = proto.pieces.add()
            piece.piece = self.backend.id_to_token(canonical_id)
            piece.type = sentencepiece_model_pb2.ModelProto.SentencePiece.NORMAL
            piece.score = 0.0  # Never used: this processor only decodes pieces.
        proto.trainer_spec.vocab_size = len(proto.pieces)
        self._native_decoder = spm.SentencePieceProcessor(model_proto=proto.SerializeToString())
        self.native_decoder_model_path = str(path)
        self.native_decoder_sha256 = hashlib.sha256(source).hexdigest()
        # Keep the original immutable artifact bytes for a later re-migration;
        # the in-memory extended decoder is never an encoding artifact.
        self.native_decoder_model_proto = source

    def get_vocab(self) -> dict[str, int]:
        return {
            self.backend.id_to_token(hf_id): model_id
            for model_id, hf_id in enumerate(self.id_map.model_to_canonical[:-1])
        }

    def text_to_ids(self, text: str) -> list[int]:
        return self.id_map.to_model(self.backend.encode(text, add_special_tokens=False).ids)

    def __call__(self, text: str) -> list[int]:
        """Support NeMo's callable tokenizer interface with native model IDs."""
        return self.text_to_ids(text)

    def ids_to_text(self, ids: Sequence[int]) -> str:
        if self._native_decoder is not None:
            return self._native_decoder.decode_pieces(self.ids_to_tokens(ids))
        return self.backend.decode(self.id_map.to_canonical(ids), skip_special_tokens=False)

    def text_to_tokens(self, text: str) -> list[str]:
        return self.ids_to_tokens(self.text_to_ids(text))

    def ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return [self.backend.id_to_token(i) for i in self.id_map.to_canonical(ids)]

    def tokens_to_ids(self, tokens: Sequence[str]) -> list[int]:
        ids = []
        for token in tokens:
            index = self.backend.token_to_id(token)
            if index is None:
                raise ValueError(f"Token not present in vocabulary: {token!r}")
            ids.append(index)
        return self.id_map.to_model(ids)

    def tokens_to_text(self, tokens: Sequence[str]) -> str:
        return self.ids_to_text(self.tokens_to_ids(tokens))

    def token_to_id(self, token: str) -> int:
        return self.tokens_to_ids([token])[0]

    def id_to_token(self, index: int) -> str:
        return self.backend.id_to_token(self.id_map.to_canonical([index], drop_blank=False)[0])


_NEMO_CLASS = None


def _register_trusted_nemo_target(model_class):
    """Permit this exact class without broadening NeMo's allowed module prefixes.

    The pinned NeMo validator has no custom-class registration API and checks
    prefixes before its exact-target set. Preserve its predicate for all other
    targets, and verify the resolved identity of our single installed class.
    """
    from importlib import import_module

    common = import_module("nemo.core.classes.common")
    original = getattr(common, "_is_target_allowed", None)
    serialization = getattr(common, "Serialization", None)
    if not callable(original) or not isinstance(serialization, type):
        raise RuntimeError("Unsupported NeMo target-validation interface")
    if not isinstance(model_class, type) or not issubclass(model_class, serialization):
        raise ValueError("The registered tokenizer model must be a NeMo Serialization subclass")
    if getattr(original, "_untok_registered_class", None) is model_class:
        return
    target_path = "untok.runtime.ExtendedNemotronRNNTModel"

    def allow_registered_model(target):
        if target == target_path:
            # An alias, replaced module attribute or similarly named class is
            # insufficient: Hydra must resolve the exact class registered here.
            return common.hydra.utils.get_class(target) is model_class
        return original(target)

    allow_registered_model._untok_registered_class = model_class
    common._is_target_allowed = allow_registered_model


def get_nemo_model_class():
    """Lazily provide the registered model class required to restore our .nemo.

    Actual NeMo integration must pass on the pinned Linux/CUDA environment;
    importing this module alone does not establish that compatibility.
    """
    global _NEMO_CLASS
    if _NEMO_CLASS is None:
        from nemo.collections.asr.models.rnnt_bpe_models_prompt import EncDecRNNTBPEModelWithPrompt

        class ExtendedNemotronRNNTModel(EncDecRNNTBPEModelWithPrompt):
            def _setup_tokenizer(self, tokenizer_cfg):
                if tokenizer_cfg.get("type") != "untok_hf_bpe":
                    return super()._setup_tokenizer(tokenizer_cfg)
                path = self.register_artifact(
                    "tokenizer.hf_tokenizer_json", tokenizer_cfg["hf_tokenizer_json"]
                )
                native_decoder = None
                if tokenizer_cfg.get("native_decoder_model") is not None:
                    native_decoder = self.register_artifact(
                        "tokenizer.native_decoder_model", tokenizer_cfg["native_decoder_model"]
                    )
                self.tokenizer_cfg = tokenizer_cfg
                self.tokenizer_dir = str(Path(path).parent)
                self.tokenizer_type = "bpe"
                self.tokenizer = HFTokenizerAdapter(path, native_decoder_model=native_decoder)

        ExtendedNemotronRNNTModel.__module__ = __name__
        ExtendedNemotronRNNTModel.__qualname__ = "ExtendedNemotronRNNTModel"
        _NEMO_CLASS = ExtendedNemotronRNNTModel
        globals()["ExtendedNemotronRNNTModel"] = _NEMO_CLASS
    _register_trusted_nemo_target(_NEMO_CLASS)
    return _NEMO_CLASS


def __getattr__(name: str):
    if name == "ExtendedNemotronRNNTModel":
        return get_nemo_model_class()
    raise AttributeError(name)
