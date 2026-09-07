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
    checks without importing its training stack. NeMo uses this protocol by
    duck typing. Language tags remain available when decoding, and consecutive
    repeated RNNT tokens are retained.
    """

    def __init__(self, tokenizer_json: str | Path):
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

    def get_vocab(self) -> dict[str, int]:
        return {
            self.backend.id_to_token(hf_id): model_id
            for model_id, hf_id in enumerate(self.id_map.model_to_canonical[:-1])
        }

    def text_to_ids(self, text: str) -> list[int]:
        return self.id_map.to_model(self.backend.encode(text, add_special_tokens=False).ids)

    def ids_to_text(self, ids: Sequence[int]) -> str:
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
                if tokenizer_cfg.get("type") != "sttok_hf_bpe":
                    return super()._setup_tokenizer(tokenizer_cfg)
                path = self.register_artifact(
                    "tokenizer.hf_tokenizer_json", tokenizer_cfg["hf_tokenizer_json"]
                )
                self.tokenizer_cfg = tokenizer_cfg
                self.tokenizer_dir = str(Path(path).parent)
                self.tokenizer_type = "bpe"
                self.tokenizer = HFTokenizerAdapter(path)

        ExtendedNemotronRNNTModel.__module__ = __name__
        ExtendedNemotronRNNTModel.__qualname__ = "ExtendedNemotronRNNTModel"
        _NEMO_CLASS = ExtendedNemotronRNNTModel
        globals()["ExtendedNemotronRNNTModel"] = _NEMO_CLASS
    return _NEMO_CLASS


def __getattr__(name: str):
    if name == "ExtendedNemotronRNNTModel":
        return get_nemo_model_class()
    raise AttributeError(name)
