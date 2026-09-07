"""Native decoding compatibility, without changing canonical HF BPE encoding."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import random

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb
from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from sttok.runtime import HFTokenizerAdapter


@pytest.fixture
def native_artifacts(tmp_path):
    pieces = ["<unk>", "▁", "a", "b", "<ctrl>", "<tag>"]
    proto = pb.ModelProto()
    proto.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    proto.trainer_spec.unk_id = 0
    proto.trainer_spec.bos_id = proto.trainer_spec.eos_id = proto.trainer_spec.pad_id = -1
    proto.trainer_spec.unk_surface = " ⁇ "
    proto.normalizer_spec.name = "identity"
    for index, text in enumerate(pieces):
        item = proto.pieces.add()
        item.piece, item.score = text, -float(index)
        item.type = {0: pb.ModelProto.SentencePiece.UNKNOWN,
                     4: pb.ModelProto.SentencePiece.CONTROL,
                     5: pb.ModelProto.SentencePiece.USER_DEFINED}.get(index, pb.ModelProto.SentencePiece.NORMAL)
    proto.trainer_spec.vocab_size = len(pieces)
    native = tmp_path / "native.model"
    native.write_bytes(proto.SerializeToString())
    backend = Tokenizer(models.BPE(vocab={piece: index for index, piece in enumerate(pieces)},
                                   merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Metaspace()
    backend.decoder = decoders.Metaspace()
    backend.add_special_tokens(["<unk>", "<pad>", "<blank>"])
    data = json.loads(backend.to_str())
    data["model"]["vocab"].update({"<pad>": 6, "<blank>": 7, "க": 8, "ള": 9, "▁க": 10})
    data["model"]["merges"] = [["▁", "க"]]
    extended = tmp_path / "extended.json"
    extended.write_text(json.dumps(data))
    return native, extended


def test_every_short_old_sequence_preserves_native_spacing_types_and_repetition(native_artifacts):
    native, extended = native_artifacts
    source = spm.SentencePieceProcessor(model_file=str(native))
    adapter = HFTokenizerAdapter(extended, native_decoder_model=native)
    # Unknowns, controls, user-defined symbols, boundaries and repeats all take
    # part in these real SentencePiece backend comparisons.
    for length in range(1, 5):
        for ids in itertools.product(range(len(source)), repeat=length):
            assert adapter.ids_to_text(ids) == source.decode_ids(list(ids))
    assert adapter.ids_to_text([0, 0]) == " ⁇  ⁇ "
    assert adapter.ids_to_text([2, 2, adapter.blank_id, 3]) == "aab"


def test_new_boundary_pieces_are_known_to_decoder_and_do_not_leak_metaspace(native_artifacts):
    native, extended = native_artifacts
    source = spm.SentencePieceProcessor(model_file=str(native))
    # Direct reuse of DecodePieces would be wrong for an absent new subword.
    assert source.decode_pieces(["▁க"]) == "▁க"
    adapter = HFTokenizerAdapter(extended, native_decoder_model=native)
    assert adapter.tokens_to_text(["▁க"]) == "க"
    assert adapter.tokens_to_text(["▁க", "ള", "▁க"]) == "கള க"
    assert adapter.tokens_to_text(["▁க", "<unk>", "▁க"]) == "க ⁇  க"


def test_hf_encoding_and_canonical_artifact_do_not_change(native_artifacts):
    native, extended = native_artifacts
    original_bytes = extended.read_bytes()
    source_bytes = native.read_bytes()
    hf_only = HFTokenizerAdapter(extended)
    native_decode = HFTokenizerAdapter(extended, native_decoder_model=native)
    for text in ("aab", "கள ab", "a <unk> b", "<tag> க", "<ctrl>"):
        assert native_decode.text_to_ids(text) == hf_only.text_to_ids(text)
        assert native_decode.backend.encode(text).ids == hf_only.backend.encode(text).ids
    assert native_decode.backend.to_str() == hf_only.backend.to_str()
    assert extended.read_bytes() == original_bytes
    assert native.read_bytes() == source_bytes
    assert native_decode.native_decoder_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert native_decode.native_decoder_model_path == str(native.resolve())
    assert native_decode.native_decoder_model_proto == source_bytes


def test_decoder_survives_removed_nemo_extraction_path(native_artifacts):
    native, extended = native_artifacts
    adapter = HFTokenizerAdapter(extended, native_decoder_model=native)
    expected_hash = adapter.native_decoder_sha256
    expected_bytes = native.read_bytes()
    native.unlink()
    assert adapter.ids_to_text([0, 1, 2]) == " ⁇  a"
    assert adapter.tokens_to_text(["▁க"]) == "க"
    assert adapter.native_decoder_sha256 == expected_hash
    assert adapter.native_decoder_model_proto == expected_bytes


def test_unrelated_or_incomplete_native_decoder_is_rejected(native_artifacts):
    native, extended = native_artifacts
    proto = pb.ModelProto()
    proto.ParseFromString(native.read_bytes())
    proto.pieces[2].piece = "c"
    native.write_bytes(proto.SerializeToString())
    with pytest.raises(ValueError, match="disagrees"):
        HFTokenizerAdapter(extended, native_decoder_model=native)
    del proto.pieces[-1]
    native.write_bytes(proto.SerializeToString())
    with pytest.raises(ValueError, match="complete original"):
        HFTokenizerAdapter(extended, native_decoder_model=native)


def test_pinned_real_native_decoder_and_all_candidate_witnesses():
    native_path = os.environ.get("STTOK_NATIVE_DECODER_MODEL")
    repo = Path(__file__).resolve().parents[1]
    extended = repo / "artifacts/nemotron-indic-v1/tokenizer.json"
    manifest_path = extended.parent / "manifest.json"
    if not native_path or not extended.exists() or not manifest_path.exists():
        pytest.skip("Supply STTOK_NATIVE_DECODER_MODEL and built candidate for the real artifact check")
    native = Path(native_path)
    assert hashlib.sha256(native.read_bytes()).hexdigest() == "ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291"
    source = spm.SentencePieceProcessor(model_file=str(native))
    adapter = HFTokenizerAdapter(extended, native_decoder_model=native)
    for index in range(len(source)):
        assert adapter.ids_to_text([index]) == source.decode_ids([index])
    rng = random.Random(0)
    for _ in range(10000):
        ids = [rng.randrange(len(source)) for _ in range(rng.randrange(1, 30))]
        assert adapter.ids_to_text(ids) == source.decode_ids(ids)
    for entry in json.loads(manifest_path.read_text())["entries"]:
        encoded = adapter.backend.encode(entry["witness"], add_special_tokens=False)
        assert adapter.ids_to_text(adapter.id_map.to_model(encoded.ids)) == adapter.backend.decode(encoded.ids, skip_special_tokens=False)
