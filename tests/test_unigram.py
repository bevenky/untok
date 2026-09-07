"""Native encoding and artifact invariants, independent of a CUDA runtime."""
import hashlib
import itertools
import json

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.unigram import NativeTokenizerAdapter, build_native_tokenizer, validate_native_prefix


@pytest.fixture
def inputs(tmp_path):
    proto = pb.ModelProto()
    proto.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    proto.trainer_spec.unk_id = 0
    proto.trainer_spec.bos_id = proto.trainer_spec.eos_id = proto.trainer_spec.pad_id = -1
    proto.normalizer_spec.name = "identity"
    proto.normalizer_spec.remove_extra_whitespaces = False
    for index, (text, score) in enumerate([
        ("<unk>", 0), ("▁", 0), ("a", -2), ("b", -3), ("ab", -4), ("<tag>", 0), ("z", -253),
    ]):
        piece = proto.pieces.add()
        piece.piece, piece.score = text, score
        piece.type = {0: pb.ModelProto.SentencePiece.UNKNOWN, 5: pb.ModelProto.SentencePiece.USER_DEFINED}.get(
            index, pb.ModelProto.SentencePiece.NORMAL)
    proto.trainer_spec.vocab_size = len(proto.pieces)
    base = tmp_path / "base.model"
    base.write_bytes(proto.SerializeToString())
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "base_tokenizer_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "additions": [{"piece": "க", "score": -2}, {"piece": "▁க", "score": -1}],
    }))
    return base, selection, tmp_path / "bundle"


def test_native_encoder_public_mapping_and_unknown_fallback(inputs):
    base, selection, output = inputs
    manifest = build_native_tokenizer(base, selection, output)
    adapter = NativeTokenizerAdapter(output)
    source = spm.SentencePieceProcessor(model_file=str(base))
    # Old IDs7/8 are public padding/blank. Native additions start at7,
    # public additions at9. No text class is inserted into the old prefix.
    assert manifest["first_new_public_id"] == 9
    assert manifest["public_pad_id"] == 7
    assert manifest["public_blank_id"] == 8
    assert manifest["native_blank_id"] == 9
    assert adapter.text_to_ids("க") == [8]
    assert adapter.text_to_public_ids("க") == [10]
    assert adapter.public_ids_to_text([10, 8]) == "க"
    with pytest.raises(ValueError, match="padding"):
        adapter.public_ids_to_text([7])
    for text in ("a", "ab", "  ab", "a  b", "🙂a", "<tag>", "ab\na"):
        assert adapter.text_to_ids(text) == source.encode(text)
        assert adapter.ids_to_text(adapter.text_to_ids(text)) == source.decode(source.encode(text))
    for length in range(1, 4):
        for ids in itertools.product(range(source.get_piece_size()), repeat=length):
            assert adapter.ids_to_text(ids) == source.decode(list(ids))
    assert adapter.ids_to_text([2, 2, adapter.blank_id, 3]) == "aab"
    assert not manifest["asr_validated"]
    assert not manifest["checkpoint_validated"]


@pytest.mark.parametrize("mutation", ["score", "type", "normalizer", "trainer", "denormalizer"])
def test_rejects_native_behavior_changes(inputs, mutation):
    base, selection, output = inputs
    build_native_tokenizer(base, selection, output)
    model = pb.ModelProto()
    model.ParseFromString((output / "tokenizer.model").read_bytes())
    if mutation == "score":
        model.pieces[2].score -= 1
    elif mutation == "type":
        model.pieces[2].type = pb.ModelProto.SentencePiece.USER_DEFINED
    elif mutation == "normalizer":
        model.normalizer_spec.remove_extra_whitespaces = True
    elif mutation == "trainer":
        model.trainer_spec.unk_surface = "?"
    else:
        model.denormalizer_spec.name = "identity"
    with pytest.raises(ValueError, match="changed"):
        validate_native_prefix(base.read_bytes(), model.SerializeToString())


@pytest.mark.parametrize("entry", [
    {"piece": "a", "score": -2}, {"piece": "", "score": -2},
    {"piece": "क", "score": -254}, {"piece": "क", "score": 1},
    {"piece": "क", "score": float("nan")}, {"piece": "<pad>", "score": -2},
    {"piece": "<blank>", "score": -2},
])
def test_rejects_invalid_additions_before_writing(inputs, entry):
    base, selection, output = inputs
    data = json.loads(selection.read_text())
    data["additions"] = [entry]
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        build_native_tokenizer(base, selection, output)
    assert not output.exists()


def test_pin_integrity_wrong_algorithm_and_no_overwrite(inputs):
    base, selection, output = inputs
    original = base.read_bytes()
    model = pb.ModelProto()
    model.ParseFromString(original)
    model.pieces[2].score -= 1
    base.write_bytes(model.SerializeToString())
    with pytest.raises(ValueError, match="base hash"):
        build_native_tokenizer(base, selection, output)
    base.write_bytes(original)
    build_native_tokenizer(base, selection, output)
    with pytest.raises(ValueError, match="empty output"):
        build_native_tokenizer(base, selection, output)
    (output / "tokenizer.model").write_bytes(original)
    with pytest.raises(ValueError, match="hash mismatch"):
        NativeTokenizerAdapter(output)


@pytest.mark.parametrize("artifact", ["vocabulary", "selection_score", "selection_order", "manifest_ids"])
def test_rejects_self_hashed_but_inconsistent_bundle_metadata(inputs, artifact):
    base, selection, output = inputs
    build_native_tokenizer(base, selection, output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if artifact == "manifest_ids":
        manifest["public_blank_id"] += 1
    else:
        name = "vocabulary.json" if artifact == "vocabulary" else "selection.json"
        path = output / name
        data = json.loads(path.read_text())
        if artifact == "vocabulary":
            data[-1]["public_id"] += 1
        elif artifact == "selection_score":
            data["additions"][0]["score"] -= 1
        else:
            data["additions"].reverse()
        path.write_text(json.dumps(data))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][name] = digest
        if name == "selection.json":
            manifest["selection_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="disagrees"):
        NativeTokenizerAdapter(output)


@pytest.mark.parametrize("bad_id", [2.5, "2", True])
def test_rejects_noninteger_ids_without_truncation(inputs, bad_id):
    base, selection, output = inputs
    build_native_tokenizer(base, selection, output)
    adapter = NativeTokenizerAdapter(output)
    for call in (adapter.ids_to_text, adapter.ids_to_tokens, adapter.public_ids_to_text):
        with pytest.raises(ValueError, match="integers"):
            call([bad_id])
    with pytest.raises(ValueError, match="integers"):
        adapter.id_to_token(bad_id)
    assert adapter.ids_to_text(iter([2, 3])) == adapter.ids_to_text([2, 3])


@pytest.mark.parametrize("algorithm", ["BPE", "json"])
def test_wrong_base_algorithm_is_reported_before_writing(inputs, algorithm):
    base, selection, output = inputs
    if algorithm == "BPE":
        model = pb.ModelProto()
        model.ParseFromString(base.read_bytes())
        model.trainer_spec.model_type = pb.TrainerSpec.BPE
        base.write_bytes(model.SerializeToString())
    else:
        base.write_text('{"model":{"type":"BPE"}}')
    data = json.loads(selection.read_text())
    data["base_tokenizer_sha256"] = hashlib.sha256(base.read_bytes()).hexdigest()
    selection.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="UNIGRAM|binary native"):
        build_native_tokenizer(base, selection, output)
    assert not output.exists()
