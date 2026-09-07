"""Native full/subset row transfer, retained logits and fail-closed migration inputs."""
import copy
import hashlib
from types import SimpleNamespace

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from untok.checkpoint import inspect_nemo_layout
from untok.native_checkpoint import (
    _prompt_registry, compare_retained_logits, initialize_native_added_rows,
    retained_row_pairs, transfer_native_state_dict, validate_source_native_tokenizer,
    verify_native_state_transfer,
)


def toy_model(vocabulary_size):
    torch = pytest.importorskip("torch")

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blank_idx = vocabulary_size
            self.blank_as_pad = True
            self.prediction = torch.nn.ModuleDict({
                "embed": torch.nn.Embedding(vocabulary_size + 1, 4, padding_idx=vocabulary_size),
                "rnn": torch.nn.LSTM(4, 4, batch_first=True),
            })

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._num_extra_outputs = 0
            self.joint_net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(4, vocabulary_size + 1))

    model = torch.nn.Module()
    model.decoder, model.joint = Decoder(), Joint()
    model.encoder = torch.nn.Linear(4, 4)
    model.prompt_kernel = torch.nn.Linear(4, 4)
    model.register_buffer("retained_buffer", torch.tensor([7, 11]))
    return model


@pytest.mark.parametrize("size,mapping,removed", [
    (6, (0, 1, 2, 3, 6), 0),
    (5, (0, None, 1, 2, 5), 1),
    (2, (0, None, 1, None, 2), 2),
])
def test_full_and_subset_transfer_preserve_precisely_retained_rows(size, mapping, removed):
    torch = pytest.importorskip("torch")
    torch.manual_seed(112)
    source, target = toy_model(4), toy_model(size)
    old_layout, new_layout = inspect_nemo_layout(source), inspect_nemo_layout(target)
    original = copy.deepcopy(source.state_dict())
    initial, policy = initialize_native_added_rows(original, target.state_dict(), old_layout, new_layout, mapping)
    target.load_state_dict(transfer_native_state_dict(original, initial, old_layout, new_layout, mapping))
    report = verify_native_state_transfer(original, target.state_dict(), old_layout, new_layout, mapping)
    assert report["passed"] and report["removed_source_text_rows"] == removed
    assert report["all_source_values_preserved"] == (removed == 0)
    assert all(torch.equal(original[k], v) for k, v in source.state_dict().items())
    kept_old, kept_new = retained_row_pairs(old_layout, new_layout, mapping)
    for key in old_layout.row_keys:
        assert torch.equal(target.state_dict()[key][list(kept_new)], original[key][list(kept_old)])
        assert torch.equal(target.state_dict()[key][new_layout.blank_id], original[key][old_layout.blank_id])
    omitted = sum(original[k][[i for i, value in enumerate(mapping) if value is None]].numel() for k in old_layout.row_keys)
    assert report["learned_values_omitted"] == omitted
    assert report["learned_values_preserved"] + omitted == sum(v.numel() for v in original.values())
    if size == 2:
        assert policy["policy"] == "no_added_rows" and policy["new_row_count"] == 0


@pytest.mark.parametrize("mapping,size", [((0, 1, 2, 3, 6), 6), ((0, None, 1, 2, 5), 5)])
def test_retained_prediction_prefixes_logits_and_new_rows_are_trainable(mapping, size):
    torch = pytest.importorskip("torch")
    torch.manual_seed(33)
    source, target = toy_model(4), toy_model(size)
    old_layout, new_layout = inspect_nemo_layout(source), inspect_nemo_layout(target)
    initial, policy = initialize_native_added_rows(source.state_dict(), target.state_dict(), old_layout, new_layout, mapping)
    target.load_state_dict(transfer_native_state_dict(source.state_dict(), initial, old_layout, new_layout, mapping))
    old_prefix = torch.tensor([[4, 0, 2, 2, 3]])
    new_prefix = torch.tensor([[mapping[i] for i in old_prefix[0].tolist()]])
    audio = torch.randn(1, 5, 4)

    def forward(model, ids):
        predicted, _ = model.decoder.prediction["rnn"](model.decoder.prediction["embed"](ids))
        return model.joint.joint_net(model.encoder(audio) + predicted)

    left, right = forward(source, old_prefix), forward(target, new_prefix)
    assert compare_retained_logits(left, right, old_layout, new_layout, mapping)["passed"]
    old_ids, new_ids = retained_row_pairs(old_layout, new_layout, mapping)
    added = policy["new_model_rows"]
    ratio = right[..., added].double().logsumexp(-1) - right[..., list(new_ids)].double().logsumexp(-1)
    assert float(ratio.detach().exp().max()) <= 1.00001e-6
    text_rows = [i for i in old_ids if i != old_layout.blank_id]
    expected = source.state_dict()[old_layout.embedding_key][text_rows].double().mean(0).float()
    assert torch.equal(target.state_dict()[new_layout.embedding_key][added[0]], expected)
    new_labels = torch.tensor([[added[0], added[1], added[0], added[1], added[0]]])
    loss = torch.nn.functional.cross_entropy(forward(target, new_labels).flatten(0, 1), new_labels.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    for param in (target.decoder.prediction["embed"].weight, target.joint.joint_net[-1].weight):
        assert torch.isfinite(param.grad[added]).all() and param.grad[added].abs().sum() > 0


def test_removed_output_can_change_predictions_despite_exact_retained_logits():
    torch = pytest.importorskip("torch")
    source, target = toy_model(4), toy_model(3)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, None, 1, None, 3)
    with torch.no_grad():
        source.joint.joint_net[-1].weight.zero_()
        source.joint.joint_net[-1].bias.zero_()
        source.joint.joint_net[-1].bias[1] = 100
    initial, _ = initialize_native_added_rows(source.state_dict(), target.state_dict(), old, new, mapping)
    target.load_state_dict(transfer_native_state_dict(source.state_dict(), initial, old, new, mapping))
    features = torch.ones(1, 4)
    before, after = source.joint.joint_net[-1](features), target.joint.joint_net[-1](features)
    assert before.argmax(-1).item() == 1 and mapping[1] is None
    assert compare_retained_logits(before, after, old, new, mapping)["passed"]
    assert after.argmax(-1).item() == 0


@pytest.mark.parametrize("mapping", [
    (0, 1, 2, 3), (0, 1, 2, 3, None), (0, 1, 2, 3, 4),
    (0, 1, 1, 3, 6), (0, True, 2, 3, 6), (0, 1.0, 2, 3, 6),
    (0, -1, 2, 3, 6), (0, 7, 2, 3, 6), (None, None, None, None, 6),
])
def test_invalid_or_ambiguous_subset_mapping_is_rejected(mapping):
    old, new = inspect_nemo_layout(toy_model(4)), inspect_nemo_layout(toy_model(6))
    with pytest.raises(ValueError):
        retained_row_pairs(old, new, mapping)


@pytest.mark.parametrize("tensor", ["encoder.weight", "prompt_kernel.weight", "decoder.prediction.embed.weight"])
def test_subset_verifier_detects_tampering_with_retained_state(tensor):
    source, target = toy_model(4), toy_model(5)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, None, 1, 2, 5)
    state = transfer_native_state_dict(source.state_dict(), target.state_dict(), old, new, mapping)
    state[tensor][0, 0] += 1
    with pytest.raises(ValueError, match=tensor.replace(".", r"\.")):
        verify_native_state_transfer(source.state_dict(), state, old, new, mapping)


def test_subset_transfer_rejects_dtypes_shapes_and_missing_keys():
    source, target = toy_model(4), toy_model(5)
    old, new = inspect_nemo_layout(source), inspect_nemo_layout(target)
    mapping = (0, None, 1, 2, 5)
    for kind in ("dtype", "shape", "keys"):
        state = copy.deepcopy(target.state_dict())
        if kind == "dtype": state["encoder.weight"] = state["encoder.weight"].double()
        elif kind == "shape": state["encoder.weight"] = state["encoder.weight"][:1]
        else: del state["encoder.weight"]
        with pytest.raises(ValueError):
            transfer_native_state_dict(source.state_dict(), state, old, new, mapping)


def test_source_checks_actual_sentencepiece_bytes_and_wrapper_encoding():
    model = pb.ModelProto()
    model.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    model.trainer_spec.unk_id = 0
    model.trainer_spec.bos_id = model.trainer_spec.eos_id = model.trainer_spec.pad_id = -1
    model.normalizer_spec.name = "identity"
    for piece, kind in [("<unk>", 2), ("▁", 1), ("a", 1)]:
        model.pieces.add(piece=piece, score=-1, type=kind)
    model.trainer_spec.vocab_size = len(model.pieces)
    data = model.SerializeToString()
    backend = spm.SentencePieceProcessor(model_proto=data)
    tokenizer = SimpleNamespace(tokenizer=backend, vocab_size=3,
        ids_to_tokens=lambda ids: [backend.id_to_piece(i) for i in ids],
        text_to_ids=lambda text: backend.encode(text, out_type=int))
    source = SimpleNamespace(tokenizer=tokenizer)
    assert validate_source_native_tokenizer(source, data)["native_tokenizer_sha256"] == hashlib.sha256(data).hexdigest()
    model.pieces[2].score = -2
    with pytest.raises(ValueError, match="pinned native base"):
        validate_source_native_tokenizer(source, model.SerializeToString())
    tokenizer.text_to_ids = lambda text: [2]
    with pytest.raises(ValueError, match="wrapper encoding"):
        validate_source_native_tokenizer(source, data)


def test_prompt_registry_keeps_old_slots_and_requires_all_22_targets():
    defaults = SimpleNamespace(num_prompts=64, prompt_dictionary={"auto": 0, "en-US": 1, "hi-IN": 2})
    registry = _prompt_registry(defaults)
    assert len(registry["target_assignments"]) == 22
    assert registry["prompt_dictionary"]["hi-IN"] == 2
    assert registry["prompt_dictionary"]["en-US"] == 1
    assert _prompt_registry(defaults, registry) == registry
    bad = copy.deepcopy(registry)
    bad["prompt_dictionary"]["en-US"] = 3
    with pytest.raises(ValueError, match="upstream prompt assignment"):
        _prompt_registry(defaults, bad)
    wrong = SimpleNamespace(num_prompts=64, prompt_dictionary={"auto": 0, "en-US": 1, "hi-IN": 3})
    with pytest.raises(ValueError, match="different pinned processor"):
        _prompt_registry(wrong, registry)
