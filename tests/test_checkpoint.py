"""CPU checks of mapping, learned state retention, and RNNT output semantics."""
import copy
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models

from untok.checkpoint import (
    artifact_preflight, compare_old_logits, inspect_nemo_layout, initialize_added_rows,
    mask_new_outputs_for_test, old_model_row_mapping, transfer_state_dict,
    verify_state_transfer,
)
from untok.runtime import HFTokenizerAdapter, build_id_map


@pytest.fixture
def artifacts(tmp_path):
    backend = Tokenizer(models.BPE(vocab={"<unk>": 0, "▁": 1, "a": 2, "b": 3}, merges=[], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    backend.add_special_tokens(["<unk>", "<pad>", "<blank>"])
    base = tmp_path / "base.json"
    backend.save(str(base))
    data = json.loads(base.read_text())
    assert {t["content"]: t["id"] for t in data["added_tokens"]} == {"<unk>": 0, "<pad>": 4, "<blank>": 5}
    # Reserve these positions in model.vocab before appending, or tokenizers
    # reassigns added-only special IDs after the new model inventory.
    data["model"]["vocab"].update({"<pad>": 4, "<blank>": 5})
    data["model"]["vocab"].update({"क": 6, "ள": 7})
    extended = tmp_path / "extended.json"
    extended.write_text(json.dumps(data))
    return base, extended


def test_hf_reservations_and_native_blank_move(artifacts):
    base, extended = artifacts
    old, new = build_id_map(base), build_id_map(extended, base)
    assert new.hf_pad_id == 4 and new.hf_blank_id == 5
    assert new.model_to_canonical == (0, 1, 2, 3, 6, 7, 5)
    assert new.canonical_to_model == (0, 1, 2, 3, None, 6, 4, 5)
    assert old_model_row_mapping(old, new) == (0, 1, 2, 3, 6)
    for ids in ([2, 3], [6, 7], [2, 6, 3, 7]):
        assert new.to_canonical(new.to_model(ids)) == ids
    with pytest.raises(ValueError, match="padding"):
        new.to_model([4])
    with pytest.raises(ValueError, match="blank"):
        new.to_model([5])
    with pytest.raises(ValueError, match="out of range"):
        new.to_model([-1])


def test_reject_changed_base_ids_and_special_behavior(artifacts):
    base, extended = artifacts
    data = json.loads(extended.read_text())
    data["model"]["vocab"]["a"], data["model"]["vocab"]["b"] = 3, 2
    extended.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Original HF ID changed"):
        build_id_map(extended, base)


def test_reject_hf_loader_reallocation_of_reserved_ids(artifacts):
    base, extended = artifacts
    data = json.loads(extended.read_text())
    del data["model"]["vocab"]["<pad>"]
    del data["model"]["vocab"]["<blank>"]
    extended.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="HF loader changed"):
        build_id_map(extended, base)


def test_adapter_retains_repeated_rnnt_outputs_and_rejects_reserved_labels(artifacts):
    base, extended = artifacts
    adapter = HFTokenizerAdapter(extended)
    assert adapter.ids_to_text(adapter.text_to_ids("aaकளbb")) == "aaकளbb"
    assert adapter.ids_to_text([2, 2, adapter.blank_id, 3, 3]) == "aabb"
    assert list(adapter.get_vocab().values()) == list(range(adapter.vocab_size))
    assert "<pad>" not in adapter.get_vocab() and "<blank>" not in adapter.get_vocab()
    assert adapter.get_vocab()["क"] == 4
    with pytest.raises(ValueError, match="padding"):
        adapter.text_to_ids("a<pad>b")
    with pytest.raises(ValueError, match="blank"):
        adapter.text_to_ids("a<blank>b")


def test_preflight_reports_upstream_style_special_id_mismatch(artifacts, tmp_path):
    base, _ = artifacts
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"blank_token_id": 4, "pad_token_id": 0, "vocab_size": 5}))
    report = artifact_preflight(base, config)
    assert report["hf_direct_compatible"] is False
    assert {r["field"] for r in report["hf_conflicts"]} == {"blank_token_id", "pad_token_id", "vocab_size"}
    assert report["checkpoint_executed"] is False


def test_adapter_callable_returns_native_labels_for_nemo_wrapper(artifacts):
    _, extended = artifacts
    adapter = HFTokenizerAdapter(extended)
    # NeMo's TokenizerWrapper calls tokenizer(text) for non-TokenizerSpec objects.
    assert adapter("aaकளbb") == [2, 2, 4, 5, 3, 3]
    assert adapter("") == []
    with pytest.raises(ValueError, match="padding"):
        adapter("a<pad>b")
    with pytest.raises(ValueError, match="blank"):
        adapter("a<blank>b")


def test_nemo_registration_allows_only_exact_resolved_model_class(monkeypatch):
    from untok.runtime import _register_trusted_nemo_target

    class Serialization:
        pass

    class TrustedModel(Serialization):
        pass

    target = "untok.runtime.ExtendedNemotronRNNTModel"
    resolved = {target: TrustedModel}
    checked = []

    def original_allowed(value):
        checked.append(value)
        return value == "nemo.collections.asr.TrustedExistingModel"

    common = ModuleType("nemo.core.classes.common")
    common.Serialization = Serialization
    common._is_target_allowed = original_allowed
    common.ALLOWED_TARGET_PREFIXES = ["nemo.collections."]
    common.hydra = SimpleNamespace(utils=SimpleNamespace(get_class=lambda value: resolved[value]))
    monkeypatch.setitem(sys.modules, common.__name__, common)

    _register_trusted_nemo_target(TrustedModel)
    validator = common._is_target_allowed
    assert validator(target) is True
    assert validator("nemo.collections.asr.TrustedExistingModel") is True
    for unrelated in ("os.system", "untok.other.Model", target + "Unsafe", target + ".Nested"):
        assert validator(unrelated) is False
        assert unrelated in checked
    assert common.ALLOWED_TARGET_PREFIXES == ["nemo.collections."]
    resolved[target] = type("DifferentModel", (Serialization,), {})
    assert validator(target) is False
    _register_trusted_nemo_target(TrustedModel)
    assert common._is_target_allowed is validator
    with pytest.raises(ValueError, match="Serialization subclass"):
        _register_trusted_nemo_target(object)
    assert common._is_target_allowed is validator


def _model(vocabulary_size):
    torch = pytest.importorskip("torch")

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blank_idx = vocabulary_size
            self.blank_as_pad = True
            self.prediction = torch.nn.ModuleDict({
                "embed": torch.nn.Embedding(vocabulary_size + 1, 5, padding_idx=vocabulary_size),
                "rnn": torch.nn.LSTM(5, 5, batch_first=True),
            })

    class Joint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._num_extra_outputs = 0
            self.pred = torch.nn.Linear(5, 5)
            self.enc = torch.nn.Linear(7, 5)
            self.joint_net = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(5, vocabulary_size + 1))

    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(8, 7)
    model.decoder = Decoder()
    model.joint = Joint()
    model.prompt_kernel = torch.nn.Linear(128, 7)
    model.register_buffer("retained_buffer", torch.tensor([7, 11]))
    return model


def test_full_state_transfer_and_old_logits_with_blank_relocation(artifacts):
    torch = pytest.importorskip("torch")
    base, extended = artifacts
    old_map, new_map = build_id_map(base), build_id_map(extended, base)
    torch.manual_seed(19)
    original, expanded = _model(old_map.model_vocab_size), _model(new_map.model_vocab_size)
    original.eval(), expanded.eval()
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(expanded)
    row_map = old_model_row_mapping(old_map, new_map)
    initial = copy.deepcopy(expanded.state_dict())
    state = transfer_state_dict(original.state_dict(), initial, old_layout, new_layout, row_map)
    expanded.load_state_dict(state, strict=True)
    report = verify_state_transfer(original.state_dict(), expanded.state_dict(), old_layout, row_map)
    assert report["tensors_checked"] == len(original.state_dict())
    for key in old_layout.row_keys:
        assert torch.equal(expanded.state_dict()[key][4:6], initial[key][4:6])
        assert torch.equal(expanded.state_dict()[key][6], original.state_dict()[key][4])
    # A real predictor prefix containing blank and repeated labels exercises the
    # retained LSTM and projections, rather than testing only copied row values.
    prefix = torch.tensor([[old_map.model_blank_id, 2, 2, 3]])
    mapped_prefix = torch.tensor([[row_map[i] for i in prefix[0].tolist()]])
    audio = torch.randn(1, 4, 8)

    def forward(model, ids):
        encoded = model.encoder(audio)
        predicted, _ = model.decoder.prediction["rnn"](model.decoder.prediction["embed"](ids))
        return model.joint.joint_net(model.joint.enc(encoded) + model.joint.pred(predicted))

    old_logits, new_logits = forward(original, prefix), forward(expanded, mapped_prefix)
    assert compare_old_logits(old_logits, new_logits, row_map)["passed"]
    masked = mask_new_outputs_for_test(new_logits, row_map)
    assert torch.allclose(old_logits.softmax(-1), masked.softmax(-1)[..., list(row_map)])
    assert not torch.allclose(old_logits.softmax(-1), new_logits.softmax(-1)[..., list(row_map)])
    # New additions remain trainable, including their embedding and output rows.
    new_prefix = torch.tensor([[4, 5, 4, 5]])
    new_loss = torch.nn.functional.cross_entropy(forward(expanded, new_prefix).flatten(0, 1), new_prefix.flatten())
    new_loss.backward()
    assert torch.isfinite(new_loss)
    assert expanded.decoder.prediction["embed"].weight.grad[4:6].abs().sum() > 0
    assert expanded.joint.joint_net[-1].weight.grad[4:6].abs().sum() > 0


def test_detects_destroyed_unchanged_layers_and_unrecognized_heads(artifacts):
    torch = pytest.importorskip("torch")
    original, expanded = _model(4), _model(6)
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(expanded)
    rows = (0, 1, 2, 3, 6)
    state = transfer_state_dict(original.state_dict(), expanded.state_dict(), old_layout, new_layout, rows)
    state["prompt_kernel.weight"][0, 0] += 1
    with pytest.raises(ValueError, match="prompt_kernel.weight"):
        verify_state_transfer(original.state_dict(), state, old_layout, rows)
    expanded.joint._num_extra_outputs = 1
    with pytest.raises(ValueError, match="Extra output"):
        inspect_nemo_layout(expanded)
    with pytest.raises(ValueError, match="blank"):
        transfer_state_dict(original.state_dict(), state, old_layout, new_layout, (0, 1, 2, 3, 4))


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_blank_anchored_additions_preserve_greedy_choice_and_bound_probability_mass(dtype_name):
    torch = pytest.importorskip("torch")
    torch.manual_seed(127)
    dtype = getattr(torch, dtype_name)
    original, expanded = _model(4).to(dtype), _model(6).to(dtype)
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(expanded)
    row_map, added = (0, 1, 2, 3, 6), [4, 5]
    source = copy.deepcopy(original.state_dict())
    untouched = copy.deepcopy(expanded.state_dict())
    initialized, policy = initialize_added_rows(source, expanded.state_dict(), old_layout, new_layout, row_map)
    assert all(torch.equal(expanded.state_dict()[key], value) for key, value in untouched.items())
    expanded.load_state_dict(transfer_state_dict(source, initialized, old_layout, new_layout, row_map))
    assert verify_state_transfer(source, expanded.state_dict(), old_layout, row_map)["passed"]
    expected_embedding = source[old_layout.embedding_key][:-1].double().mean(0).to(dtype)
    assert torch.equal(expanded.decoder.prediction["embed"].weight[added], expected_embedding.expand(2, -1))
    expected_weight = source[old_layout.output_weight_key][old_layout.blank_id]
    assert torch.equal(expanded.joint.joint_net[-1].weight[added], expected_weight.expand(2, -1))
    features = torch.cat([torch.zeros(1, 5, dtype=dtype)] +
                         [torch.randn(73, 5, dtype=dtype) * scale for scale in (1e-3, 1, 100, 10000)])
    old_logits = original.joint.joint_net[-1](features)
    new_logits = expanded.joint.joint_net[-1](features)
    assert torch.equal(new_logits.argmax(-1), torch.tensor(row_map)[old_logits.argmax(-1)])
    log_ratio = (torch.logsumexp(new_logits[:, added].double(), -1)
                 - torch.logsumexp(new_logits[:, list(row_map)].double(), -1))
    assert torch.isfinite(new_logits).all()
    # Large features magnify float rounding when the bias is added to logits.
    # Check the exact statewise formula with an explicit dtype/scale tolerance.
    tolerance = 4 * torch.finfo(dtype).eps * max(1, float(new_logits.detach().abs().max()))
    expected_ratio = (math.log(len(added)) - policy["applied_bias_margin"]
                      + old_logits[:, old_layout.blank_id].double() - old_logits.double().logsumexp(-1))
    assert torch.allclose(log_ratio, expected_ratio, atol=tolerance, rtol=0)
    assert float(log_ratio.detach().max()) <= math.log(policy["max_new_to_old_softmax_mass_ratio"]) + tolerance
    assert policy["old_output_rows_including_blank"] == 5
    assert policy["embedding_mean_rows_excluding_blank"] == 4
    assert policy["reference_old_blank_row"] == old_layout.blank_id


@pytest.mark.parametrize("nonblank_bias", [0., -100.])
def test_added_probability_tracks_old_blank_and_bound_is_tight_when_blank_dominates(nonblank_bias):
    torch = pytest.importorskip("torch")
    original, expanded = _model(4), _model(6)
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(expanded)
    with torch.no_grad():
        original.joint.joint_net[-1].weight.zero_()
        original.joint.joint_net[-1].bias.fill_(nonblank_bias)
        original.joint.joint_net[-1].bias[-1] = 0
        original.decoder.prediction["embed"].weight[-1].fill_(1000)
    row_map = (0, 1, 2, 3, 6)
    initialized, policy = initialize_added_rows(original.state_dict(), expanded.state_dict(), old_layout, new_layout, row_map)
    expanded.load_state_dict(transfer_state_dict(original.state_dict(), initialized, old_layout, new_layout, row_map))
    logits = expanded.joint.joint_net[-1](torch.full((1, 5), 10000.0)).double()
    probabilities = logits.softmax(-1)
    ratio = float((probabilities[:, 4:6].sum() / probabilities[:, list(row_map)].sum()).detach())
    old_blank_probability = 1 / (1 + 4 * math.exp(nonblank_bias))
    assert ratio == pytest.approx(policy["exact_arithmetic_mass_ratio_bound"] * old_blank_probability, rel=1e-6)
    if nonblank_bias == -100:
        assert 0.99999e-6 < ratio <= 1e-6
    assert policy["exact_arithmetic_mass_ratio_bound"] <= 1e-6
    assert torch.equal(expanded.decoder.prediction["embed"].weight[4],
                       original.decoder.prediction["embed"].weight[:4].double().mean(0).float())
    # Blank must remain preferred even though it moves after the newly added IDs.
    with torch.no_grad():
        original.joint.joint_net[-1].bias[-1] = 4
    initialized, _ = initialize_added_rows(original.state_dict(), expanded.state_dict(), old_layout, new_layout, row_map)
    expanded.load_state_dict(transfer_state_dict(original.state_dict(), initialized, old_layout, new_layout, row_map))
    assert expanded.joint.joint_net[-1](torch.ones(1, 5)).argmax(-1).item() == new_layout.blank_id


def test_new_row_initialization_is_seed_independent_and_remains_trainable():
    torch = pytest.importorskip("torch")
    torch.manual_seed(731)
    original = _model(4)
    row_map = (0, 1, 2, 3, 6)
    old_layout = inspect_nemo_layout(original)
    states = []
    for seed in (22, 99):
        torch.manual_seed(seed)
        expanded = _model(6)
        new_layout = inspect_nemo_layout(expanded)
        initialized, _ = initialize_added_rows(original.state_dict(), expanded.state_dict(), old_layout, new_layout, row_map)
        state = transfer_state_dict(original.state_dict(), initialized, old_layout, new_layout, row_map)
        states.append(state)
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
    expanded.load_state_dict(states[-1])
    prefix = torch.tensor([[4, 4, 5]])
    embedding = expanded.decoder.prediction["embed"](prefix)
    prediction, _ = expanded.decoder.prediction["rnn"](embedding)
    logits = expanded.joint.joint_net(expanded.joint.pred(prediction))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([4, 5, 4]))
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (expanded.decoder.prediction["embed"].weight, expanded.joint.joint_net[-1].weight,
                      expanded.joint.joint_net[-1].bias):
        gradients = parameter.grad[4:6]
        assert parameter.requires_grad and torch.isfinite(gradients).all()
        assert (gradients.reshape(2, -1).abs().sum(-1) > 0).all()
    assert not torch.equal(expanded.joint.joint_net[-1].weight.grad[4], expanded.joint.joint_net[-1].weight.grad[5])


@pytest.mark.parametrize("failure", ["no_bias", "nan", "zero_ratio", "infinite_ratio", "bad_mapping"])
def test_added_row_initialization_rejects_unsupported_or_nonfinite_inputs(failure):
    torch = pytest.importorskip("torch")
    original, expanded = _model(4), _model(6)
    if failure == "no_bias":
        original.joint.joint_net[-1].bias = None
        expanded.joint.joint_net[-1].bias = None
    old_layout, new_layout = inspect_nemo_layout(original), inspect_nemo_layout(expanded)
    source = copy.deepcopy(original.state_dict())
    if failure == "nan":
        source[old_layout.output_weight_key][0, 0] = float("nan")
    ratio = 0 if failure == "zero_ratio" else float("inf") if failure == "infinite_ratio" else 1e-6
    rows = (0, 1, 2, 3, 4) if failure == "bad_mapping" else (0, 1, 2, 3, 6)
    with pytest.raises(ValueError, match="bias|non-finite|mass ratio|mapping"):
        initialize_added_rows(source, expanded.state_dict(), old_layout, new_layout, rows, max_new_mass_ratio=ratio)


def test_pinned_nvidia_artifact_contract_if_available():
    """Use the actual locally fetched source inventory without network access."""
    root = Path(__file__).resolve().parents[1]
    candidates = [root / ".cache/sources/nvidia/tokenizer.json", root / ".cache/sources/nvidia/nemotron-tokenizer.json"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        pytest.skip("Pinned source cache not populated; synthetic contracts are checked separately")
    mapping = build_id_map(path)
    assert mapping.tokenizer_sha256 == "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246"
    assert (mapping.hf_pad_id, mapping.hf_blank_id, mapping.model_blank_id) == (13087, 13088, 13087)
    assert mapping.model_output_size == 13088
