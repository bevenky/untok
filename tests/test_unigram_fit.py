"""Numerical oracles for fitting additions with native scores held fixed."""
from collections import Counter
import io
import math

import pytest
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as pb

from sttok.unigram_fit import Lattice, NativeFitter, _allocate, _trie, prepare_records


def test_forward_backward_matches_exhaustive_segmentations_and_derivatives():
    scores = {"a": -2.0, "b": -3.0, "ab": -1.0, "ba": -1.5}

    def enumerate_paths(text):
        if not text:
            return [([], 0.0)]
        result = []
        for piece, score in scores.items():
            if text.startswith(piece):
                result.extend(([piece] + tail, score + value)
                              for tail, value in enumerate_paths(text[len(piece):]))
        return result

    paths = enumerate_paths("ababa")
    partition = sum(math.exp(score) for _, score in paths)
    expected = Counter()
    for pieces, score in paths:
        for piece in pieces:
            expected[piece] += math.exp(score) / partition
    lattice = Lattice("ababa", _trie(scores), -263)
    actual_partition, actual_counts = lattice.expected(scores)
    assert actual_partition == pytest.approx(math.log(partition), abs=1e-12)
    for piece in scores:
        assert actual_counts[piece] == pytest.approx(expected[piece], abs=1e-12)
        plus, minus = dict(scores), dict(scores)
        plus[piece] += 1e-5
        minus[piece] -= 1e-5
        derivative = (lattice.expected(plus)[0] - lattice.expected(minus)[0]) / 2e-5
        assert derivative == pytest.approx(expected[piece], abs=1e-8)


def test_unknown_fallback_activates_when_a_single_character_is_pruned():
    lattice = Lattice("c", _trie(["c"]), -263)
    assert lattice.best({"c": -4}) == (-4, ["c"])
    assert lattice.best({}) == (-263, [None])
    assert lattice.expected({})[0] == -263
    assert lattice.best({"c": -4}, excluded="c") == (-263, [None])


def test_fixed_mass_allocation_respects_active_floor_and_ceiling():
    scores = _allocate({"a": 1.0, "b": 2.0, "c": 7.0}, 0.75,
                       {"a": 0.2, "b": 0.01, "c": 0.01}, 0.4)
    probabilities = {p: math.exp(s) for p, s in scores.items()}
    assert sum(probabilities.values()) == pytest.approx(0.75, abs=1e-7)
    assert probabilities["a"] == pytest.approx(0.2, abs=1e-7)
    assert probabilities["c"] == pytest.approx(0.4, abs=1e-7)
    assert probabilities["b"] == pytest.approx(0.15, abs=1e-7)


@pytest.mark.parametrize("mass", [0.05, 2.1])
def test_infeasible_mass_is_rejected_instead_of_silently_adjusted(mass):
    with pytest.raises(ValueError, match="Infeasible"):
        _allocate({"a": 1.0, "b": 2.0}, mass, {"a": 0.1, "b": 0.1}, 1.0)


@pytest.mark.parametrize("count", [float("inf"), float("nan"), -1.0])
def test_nonfinite_or_negative_counts_fail_before_multiplier_search(count):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _allocate({"a": count}, 0.5, {"a": 0.1}, 1.0)


def test_tiny_infeasible_mass_is_not_hidden_by_absolute_tolerance():
    with pytest.raises(ValueError, match="Infeasible"):
        _allocate({"a": 1.0}, 1e-30, {"a": 1e-20}, 1.0)


def test_large_finite_count_rescaling_preserves_the_true_optimum():
    scores = _allocate({"a": 1e308, "b": 5e307}, 0.25,
                       {"a": 1e-110, "b": 1e-110}, 1.0)
    assert math.exp(scores["a"]) == pytest.approx(1/6, rel=2e-7)
    assert math.exp(scores["b"]) == pytest.approx(1/12, rel=2e-7)


def test_tiny_feasible_mass_stays_proportional_after_float32_rounding():
    scores = _allocate({"a": 2.0, "b": 1.0}, 1e-100,
                       {"a": 1e-110, "b": 1e-110}, 1.0)
    assert math.exp(scores["a"]) == pytest.approx(2e-100/3, rel=2e-5, abs=0)
    assert math.exp(scores["b"]) == pytest.approx(1e-100/3, rel=2e-5, abs=0)


def test_normalization_cannot_introduce_unhandled_user_defined_controls():
    output = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello tag word other text"]), model_writer=output,
        model_type="unigram", vocab_size=30, hard_vocab_limit=False,
        user_defined_symbols=["<tag>"], minloglevel=2,
    )
    native = pb.ModelProto()
    native.ParseFromString(output.getvalue())
    raw = "＜ｔａｇ＞"
    assert spm.SentencePieceProcessor(model_proto=output.getvalue()).normalize(raw) == "▁<tag>"
    with pytest.raises(ValueError, match="Normalization produced a USER_DEFINED"):
        prepare_records(native, [{"text": raw, "language": "en", "source": "fixture"}])


def test_small_complete_fit_preserves_native_prefix_and_validates_actual_usage():
    native = pb.ModelProto()
    native.trainer_spec.model_type = pb.TrainerSpec.UNIGRAM
    native.trainer_spec.unk_id = 0
    native.trainer_spec.bos_id = native.trainer_spec.eos_id = native.trainer_spec.pad_id = -1
    native.normalizer_spec.name = "identity"
    native.normalizer_spec.remove_extra_whitespaces = False
    for text, score, kind in [("<unk>", 0, 2), ("▁", 0, 1), ("a", -2, 1),
                              ("b", -3, 1), ("ab", -4, 1), ("z", -253, 1)]:
        p = native.pieces.add()
        p.piece, p.score, p.type = text, score, kind
    native.trainer_spec.vocab_size = len(native.pieces)
    raw = "aba aba ba aa"
    fitter = NativeFitter(
        native, [{"text": raw, "language": "hi", "source": "toy", "weight": 2.0}],
        [{"piece": p, "score": -3.0, "groups": ["g"]} for p in ["aba", "ba", "aa"]],
        {"g": 2}, {"g": ["ba"]}, balance_language_source=False,
    )
    result = fitter.fit(added_score_mass=0.2, em_passes=2, refill_rounds=4)
    model, report = result["model_proto"], result["report"]
    assert all(report["gates"].values())
    assert report["final_group_counts"] == {"g": 2}
    assert report["added_score_mass_actual"] == pytest.approx(0.2, rel=1e-6)
    assert [p.SerializeToString() for p in model.pieces[:len(native.pieces)]] == [
        p.SerializeToString() for p in native.pieces
    ]
    assert model.normalizer_spec.SerializeToString() == native.normalizer_spec.SerializeToString()
    processor = spm.SentencePieceProcessor(model_proto=model.SerializeToString())
    counts = Counter(processor.encode(raw, out_type=str))
    for row in result["selected_pieces"]:
        assert row["weighted_training_usage"] == 2.0 * counts[row["piece"]]
    for phase in report["trace"]:
        for step in phase.get("em", []):
            assert step["objective_after_em"] >= step["objective_before"] - 1e-6
