"""Fake-NeMo API tests, not evidence of real RNNT or audio training."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models

import untok.training_smoke as smoke
from untok.runtime import HFTokenizerAdapter


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tokenizer = Tokenizer(models.BPE(vocab={"<unk>": 0, "a": 1}, merges=[], unk_token="<unk>"))
    tokenizer.decoder = decoders.Fuse()
    tokenizer.add_special_tokens(["<unk>", "<pad>", "<blank>"])
    base, extended = tmp_path / "base.json", tmp_path / "extended.json"
    tokenizer.save(str(base))
    data = json.loads(base.read_text())
    data["model"]["vocab"].update({"<pad>": 2, "<blank>": 3, "क": 4})
    extended.write_text(json.dumps(data))
    adapter = HFTokenizerAdapter(extended)

    class FakeDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blank_as_pad, self.blank_idx = True, adapter.blank_id
            self.prediction = torch.nn.ModuleDict({"embed": torch.nn.Embedding(adapter.vocab_size + 1, 4, padding_idx=adapter.blank_id)})

        def forward(self, targets, target_length):
            return self.prediction["embed"](targets).transpose(1, 2), target_length, None

    class FakeJoint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._num_extra_outputs, self.fuse_loss_wer = 0, False
            self.joint_net = torch.nn.Sequential(torch.nn.Tanh(), torch.nn.Linear(4, adapter.vocab_size + 1))

        def forward(self, encoder_outputs, decoder_outputs, **kwargs):
            logits = self.joint_net(decoder_outputs.transpose(1, 2))
            if self.fuse_loss_wer:
                loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), kwargs["transcripts"].flatten())
                return loss, None, None, None
            return logits

    class FakeNeMo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.tokenizer, self.decoder, self.joint = adapter, FakeDecoder(), FakeJoint()
            self.cfg = {"model_defaults": {"prompt_dictionary": {"hi-IN": 6}}}

        def forward(self, input_signal, input_signal_length, prompt_indices):
            return input_signal.unsqueeze(1), input_signal_length

        def loss(self, log_probs, targets, input_lengths, target_lengths):
            # This cross-entropy is solely a fake API fixture. Production calls
            # the restored NeMo model's actual RNNT loss implementation.
            return torch.nn.functional.cross_entropy(log_probs.flatten(0, 1), targets.flatten())

    model = FakeNeMo()
    checkpoint = tmp_path / "fixture.nemo"
    checkpoint.write_bytes(b"fake checkpoint; no real model is loaded by these tests")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    labels = torch.tensor([adapter.text_to_ids("कaक")])
    batch = (torch.ones(1, 8), torch.tensor([8]), labels, torch.tensor([3]), torch.tensor([6]))
    monkeypatch.setattr(smoke, "_load_model", lambda path, device: model)
    kwargs = {"checkpoint_path": checkpoint, "batch": batch, "texts": ["कaक"], "target_langs": ["hi-IN"],
              "base_tokenizer_json": base, "checkpoint_sha256": digest, "tokenizer_sha256": adapter.id_map.tokenizer_sha256}
    return SimpleNamespace(model=model, kwargs=kwargs, batch=batch, adapter=adapter)


@pytest.mark.parametrize("fused", [False, True])
def test_checks_gradients_without_optimizer_or_checkpoint_write(setup, fused):
    torch = pytest.importorskip("torch")
    setup.model.joint.fuse_loss_wer = fused
    before = {key: value.clone() for key, value in setup.model.state_dict().items()}
    report = smoke.run_training_smoke(**setup.kwargs)
    assert report["passed"] is True and report["optimizer_steps"] == 0
    assert report["checkpoint_saved"] is False and report["asr_accuracy_evaluated"] is False
    assert report["release_ready"] is False
    assert report["exercised_new_model_rows"] == [2]
    assert report["loss_path"] == ("fused_rnnt_joint_loss" if fused else "rnnt_loss")
    assert "FakeNeMo" in report["model_class"]
    assert all(torch.equal(before[key], value) for key, value in setup.model.state_dict().items())
    assert all(parameter.grad is None for parameter in setup.model.parameters())


@pytest.mark.parametrize("corruption", ["blank", "wrong_prompt", "wrong_labels", "no_new_rows", "wrong_hash"])
def test_invalid_training_batch_is_not_reported_as_passing(setup, corruption):
    torch = pytest.importorskip("torch")
    kwargs = dict(setup.kwargs)
    batch = list(setup.batch)
    if corruption == "blank":
        batch[2] = batch[2].clone()
        batch[2][0, 0] = setup.adapter.blank_id
    elif corruption == "wrong_prompt":
        batch[4] = torch.tensor([7])
    elif corruption == "wrong_labels":
        kwargs["texts"] = ["aaa"]
    elif corruption == "no_new_rows":
        batch[2] = torch.tensor([setup.adapter.text_to_ids("aaa")])
        kwargs["texts"] = ["aaa"]
    else:
        kwargs["checkpoint_sha256"] = "0" * 64
    kwargs["batch"] = tuple(batch)
    with pytest.raises(ValueError):
        smoke.run_training_smoke(**kwargs)


def test_nonfinite_loss_does_not_pass(setup, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(setup.model, "loss", lambda **kwargs: torch.tensor(float("nan"), requires_grad=True))
    with pytest.raises(ValueError, match="finite differentiable scalar"):
        smoke.run_training_smoke(**setup.kwargs)


def test_frozen_new_embeddings_are_detected(setup):
    setup.model.decoder.prediction["embed"].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="No gradients reached"):
        smoke.run_training_smoke(**setup.kwargs)
