# Run checkpoint compatibility checks on real audio

`validate_checkpoint_pair()` restores a local original `.nemo` and a local
expanded `.nemo`. It verifies all old tensors and model/tokenizer mappings,
transcribes the same audio with the same existing language prompts, and records
both the restricted migration check and a separate run with new outputs active.
It never downloads a checkpoint or substitutes example predictions.

This command targets an expanded **untrained** checkpoint. A fine-tuned model
will normally fail exact old-weight equality; its accuracy belongs in the
separate speech evaluation workflow.

## Manifest

Use an explicit nonempty local corpus and actual SHA-256 hashes. Relative audio
paths resolve against the manifest directory. Every `target_lang` must exist in
both checkpoints. New language prompts cannot be silently replaced with `auto`.

```json
{
  "schema_version": 1,
  "run_id": "migration-offline-greedy-001",
  "source_checkpoint_sha256": "<64 lowercase hex characters>",
  "expanded_checkpoint_sha256": "<64 lowercase hex characters>",
  "base_tokenizer_sha256": "<64 lowercase hex characters>",
  "expanded_tokenizer_sha256": "<64 lowercase hex characters>",
  "source_runtime_tokenizer_sha256": "<64 lowercase hex characters>",
  "settings": {
    "mode": "offline",
    "dtype": "float32",
    "device": "cuda",
    "batch_size": 1,
    "num_workers": 0,
    "decoder": "greedy_batch"
  },
  "utterances": [
    {
      "id": "english-001",
      "audio": "audio/english-001.wav",
      "audio_sha256": "<64 lowercase hex characters>",
      "target_lang": "en-US"
    }
  ]
}
```

The source runtime tokenizer hash describes the actual native SentencePiece
artifact embedded in `.nemo`, not the canonical HF file. For the previously
audited native artifact it is
`ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`;
the runner independently checks the loaded tokenizer's serialized bytes.

## Execution

Install the project and its compatible pinned NeMo runtime on Linux. Supply
sufficient RAM/GPU memory for both models. Tokenizer dependencies alone cannot
run this validation. Use the same declared device in the manifest and function.

```python
from sttok.checkpoint_validation import validate_checkpoint_pair

report = validate_checkpoint_pair(
    source_checkpoint="/models/original.nemo",
    expanded_checkpoint="/models/indic-expanded.nemo",
    base_tokenizer_json="/artifacts/nvidia/tokenizer.json",
    extended_tokenizer_json="/artifacts/indic/tokenizer.json",
    manifest_path="/evaluation/checkpoint-validation.json",
    output_path="/evaluation/results/migration-001.json",
    device="cuda",
)
```

The runner requires the checkpoints' actual greedy strategy to match the
manifest. It compares encoder, preprocessor, decoder, joint, prompt-dimension
and decoding configuration, excluding only approved vocabulary fields. Explicit
graph/compiled decoding settings are rejected because Python forward hooks
cannot be assumed to run during graph replay. If the actual joint hook does not
execute, validation stops rather than claiming success.

## Checks performed

1. Every supplied artifact/audio hash matches; old HF IDs and special-token
   behavior remain intact; the restored expanded tokenizer has the expected ID
   mapping and hash.
2. Every original learned tensor, buffer, embedding row and joint-output row
   matches exactly, including the relocated blank.
3. Original audio transcription runs through the real final joint `Linear`
   layer. Its hook captures up to eight calls and eight vectors per call, keeping
   trace memory bounded.
4. Expanded transcription runs with added outputs set to negative infinity by a
   hook on that same final `Linear`, before the runtime applies softmax. It must
   return the same canonical token sequence and text. Consecutive RNNT tokens
   are retained. Only actual `Hypothesis.text` and `Hypothesis.y_sequence` are
   accepted.
5. Captured joint states and mapped old-token/blank raw logits are compared.
   Captured original joint inputs are also replayed into the expanded final head
   for an independent fixed-input logit check. This is a bounded probe of real
   decoding states, not a claim to have tested every possible prefix.
6. The hook is removed and the expanded model transcribes each audio again with
   all outputs enabled. Changed token sequences and transcripts are reported
   separately. These changes do not get hidden by the migration mask.

`passed=true` means that the migration checks passed **on this supplied offline
corpus**. It does not mean unchanged ASR accuracy with the new outputs active,
and it never sets `release_ready=true`. The report leaves real RNNT
forward/backward with new labels, streaming regression, multilingual speech
fine-tuning and held-out accuracy evaluation pending.

Missing corpus/metadata or incompatible runtime behavior raises an error. Actual
token/text or captured-state disagreement produces a failed compatibility report.
Existing report files are never overwritten.

## Test boundary

```sh
python -m pytest tests/test_checkpoint_validation.py -q
```

These tests use clearly labelled fake NeMo objects and fixture bytes. They test
the runner's failure handling, real PyTorch hook/softmax behavior, hook cleanup,
separate active-output run, metadata checks and reporting. They are not evidence
that a real NVIDIA checkpoint or audio evaluation has run.
