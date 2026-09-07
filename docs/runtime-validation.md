# Legacy BPE checkpoint validation

These commands and scripts use the BPE adapter. Native Unigram checkpoint
integration and speech tests remain pending.

These checks use the complete NVIDIA checkpoint and real FLEURS recordings.
They test migration and training integration. They do not measure whether the
expanded model recognizes the new languages accurately.
See [the recorded results](runtime-results.md) for the completed run and its limits.

## Tested runtime

The September 2026 run used Ubuntu, Python 3.12.3, an H100 NVL with 96 GB
of GPU memory, PyTorch 2.8.0 with CUDA 12.8, and NeMo Speech revision
`ca4daa1470f6c01068c4e6a9a73b19b9a91dc366`.

Use an isolated environment with this NeMo revision and install this repository
with `python -m pip install -e '.[legacy-bpe,test,checkpoint]'`. The tested loss backend
was NeMo's native Numba RNNT loss with its original FastEmit configuration.
The runtime used numba 0.67.0, numba-cuda 0.13.0, cuda-python 12.8.0,
SoundFile 0.14.0 and lhotse 2.0.0a3. The checkpoint extra alone does not
install this training stack. Retain a package freeze with each run.

To include the optional CPU decoding check, set `STTOK_NATIVE_DECODER_MODEL`
to the original `.model` extracted from the pinned native checkpoint when
running pytest. Without that artifact, this one test is explicitly skipped.

Build the tokenizer and migrate the checkpoint using
[the checkpoint instructions](checkpoint.md). Always write a new checkpoint
and retain the source and migration report.

## Small development corpus

From the repository root:

```sh
python scripts/prepare_fleurs_smoke.py \
  --output .cache/fleurs-smoke \
  --tokenizer artifacts/nemotron-indic-v1/tokenizer.json \
  --source-processor .cache/sources/nvidia/processor_config.json

python scripts/prepare_fleurs_smoke.py \
  --output .cache/fleurs-new-prompts --configs pa_in or_in \
  --eval-per-language 0 --train-per-language 1 \
  --tokenizer artifacts/nemotron-indic-v1/tokenizer.json \
  --source-processor .cache/sources/nvidia/processor_config.json
```

The script pins Google FLEURS revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, keeps the original transcripts
and audio, and records each selected file's hash and source. It downloads
only enough of each archive to find short clips. It records the published
whole-archive hash but does not claim to have verified the undownloaded bytes.
Selection does not use model predictions or loss.

The default corpus has two test and one train clip for English, Hindi,
Tamil, Malayalam, Marathi and Kannada. The second command adds Punjabi
and Odia train clips, whose prompt slots were unused in the source model.
These are development samples. Once used to debug migration, they are not
an independent accuracy benchmark.

## Checks

Run [paired checkpoint validation](checkpoint-validation.md) with the test
records from `migration-clips.jsonl`. Its manifest must bind the original
and migrated checkpoint hashes, both tokenizer hashes and each audio hash.
Both models receive the same explicitly verified language prompt. Review
the run with all new outputs enabled as well as the masked migration run.

Run actual dataloader and backward checks:

```sh
python scripts/real_training_probe.py \
  --checkpoint /models/indic-expanded.nemo \
  --tokenizer artifacts/nemotron-indic-v1/tokenizer.json \
  --base-tokenizer .cache/sources/nvidia/tokenizer.json \
  --prompts artifacts/nemotron-indic-v1/prompts.json \
  --source-processor .cache/sources/nvidia/processor_config.json \
  --corpus-dir .cache/fleurs-smoke .cache/fleurs-new-prompts \
  --output reports/training-probes.json --device cuda
```

This exercises seven Indic languages with new labels through NeMo's actual
Lhotse dataloader and native RNNT loss. It checks finite loss, gradients on
new embedding/output rows and the requested prompt column. It performs no
optimizer steps and does not save trained weights.

`scripts/diagnose_loss_scale.py` compares original native labels with BPE
labels on the English and Hindi train clips. It checks old logits, masked
and unmasked native loss, fused/direct loss agreement and new-token probability
mass. Supply `--source`, `--expanded`, `--base-tokenizer`, `--tokenizer`,
`--corpus-dir`, `--output` and `--device cuda` using the same artifacts above.
This is a read-only diagnostic with no backward pass or saved checkpoint.

Run one native streaming comparison:

```sh
python scripts/real_streaming_probe.py \
  --source /models/nemotron-3.5-asr-streaming-0.6b.nemo \
  --expanded /models/indic-expanded.nemo \
  --migration-report /models/indic-expanded.migration.json \
  --base-tokenizer .cache/sources/nvidia/tokenizer.json \
  --tokenizer artifacts/nemotron-indic-v1/tokenizer.json \
  --corpus-dir .cache/fleurs-smoke \
  --output reports/streaming-probe.json --device cuda
```

This follows the pinned NeMo cache-aware streaming API at a supported
1,120 ms chunk setting. It compares chunk caches, carried RNNT hypotheses,
tokens and old logits. The official buffer precomputes frontend features;
this is a streaming simulation, not a microphone or latency benchmark.

## Remaining evaluation

Use separate, sufficiently sized speech sets to evaluate all original
locales and new target profiles, then repeat after fine-tuning. Track WER/CER,
code-switching and script errors separately. Check mixed precision, batching,
CUDA graphs and production streaming settings before deploying those modes.
Exact old weights and short smoke tests are not an all-language accuracy
guarantee. Native SentencePiece and HF BPE can also produce different training
labels, despite matching original piece IDs.
