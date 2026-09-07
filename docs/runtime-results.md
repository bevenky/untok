# Runpod validation results

Run date: 7 September 2026. Hardware: NVIDIA H100 NVL. Runtime and repeatable
commands are in [runtime validation](runtime-validation.md).

The tested candidate retains 16,481 canonical IDs. These results cover a
small development corpus and an untrained vocabulary expansion, not an ASR
release or a fine-tuned multilingual model.

## Completed checks

| Check | Result |
| --- | --- |
| Complete CPU suite | 171 tests pass on macOS and Linux, including the optional real native decoder artifact check |
| Fresh tokenizer build on Linux | Exact same canonical tokenizer SHA-256 |
| Full checkpoint migration and save/reload | All 657 original state tensors and 638,030,384 original values preserved exactly |
| New output rows and native decoder artifact | Verified before saving and after reloading |
| Fresh-process re-migration | The expanded checkpoint restores and saves again with no new additions; all existing values remain exact |
| Paired offline audio | 12 FLEURS recordings, 78.66 seconds, two each in English, Hindi, Tamil, Malayalam, Marathi and Kannada |
| Masked migration run | All 12 canonical token sequences and raw transcripts match the source |
| All new outputs enabled | All 12 canonical token sequences and raw transcripts still match the source |
| Captured joint states and old logits | Maximum absolute error 0, including independent fixed-input head replay |
| Native streaming simulation | One 5.76-second English clip, six cached chunks at a 1,120 ms setting; caches, old logits and token/text hypotheses match |
| Streaming with all new outputs enabled | All six chunk hypotheses match the source |
| Public offline inference command | Real audio succeeds with verified known-language and automatic prompt inputs |
| Real RNNT backward checks | Seven Indic languages pass through the actual Lhotse dataloader, native loss and prompt projection |

The backward checks use original FLEURS train recordings and new labels.
They verify finite loss and nonzero gradients on new prediction embedding and
output rows. They perform **zero optimizer steps** and save no trained weights.

| Language | Distinct new rows exercised | Prompt slot | Previously unused slot |
| --- | ---: | ---: | --- |
| Hindi | 13 | 6 | No |
| Tamil | 38 | 39 | No |
| Malayalam | 27 | 44 | No |
| Marathi | 3 | 41 | No |
| Kannada | 25 | 43 | No |
| Punjabi | 46 | 79 | Yes |
| Odia | 34 | 78 | Yes |

Having a prompt key in the source checkpoint does not establish pretrained
recognition support for that language. The new-language recordings above test
migration and gradient flow, not transcription accuracy.

## Problems found and corrected

The original constructor initialization made the new output rows dominate
decoding. All 12 unrestricted transcripts changed, often with repeated output.
That initial expanded checkpoint was rejected. New rows now start below the
learned blank output and remain independently trainable. On two real clips,
the old-label native RNNT loss changes by less than 0.000096 with additions
enabled; the masked loss is exactly equal. The mathematical probability bound
is qualified for floating-point arithmetic.

HF and native SentencePiece render unknown tokens differently. Before the
decoder fix, token/logit parity passed but raw text matched on only 5 of 12
clips. Retaining the native decoder semantics raises raw-text parity to 12 of
12 without changing HF encoding or canonical tokenizer bytes.

The run also required a callable adapter for NeMo's tokenizer wrapper, exact
custom-class registration for restoration, explicit waveform/prompt handling,
verification that CUDA graphs were disabled during hook-based comparisons,
and enum-safe report serialization.

One diagnostic run crashed during a timed Python stack dump after six
transcription calls. Its result was discarded. A fresh run of the public
paired-validation CLI without that timer completed all 36 calls and passed.
The native crash's cause was not established; these runs are not a runtime
stress or stability certification.

## Training-label difference that remains

The actual native tokenizer explicitly declares SentencePiece **Unigram**
(`model_type=1`). The published HF artifact declares **BPE**, with 3,621 merges.
Both facts were checked in the pinned artifacts, rather than inferred from
the NeMo class name.

For one real English train clip, native segmentation has 53 labels and both
the original and extended BPE have exactly the same 46 labels, using no added
IDs. The normalizer output and decoded text match. Native RNNT loss is 35.10
with the original labels versus 587.85 with the BPE labels. This is a
teacher-forcing/segmentation difference, not an English recognition-accuracy
measurement or a regression caused by the Indic additions.

For example, native Unigram encodes `a` as `▁` and `a`; BPE merges them into
`▁a`. Native can emit `▁which`, but the published BPE has no merge producing
that vocabulary entry and emits `▁w`, `h`, `ich`. Preserved IDs do not remove
these training-label differences. The approved BPE encoding remains unchanged.

## Scope still pending

No full fine-tuning, optimizer/resume test, all-locale WER/CER regression,
all-22-profile speech evaluation, mixed-precision test or production latency
benchmark was performed. The streaming test uses NeMo's buffer with precomputed
frontend features, not a live audio source. Keep separate held-out speech for
accuracy evaluation; these short clips have been used for debugging.

## Artifact identity

| Artifact | SHA-256 |
| --- | --- |
| Original NVIDIA checkpoint | `210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74` |
| Corrected expanded checkpoint | `dacc899172738d4bcdce34f18c1c31650a58d59083ed38f71074e87e49a1b719` |
| Original native decoder | `ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291` |
| Extended canonical BPE tokenizer | `ef619be3ac1f85de60a3e9ab8c9f2deaddcbcfd71c8c9f1a55d9a9fc26d6a9c4` |

The source model revision is `1c8deaecc64b91f034d73e08dd8b64625eb3395d`.
The NeMo revision is `ca4daa1470f6c01068c4e6a9a73b19b9a91dc366`.
The FLEURS revision is `70bb2e84b976b7e960aa89f1c648e09c59f894dd`.
Raw JSON reports, audio provenance and logs remain with the local run evidence;
model weights and corpus records are not committed to this repository.
