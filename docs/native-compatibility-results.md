# Native compatibility results

Validation date: 7 September 2026. These checks use the pinned original
Nemotron 3.5 ASR streaming 0.6B checkpoint and its actual native Unigram model.
No acoustic training, model backward pass or optimizer step was performed
during these runtime checks.

## Tokenizer bundles

| Bundle | Native text IDs | Native acoustic outputs | Original text entries retained |
| --- | ---: | ---: | ---: |
| Latin | 2,916 | 2,917 | 2,664 |
| Latin plus Indic | 10,572 | 10,573 | 3,109 |
| Full | 20,550 | 20,551 | 13,087 |

Latin plus Indic retains all 7,463 approved additions. Latin retains 252 added
pieces, including the 190 rare Latin characters and shared characters. These
are not new Latin subword banks. Both reduced bundles preserve the native
normalizer, retained piece scores and score extrema. Full files remain
byte-identical to the frozen candidate. All three ZIP archives and packaging
receipts are byte-identical between macOS and Linux.

Latin plus Indic passed mapped-token and decoded-text comparisons on 33,027
development and 20,055 reserved records across all 22 profiles. Reserved
records contain no unknown tokens in the reduced bundle. These are the
previously evaluated frozen text sets, not additional independent holdouts.

Both reduced bundles also matched Full on all 647 English FLEURS reference
records in raw and publisher-normalized form, with no unknown tokens. The two
forms are the same underlying records, not independent samples.

## Checkpoint preservation

All three native checkpoints passed exact tensor verification before saving
and after restoring their `.nemo` archives. Each checked all 657 state tensors.

| Bundle | Original tensor values retained | Original tensor values deliberately omitted |
| --- | ---: | ---: |
| Full | 638,030,384 | 0 |
| Latin | 624,678,521 | 13,351,863 |
| Latin plus Indic | 625,248,566 | 12,781,818 |

Removed values belong to omitted vocabulary rows. All shared acoustic tensors
and retained vocabulary rows are exact copies. Blank is explicitly relocated;
every saved tokenizer artifact and prompt assignment is checked after reload.
The source checkpoint and tokenizer files are not rewritten.

## Full checkpoint audio checks

The Full checkpoint passed 80 offline comparisons across all 40 original
prompt settings. Original, old-output-only and all-output decoding produced
identical native token sequences and text. The 640 captured audio-derived
joint probes and their fixed-state replays had zero old-logit error.

Streaming also passed all 40 prompt settings, using one recording per setting
and 292 source encoder chunks. Mapped tokens, partial text, input chunks and
encoder caches matched with both restricted and unrestricted added outputs.
Captured old logits had zero error.

Thirty-three source locale profiles have matching published FLEURS locale
recordings; seven use explicitly recorded regional/language proxies. The
original model describes eight of its 40 locales as adaptation-ready. The
checks validate prompt paths and migration behavior on the supplied recordings;
they do not establish accuracy across every dialect or possible recording.

Separate inference completed on 44 recordings covering all 22 Indic profiles.
FLEURS supplies 13 expected-script profiles and IndicVoices supplies nine.
All source references contain zero unknown tokens in the new tokenizer.
Successful inference is not evidence that untrained additions have learned
new-language recognition.

## Reduced checkpoint audio checks

Both reduced checkpoints passed offline and streaming comparisons against
the original checkpoint with removed output rows masked. Added rows were
tested both masked and enabled in the reduced model.

| Bundle | Locale paths in each mode | Streaming chunks | Largest retained raw-logit difference |
| --- | ---: | ---: | ---: |
| Latin | 29 | 197 | 0.00048828125 |
| Latin plus Indic | 52 | 336 | 0 |

Mapped token sequences, decoded text and streaming encoder caches matched
their controls for every recording. Latin's small floating-point differences
passed the declared `rtol=1e-5`, `atol=1e-6` comparison; its raw logits are not
bit-identical. Latin plus Indic had zero retained-logit error.

The 52 paths include the Latin settings, all 22 Indic profiles and retained
Arabic. Twelve Indic prompt identities are absent from the original model;
those comparisons use the same explicit `auto` prompt on both models. The
new requested prompt is exercised separately as an inference diagnostic.

Pruning does not guarantee the unrestricted original prediction is preserved.
On one Odia recording, the original model emitted Greek pieces under `auto`.
Removing Greek outputs changed that prediction. The reduced model exactly
matched the original with those removed outputs masked. All 29 Latin offline
predictions also matched the unrestricted original.

## Findings

The original tokenizer remains available unchanged as `base-tokenizer.model`.
The extended tokenizer does not promise identical encoding wherever an added
piece can match. Hindi changes were expected. The audio-reference text check
also found one changed Arabic segmentation because Urdu/Kashmiri additions
share that script: `ائي` + `ة` became `ائ` + `ية`, with the same decoded text
and total token count. French and Maltese references gained coverage for
characters that were unknown in the base model. These changes did not change
the corresponding Full checkpoint audio hypotheses in this evaluation.

Reduced-model packaging initially wrote implicit protobuf special-ID defaults
explicitly. The builder now preserves field presence unless an ID changes.
This changed serialization hashes, not vocabulary selection or scores. The
final models were rebuilt and their text comparisons repeated.

An initial reduced streaming comparison failed because NeMo's transcription
teardown re-enabled training mode in encoder, decoder and joint submodules
after restoring the parent mode. The first divergence was in encoder caches,
before the tokenizer output head. Explicit recursive evaluation mode is now
required before streaming; the public native transcription helper also resets
it after offline transcription. The rerun restores fresh checkpoints so
in-memory buffers from the failed harness cannot carry forward. No optimizer
or backward pass was involved.

See [usage and reproduction commands](native-checkpoint.md). Detailed reports,
audio provenance and checkpoint receipts are distributed separately from Git.
They include file hashes and explicit coverage limits.
