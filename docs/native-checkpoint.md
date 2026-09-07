# Native checkpoint usage

Use the native `.nemo` checkpoint from
`nvidia/nemotron-3.5-asr-streaming-0.6b`, revision
`1c8deaecc64b91f034d73e08dd8b64625eb3395d`.

Expected checkpoint SHA256:
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`.
The embedded SentencePiece model is Unigram. NeMo's class names use `BPE`
for this SentencePiece integration, but that does not determine the model's
actual segmentation algorithm.

## Package and migrate

Tokenizer packaging runs on CPU with the normal installation:

```sh
untok package --bundle artifacts/nemotron-indic-unigram-v1 --output dist
untok check --bundle dist/latin-indic
```

Migration requires the compatible NVIDIA NeMo Speech runtime. The verified
environment uses Python 3.12, PyTorch 2.8.0 with CUDA 12.8, SentencePiece 0.2.1
and NeMo Speech revision `ca4daa1470f6c01068c4e6a9a73b19b9a91dc366`.
The `checkpoint` extra supplies tensor utilities, not the complete NeMo stack.

```sh
untok migrate \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --source-sha256 210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74 \
  --bundle dist/latin-indic \
  --output nemotron-latin-indic.nemo
```

Choose `dist/full` or `dist/latin` to migrate the other bundles. The destination
and migration report must not exist. The source checkpoint is never overwritten.

Migration verifies the source hash, actual native tokenizer bytes, every
retained tensor row, new-row initialization, saved tokenizer artifacts and
restored checkpoint. The adjacent `.migration.json` records the checks and
the explicit source-to-target row map. A removed source text row maps to `null`;
the original RNNT blank always maps to the target's final output row.

Restore a migrated checkpoint with the installed `untok` package:

```python
from untok.native_runtime import get_native_nemo_model_class, transcribe_native_file

model = get_native_nemo_model_class().restore_from(
    "nemotron-latin-indic.nemo", map_location="cuda"
)
hypotheses, prompt_evidence = transcribe_native_file(
    model, "speech.wav", target_lang="hi-IN"
)
print(hypotheses[0].text)
```

This helper verifies the language prompt actually supplied to the model. The
pinned NeMo file-list loader can select a unified prompt, so this path loads
the file as tensor audio and sets the requested prompt explicitly. It also
leaves all model layers in evaluation mode. NeMo's transcription teardown can
otherwise re-enable training mode in submodules before a later streaming call.

## Language preferences and choosing two languages

Language preferences belong in model inference and decoding, not in
SentencePiece normalization or piece scores. Each checkpoint has one shared
output vocabulary. A language prompt changes the model's conditioning without
automatically masking other languages' pieces.

The current helper accepts one `target_lang` string. For example,
`target_lang="en-US"` uses English prompt slot 0, and `target_lang="hi-IN"`
uses Hindi slot 6. `target_lang="auto"` selects slot 101; it is an explicit
conditioning mode, not an absent or all-zero prompt. The default greedy
decoder chooses the highest-scoring output rather than sampling randomly.

| Desired behavior | Where it belongs | Available now |
| --- | --- | --- |
| Nemotron's learned English preference | Pass `target_lang="en-US"` to the inference helper | Yes |
| An adjustable additional English preference | A decoder policy that adjusts token scores before selection | No |
| Restrict available pieces to an English-Hindi set | A decoder policy that excludes pieces outside the permitted set | No |

An additional decoder policy could be exposed through
[`transcribe_native_file`](../src/untok/native_runtime.py), with its scoring
logic in a separate decoder module. It would operate at runtime without
rewriting the tokenizer or checkpoint weights. It would need validation for
both offline and streaming decoding, including any enabled CUDA graph path.
The masks in the existing checkpoint tests are validation controls, not a
supported language-lock feature.

For an English-Hindi option, keep the model prompt and the permitted output
set separate. A future `allowed_languages=["en-US", "hi-IN"]` setting could
be paired with `target_lang="auto"`, or with one primary language such as
`"hi-IN"`. This setting is a proposal and is not an accepted argument today.
Do not pass a list to `target_lang` or activate two one-hot positions; the
current interface and checkpoint expect one prompt index.

A practical output filter could keep Latin and Devanagari pieces, shared
punctuation, numbers, word-boundary pieces and RNNT blank. Output language tags
need explicit handling as well. This is a script restriction, not a guarantee
of English and Hindi wording: Latin also represents French and romanized
Hindi, while Devanagari is shared with Marathi, Nepali and other languages.
Shared pieces must not be assigned exclusively to one language.

## What preservation means

Full keeps all original text IDs, weights and native normalization. Added
Unigram pieces may still change encoding, including Hindi and shared-script
Arabic text. Reduced bundles compact retained text IDs, copy the corresponding
predictor/output rows, and remove other rows. Their retained raw logits can
stay equal while their softmax probabilities and unconstrained predictions change.

The full tokenizer has 20,550 text IDs; Latin has 2,916; Latin plus Indic has
10,572. Native RNNT adds one blank output. Public padding/blank layouts are
separate and must never be used directly as native training labels.

New output weights initially copy the original blank row with a lower bias.
The combined new-output mass is bounded relative to retained old outputs by
`1e-6`, before training. New predictor rows start from the mean retained text
embedding. These rows remain independent and trainable.

The original prompt slots stay fixed. Missing target identities receive unused
slots in the existing prompt dimension. This enables the conditioning path;
it does not teach a new language or impose an output-language mask.

## Using newly added Hindi pieces

Migration makes the new IDs valid decoder outputs, but does not teach the
model when to emit them. This applies to new pieces for Hindi even though the
original checkpoint already recognizes Hindi through its original vocabulary.

For example, the actual native tokenizers encode `भारत` as follows. The `▁`
symbol marks a word boundary.

| Tokenizer | Pieces |
| --- | --- |
| Original | `▁`, `भा`, `र`, `त` |
| Extended | `▁भारत` |

The extended text tokenizer can select `▁भारत` immediately. The acoustic
model's new output row has no learned association with that word yet. Under
the current initialization, new rows have the blank row's weights and a lower
bias, so ordinary greedy decoding will not select them without subsequent
weight changes or an additional scoring intervention. Changing the language
prompt does not close that score gap. Simply boosting a new row's score would
not teach it the word's acoustic meaning.

To learn meaningful use of the new pieces, a future checkpoint fine-tuning
step would use speech transcripts encoded by the updated tokenizer. This
teaches the new predictor and output rows; it is not another tokenizer change.
The Full and Latin-plus-Indic checkpoints can continue emitting their original
Hindi pieces. The Latin-only checkpoint excludes Devanagari. Passing
compatibility checks establishes preservation on the tested inputs, not that
the new pieces have been learned.

## Reproduce speech integration checks

`scripts/native_checkpoint_probe.py` runs bounded `offline`, `streaming` and
`inference` modes. These checks do not train or update model weights.
Supply the original and migrated checkpoint,
its migration receipt, an immutable audio JSONL manifest and a new report path.

```sh
python scripts/native_checkpoint_probe.py offline \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --checkpoint nemotron-full.nemo \
  --migration nemotron-full.migration.json \
  --manifest audio-evaluation.jsonl \
  --output reports/native-offline.json
```

`offline` compares original, old-output-only and all-output greedy predictions
with real audio-derived joint states. `streaming` checks cached encoder chunks
and carried hypotheses. Both require a Full migration with all source rows.

`inference` exercises explicit target prompts and reports raw diagnostic WER/CER.
Its success flag means inference executed, not that recognition accuracy passed.
Small samples and regional proxy recordings do not establish all-locale accuracy.

Use `scripts/native_subset_probe.py` for reduced checkpoints:

```sh
python scripts/native_subset_probe.py --mode both \
  --source nemotron-3.5-asr-streaming-0.6b.nemo \
  --checkpoint nemotron-latin-indic.nemo \
  --migration nemotron-latin-indic.migration.json \
  --manifest audio-evaluation.jsonl \
  --output reports/native-subset.json
```

This compares the reduced model with the original model's retained outputs.
The manifest should include the desired locale paths and immutable audio hashes.
Removed output rows can change unrestricted predictions even when every
retained row was copied exactly.

## Work required for a trained release

Fine-tuning is a separate task, outside compatibility validation.

Use the Full migration for the initial continuation-training experiment, with
new-language speech and replay from original languages. Keep speaker-disjoint
development and test audio, preserve complete transcript labels, and record
the training hours and source mix per language. Tune learning rate, sampling
and stopping only on development audio.

Evaluate final WER/CER against the original checkpoint on sufficiently large
held-out sets for the original locales and the 22 new text profiles. Include
conversation, read speech, code-switching and streaming. Report absent dialects
or scripts explicitly. A few optimizer steps establish the training path but
cannot replace this fine-tuning and accuracy evaluation.
