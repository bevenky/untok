# Native Unigram tokenizer

This variant extends the SentencePiece Unigram model embedded in NVIDIA's
`nemotron-3.5-asr-streaming-0.6b.nemo`. It is separate from the BPE variant built
from NVIDIA's published `tokenizer.json`.

The native source is pinned to
[revision 1c8deae](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d).
Its tokenizer SHA256 is
`ce3895e40806f02a26c3a225161b96ef682d6c0054bae32a245dec4258d7d291`.

## Selection

The shared Devanagari bank has 1,400 selected candidates. The shared Bengali
and Assamese bank has 500. Gujarati, Kannada, Malayalam, Odia, Punjabi,
Tamil, Telugu, Kashmiri, Urdu, Manipuri and Santali each have 500.

Required script characters count inside these budgets. The 103 approved Hindi
pieces count inside Devanagari. Shared strings and pieces already in the native
model do not consume additional IDs. The 190 approved rare Latin characters
are included once globally. No new Latin multi-character pieces are introduced.

Selection uses the original native vocabulary during fitting and pruning.
Old scores stay fixed. New scores must stay within the native score range.
Required pieces need encoding witnesses; optional additions need actual
training usage. Text efficiency is measured separately on held-out data.

The fitter maximizes a weighted segmentation objective with a fixed total
`sum(exp(new_score))` for additions. This anchors new scores against the
unchanged native scores; it is not a normalized language-model probability.
Forward-backward expected counts fit the scores, and deletion-loss pruning
reduces the candidate inventory to the quotas. Fresh and donor-seeded pools,
two seeds and two score-mass settings are compared on development text.
Identical clamped settings are reported as duplicates.

[Corpus preparation](native-unigram-data.md) describes the frozen training
and evaluation sets. The `untok.unigram_fit.NativeFitter` library accepts
explicit text records, candidate strings and scores, group memberships,
budgets and protected pieces. It performs no downloads. Text must not already
be SentencePiece-normalized. Supply `balance_language_source=False` when
records already carry the frozen source weights.

Candidate-pool training uses already normalized script spans and an identity
rule to avoid applying Unicode normalization twice. That temporary training
model is only a source of candidate strings. Final score fitting uses complete
raw lexical records with native normalization once, and the shipped model
retains the original native normalizer bytes.

## Build and use

Rebuild a candidate from its original model and scored selection:

```sh
untok build \
  --base candidate/base-tokenizer.model \
  --selection candidate/selection.json \
  --output artifacts/nemotron-indic-unigram-v1

untok check --bundle artifacts/nemotron-indic-unigram-v1
```

`check` verifies file hashes, preserved native metadata and the ID map.
It does not run corpus evaluation or speech recognition tests.

To run the full CPU tokenizer checks with your frozen text manifest:

```sh
untok validate \
  --bundle artifacts/nemotron-indic-unigram-v1 \
  --policy configs/native-unigram-validation.json \
  --corpora /path/to/frozen-data/manifest.json \
  --phase dev \
  --output reports/native-unigram-dev.json
```

The self-contained policy pins the native source, normalizer, alphabets and
protected pieces. A corpus manifest pins relative `dev/{language}.jsonl` and
`reserve/{language}.jsonl` files by SHA256. Each row needs a `text` string;
an optional `language` field must match the file's profile. Raw corpora are
not distributed in the tokenizer bundle. Missing or empty profile data gives
`incomplete` and exit code 2.

The older `build-unigram`, `check-unigram` and `validate-unigram` command names
remain aliases. Historical BPE commands require the explicit `legacy-bpe`
prefix; see [legacy notes](legacy-bpe.md).

Reserve evaluation uses `--phase reserve --selection-receipt receipt.json`.
Before examining reserved text, freeze a receipt containing `tokenizer_sha256`,
`data_manifest_sha256`, `bundle_manifest_sha256`, `selection_sha256` and
`policy_sha256`. The validator rejects mismatches and omits reserve text
examples from reports. A receipt records artifact identity; its hash alone
does not independently prove when selection happened.

Checks include actual encoding witnesses, preserved old-ID decoding, encoding
parity on probes with no added-piece match, normalizer equality and normalized
round trips. Unknown characters outside the declared finite coverage are
counted explicitly. These checks do not assert identical segmentation where
new pieces match or unchanged speech accuracy.

```python
from untok.unigram import NativeTokenizerAdapter

tokenizer = NativeTokenizerAdapter("artifacts/nemotron-indic-unigram-v1")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

These are native text IDs. For the public ID layout, use
`text_to_public_ids()` and `public_ids_to_text()` explicitly.

## Bundle files

| File | Contents |
| --- | --- |
| `tokenizer.model` | Extended native Unigram encoder and decoder |
| `base-tokenizer.model` | Exact original native tokenizer |
| `selection.json` | Ordered additions, fitted scores and provenance |
| `vocabulary.json` | Pieces, scores, types and both ID layouts |
| `nemo-id-map.json` | Public-to-native mapping, including RNNT blank |
| `manifest.json` | Hashes, structural checks and validation status |

Original native IDs `0..13086` are preserved. Public padding and blank remain
`13087` and `13088`, so public additions start at `13089`. Native additions
start at `13087`; native RNNT blank moves to the end of the expanded inventory.
Padding and blank are not inserted as SentencePiece text pieces.

## Limits

Normalizer 1A is unchanged, including its treatment of joiners. Source-specific
noise annotations may be removed from corpus text before fitting; their raw
forms and cleanup policy must be retained. Such cleaned partial transcripts
are not automatically suitable as complete paired audio training labels.

Preserved IDs and scores do not guarantee unchanged segmentation where new
pieces match, particularly in Hindi and Arabic-script text. An expanded
checkpoint must copy and remap its existing rows and add trainable rows for
new tokens. The tokenizer bundle alone does not perform that migration.

Use a matching tokenizer and checkpoint variant. Native migration now verifies
the source pin, native tokenizer, retained weights, relocated blank and complete
save/reload behavior. See [checkpoint usage](native-checkpoint.md) and
[compatibility results](native-compatibility-results.md) for the Full, Latin and
Latin-plus-Indic variants and their paired offline/streaming controls.

The current validation scope excludes acoustic training. A future trained
release would need a separately scoped speech corpus, training recipe and
held-out WER/CER evaluation. Text coverage and migration compatibility do not
establish learned recognition accuracy for newly added languages.
