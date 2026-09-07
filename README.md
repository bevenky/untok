# sttok

Extend the native SentencePiece Unigram tokenizer in
[NVIDIA Nemotron 3.5 ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
with Indic characters and subwords.

The current candidate has **20,550 native text IDs**, including **7,463
additions**. It preserves all 13,087 original entries, their scores and the
native normalizer. Public padding and blank bring the public inventory to
20,552 IDs.

Tokenizer validation is complete. Native checkpoint migration, fine-tuning
and speech accuracy tests are still pending. New pieces can change Hindi
segmentation; preserved IDs alone do not guarantee unchanged recognition.
See [validation results](docs/native-unigram-results.md).

## Install

Use Python 3.11 or later, from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Use the tokenizer

Generated tokenizer files are excluded from Git. A clean clone needs the
candidate bundle separately. Copy the six files from its `tokenizer/` folder
into `artifacts/nemotron-indic-unigram-v1/`, then verify them:

```sh
sttok check --bundle artifacts/nemotron-indic-unigram-v1
```

```python
from sttok.unigram import NativeTokenizerAdapter

tokenizer = NativeTokenizerAdapter("artifacts/nemotron-indic-unigram-v1")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

These are native text IDs. Use `text_to_public_ids()` and
`public_ids_to_text()` for the public layout, which reserves padding and blank
at 13,087 and 13,088. New public IDs start at 13,089.

Rebuild the same tokenizer from the bundle's original model and scored
selection, writing to a new directory:

```sh
sttok build \
  --base artifacts/nemotron-indic-unigram-v1/base-tokenizer.model \
  --selection artifacts/nemotron-indic-unigram-v1/selection.json \
  --output artifacts/rebuilt-native
```

This packages the selected pieces and scores. It does not repeat corpus
selection or train an acoustic model.

## Language coverage

The 22 target text profiles are Assamese, Bengali, Bodo, Dogri, Gujarati,
Hindi, Kannada, Konkani, Kashmiri (Arabic), Maithili, Malayalam, Manipuri
(Meetei Mayek), Marathi, Nepali, Odia, Punjabi (Gurmukhi), Sanskrit, Santali
(Ol Chiki), Sindhi (Devanagari), Tamil, Telugu and Urdu.

The selected banks use 1,400 memberships for shared Devanagari, 500 for shared
Bengali/Assamese and 500 for each remaining language, including separate
Kashmiri and Urdu quotas. Shared strings and native overlap reduce new IDs.
Text coverage does not imply trained speech support or every alternate script.

## Files and folders

| Path | Contents |
| --- | --- |
| `src/sttok/unigram*.py` | Native builder, fitter, adapter and validator |
| `configs/` | Pinned inputs, approved pieces and validation policy |
| `tests/` | Native tests and retained legacy regression checks |
| `docs/` | Methods, measured results and checkpoint work still pending |
| `artifacts/` | Generated bundles, excluded from Git |
| `.cache/` | Local source snapshots, excluded from Git |
| `reports/` | Generated validation reports, excluded from Git |
| `scripts/` | Legacy BPE audio probes |

See [native usage and validation](docs/native-unigram.md),
[data preparation](docs/native-unigram-data.md) and
[source provenance](THIRD_PARTY.md).

## Tests

```sh
python -m pip install -e '.[test]'
python -m pytest tests/test_unigram.py tests/test_unigram_fit.py tests/test_unigram_validation.py -q
```

`sttok validate` runs the native checks against an explicit policy and corpus
manifest. Missing corpus evidence is reported as incomplete. Reserved text
requires a receipt binding the finalized model and evaluation inputs.

## Earlier BPE work

Unigram is the active workflow. Earlier BPE code remains in this repository
for reproducing past experiments and reusing tested checkpoint utilities.
Its commands require `sttok legacy-bpe`; see [legacy BPE notes](docs/legacy-bpe.md).
BPE and Unigram must each use a matching checkpoint, not an interchangeable
runtime setting.
