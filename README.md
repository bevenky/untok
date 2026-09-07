# untok

Extend the native SentencePiece Unigram tokenizer from
[NVIDIA Nemotron 3.5 ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
with Indic characters and subwords. Keep the original normalizer and package
the vocabulary you need.

## Install

Use Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

## Choose a bundle

| Bundle | Text IDs | Includes |
| --- | ---: | --- |
| `latin` | 2,916 | Latin pieces, shared punctuation and special tokens |
| `latin-indic` | 10,572 | Latin plus all 22 target Indic profiles |
| `full` | 20,550 | Original Nemotron vocabulary plus all approved additions |

The full bundle preserves all 13,087 original entries and their IDs and scores.
Reduced bundles use compact IDs and explicit checkpoint row maps. For ASR,
each needs a matching migrated checkpoint with one additional RNNT blank output.
New Indic pieces require speech fine-tuning before recognition quality can be
claimed for the added languages.

Extract the separately supplied `full.zip`, then create all three bundles and
ZIP archives:

```sh
unzip full.zip -d artifacts/nemotron-indic-unigram-v1
untok package --bundle artifacts/nemotron-indic-unigram-v1 --output dist
```

To package just one, add `--profiles latin-indic`. Generated tokenizer bundles
are distributed separately from the Git repository.

## Use the tokenizer

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("dist/latin-indic")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

These are native text IDs for the matching model. Public padding and blank IDs
have a separate mapping; use the adapter's explicit public-ID methods when needed.

## Indic coverage

Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Konkani, Kashmiri
(Arabic), Maithili, Malayalam, Manipuri (Meetei Mayek), Marathi, Nepali, Odia,
Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi (Devanagari), Tamil,
Telugu and Urdu.

Devanagari is shared across its languages, and Bengali and Assamese share a
bank. Text coverage does not imply support for every alternate script or dialect.

See [checkpoint usage](docs/native-checkpoint.md) and
[source provenance](THIRD_PARTY.md).
