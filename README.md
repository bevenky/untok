# untok

A SentencePiece Unigram tokenizer for speech-to-text (ASR) and TTS models. The Latin vocabulary is based on NVIDIA Nemotron, extended with Indic language support.

## Install

On Linux or macOS, clone the repository:

```sh
git clone https://github.com/bevenky/untok.git
cd untok
```

Install with [uv](https://docs.astral.sh/uv/getting-started/installation/).
It manages the Python environment for you:

```sh
uv sync --locked
```

Or use pip with Python 3.11 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Choose a bundle

All three bundles are included. Choose one when loading the tokenizer.

| Bundle | Text IDs | Includes |
| --- | ---: | --- |
| `latin` | 2,916 | Latin pieces, shared punctuation and special tokens |
| `latin-indic` | 10,572 | Latin plus all 22 target Indic profiles |
| `full` | 20,550 | Original Nemotron vocabulary plus all approved additions |

The full bundle preserves the original 13,087 entries, IDs and scores. The smaller
bundles use compact IDs.

## Use the tokenizer

Save this as `example.py`. Run `uv run example.py`, or `python example.py` if
you installed with pip:

```python
from untok.bundles import load_tokenizer

tokenizer = load_tokenizer("latin-indic")
ids = tokenizer.text_to_ids("நான் office போகிறேன்")
print(ids)
print(tokenizer.ids_to_text(ids))
```

Use `"latin"` or `"full"` to select another bundle. You can also pass the path to
a custom bundle. The example uses native text IDs.

## Languages

These counts describe tokenizer text coverage. Regional variants count once;
Norwegian includes Bokmål and Nynorsk.

**`latin`: 24 languages.** Croatian, Czech, Danish, Dutch, English, Estonian,
Finnish, French, German, Hungarian, Italian, Latvian, Lithuanian, Maltese,
Norwegian, Polish, Portuguese, Romanian, Slovak, Slovenian, Spanish, Swedish,
Turkish and Vietnamese.

**`latin-indic`: 47 languages.** All 24 Latin languages above, Arabic, and 22 Indic languages below:
Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri (Arabic script),
Konkani, Maithili, Malayalam, Manipuri (Meetei Mayek), Marathi, Nepali, Odia,
Punjabi (Gurmukhi), Sanskrit, Santali (Ol Chiki), Sindhi (Devanagari),
Tamil, Telugu and Urdu.
Arabic is retained because its script is shared with Kashmiri and Urdu.
Devanagari is shared across its languages; Bengali and Assamese share a vocabulary bank.

**`full`: 56 languages.** All 47 languages above, plus Bulgarian, Greek, Hebrew,
Japanese, Korean, Mandarin Chinese, Russian, Thai and Ukrainian.

## Limitations

- Text coverage does not mean a model can recognize or generate speech in those
  languages.
- Using this with an existing Nemotron model requires a matching migrated checkpoint; new pieces need
  speech training before the model can use them reliably. See [checkpoint usage](docs/native-checkpoint.md).
- Nemotron's original normalization is retained, including its joiner handling.
  Decoding may not reproduce the raw input exactly. Keeping original IDs does not guarantee identical
  segmentation or speech accuracy. Added pieces can change segmentation, including Hindi.
- The vocabulary is finite. Uncovered characters or emoji can produce `<unk>`;
  alternate scripts and every dialect are not validated.
- Selecting a bundle selects a vocabulary, not an inference language lock.

See [source provenance](THIRD_PARTY.md) for tokenizer and data sources.
