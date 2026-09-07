# sttok

Extend the BPE tokenizer from
[NVIDIA Nemotron 3.5 ASR streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
with selected AI4Bharat IndicConformer pieces. The repository also includes
NeMo checkpoint migration and speech validation tools. The output is a
`tokenizer.json` file for the Hugging Face `tokenizers` library.

The current tokenizer has **16,481 IDs**. Original IDs `0` through `13,088`
are preserved, and additions start at `13,089`. It keeps NVIDIA's normalizer
and original merge rules, with 103 additional Hindi pieces and 190 rare Latin
characters. No Latin subword merges are added.

The tokenizer is built and text-tested. Real checkpoint migration, 12 paired
audio checks, a small streaming check and seven-language gradient checks pass.
See [measured results](docs/runtime-results.md). Fine-tuning and multilingual
speech accuracy validation remain pending. Adding tokens alone does not teach
the model to recognize new languages.

## Build

Use Python 3.11 or later. Run these commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
sttok fetch
sttok build
sttok id-map
sttok prompts
```

`fetch` downloads the pinned tokenizer inputs and checks their SHA-256 hashes.
The remaining commands write the tokenizer, build manifest, NeMo ID mapping
and language prompt registry to `artifacts/nemotron-indic-v1/`.

## Use the tokenizer

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("artifacts/nemotron-indic-v1/tokenizer.json")
encoded = tokenizer.encode("நான் office போகிறேன்")
print(encoded.ids)
print(tokenizer.decode(encoded.ids))
```

The 22 target text profiles are Assamese, Bengali, Bodo, Dogri, Gujarati,
Hindi, Kannada, Konkani, Kashmiri (Arabic), Maithili, Malayalam, Manipuri
(Meetei Mayek), Marathi, Nepali, Odia, Punjabi (Gurmukhi), Sanskrit, Santali
(Ol Chiki), Sindhi (Devanagari), Tamil, Telugu and Urdu. Alternate scripts
and arbitrary Unicode coverage are not implied.

The base normalizer is unchanged, including its ZWNJ-to-space behavior.
Added Hindi pieces can change segmentation in Hindi and other Devanagari
languages. Preserving old IDs does not guarantee unchanged recognition accuracy.

## Files and folders

| Path | Contents |
| --- | --- |
| `src/sttok/` | Tokenizer builder, CLI, checkpoint adapter and validation code |
| `configs/` | Source hashes, build settings, character lists and corpus manifests |
| `tests/` | Tokenizer, migration and evaluation tests |
| `scripts/` | Small real-audio training and streaming checks |
| `docs/` | Validation results, data formats and checkpoint instructions |
| `.github/workflows/` | Automated CPU tests |
| `artifacts/nemotron-indic-v1/` | Generated tokenizer and integration files |
| `.cache/` | Downloaded inputs and local corpus snapshots |
| `reports/` | Generated validation reports |

The generated artifact folder contains:

- `tokenizer.json`: the extended NVIDIA BPE tokenizer used in the example above.
- `manifest.json`: source hashes, token provenance and build checks.
- `nemo-id-map.json`: mapping from tokenizer IDs to NeMo model IDs.
- `prompts.json`: existing and added language prompt assignments.

Generated artifacts, caches, reports and model weights are excluded from Git.
Some corpus manifests refer to local audit snapshots that are not bundled.
See [source provenance](THIRD_PARTY.md) for upstream sources and licenses.

## Validate and integrate

To run the full local test suite, install the test and checkpoint dependencies:

```sh
python -m pip install -e '.[test,checkpoint]'
python -m pytest -q
```

The recorded build passes standard-alphabet checks for all 22 profiles. The
latest complete test run passed 171 tests, including the optional check against
the actual native tokenizer artifact. See [build evidence](docs/build-evidence.md) for corpus counts,
exclusions and remaining work, and [text validation](docs/validation.md)
for corpus commands and checks before extending a previous release.

To use the tokenizer with Nemotron, follow the
[checkpoint migration instructions](docs/checkpoint.md). The native model
uses a different padding/blank layout, so replacing its tokenizer file alone
is insufficient. Migration needs the complete checkpoint and a compatible
NeMo installation, which the `checkpoint` extra does not install. Migration
and small inference checks can run on CPU; substantial training should use GPU.

[Checkpoint verification](docs/checkpoint-validation.md) compares weights and
real audio outputs. [Speech evaluation](docs/evaluation.md) describes
per-language accuracy checks. See [real runtime checks](docs/runtime-validation.md)
for the development audio, training and streaming probes. The general inference
command is offline.

NVIDIA's published JSON uses BPE; the native `.nemo` embeds a SentencePiece
Unigram tokenizer. They share old piece IDs but can produce different training
labels. This project keeps BPE encoding and preserves native text decoding in
the migrated checkpoint. Continued training must account for that difference.
