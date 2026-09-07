# Legacy BPE tokenizer validation

These instructions reproduce the BPE experiment. For the active native
tokenizer, see [Unigram usage and validation](native-unigram.md).

## Run the text audit

After `sttok legacy-bpe fetch` and `sttok legacy-bpe build`, run from the repository root:

```sh
sttok legacy-bpe fetch-corpora
sttok legacy-bpe validate --corpora .cache/bhasha/manifest.json --output reports/bhasha-full.json
sttok legacy-bpe scope-corpora --corpora .cache/bhasha/manifest.json --output .cache/scoped-bhasha
sttok legacy-bpe validate --corpora .cache/scoped-bhasha/manifest.json --output reports/bhasha-scoped.json
```

Keep both the full and scoped reports. The full audit exposes unsupported
characters. Scope filtering excludes whole records with a recorded reason;
it does not remove individual characters to make a record pass.

Bhasha covers 21 declared profiles and lacks Devanagari Sindhi. These commands
alone therefore cannot pass the all-22 requirement. Add separately sourced
Devanagari Sindhi text using the manifest format below. The manifests in
`configs/corpora-independent-all22.json` and `configs/corpora-sindhi.json`
describe local audit snapshots, with file hashes and source notes. Those
snapshots are not bundled or downloaded by `fetch-corpora`.

`validate` exits 0 only for `passed_on_supplied_text`; failed or incomplete
evidence exits 2. See [build-evidence.md](build-evidence.md) for the recorded
results, exclusions and limitations of the current candidate.

## Extend a previous release

Preserve the previous artifact directory and pass it to the builder:

```sh
sttok legacy-bpe build --previous /path/to/previous-release --output /path/to/new-release
sttok legacy-bpe id-map --tokenizer /path/to/new-release/tokenizer.json --output /path/to/new-release/nemo-id-map.json
sttok legacy-bpe prompts --previous /path/to/previous-release/prompts.json --output /path/to/new-release/prompts.json
```

Existing IDs and merges are validated before additions are appended. The
previous prompt registry preserves existing assignments. Keep the source locks
and previous artifacts with each release; rebuilding from an updated donor list
alone does not establish compatibility.

## Validation API

`sttok.validation.validate_tokenizer(base_path, extended_path, manifest_path,
corpora=None, required_targets=None, max_examples=20)` returns a JSON-compatible
report. It does not write files. The CLI can save that report separately.

The three artifact paths name the pinned original HF BPE JSON, the built HF BPE
JSON and the builder manifest. Every original ID, special-token record, model
setting, normalizer and other text-processing component must stay unchanged.
Existing merge rules must be an exact prefix. New Latin/mixed-Latin merges and
new merges containing only shared symbols/digits are rejected.
Declared artifact hashes and vocabulary/merge counts are checked against the
files actually loaded. When present, the Hindi103 and Latin190 inventory fields
are checked independently for their stated size, new-entry membership and Latin
character normalization/round-trip behavior.

The builder manifest supplies `entries` with `piece`, `id`, `witness` (raw text)
and optionally `reachable`. The validator actually encodes each witness and
requires its new ID in the result. Merely storing a piece or presenting a merge
graph does not establish reachability. Every new ordinary entry needs a witness.
An optional `required_characters` list accepts character strings or objects with
`character`, `language`, `script` and an optional explicit `expected` value.
Optional `normalizer_controls` contain `text` and an explicit `expected` value.

## Corpus inputs

Pass a list of corpus specifications, or the path to a JSON manifest containing
that list under `corpora`. Relative paths are resolved against the manifest's
directory. For an inline list they are resolved against the working directory.

```json
{
  "corpora": [
    {
      "path": "heldout/ta.jsonl",
      "format": "jsonl",
      "language": "ta",
      "script": "Taml",
      "source": "Publisher, dataset revision and split",
      "sha256": "the full SHA256 of this exact local file",
      "text_field": "text",
      "expected_field": "expected_text"
    }
  ]
}
```

Supported formats are JSONL, CSV, JSON arrays (or objects with a `records` list),
and UTF-8 text with one record per line. Text-file record delimiters are removed;
other whitespace is preserved. JSONL/CSV/JSON rows can override `language` and
`script`. Both are required. A row can supply an independently reviewed
`expected_text`; an empty explicit expected string is respected.
Set `language_field: null` and `script_field: null` to use fixed manifest values
when source rows contain incompatible display names instead of language/script
codes. These fields can also name alternative columns in a multilingual corpus.
Known display names such as `Tamil`/`Tamil` are canonicalized to `ta`/`Taml`;
report keys use language codes and ISO15924 script codes. Distinct scripts remain
distinct requirements.

`expect_unchanged: true` additionally requires exact base/extended token IDs for
every record in that corpus. Use it for protected existing text, not for the
approved Hindi sequence changes or newly covered scripts. Source hashes are
checked when supplied. A mismatch fails validation. Without an input hash the
report records the hash actually tested and warns that its provenance was not
verified. The validator cannot prove dataset independence, licensing or lack of
training/evaluation leakage from a URL or publisher label.

Keep training transcripts, independent held-out prose, original-language text,
code-switched text and synthetic edge cases in separately identified sources.
Freeze held-out manifests before fitting or selecting additional pieces. Retain
raw reference text so cleanup cannot silently redefine a failing test as a pass.

## Expected text and loss metrics

For this project's Metaspace pipeline, the automatic reference applies the
**base** normalizer, maps spaces to the configured boundary marker, applies the
prefix convention, then renders boundaries as spaces. This reference calculation
does not call tokenization or decoding. It deliberately preserves the existing
pipeline's treatment of leading/repeated/trailing whitespace. Other pipelines
need explicit expected text rather than an assumed generic decoder inverse.

Normalization equality is checked separately. Thus an accidental extension-side
normalizer change cannot hide inside its own round-trip reference. For
joiner-bearing target spellings, explicit reviewed references additionally test
whether the chosen 1A spelling convention is suitable; a compatibility pass alone
does not make the convention linguistically desirable.

Unknown tokens are not sufficient coverage evidence: `fuse_unk` can collapse five
uncovered characters into one token. The report separately counts Unicode
characters in unknown spans from the BPE model's UTF-8 offsets after normalization
and pre-tokenization. It reports the missing inventory and occurrence counts.
This diagnostic runs before special-token interception; use ordinary transcript
text for corpus loss metrics and the separate special-token parity probes for
literal special strings. No raw character is treated as an audio prediction.

Round-trip checks run on records without unknown tokens. Records with unknowns
already fail coverage and are explicitly excluded from the eligible round-trip
count. Explicit expected references are preferred for important linguistic and
whitespace cases. Token-length and fertility statistics are reported separately:
mean, p50, p95, p99 and maximum; increased compression is not claimed as an ASR
accuracy improvement.

## Language/script completeness and result status

The default requirement is all 22 donor language/script pairs: Assamese/Bengali,
Bengali/Bengali, Bodo/Devanagari, Dogri/Devanagari, Gujarati/Gujarati,
Hindi/Devanagari, Kannada/Kannada, Konkani/Devanagari, Kashmiri/Arabic,
Maithili/Devanagari, Malayalam/Malayalam, Manipuri/Meetei Mayek,
Marathi/Devanagari, Nepali/Devanagari, Odia/Odia, Punjabi/Gurmukhi,
Sanskrit/Devanagari, Santali/Ol Chiki, Sindhi/Devanagari, Tamil/Tamil,
Telugu/Telugu and Urdu/Arabic. The builder may supply `expected_targets`, and
the caller can override it with `required_targets` for a deliberately scoped run.

An Arabic-Sindhi corpus does not satisfy the Devanagari-Sindhi requirement.
Missing or empty corpora produce `missing_corpus`, not a zero-error success.
Any failed structural, witness, character, reference or protected-text check
makes the overall status `failed`. Otherwise absent required corpora or no corpus
run makes it `incomplete`. Only a completed supplied-text run can return
`passed_on_supplied_text`.

Passing one small file per language is still evidence about those files only.
This status does not certify the completeness or representativeness of a corpus.
The report always states `asr_validation: not_run`: checkpoint mapping, weight
preservation, masked model equivalence, training and per-language held-out audio
evaluation remain separate release gates.

## Tests

`tests/test_validation.py` builds small actual BPE models and checks valid
extensions, ID corruption, changed normalization, dead vocabulary entries,
unwanted Latin merges, fused unknown spans, explicit reference failures,
language/script gaps, input formats, pinned hashes and deterministic read-only
execution. These tests do not substitute for running the same validator against
the finished NVIDIA/Indic artifact and declared corpora.
