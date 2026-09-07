# Speech evaluation contract

The scorer consumes actual predictions from the original NVIDIA model, the
expanded untrained model, and the fine-tuned model. It does not run models,
download audio, train a checkpoint, or turn tokenizer coverage into an ASR claim.
The checkpoint migration tests separately establish tensor and decoding parity.

Use `untok.evaluation.evaluate_predictions(manifest, predictions)` and
`write_report(report, path)`; the loaders accept a JSON manifest and JSONL
predictions. The return value is a JSON-serializable, deterministic report.

## Manifest

This small **development fixture** illustrates the schema. Development reports
are always blocked from release, even when every prediction is perfect.

```json
{
  "schema_version": 1,
  "purpose": "development",
  "normalization": {
    "unicode_form": "none",
    "casefold": false,
    "punctuation": "preserve",
    "cer_remove_spaces": false
  },
  "profiles": [
    {
      "id": "hi-IN-Deva",
      "locale": "hi-IN",
      "language": "hi",
      "script": "Deva",
      "cohorts": ["existing_asr", "indic"],
      "protected": true,
      "required_conditions": ["known_language_streaming_1120ms"]
    }
  ],
  "utterances": [
    {
      "id": "heldout-hi-0001",
      "profile_id": "hi-IN-Deva",
      "condition_id": "known_language_streaming_1120ms",
      "reference": "नमस्ते",
      "audio": "/path/to/heldout-hi-0001.wav",
      "audio_sha256": "replace_with_actual_sha256",
      "dataset": "actual_dataset_name_and_revision",
      "split": "test",
      "cluster_id": "speaker-or-recording-session-001"
    }
  ]
}
```

For each manifest utterance, supply exactly one row for every phase:

```json
{"utterance_id":"heldout-hi-0001","phase":"baseline","hypothesis":"नमस्ते","run_id":"original-run-001","language_mode":"known","language_prompt":"hi-IN"}
{"utterance_id":"heldout-hi-0001","phase":"expanded_untrained","hypothesis":"नमस्ते","run_id":"expanded-run-001","language_mode":"known","language_prompt":"hi-IN"}
{"utterance_id":"heldout-hi-0001","phase":"fine_tuned","hypothesis":"नमस्ते","run_id":"trained-run-001","language_mode":"known","language_prompt":"hi-IN"}
```

Never replace missing or failed inference with an empty hypothesis. An empty
hypothesis means the model actually emitted no text. Keep absent runs absent so
the report exposes the missing evidence. Empty references are legitimate silence
examples, and hallucinated words contribute insertion errors.

## Release coverage and provenance

Set `purpose` to `release` only for real evaluation. It requires:

- All 32 exact pretrained ASR locales in `BASE_ASR_LOCALES` and all eight exact
  adaptation locales in `ADAPTATION_LOCALES`, matching the
  [pinned NVIDIA model card](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/1c8deaecc64b91f034d73e08dd8b64625eb3395d/README.md).
  Every existing ASR profile must be protected. Adaptation locales are reported
  separately; the manifest may also mark them protected when required.
- All 22 selected Indic language/script pairs in `INDIC_SCRIPTS`: Assamese and
  Bengali/Beng; Bodo, Dogri, Hindi, Konkani, Maithili, Marathi, Nepali, Sanskrit,
  Sindhi/Deva; Gujarati/Gujr; Kannada/Knda; Kashmiri and Urdu/Arab;
  Malayalam/Mlym; Manipuri/Mtei; Odia/Orya; Punjabi/Guru; Santali/Olck;
  Tamil/Taml; Telugu/Telu. One profile can belong to multiple cohorts, so Hindi
  does not require duplicate audio. Alternate scripts are not implied.
- Held-out audio for every declared profile and required condition, with an
  actual audio SHA-256, dataset revision, split (`test`, `validation`, or
  `held_out`), and speaker/session cluster. Select utterances independently of
  observed system errors; audit train/evaluation overlap before running.
- Three `runs` entries keyed by phase. Each entry contains `run_id`,
  `checkpoint_sha256`, `tokenizer_sha256`, and a `settings` object recording the
  actual software version, decoding algorithm, beam settings, streaming chunk,
  device/dtype, and other inference settings. All predictions must name the
  corresponding run. Artifact hashes refer to exact deployed files; for a
  multifile checkpoint use a documented hash of its sorted checksum manifest.
- Each prediction records its actual `language_mode` (`known` or `automatic`)
  and, for known-language mode, the actual `language_prompt`. The original
  model cannot accept a newly allocated prompt. Evaluate that baseline in an
  actually supported automatic mode and label the protocol difference. Such a
  comparison describes the new system but cannot establish preservation of a
  protected baseline. Separate profile conditions prevent automatic and known
  language results, or different streaming settings, from being pooled.
- Explicit `indic_accuracy_targets`, keyed by profile ID, each containing
  predeclared fractional `max_wer` and `max_cer`. No universal threshold is
  invented here: agree realistic per-language acceptance targets and adequate
  sample sizes before observing results. Missing targets block an ASR release.

Example additional run/target fields (hash placeholders must be replaced):

```json
{
  "runs": {
    "baseline": {
      "run_id": "original-run-001",
      "checkpoint_sha256": "actual_64_lowercase_hex_characters",
      "tokenizer_sha256": "actual_64_lowercase_hex_characters",
      "settings": {"decoder": "greedy", "chunk_ms": 1120, "dtype": "float32"}
    }
  },
  "indic_accuracy_targets": {
    "hi-IN-Deva": {"max_wer": 0.1, "max_cer": 0.05}
  }
}
```

The numbers above only illustrate the schema; they are not accepted project
quality targets. Add equivalent run provenance for the other two phases.
The model's native streaming settings include 80, 160, 320, 560, and 1120 ms;
the HF path supports a subset. Do not infer an unsupported 112 ms mode from
these examples. The evaluator accepts saved streaming predictions, but the
minimal runner described below implements offline inference only.

## Creating predictions with a local NeMo checkpoint

`untok.inference.run_nemo_inference(manifest_path, checkpoint_path, phase,
output_path, device="cpu")` restores a local `.nemo` checkpoint and transcribes
the manifest's local audio. It uses the actual `target_lang` parameter in
[NVIDIA's prompt RNNT API](https://github.com/NVIDIA-NeMo/Speech/blob/main/nemo/collections/asr/models/rnnt_bpe_models_prompt.py),
validates prompt keys against the restored dictionary, and chooses the untok
restoration class when the checkpoint declares the custom tokenizer.

The runner requires a separate **offline** condition, an explicit request for
each profile/phase/condition, and corresponding run settings:

```json
{
  "required_conditions": ["known_language_offline"],
  "inference": {
    "baseline": {
      "known_language_offline": {
        "mode": "offline",
        "language_mode": "known",
        "language_prompt": "en-US"
      }
    }
  }
}
```

This object extends the relevant profile. Each utterance's `condition_id` must
match it. For automatic mode, set `language_mode` to `automatic` and omit
`language_prompt`; the runner actually sends `target_lang="auto"` and verifies
that registry entry exists. Add requests for the other phases explicitly.
There is no fallback from an unsupported known prompt to automatic mode.

The matching `runs[phase].settings` must include all of:

```json
{
  "mode": "offline",
  "device": "cpu",
  "dtype": "float32",
  "batch_size": 1,
  "num_workers": 0,
  "decoder": "greedy_batch"
}
```

The decoder must match the restored model configuration; this runner does not
change decoding strategies. The device can be changed explicitly to the actual
run device. Single-utterance calls retain repeated words and avoid sharing
partial hypotheses across recordings. Streaming/chunk options and unknown
settings fail instead of being ignored. Returned text is saved verbatim; no
language-tag removal or transcript cleanup is silently applied.

Audio hashes and the checkpoint hash must match the manifest before inference.
The declared tokenizer hash must match the **actually restored tokenizer**:
the original native checkpoint's SentencePiece model hash differs from the
canonical HF JSON hash. The extended checkpoint uses the HF adapter's verified
JSON hash. NeMo can delete its extraction directory after restoration. The HF
adapter retains the verified hash captured while loading; if its source file
still exists, the runner additionally checks that its bytes did not change.
For native SentencePiece, the runner hashes the processor's in-memory serialized
model, so no surviving temporary path is required. Do not label native
predictions with the HF tokenizer's hash.
Relative audio paths resolve against the manifest directory.

The output JSONL includes actual audio/checkpoint/tokenizer hashes and actual
decoding settings, including hashes of the complete decoding and encoder
configurations, the encoder's attention context, and runtime package/Python
versions. A missing package version is recorded as `null`, never invented.
Optional `decoding_sha256`, `encoder_config_sha256`, `encoder_context`, and
`runtime_versions` settings can pin those actual values in advance; declared
values must match the restored environment. A companion `.jsonl.run.json`
records the run.
The evaluator compares actual settings when present, so different decoding
settings cannot silently pass the protected-language comparison. Existing
outputs are not overwritten. An exception, malformed response, or failed
utterance produces no partial predictions file and is never replaced by an
empty hypothesis. An actual empty model output remains a valid hypothesis.

The offline runner's native integration remains unverified until it executes
with the pinned NeMo environment and real checkpoint. Runner tests use an
explicit fake model and synthetic files. Additional tests exercise actual HF
and SentencePiece tokenizer backends after deleting their source files; these
verify artifact handling, not real speech recognition.

## Scores and gates

WER uses whitespace-separated tokens. This is not a linguistically correct
word segmenter for every language, particularly Chinese, Japanese, and Thai;
CER therefore remains a mandatory gate. CER counts Unicode code points, not
rendered grapheme clusters. Token repetitions are preserved. Error counts are
summed over reference lengths; per-utterance percentages are not averaged.
Insertion errors can make rates exceed 1.0. A zero reference denominator is
reported as JSON `null`, never as a false zero or nonstandard infinity.

Scoring normalization is explicit and fixed across systems. Whitespace is
collapsed and stripped. Case folding, NFC/NFKC, punctuation removal, and space
removal for CER are opt-in. Joiners are never removed by this scorer. A second
raw score always preserves punctuation, case, and Unicode spelling (with the
same whitespace convention), exposing changes that the chosen scoring policy
would otherwise hide. This scoring policy is independent of tokenizer 1A/1C.

Every profile and required condition is reported separately. Fine-tuned WER and
CER must not increase over baseline for **each protected profile/condition**.
Silent-utterance word/character insertions must not increase either, so improved
speech elsewhere cannot hide new silence hallucinations. The expanded untrained
phase is measured and compared, but its full-output-space accuracy is not
assumed identical merely because original weights were copied.

The paired bootstrap resamples the same speaker/session clusters in both
systems and reports deterministic 95% percentile intervals for
candidate-minus-baseline WER/CER. Positive values mean worse. It reports the
cluster count and valid resamples; silence-only samples have undefined rates
and are omitted from the interval. The strict no-worsening gate uses the point
estimates, not statistical nonsignificance. A wide interval warrants more data;
passing does not prove population-wide equivalence.

`release_status` is `blocked` when evidence, required coverage, provenance, or
fixed evaluation requirements are missing; `failed` when complete evidence
violates a quality gate; and `passed` only when both sets of checks pass.
Regressions remain listed even in a blocked report. No aggregate average can
hide a failed locale. Silence-only profiles cannot establish ASR accuracy.

This evaluation should include public third-party held-out speech and actual
project recordings, all kept separate from training. Suitable sources include
[FLEURS](https://huggingface.co/datasets/google/fleurs) and
[IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices), subject to
verified locale/script coverage and split independence. Do not claim missing
locales were validated. No real speech run is included in the unit-test results.
