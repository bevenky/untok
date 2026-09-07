# Native Unigram candidate results

Build date: 7 September 2026. The candidate contains **20,550 native text IDs**
and **20,552 public IDs**, including public padding and blank. It extends the
actual SentencePiece Unigram tokenizer embedded in the pinned Nemotron 3.5 ASR
streaming 0.6B checkpoint. It does not replace the separate BPE artifact.

This is a validated tokenizer candidate. Native checkpoint migration and
bounded speech compatibility checks are documented in
[native compatibility results](native-compatibility-results.md).
Acoustic fine-tuning and broad accuracy evaluation are separate tasks.

## Inventory

| Component | Physical IDs |
| --- | ---: |
| Original native entries, preserved exactly | 13,087 |
| Newly selected Indic pieces | 7,219 |
| Approved rare Latin characters | 190 |
| Additional approved shared punctuation, currency and brackets | 54 |
| Native text inventory | 20,550 |
| Public padding and blank | 2 |
| Public inventory | 20,552 |

The selection has 7,400 group memberships: 1,400 shared Devanagari, 500 shared
Bengali/Assamese, and 500 each for Gujarati, Kannada, Malayalam, Odia, Punjabi,
Tamil, Telugu, Kashmiri, Urdu, Manipuri and Santali. Required characters and the
103 approved Hindi pieces count inside those budgets.

The 7,400 memberships contain 87 repeated memberships and 94 unique strings
already in the native model, leaving 7,219 additions. The shared character list
contains 56 characters; Arabic semicolon and double danda are already selected
in the banks, so only 54 need extra IDs. No further alphabet additions or broad
Unicode fallback were needed. No new Latin multi-character pieces were added.

Original native IDs remain `0..13086`. New native text IDs start at `13087`.
Public padding and blank remain `13087` and `13088`, so new public IDs start at
`13089`. The migrated Full native RNNT checkpoint uses blank `20550` and 20,551
output classes. Public IDs are not native training labels.

## Selection and data

The main study evaluated 96 nominal configurations across 12 fitting jobs.
They produced **47 distinct model files**, with 49 repeated model rows. Full
data seed pairs were identical, so they are not independent search outcomes.
All configurations passed their fitting gates. The selected donor-seeded
recipes used mass multiplier 1.2 for Devanagari and 0.8 for the other banks.

Candidate strings came from fresh corpus fitting, IndicConformer and
IndicBART/IndicBARTSS. Donor scores were not copied. Added scores were fitted
against the unchanged native vocabulary, with a fixed added-score mass and
deletion-loss pruning to each quota. This is the best result among the tested
recipes, not a proof of a global optimum.

The frozen corpus contains **280,471 training**, **33,027 development** and
**20,055 reserved** records across all 22 target profiles. Training combines
IndicVoices conversation, extempore and read speech with SPRING, Omnilingual ASR,
UrduSpeech and written text. Source/language weights prevent larger sources
from dominating. See [data preparation](native-unigram-data.md) for sampling,
speaker overlap removal, deduplication and source-specific limitations.
Development and reserved text are held out from this fitting process; absence
from the native model's or donors' upstream training is not established.

Vaani was downloaded successfully but excluded because its mixed annotation
format lacks a verified lexical extraction rule. IN22-Conv was unavailable to
the supplied account. Nine profiles have only one independent speech origin;
the corpus does not establish complete dialect coverage.

The 72 partial-data fits cover 25%, 50% and 75% of training groups with two
seeds per bank. Together with full-data references, all 176
language/fraction/seed measurements passed their fitting and development
guards. The selected candidate family and mass multiplier stayed fixed;
absolute added-score mass was recalculated for each subset.

| Requested training fraction | Actual records across all profiles | Mean relative token length versus full data |
| --- | ---: | ---: |
| 25% | 70,845 to 70,848 | +1.38% |
| 50% | 139,998 to 140,783 | +0.49% |
| 75% | 210,410 to 210,649 | +0.11% |
| 100% | 280,471 | Reference |

These are recipe learning curves on the same development set. Ten profiles
are slightly non-monotonic from 75% to 100%, and Gujarati still improves by
about 1.09%. The small average final gain does not prove data saturation or
complete conversational/dialect coverage.

The 36 source-removal fits also cover all 22 profiles. They keep the selected
candidate family and matched full-mixture score mass, refit on the remaining
training sources and evaluate the unchanged development set.

| Training variant | Equal-language mean token-length increase | Profiles with more than 2% p95 regression |
| --- | ---: | ---: |
| IndicVoices only | +1.04% | 1 |
| Speech sources only | +1.57% | 5 |
| Without IndicVoices | +5.10% | 14 |

Unknown counts did not increase. The tail regressions above use the matched
full-mixture model as reference; all variants passed the weaker original fresh
reference guards. IndicVoices-only improves its own development domain by
1.54% but worsens omitted source domains by 4.21%. This supports retaining the
mixture for broader text coverage. Removing a source also changes corpus
quantity, and donor pretraining may contain that source, so this is not an
isolated causal estimate of data diversity.

## Validation

| Check | Result |
| --- | --- |
| Original native pieces, IDs, scores and types | Exact preservation |
| Original metadata | Preserved except the required vocabulary-size update |
| Native normalizer 1A | Byte-identical |
| All 22 standard target alphabets | Pass |
| Required Hindi 103 and Latin 190 encoding witnesses | Pass |
| Optional additions used on full training text | Every piece used |
| Strictly dominated additions after merging | None |
| Group quotas and shared-piece collisions | Pass |
| All 22 merged development unknown/tail guards | Pass |
| Development corpus | 33,027 records, zero normalizer or representable round-trip failures |
| Actual bundle on Linux | Same hashes and all development metrics as macOS |
| Repository suite | 235 tests passed locally and across the Linux runs |
| Reserved corpus | 20,055 records across all 22 profiles; zero unknown tokens, normalizer mismatches or round-trip failures |

The merge scans complete training records, including mixed-script text. A
piece shared by several banks receives one physical ID. The combined model
must still satisfy actual usage, witness and development checks after that
deduplication. Every merged profile passed unknown nonincrease and the p95
token-per-character bound against both its bank winner and the common fresh
reference.

All 6,280 optional additions occur in training, at least 3 times each, with a
median of 435 occurrences. All approved Hindi additions occur in training.
Another 221 required coverage pieces have no training occurrences, including
161 rare Latin characters. They remain deliberately available and pass direct
encoding witnesses; this does not establish learned acoustic behavior.

The reserved corpus was evaluated once after the final model, selection,
policy, corpus manifest and study hashes were frozen. No tokenizer tuning
followed that evaluation. All reserved records come from IndicVoices, so this
is held-out coverage within that source, not a new-source generalization test.

The development set contains 46 records with remaining unknowns, including
phonetic notation, foreign-script quotations, soft hyphens and symbols outside
the declared coverage. Standard alphabet checks pass; arbitrary Unicode
coverage is not claimed. Round trips are checked against native-normalized
text when no unknown token is present, not against unnormalized spelling.

In the pinned native normalizer, ZWNJ (`U+200C`) becomes a space, while ZWJ
(`U+200D`) is retained. The candidate includes a ZWJ piece shared across the
banks. That adds character coverage without changing normalizer 1A; it does
not fix 1A's ZWNJ-to-space behavior.

## What changes and what stays compatible

The original `a` still encodes as `▁`, `a`; `which` still encodes as `▁which`.
Old-ID decoding and the native normalization behavior are preserved. The
checks also include 10,000 random old-ID sequences and encoding probes where
no added piece can match.

A separate English FLEURS text control checked 647 records with 350 distinct
transcripts, using both raw and publisher-normalized forms. All 531 raw and
613 normalized records that the original model could fully represent kept
identical token IDs. The other 116 raw and 34 normalized records gained
coverage for approved punctuation, eliminating their unknown tokens. These
are two forms of the same corpus, not independent samples or an audio test.

Hindi segmentation changes are substantial: 1,531 of 1,552 development
records fully representable by the base model changed their ID sequences.
For example, `भारत में आज मौसम अच्छा है।` uses 15 native tokens and 8 extended
tokens, with identical decoded text. New Devanagari pieces are shared across
Hindi, Marathi, Bodo, Dogri, Konkani, Maithili, Nepali, Sanskrit and Devanagari
Sindhi. Such changes require fine-tuning and speech regression tests.

Text evidence alone does not establish unchanged recognition accuracy across
the original 40 locales. The later [native compatibility checks](native-compatibility-results.md)
verify migrated weights and paired audio behavior on explicitly bounded
recordings. New-language acoustic learning remains a separate task. No acoustic
training was performed in that compatibility phase.

Model SHA256:
`f987a99ce9448ca72bb2da11f36744254f9f9b12f5596fcb742ddedf950886a8`.
The candidate bundle includes the original model, exact scored selection,
both ID layouts and their hashes so the tokenizer can be rebuilt without
rerunning selection. Raw corpora and audio are not included.
