# Candidate build evidence

Measured locally on 6 September 2026. This is tokenizer/text evidence; no full Nemotron checkpoint or real audio inference ran.

This is a historical report. Source paths in `build-evidence.json` follow the
current `untok` layout, while the hashes still describe the source snapshot
from the measurement date. The original report is preserved in Git history.

Candidate: **16,481 canonical HF IDs**, including **3,392 additions**, with **5,848 BPE merges**. Both a second build and a build from 33 freshly downloaded, hash-verified source files produced byte-identical tokenizer and manifest files.

SHA-256: `ef619be3ac1f85de60a3e9ab8c9f2deaddcbcfd71c8c9f1a55d9a9fc26d6a9c4`.

**136 tests passed.** They include actual HF BPE construction/serialization, corpus validation, prompt/ID contracts and synthetic PyTorch migration/gradient checks. NeMo runner tests use explicit fake models. The two warnings are upstream SentencePiece SWIG deprecations.

Every one of the 3,392 new IDs has a verified encoding witness. All old IDs, original merge order, added-token descriptors, and normalization/pre-tokenization/decoding settings pass preservation checks. The 54 protected probes pass; these are English/control examples and existing special/tag strings, not a corpus proving all-40-locale parity.

| Corpus | Profiles | Records | Records containing unknowns | Round-trip failures on unknown-free records | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| raw-bhasha | 21 | 80,088 | 16 | 0 | failed |
| scoped-bhasha-with-sindhi | 22 | 80,317 | 0 | 0 | passed_on_supplied_text |
| raw-independent | 22 | 1,442 | 16 | 0 | failed |
| scoped-independent | 22 | 1,426 | 0 | 0 | passed_on_supplied_text |

Bhasha excludes one whitespace-only record before counting language coverage. Its full audit retains 16 records containing unsupported soft hyphens, an unassigned Gujarati codepoint, bullet/private-use symbols or invisible separators. The independent full audit retains 16 records containing unsupported IPA notation or foreign-script quotations. Scope selection excludes whole records and records their hashes and out-of-policy codepoints; it never strips individual characters to create passing text. All assigned characters in declared script ranges remain eligible even when absent from the tokenizer.

The 80,317-record combined run contains 80,072 scoped Bhasha records and 245 Devanagari Sindhi paragraphs. Those Sindhi paragraphs also appear in the independent run. These totals must not be summed as unique records.

## Per-language scoped evidence

| Language | Script | Bhasha or Sindhi records | Independent records | Unknowns / failed round trips |
| --- | --- | ---: | ---: | --- |
| as | Beng | 1,524 | 71 | 0 / 0 |
| bn | Beng | 5,612 | 80 | 0 / 0 |
| brx | Deva | 1,502 | 21 | 0 / 0 |
| doi | Deva | 1,498 | 8 | 0 / 0 |
| gu | Gujr | 5,800 | 71 | 0 / 0 |
| hi | Deva | 5,628 | 76 | 0 / 0 |
| kn | Knda | 5,861 | 58 | 0 / 0 |
| kok | Deva | 1,500 | 38 | 0 / 0 |
| ks | Arab | 2,518 | 24 | 0 / 0 |
| mai | Deva | 2,514 | 62 | 0 / 0 |
| ml | Mlym | 5,639 | 51 | 0 / 0 |
| mni | Mtei | 1,502 | 15 | 0 / 0 |
| mr | Deva | 5,629 | 60 | 0 / 0 |
| ne | Deva | 2,514 | 109 | 0 / 0 |
| or | Orya | 1,522 | 95 | 0 / 0 |
| pa | Guru | 5,794 | 55 | 0 / 0 |
| sa | Deva | 2,524 | 81 | 0 / 0 |
| sat | Olck | 2,504 | 27 | 0 / 0 |
| sd | Deva | 245 | 245 | 0 / 0 |
| ta | Taml | 5,814 | 60 | 0 / 0 |
| te | Telu | 5,766 | 58 | 0 / 0 |
| ur | Arab | 6,907 | 61 | 0 / 0 |

## Standard alphabet check

The final CLDR48 standard-exemplar audit found six characters absent from the corpus-derived inventory: Bengali `ৡ ৢ ৣ`, Kannada `ೡ`, and Telugu `ౕ ౡ`. They were added as ordinary character tokens and tested in isolation and context. The three source XML files are pinned by commit and SHA-256 in the source lock. All **22 target profiles and 1,348 standard exemplar elements** now pass unknown-character and normalized round-trip checks, in isolation and context. Script inheritance is resolved explicitly, including Sindhi Devanagari and Manipuri Meetei Mayek. See [standard-exemplar-audit.json](standard-exemplar-audit.json). Auxiliary and historical block characters were not imported wholesale.

## Segmentation and generalization limits

**Hindi segmentation changes in 5,624 of 5,628 Bhasha records.** The 103 approved missing Hindi pieces are active, and shared Devanagari languages can change too. Stable original IDs preserve the meaning of learned rows; they do not imply stable encoded labels. Added Latin characters have no new Latin merge rules, but previously unknown characters can now produce new IDs.

The required character inventory was refined using these public audits. The results therefore demonstrate coverage of the measured material and reveal out-of-scope characters; they are not an untouched held-out benchmark. The independent Sindhi sample is single-author prose, and some other language samples are small. Native-speaker orthography review, unseen transcript coverage and representative speech evaluation remain necessary.

## Remaining model gates

The source HF configuration conflicts with canonical padding/blank/size. The explicit NeMo mapping and checkpoint migration code handle the intended native layout, but actual migration/save/reload, old-logit checks, masked and unmasked audio inference, real RNNT backward, streaming evaluation and fine-tuning are still unexecuted. The source model has not been changed. No published model-quality result is inferred from this report.

Machine-readable counts, per-language token-length statistics and precise environment versions are in [build-evidence.json](build-evidence.json). Full local reports and corpus snapshots remain in ignored `reports/` and `.cache/`; compact evidence contains no raw paragraphs.
