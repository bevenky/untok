# Native Unigram corpus preparation

Manifest SHA256: `b31dfd0d45542614ab51bb9c49621d2bbb100747bfe8f93f83a28df599973188`. All 67 artifact hashes verified. The 22 profiles retain 280,471 training rows, 33,027 development rows and 20,055 unique reserved representatives.

| Language | Train | Dev | Reserve | IV train | Extra speech train |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hindi | 14,497 | 1,753 | 898 | 10,185 | 0 |
| Marathi | 13,955 | 1,596 | 902 | 9,203 | 497 |
| Bodo | 12,017 | 1,346 | 899 | 8,733 | 378 |
| Dogri | 12,843 | 1,475 | 933 | 9,935 | 0 |
| Konkani | 11,044 | 1,255 | 928 | 7,563 | 568 |
| Maithili | 11,675 | 1,398 | 893 | 8,860 | 0 |
| Nepali | 11,172 | 1,458 | 907 | 8,345 | 0 |
| Sanskrit | 11,541 | 1,406 | 939 | 8,718 | 0 |
| Sindhi (Devanagari) | 7,849 | 973 | 872 | 7,722 | 0 |
| Assamese | 13,154 | 1,541 | 901 | 10,400 | 493 |
| Bengali | 14,732 | 1,687 | 904 | 9,900 | 497 |
| Gujarati | 13,558 | 1,561 | 937 | 8,675 | 498 |
| Kannada | 15,684 | 1,773 | 894 | 10,741 | 500 |
| Malayalam | 14,432 | 1,646 | 928 | 9,599 | 491 |
| Odia | 12,332 | 1,446 | 907 | 9,602 | 498 |
| Punjabi | 13,506 | 1,581 | 905 | 8,831 | 292 |
| Tamil | 14,439 | 1,638 | 903 | 9,583 | 497 |
| Telugu | 13,816 | 1,631 | 925 | 9,453 | 0 |
| Kashmiri | 12,364 | 1,362 | 951 | 9,154 | 670 |
| Urdu | 15,313 | 2,138 | 901 | 8,923 | 1,489 |
| Manipuri (Meetei) | 10,868 | 1,230 | 898 | 8,588 | 0 |
| Santali (Ol Chiki) | 9,680 | 1,133 | 930 | 7,328 | 0 |

## Material exclusions

The original official train and valid splits share source speaker identifiers. Metadata-only verification found 68,603 raw training rows with direct reserve-speaker overlap and zero repeated recording paths. After lexical marker exclusions, 68,549 lower-priority rows were removed by connected speaker/recording components. This is necessary for the requested speaker separation; official split names alone were insufficient.

Other exclusions: 39,676 exact audit-fingerprint duplicates, 238 verified near duplicates, 756 marker-only rows (700 training intake and 56 reserve), four malformed angle annotations, and 16 wrong-script written rows. All 17,067 identifiable FLORES records were excluded before preparation. Original input files are unchanged.

The 22,000 raw reserve records became 20,055 representatives after 56 marker-only rows and 1,889 exact duplicate records were excluded. Equivalent record provenance is attached to the retained representatives. No near-duplicate reserve records were found by the approximate screen.

## Source coverage and limits

The final IndicVoices training contribution is 200,041 rows: 68,588 labeled Conversation, 101,034 Extempore and 30,419 Read. Extra speech training includes 4,263 SPRING rows across nine profiles, 1,616 Meta rows across Bodo/Konkani/Kashmiri, and 1,489 UrduSpeech rows. A separate 499-row UrduSpeech domain is development data.

- No dialect-representative sampling claim. Region metadata is not proof of dialect coverage.

- Nine profiles have no second independently collected speech source in this freeze: hi, doi, mai, ne, sa, sd, te, mni, sat.

- All final reserve rows come from IndicVoices. The additional Meta/SPRING sources have no independent within-source dev split because prompt/speaker metadata connects or cannot separate them.

- UrduSpeech has one domain held out (499 rows), three domains training (1489 rows); generic speaker labels are not verified people.

- Written data was used in previous research. Exact duplicate removal discards repeated occurrence frequencies; exposure weights are constant per source.

- Near duplicate discovery is approximate, with exact Jaccard verification of discovered candidates, not an exhaustive zero-near-duplicate guarantee.

- The corpus has tokenizer-only partial transcripts after known annotation removal; not complete paired audio labels.

- Vaani is excluded because lexical gloss/noise annotation conventions are unresolved and source speaker identity is absent.

Meta's 1,742 original records form one cross-language component through shared prompt IDs. Keeping this component whole is intentional. SPRING stays one source group per language because the acquired view does not identify speakers/sessions. Source speaker labels are reported as labels, not verified independent individuals.

## Frozen input contract

One wording correction applies to the immutable manifest: its `normalized_character_definition` says `normalize(raw_text)`, but the implemented and tested calculation uses `native.normalize(row['text'])` after source annotation cleanup. The `raw_text` field preserves source annotations. This clarification changes no frozen bytes, weights, source rows or model normalization behavior.

The freeze meets the mechanical gates: every profile has train/dev/reserve records, native model and normalizer hashes are pinned, source annotation policy v2 is fixed, and no final metadata group, exact normalized hash or audit fingerprint crosses roles. The native normalizer and checkpoint were not modified. Keep this manifest immutable, preserve supplied source weights in fitting, and use reserve only after selecting the candidate on development data.
