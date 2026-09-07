# Legacy BPE experiments

The active tokenizer workflow uses the native Unigram model embedded in the
Nemotron checkpoint. The earlier BPE experiment used NVIDIA's separately
published `tokenizer.json`. The two encoders can segment the same text
differently, even when their original vocabulary IDs match.

Keep the BPE code here to reproduce its recorded results and reuse tested ID
mapping and checkpoint utilities. It is not a second default for the native
checkpoint, and switching tokenizer algorithms is not a runtime option for an
already trained model.

## Reproduce the old workflow

All historical commands require the explicit `legacy-bpe` prefix:

```sh
python -m pip install -e '.[legacy-bpe]'
untok legacy-bpe fetch
untok legacy-bpe build
untok legacy-bpe id-map
untok legacy-bpe prompts
untok legacy-bpe --help
```

Their arguments and default paths are unchanged. The BPE artifact remains in
`artifacts/nemotron-indic-v1/`; the active Unigram artifact is separate at
`artifacts/nemotron-indic-unigram-v1/`.

Plain `untok build`, `untok check` and `untok validate` now use Unigram.
The `build-unigram`, `check-unigram` and `validate-unigram` aliases remain
available for existing native instructions and scripts. There is no native
checkpoint migration command yet. For historical BPE migration, use
`untok legacy-bpe migrate` and the corresponding BPE documentation.

## Historical evidence

- [BPE tokenizer build](build-evidence.md)
- [BPE text validation](validation.md)
- [BPE checkpoint migration](checkpoint.md)
- [BPE audio verification](checkpoint-validation.md)
- [BPE runtime setup](runtime-validation.md)
- [BPE Runpod results](runtime-results.md)

The BPE run's successful tensor-copying and audio comparisons do not validate
the new Unigram checkpoint. Native integration, training and speech evaluation
are tracked separately in [native Unigram details](native-unigram.md).

The complete test suite includes these retained helpers:

```sh
python -m pip install -e '.[legacy-bpe,test,checkpoint]'
python -m pytest -q
```
