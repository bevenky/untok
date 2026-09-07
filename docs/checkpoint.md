# BPE checkpoint integration

This page documents the BPE variant. Native Unigram checkpoint integration is
still pending; see [native Unigram details](native-unigram.md).

The canonical artifact is the approved HF BPE `tokenizer.json`. The runtime
adapter keeps this artifact intact and maps its IDs into native NeMo RNNT's
contiguous, blank-last output space. The original checkpoint remains unchanged.

| Symbol | Canonical HF ID | Original native model row | Expanded native model row |
| --- | ---: | ---: | ---: |
| Existing text/tag piece | 0–13086 | Same ID | Same ID |
| HF padding | 13087 | No text/output row | No text/output row |
| Blank | 13088 | 13087 | Last output row |
| First appended token | 13089 | Absent | 13087 |

Do not pass canonical IDs directly to the native model. `IdMap.to_model()`
rejects padding and blank in transcript labels; `HFTokenizerAdapter` exposes
model IDs at its public boundary. Native batch padding uses the model's blank
index and must be excluded by transcript lengths. Existing language tags remain
available in decoded text. Repeated RNNT tokens are retained.

## Upstream artifact gate

The pinned HF model config declares blank 13087, padding 0 and vocabulary size
13088, while its tokenizer declares padding 13087 and blank 13088. The
`artifact_preflight()` report records this mismatch without silently changing
the source. A direct HF training call may receive the tokenizer's blank outside
the model's embedding range. HF load/inference success alone is not a training
compatibility test.

The actual native checkpoint embeds a SentencePiece **Unigram** model
(`model_type=1`); the published `tokenizer.json` uses **BPE**. They share old
piece IDs but encode some text differently, including English. The adapter
intentionally uses the approved BPE for encoding. On one real English clip,
native labels used 53 tokens and both the original and extended BPE used the
same 46 tokens, with zero additions. This difference must be evaluated before
continued training. Exact old weights and decoded speech do not establish
training-label equivalence.

Migrated checkpoints also contain the original native decoder artifact.
The runtime copies its vocabulary in memory and appends new ordinary pieces
solely to decode token strings, preserving native unknown-token rendering,
spacing and existing piece types. This decoder never segments text or uses
the appended scores. The canonical BPE file and encoding remain unchanged.

Sources: [pinned HF configuration](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/blob/1c8deaecc64b91f034d73e08dd8b64625eb3395d/config.json),
[HF processor's decoder-input construction](https://github.com/huggingface/transformers/blob/main/src/transformers/models/nemotron3_5_asr/processing_nemotron3_5_asr.py),
[NeMo vocabulary replacement](https://github.com/NVIDIA-NeMo/Speech/blob/ca4daa1470f6c01068c4e6a9a73b19b9a91dc366/nemo/collections/asr/models/rnnt_bpe_models.py).

## APIs

- `runtime.build_id_map(tokenizer_json, base_tokenizer_json=None)` validates
  canonical IDs, loader stability and, when supplied, every original ID and
  added-token descriptor. The JSON manifest records both mapping directions,
  reserved IDs and source hashes.
- `checkpoint.artifact_preflight(tokenizer_json, hf_config_json=None)` produces
  a read-only contract report without importing torch/NeMo.
- `checkpoint.migrate_nemo_checkpoint(source, tokenizer_json,
  base_tokenizer_json, output, seed=0, prompt_dictionary=None)` consumes a
  complete local `.nemo`, constructs the adapter-aware model, copies every old
  tensor, saves a new checkpoint, restores it, and checks every copied value
  again. It produces `<output>.migration.json`. It never downloads weights or
  overwrites a checkpoint. The installed `sttok` package is needed for restoring
  the custom `sttok.runtime.ExtendedNemotronRNNTModel` class.
  The pinned NeMo runtime rejects external target namespaces by default.
  `get_nemo_model_class()` permits only this exact installed class, verifies its
  identity and NeMo base class, and leaves validation of all other targets
  unchanged. It does not allow arbitrary targets under the `sttok` namespace.
- `inspect_nemo_layout`, `transfer_state_dict` and `verify_state_transfer`
  separate live module discovery, copying, and independent verification. All
  unaffected tensors and buffers must have unchanged names, shapes and dtype.
  Only predictor embedding and final joint weight/bias rows can expand. The
  output blank and embedding blank row both move explicitly.
- `compare_old_logits()` checks supplied raw logits from the same features and
  mapped old prefixes. `mask_new_outputs_for_test()` suppresses additions before
  softmax for migration-only regression tests. This is not a product language
  lock, and it must not be used to claim expanded-output ASR parity.

The migration refuses hybrid/CTC heads, duration/extra-output layouts, tied
vocabulary parameters, different source token inventories and unexpected tensor
changes. It does not migrate optimizer state. New prediction embeddings use
the mean of old nonblank embeddings. New joint-output weights copy the old
blank row, with a lower bias so additions do not dominate before training.
For 3,392 additions the bias margin is about 21.945, targeting a combined
new-to-old softmax mass ratio of at most 1e-6 in exact arithmetic. Actual
floating-point decoding still requires validation. Rows remain independently
trainable. All old encoder, predictor recurrent, prompt and joint-projection
weights are copied exactly from the source. Migration verifies the added rows
and native decoder artifact before saving and after reloading.

## CPU tests

From the repository, after installing the package's test/checkpoint dependencies:

```sh
python -m pytest tests/test_checkpoint.py -q
```

These tests exercise an actual PyTorch LSTM predictor and joint projections with
synthetic weights. They verify relocated blank rows, unchanged logits for mapped
prefixes, exact state preservation, finite gradients on new rows, padding/blank
label rejection, repeated-token decoding, and detection of accidental HF
special-ID reassignment. A cached NVIDIA tokenizer contract is checked when
available. These are not audio accuracy measurements or an actual NeMo model
integration run.

## Run migration and model checks

Use the project-pinned NeMo revision in a compatible Linux training environment
with its RNNT loss dependencies. A Python 3.14 macOS tokenizer/test environment
does not establish that this NeMo stack works. Provision the complete original
`.nemo` locally and sufficient RAM for original, expanded and reloaded states;
the migration does not infer a complete checkpoint from an archive header.

The `checkpoint` package extra supplies PyTorch helpers, not NeMo itself.
Install the `legacy-bpe` extra for this historical BPE workflow as well.
Migration restores, expands and verifies the checkpoint on CPU. Small offline
audio comparisons can also run on CPU with a compatible runtime. CUDA is needed
for testing GPU behavior and is recommended for substantial training and speech
evaluation workloads.

From the repository root, after building the tokenizer, ID map and prompts:

```sh
sttok legacy-bpe migrate --source /models/original.nemo --output /models/indic-expanded.nemo --prompts artifacts/nemotron-indic-v1/prompts.json
sttok legacy-bpe verify-checkpoint --source /models/original.nemo --expanded /models/indic-expanded.nemo --manifest /data/migration-check.json --output reports/migration-audio.json --device cpu
```

Use the same device in the audio manifest. The manifest format and verification
limits are described in [checkpoint-validation.md](checkpoint-validation.md).

The migration function is callable directly:

```python
from sttok.checkpoint import migrate_nemo_checkpoint

report = migrate_nemo_checkpoint(
    source="/models/original.nemo",
    tokenizer_json="/artifacts/extended/tokenizer.json",
    base_tokenizer_json="/artifacts/nvidia/tokenizer.json",
    output="/models/indic-expanded.nemo",
    seed=0,
)
```

After migration, restore the model, configure real training data, and run a
forward/backward step with new-script labels and valid prompt indices. Require
finite RNNT loss, correct target lengths, no blank/pad transcript labels and
nonzero gradients on relevant new rows. Then run paired old-output-only audio
tests on identical audio, prompts and decoding settings. The current runner is
offline; streaming requires separate integration and checks. Finally, enable all
outputs and evaluate the untrained expansion and the fine-tuned model on held-out
speech for every declared language/profile. None of these checks is substituted
by successful tensor copying. The migration report leaves these gates explicitly
pending until real evidence is collected.

Fine-tuning data configuration is deliberately omitted from the migrated model:
the source's dataset manifests must not be reused accidentally. Supply the
project's audited transcripts, label convention and language-prompt registry.
Added prompt entries must preserve every original alias/index, use unused slots,
and fit the unchanged 128-dimensional prompt representation. Input prompts are
separate from any language-tag tokens learned for automatic detection.
