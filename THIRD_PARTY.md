# Source provenance

BPE input pins are in `configs/sources.lock.json` and `configs/corpora.lock.json`.
Native input pins are recorded in its generated selection, frozen corpus
manifest and study reports. Build manifests record consumed hashes. This
project does not assert NVIDIA, AI4Bharat or Meta endorsement.

| Input | Source and published license information |
| --- | --- |
| Base tokenizer/configuration | [NVIDIA Nemotron 3.5 ASR 0.6B pinned release](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d), Open Model Definition and Weights License 1.1 |
| 22 BPE donors | [AI4Bharat IndicVoices tokenizer artifacts](https://github.com/AI4Bharat/IndicVoices/tree/d50726d3123bc5066994b94bf6c0cfbc5ab961d8/artifacts/tokenizers); [IndicConformer model card](https://huggingface.co/ai4bharat/indicconformer_stt_multi), MIT |
| Selected rare Latin character inventory | Meta `omniASR_tokenizer_written_v2.model`; [Omnilingual ASR documentation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/README.md), [Apache 2.0 license](https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE) |
| Assigned codepoints, script properties and standard exemplars | [Unicode 17 UCD](https://www.unicode.org/Public/17.0.0/ucd/), [CLDR48 pinned sources](https://github.com/unicode-org/cldr/tree/acd6d88ae493633240e19a87a721076a8a75c310/common/main), [Unicode License V3](https://www.unicode.org/license.txt) |
| Public primary audit | [Bhasha-Abhijnaanam v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0); source-specific provenance retained in records |
| Independent text samples | Wikimedia revision links, Unicode UDHR and author-published Devanagari Sindhi prose; raw snapshots remain local and are not redistributed in the candidate bundle |
| Development speech checks | [Google FLEURS pinned revision](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md), CC-BY-4.0; per-clip hashes and source transcripts remain in local run evidence |
| Native Unigram donor strings | [IndicBARTSS](https://huggingface.co/ai4bharat/IndicBARTSS/tree/4b2669d25bc24a46ad2501c2b759451b7a4a1a26) and [IndicBART](https://huggingface.co/ai4bharat/IndicBART/tree/78466a0c0e29f9229f7005623ecd6bc4243c0ae0), MIT; candidate strings only, with scores fitted against the native model |
| Primary Unigram transcript corpus | [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices/tree/c96f9088f138cf89d419da7e8e643e1f05c00a87), CC-BY-4.0 |
| Additional Bodo, Konkani and Kashmiri transcripts | [Omnilingual ASR corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus/tree/8648ba8946377697b427ae952076e49fc0e5e44d), CC-BY-4.0 |
| Additional Urdu transcripts | [UrduSpeech](https://huggingface.co/datasets/ASLP-lab/UrduSpeech/tree/16dd380cfd9049a3db7f06a98e878086916bf833), publisher-declared CC-BY-4.0 |
| Additional SPRING R1 transcripts | [Primary author release](https://github.com/Speech-Lab-IITM/SPRING_INX_ESPnet_Recipe/blob/6e30c6ab949211bb573ac9bc034f61eb5114db28/README.md) describes the original audio and manually transcribed text as public domain; the recipe's MIT license is not treated as the data license |

Before distribution, carry the applicable upstream license/notice files with
derived artifacts and set the distribution metadata accordingly. The source
code's license and the derived model/data terms are separate decisions. No
project-wide license is inferred merely from a donor's license.

No upstream model weights or third-party corpus text are committed here.
The selected 190-character list is a finite repertoire; it is not universal
Unicode fallback. The IndicConformer donor normalizers are not imported into
the final tokenizer. The BPE variant retains the published JSON normalizer;
the native Unigram variant retains the embedded SentencePiece normalizer.

See [native corpus preparation](docs/native-unigram-data.md) for sampling,
split checks, source-specific annotation cleanup and exclusions. Corpus text
is not included in tokenizer bundles. Vaani was downloaded for inspection
but excluded from native fitting because its lexical extraction policy is
unresolved. IN22-Conv was unavailable to the supplied account and is excluded.
