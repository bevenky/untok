# Source provenance

Exact tokenizer/data URLs, revisions and SHA-256 values are in
`configs/sources.lock.json` and `configs/corpora.lock.json`. Build manifests
record the consumed hashes. This project does not assert NVIDIA, AI4Bharat or
Meta endorsement.

| Input | Source and published license information |
| --- | --- |
| Base tokenizer/configuration | [NVIDIA Nemotron 3.5 ASR 0.6B pinned release](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b/tree/1c8deaecc64b91f034d73e08dd8b64625eb3395d), Open Model Definition and Weights License 1.1 |
| 22 BPE donors | [AI4Bharat IndicVoices tokenizer artifacts](https://github.com/AI4Bharat/IndicVoices/tree/d50726d3123bc5066994b94bf6c0cfbc5ab961d8/artifacts/tokenizers); [IndicConformer model card](https://huggingface.co/ai4bharat/indicconformer_stt_multi), MIT |
| Selected rare Latin character inventory | Meta `omniASR_tokenizer_written_v2.model`; [Omnilingual ASR documentation](https://github.com/facebookresearch/omnilingual-asr/blob/main/src/omnilingual_asr/models/README.md), [Apache 2.0 license](https://github.com/facebookresearch/omnilingual-asr/blob/main/LICENSE) |
| Assigned codepoints, script properties and standard exemplars | [Unicode 17 UCD](https://www.unicode.org/Public/17.0.0/ucd/), [CLDR48 pinned sources](https://github.com/unicode-org/cldr/tree/acd6d88ae493633240e19a87a721076a8a75c310/common/main), [Unicode License V3](https://www.unicode.org/license.txt) |
| Public primary audit | [Bhasha-Abhijnaanam v1.0](https://github.com/AI4Bharat/IndicLID/releases/tag/v1.0); source-specific provenance retained in records |
| Independent text samples | Wikimedia revision links, Unicode UDHR and author-published Devanagari Sindhi prose; raw snapshots remain local and are not redistributed in the candidate bundle |

Before distribution, carry the applicable upstream license/notice files with
derived artifacts and set the distribution metadata accordingly. The source
code's license and the derived model/data terms are separate decisions. No
project-wide license is inferred merely from a donor's license.

No upstream model weights or third-party corpus text are committed here.
The selected 190-character list is a finite repertoire; it is not universal
Unicode fallback. The IndicConformer donor normalizers are not imported into
the final tokenizer: the selected NVIDIA HF normalizer remains exact.
