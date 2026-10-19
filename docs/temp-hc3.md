# Temp-HC3

**A temperature-controlled dataset for studying human/AI text detection across generator families and decoding temperatures.**

[Download the complete dataset](https://github.com/twzydn20000928/T-Aware/releases/download/v0.1.0/temp-hc3-v0.1.0.tar.gz) · [Release and checksums](https://github.com/twzydn20000928/T-Aware/releases/tag/v0.1.0) · [Machine-readable manifest](../configs/assets.json)

## Dataset contribution

Generation temperature changes the distribution of AI-written text. Temp-HC3 makes
this variation explicit: a source question and human answer are paired with AI
answers from three generator families at 13 temperatures. Keeping prompt identity,
generator identity, and temperature labels enables controlled studies of detection
robustness and temperature-related text statistics.

Temp-HC3 is useful independently of the T-Aware detector. It supports supervised
human/AI classification, auxiliary temperature prediction, temperature-conditioned
evaluation, cross-generator transfer, and entropy/surprisal analysis.

| Property | Description |
| --- | --- |
| Language and task | English question answering; human/AI text classification |
| Source corpus | HC3 question-answer prompts and human answers |
| Generator families | Llama-3, Mistral, Gemma-2 |
| AI temperatures | 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5 |
| Records | 418,110: 388,245 AI generations and 29,865 human anchors |
| Organization | Three generators × three fixed splits, supplied as 9 JSONL files |
| Labels | Human=0, AI=1; generation temperature retained on AI examples |
| Download | One complete archive, approximately 441 MiB |

Human anchors repeat across generator partitions. Record counts therefore differ
from the number of independent source prompts. Each generator partition contains
9,955 source groups across its train, validation, and test files.

## Construction

1. Prepare HC3 question-answer source partitions, retaining a source identifier,
   question, and human-written answer.
2. For each generator family, produce one AI answer per source question at each of
   the 13 temperatures. Retain one human answer as an anchor with `generation_T=-1`.
3. Preserve the source-group identifier across temperature variants and retain the
   predefined training, validation, and test partitions.
4. Export generator, temperature, prompt, and answer fields in JSONL format.

This distribution preserves the supplied research files byte for byte. The
[generation command](data.md#generate-temperature-controlled-examples) can create
compatible examples for further experiments; its model and sampling options are
recorded with each run.

## Split statistics

For **each** of Llama-3, Mistral, and Gemma-2:

| Split | Source groups | AI generations | Human anchors | Records |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,000 | 104,000 | 8,000 | 112,000 |
| Validation | 975 | 12,675 | 975 | 13,650 |
| Test | 980 | 12,740 | 980 | 13,720 |
| Total | 9,955 | 129,415 | 9,955 | 139,370 |

The nine files together contain 418,110 records. The complete archive includes
**all** splits; the validation and test files are not a separate download or a
different dataset. Keep temperature variants from the same source group together
when constructing any additional splits.

## Record schema

```json
{
  "id": "train_0001_llama3_T1.2",
  "base_id": "train_0001",
  "prompt": "Example question",
  "model": "llama3",
  "generation_T": 1.2,
  "generated_text": "An AI-generated answer."
}
```

| Field | Meaning |
| --- | --- |
| `id` | Record identifier |
| `base_id` | Source-group identifier shared by a prompt's temperature variants |
| `prompt` | Source question |
| `model` | Generator identifier or `Human` |
| `generation_T` | AI generation temperature; -1 for a human anchor |
| `generated_text` | Human-written or AI-generated answer |

Derive the source label as `int(generation_T >= 0)`. Human anchors have no regression
target. AI temperatures are not required as detector inputs during inference.

## Download and use

```bash
# Download only Temp-HC3, verify its hashes, and extract all 9 files
python scripts/download_assets.py --assets temp-hc3 --output /path/to/t-aware-data
```

```text
t-aware-data/
├── dataset/
│   ├── dataset.llama3.train.jsonl
│   ├── dataset.llama3.val.clean.jsonl
│   ├── dataset.llama3.test.clean.jsonl
│   ├── dataset.mistral.train.jsonl
│   ├── dataset.mistral.val.clean.jsonl
│   ├── dataset.mistral.test.clean.jsonl
│   ├── dataset.gemma2.train.jsonl
│   ├── dataset.gemma2.val.clean.jsonl
│   └── dataset.gemma2.test.clean.jsonl
└── metadata/
    ├── temp-hc3.json
    └── temp-hc3-README.md
```

Load a file without installing any additional dataset library:

```python
import json

with open("/path/to/t-aware-data/dataset/dataset.llama3.train.jsonl", encoding="utf-8") as stream:
    example = json.loads(next(stream))
print(example["base_id"], example["generation_T"], example["generated_text"])
```

The release retains 223 empty answer records across the nine files. The manifest
gives counts per file. Use the documented `keep`, `skip`, or `error` policy for your
experiment; filtering changes the effective counts. The default T-Aware training
configuration retains these records and repeats human examples for class balance.

## Companion robustness resources

The [robustness archive](https://github.com/twzydn20000928/T-Aware/releases/download/v0.1.0/t-aware-robustness-v0.1.0.tar.gz)
is a separate, optional resource. It contains prepared HC3/ChatGPT attack sets,
generator-transfer and domain-transfer sets, attack validation, recursive and
synonym variants, variance annotations, and prepared source files. See
[Data preparation](data.md#evaluation-records) for the evaluation schema.

These populations have different memberships and sample counts. The supplied
generator-specific recursive filenames retain their original `k5` names while
their AI type metadata contains `k2`; retain the metadata when interpreting depth.
The standalone Temp-HC3 dataset does not require this companion package.

## Attribution, terms, and citation

Temp-HC3 builds on **HC3**, by Biyang Guo and colleagues, introduced in *How Close is
ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection* (2023).
[HC3 dataset](https://huggingface.co/datasets/Hello-SimpleAI/HC3) ·
[HC3 paper](https://arxiv.org/abs/2301.07597) ·
[HC3 source copyright](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection#dataset-copyright).

HC3 identifies CC BY-SA 4.0, subject to stricter original-source terms. Original
source rights and attribution are retained. Temp-HC3 adds controlled generations,
human anchors, generator/temperature metadata, and experimental partitions;
HC3-derived material is distributed under the applicable upstream terms.
Generation models and the companion benchmarks retain their respective terms.

When using Temp-HC3, cite the T-Aware paper and HC3. This entry identifies the
versioned dataset distribution:

```bibtex
@misc{temphc3,
  title = {Temp-HC3: A Temperature-Controlled Dataset for AI-Text Detection},
  author = {{T-Aware Authors}},
  year = {2026},
  howpublished = {T-Aware research assets, version 0.1.0},
  url = {https://github.com/twzydn20000928/T-Aware/releases/tag/v0.1.0}
}
```

Dataset metadata and checksums are versioned with the code and release. Please
report any data issue through the repository's issue tracker, including the file
name and record identifier.
