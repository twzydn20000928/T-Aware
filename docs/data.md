# Data preparation

T-Aware reads one JSON object per line. Labels use **0 = human** and **1 = AI**. Set an external data directory with `--data-root /path/to/t-aware-data` or `T_AWARE_DATA_ROOT`; configuration paths are resolved relative to that directory.

## Availability and directory layout

The [v0.1.0 release](https://github.com/twzydn20000928/T-Aware/releases/tag/v0.1.0) provides the complete Temp-HC3 dataset and a separate companion robustness benchmark. Temp-HC3 is distributed as one archive containing all generators and all splits; it can be used independently of T-Aware. See the [Temp-HC3 dataset card](temp-hc3.md) for its contribution, construction, detailed schema, and citation.

```bash
python scripts/download_assets.py --assets temp-hc3 robustness-benchmark --output /path/to/t-aware-data
export T_AWARE_DATA_ROOT=/path/to/t-aware-data
```

The downloader checks archive and individual-file SHA-256 values and extracts files into the layout below. You can also use the release's direct archive links and extract them into the same directory with `tar -xzf ARCHIVE -C /path/to/t-aware-data`.

```text
t-aware-data/
├── dataset/             # generated train/validation/test JSONL
├── test_data/           # validation attacks and benchmark JSONL
└── saved_models_epoch50/ # reference detector weights
```

## Training records

AI examples need text, a source-group identifier, and the generation temperature:

```json
{"id":"train_0001_llama3_T1.2","base_id":"train_0001","prompt":"Example question","model":"llama3","generation_T":1.2,"generated_text":"An AI-generated answer."}
{"id":"train_0001_Human_Anchor","base_id":"train_0001","prompt":"Example question","model":"Human","generation_T":-1.0,"generated_text":"A human-written answer."}
```

`generation_T < 0` denotes human text. Nonnegative temperatures denote AI text. An explicit `label` is also accepted; if present alongside a temperature, the two must agree. The loader accepts either `text` or `generated_text`.

Retain the same `base_id` for temperature variants of one source prompt. Training and validation must have disjoint source groups. If no `base_id` is supplied, the loader can derive groups from `prompt`, `source_id`, `origin_id`, or `id`.

## Temp-HC3

The benchmark covers Llama-3, Mistral, and Gemma-2 at temperatures `0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5`.

| Split per generator | Source groups | Human | AI | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 8,000 | 8,000 | 104,000 | 112,000 |
| Validation | 975 | 975 | 12,675 | 13,650 |
| Test | 980 | 980 | 12,740 | 13,720 |

There are 418,110 records across the three generators, including repeated human answers across generator partitions. The default training configuration uses Llama-3 generations. Human repetition during training balances the 13:1 AI-to-human record ratio.

Expected file names include `dataset.llama3.train.jsonl`, `dataset.llama3.val.clean.jsonl`, and `dataset.llama3.test.clean.jsonl`. Generator-transfer evaluation also uses the corresponding `mistral` and `gemma2` test files.

## Evaluation records

Detection evaluation requires text and a source label; temperature is optional:

```json
{"id":"example_01_human","source_id":"example_01","text":"A human answer.","label":0}
{"id":"example_01_attacked","source_id":"example_01","text":"A paraphrased AI answer.","label":1}
```

For rewrite-quality analysis, AI records additionally need `origin_text`, containing the passage before paraphrasing. Variance probing needs the numeric `variance` field and source-group metadata.

| Evaluation population | Data |
| --- | --- |
| HC3 main benchmark | `dataset.no_attack.hc3.jsonl`, `dataset.dipper.hc3.jsonl`, `dataset.minp.mistral.hc3.jsonl`, `dataset.tp.hc3.jsonl`, `dataset.recursive_k5.hc3.jsonl` |
| Attack validation | `dataset.tp.hc3.val.jsonl` |
| Domain transfer | corresponding attack files with `squad` and `xsum` suffixes |
| Generator transfer | corresponding attack files with `llama3`, `mistral`, and `gemma2` suffixes |

The HC3 TP test set has 1,920 records, balanced between classes; TP validation has 1,916. Generator-specific TP files contain about 1,960 records each. The generator-specific DIPPER, Min-P, and recursive subsets contain 72 records each. Each benchmark output records the actual loaded and skipped counts.

## Validate inputs

```bash
python scripts/run.py check-data \
  "$T_AWARE_DATA_ROOT/dataset/dataset.llama3.train.jsonl" \
  "$T_AWARE_DATA_ROOT/dataset/dataset.llama3.val.clean.jsonl" \
  --empty-text keep \
  --output outputs/data-check.json
```

Empty-text policies are `error` (reject), `keep` (retain), and `skip` (filter and count). Default training retains empty records; attack evaluation configurations skip them. Malformed JSON and conflicting labels raise an error.

## Generate temperature-controlled examples

Prepare source rows containing `id`, `prompt`, and `human_answer`:

```json
{"id":"train_0001","prompt":"Example question","human_answer":"A human-written answer."}
```

Then generate AI answers at all 13 temperatures and retain one human anchor per source:

```bash
python -m pip install -e '.[generation]'
python scripts/run.py generate \
  --input /path/to/source-train.jsonl \
  --output outputs/llama3-train.jsonl \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --model-alias llama3 \
  --chat-template \
  --device cuda \
  --dtype bfloat16
```

Add `--resume` to continue an interrupted run with the same settings. Generation records the input fingerprint and settings and uses a per-source/temperature seed. Access to gated generation models must be configured separately.

Standalone no-attack, Min-P, and synonym-replacement producers are available under `scripts/data/`; inspect each script's `--help` for its model and input options. The detector accepts prepared paraphrasing outputs through the evaluation schema above.
