# T-Aware & Temp-HC3 v0.1.0

This release provides the complete Temp-HC3 dataset, companion robustness benchmarks,
and two trained detector checkpoints for the T-Aware implementation.

## Downloadable resources

| Asset | Contents |
| --- | --- |
| `temp-hc3-v0.1.0.tar.gz` | Complete Temp-HC3: all three generators, 13 temperatures, and all fixed train/validation/test splits; 418,110 records in 9 JSONL files |
| `t-aware-robustness-v0.1.0.tar.gz` | Companion attack/domain/generator evaluations, attack validation, variance annotations, and prepared source files |
| `t-aware-roberta-base-v0.1.0.tar.gz` | T-Aware checkpoint with lambda=10 and the matching RoBERTa configuration/tokenizer |
| `t-aware-classification-only-v0.1.0.tar.gz` | Classification-only checkpoint with lambda=0 and the matching configuration/tokenizer |
| `assets-manifest.json` | Archive and individual-file checksums, byte sizes, fields, counts, class labels, and temperature coverage |
| `SHA256SUMS` | SHA-256 checksums for the four archives and manifest |

Temp-HC3 is a standalone dataset; its own validation and test splits are included in
the complete dataset download. The robustness package is an optional companion resource.

## Quick start

```bash
git clone https://github.com/twzydn20000928/T-Aware.git
cd T-Aware
python -m pip install -e .
python scripts/download_assets.py --assets all --output /path/to/t-aware-data
export T_AWARE_DATA_ROOT=/path/to/t-aware-data
python examples/detect.py \
  --checkpoint "$T_AWARE_DATA_ROOT/saved_models_epoch50/best_defense_model_FINAL_TESTED.pt" \
  --model-config "$T_AWARE_DATA_ROOT/roberta-base" \
  --tokenizer "$T_AWARE_DATA_ROOT/roberta-base" \
  --text "A passage to classify." --device cpu
```

The downloader verifies archive and extracted-file hashes. It can resume interrupted
downloads and reuses matching files. To fetch Temp-HC3 independently, select
`--assets temp-hc3`; to fetch only the main detector, select `--assets t-aware`.

See the [Temp-HC3 dataset card](https://github.com/twzydn20000928/T-Aware/blob/main/docs/temp-hc3.md)
for construction, schema, split statistics, and source attribution. Original HC3 and
other upstream source rights are retained. Detector weights and research text should
be used in accordance with the corresponding source and model terms.
