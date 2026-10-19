# Experiment guide

All experiments use `python scripts/run.py`. An editable installation also provides the equivalent `t-aware` command. Set `T_AWARE_DATA_ROOT` or pass `--data-root` for commands that use JSON experiment configurations.

## Train T-Aware and the classification ablation

```bash
export T_AWARE_DATA_ROOT=/path/to/t-aware-data

python scripts/run.py train \
  --config configs/train.json \
  --device cuda \
  --output outputs/t-aware

python scripts/run.py train \
  --config configs/train_classification_only.json \
  --device cuda \
  --output outputs/classification-only
```

| Setting | Default |
| --- | --- |
| Encoder | `roberta-base` |
| Classifier | linear layer, 2 logits, cross-entropy |
| Temperature regressor | linear layer, AI-only MSE |
| Temperature targets | raw `generation_T` |
| Loss weight | 10 for T-Aware, 0 for classification only |
| Optimizer | AdamW, learning rate 2e-5, weight decay 0.01 |
| Batch size / max length | 128 / 128 tokens |
| Maximum epochs / patience | 50 / 15 |
| Warmup | 0 |
| Seed | 42 |
| Checkpoint selection | highest AI recall on attack validation |

Human records are repeated using `max(1, N_ai // N_human)`. Training and validation files must have disjoint source groups. The standard validation set is balanced by human repetition; the attack validation set is evaluated as supplied. Test data is evaluated in a separate command.

Reduce `batch_size` in a copy of a training configuration if needed. Use a fresh `--output` directory for each experiment; existing run directories are protected against overwrite.

## Auxiliary-task ablations

```bash
python scripts/run.py train --config configs/train_lambda_sweep.json --device cuda --output outputs/lambda-sweep
python scripts/run.py train --config configs/train_random_temperature.json --device cuda --output outputs/random-temperature
python scripts/run.py train --config configs/train_normalized_temperature.json --device cuda --output outputs/normalized-temperature
```

The lambda sweep resets the same seed for each loss weight. The random-temperature configuration resamples AI targets from `U(0.2, 3.5)` on access and runs for five epochs, saving the final checkpoint. The optional normalized-target configuration uses `(T-0.2)/3.3`, five epochs, batch size 16, 10% warmup, and final-epoch checkpointing. These options are separate experimental configurations; compare runs using the settings recorded in their manifests.

## Evaluate a selected checkpoint

For a default run:

```bash
python scripts/run.py evaluate \
  --data "$T_AWARE_DATA_ROOT/test_data/dataset.tp.hc3.jsonl" \
  --checkpoint outputs/t-aware/lambda_10/model.pt \
  --model-config outputs/t-aware/lambda_10 \
  --tokenizer outputs/t-aware/tokenizer \
  --max-length 128 \
  --batch-size 32 \
  --device cuda \
  --empty-text skip \
  --output outputs/t-aware-tp.json
```

For a sweep, read the chosen path from `selection.json`. Output includes accuracy, AI/human recall, false-positive rate, AI precision, `f1_ai`, `f1_macro`, confusion matrix, per-example predictions, and data/checkpoint fingerprints. A metric is `null` when the required true class is absent.

## Benchmark matrices

`configs/evaluate_main.json` covers HC3 attacks and SQuAD/XSum domain transfer. `configs/evaluate_cross_generator.json` covers Llama-3, Mistral, and Gemma-2. Standard generator test sets repeat human examples for balance; attack sets use their supplied class distribution.

To benchmark newly trained weights, create a copy of a matrix and replace its `models` section. Use absolute paths or paths relative to the data root:

```json
{
  "name": "t_aware",
  "checkpoint": "/path/to/outputs/t-aware/lambda_10/model.pt",
  "model_config": "/path/to/outputs/t-aware/lambda_10/config.json",
  "tokenizer": "/path/to/outputs/t-aware/tokenizer"
}
```

```bash
python scripts/run.py benchmark --config configs/evaluate_main.json --device cuda --output outputs/main.json
python scripts/run.py benchmark --config configs/evaluate_cross_generator.json --device cuda --output outputs/transfer.json
```

The benchmark saves each model/dataset pair separately, including its population, counts, class-specific metrics, and fingerprints. Keep domain-transfer and generator-transfer results separate when reporting them.

## Token statistics

```bash
python -m pip install -e '.[analysis]'
python scripts/run.py statistics \
  --data "$T_AWARE_DATA_ROOT/dataset/dataset.llama3.test.clean.jsonl" \
  --scorer meta-llama/Meta-Llama-3-8B-Instruct \
  --limit-groups 20 \
  --device cuda \
  --dtype bfloat16 \
  --empty-text skip \
  --output outputs/statistics.json
```

The output distinguishes predictive-distribution entropy, mean token surprisal, and unbiased sample surprisal variance. Results are grouped by generation temperature, with human examples in a separate group.

## Representation probing

```bash
python scripts/run.py probe \
  --data "$T_AWARE_DATA_ROOT/dataset/dataset.llama3.test.with_variance.jsonl" \
  --checkpoint outputs/t-aware/lambda_10/model.pt \
  --model-config outputs/t-aware/lambda_10 \
  --tokenizer outputs/t-aware/tokenizer \
  --ai-only \
  --device cuda \
  --output outputs/probe.json
```

The probe extracts frozen first-token features, holds out 20% of source groups, fits scaling on training groups, and trains Ridge regression. Compare models with the same data and seed.

## Rewrite quality

```bash
python -m pip install -e '.[quality]'
python scripts/run.py quality --data /path/to/paired-rewrites.jsonl --device cuda --output outputs/quality.json
```

AI records need `text` and `origin_text`. The command reports unrescaled RoBERTa-base BERTScore F1 and GPT-2 perplexity, aggregated as the exponential of mean passage NLL.

## Plot benchmark outputs

```bash
python scripts/plot_results.py \
  --input outputs/main.json \
  --protocol 'HC3/ChatGPT main; Min-P uses Mistral' \
  --metric ai_recall \
  --output outputs/ai-recall.png
```

Plots read measured benchmark JSON. Choose an exact protocol string from that output. All result commands require a new output filename to protect existing measurements.
