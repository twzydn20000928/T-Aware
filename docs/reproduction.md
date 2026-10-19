# Reproduction workflow

1. Keep an untouched original asset tree. Record the actual AutoDL environment before installing or updating dependencies.
2. Run `scripts/server_check.py`; inspect counts and hashes against the local manifest. A no-card instance can run these read-only CPU checks.
3. Confirm which checkpoint and dataset populations produced each target table row. Use the available lambda=10 checksum for checkpoint-compatible inference. Do not replace missing v2 weights with a differently trained model under the same name.
4. Run main/domain and generator-transfer benchmarks separately, with a new output filename for each run. Inspect loaded/skipped counts, Human recall, AI recall and F1 convention.
5. Only retrain if needed. `train_observed_legacy.json` follows the verified server defaults. `train_hypothetical_normalized.json` is an optional new experiment based on unsupported manuscript settings; it is not used to explain historical results. It rejects existing empty texts, so prepare a versioned clean copy or make an explicit recorded empty-text-policy decision first.
6. Training reads training and validation files only. The selected checkpoint is recorded in `selection.json`; test evaluation is a separate invocation. Do not use a test score to select a lambda, epoch, threshold, seed or normalization formula.
7. Save data/checkpoint hashes, exact model revisions, runtime package versions and commands with every result. Model snapshots and tokenizers should be cached for repeatable offline inference.

## Result interpretation

The main HC3/ChatGPT protocol is different from generated Llama/Mistral/Gemma transfer and the separate prompted LLM-judge evaluation. Numbers from these populations are not interchangeable. Some attack subsets have only 36 AI passages; report sample counts and uncertainty where appropriate.

Report `f1_ai` or `f1_macro` by name. Macro F1 is undefined here when a true class is absent; it is null. API-judge results include coverage and invalid/failed statuses, with metrics conditional on valid requests. Retry or rerun failed requests before describing the result as full-dataset performance.

## Analysis

Historical scripts called mean token negative log-probability “entropy.” The new statistics command separately reports mean surprisal, unbiased sample surprisal variance and mean predictive-distribution entropy. This clarifies quantities; it does not retroactively change the manuscript figure's measurements.

Ridge probes use held-out source groups, with scaling fitted on training data only. Compare lambda=0 and lambda=10 using exactly the same grouped split. AutoDL's historical probe used RidgeCV for alpha selection, then fitted and scored on the same full feature set. Its R2 is in-sample, not an independent evaluation. The new split need not reproduce R2=0.810; recoverable variance does not establish causal use or disentanglement.

`quality` requires paired AI `text` and `origin_text`, calculates unrescaled RoBERTa-base BERTScore F1 and GPT-2 perplexity, and retains the historical passage-mean NLL aggregation. Perplexity is not proof that a rewrite preserves meaning.

## Generation

```bash
python scripts/run.py generate --input /path/to/hc3.train.jsonl --output outputs/new-generations.jsonl --model meta-llama/Meta-Llama-3-8B-Instruct --model-alias llama3 --device cuda --dtype bfloat16
```

Source input requires nonempty string `id`, `prompt` and `human_answer`. Resume with the same command plus `--resume`. Resume keys include both source and temperature, and configuration/input hashes must match. Per-source/temperature seeds make missing entries reproducible independently of the number of prior entries. This explicitly seeded implementation is a new generation pipeline; it does not claim byte-identical reconstruction of historical model-default sampling.

The original standalone No Attack, Min-P and synonym scripts are retained under `scripts/data/` with an added guard against overwriting existing outputs. Their additional models/NLTK dependencies are optional and have not been run in the release's offline validation. The exact historical TempParaphraser/DIPPER/recursive generation scripts were not found in either the inspected local copy or the inspected server source tree; recover their exact versions and configurations before claiming dataset reconstruction.
