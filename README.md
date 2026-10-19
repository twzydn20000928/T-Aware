# T-Aware

**Temperature-Supervised Learning for Robust AI-Text Detection**

T-Aware shares a RoBERTa encoder between source classification and an auxiliary generation-temperature regressor. Human texts contribute only classification loss. Inference needs text only.

This repository organizes the existing research implementation associated with the EMNLP 2026 submission and ICASSP 2027 draft. When a manuscript conflicts with the experiments, the owner's AutoDL source, assets and logs take precedence. It includes checkpoint-compatible training and evaluation, dataset checks, generation, baseline adapters, variance analysis, and reproduction notes. It does **not** bundle datasets, model weights, private manuscript/review files, or credentials.

**Release status:** Local and AutoDL CPU tests and strict compatibility with the actual historical checkpoint have passed; evidence is recorded in [validation](docs/validation.md). Reproduction of all paper tables remains incomplete. Manuscript claims about normalized targets, batch size, epochs and warmup disagree with the server implementation; see [paper alignment](docs/paper_alignment.md).

## Quick start

Use an environment with a suitable PyTorch build. Core commands can run directly from the checkout without installing this package:

```bash
git clone https://github.com/twzydn20000928/T-Aware.git
cd T-Aware
pip install -r requirements.txt
python scripts/run.py --help
python scripts/run.py smoke
python -m unittest discover -s tests -v
```

The offline `smoke` command uses a tiny random CPU model: it does not download RoBERTa or train on research data. For an editable installation and the `t-aware` command, use `pip install -e .`. Optional extras are `analysis`, `quality`, `generation`, `sa`, and `llm`.

## Existing data and checkpoints

Keep the original project directory intact. `--data-root` points to that directory, which should contain `dataset/`, `test_data/`, and `saved_models_epoch50/`. It may be on another drive or outside this clone. You do not need to copy the 16 GB asset directory into Git.

```bash
python scripts/run.py check-data /path/to/original/dataset/dataset.llama3.train.jsonl --empty-text keep
python scripts/run.py benchmark --config configs/evaluate_main.json --data-root /path/to/original --device cpu --output outputs/main.json
python scripts/run.py benchmark --config configs/evaluate_cross_generator.json --data-root /path/to/original --device cpu --output outputs/generator-transfer.json
```

No-card AutoDL is sufficient for environment/data checks and tiny CPU tests. The inspected instance has a **2 GiB container memory limit** despite reporting much more host memory. Run checks serially and use a small inference batch. Modern PyTorch loads zip-format checkpoints through memory mapping to avoid a duplicate resident weight copy. Generation and training need an appropriate GPU environment. The original `saved_models_epoch50/best_defense_model_FINAL_TESTED.pt` and `model_lambda_10.0_best.pt` have identical SHA-256 checksums both locally and on AutoDL. See [checkpoints](docs/checkpoints.md).

`evaluate_main.json` separates HC3/ChatGPT source text from XSum/SQuAD conditions. `evaluate_cross_generator.json` separately evaluates Llama-3, Mistral and Gemma-2. Different populations and subset sizes must not be merged into one reported experiment.

## Training

```bash
python scripts/run.py train --config configs/train_observed_legacy.json --data-root /path/to/original --device cuda --output outputs/observed-run-01
```

This profile documents the available code's raw temperature targets, human oversampling and validation-based selection. Every lambda resets the same seed. Training outputs contain model config, tokenizer, checkpoint, validation history, data fingerprints and selection metadata. **Training never reads test sets**; evaluate the selected checkpoint in a separate command:

```bash
python scripts/run.py evaluate --data /path/to/original/test_data/dataset.tp.hc3.jsonl --checkpoint outputs/observed-run-01/lambda_10/model.pt --model-config outputs/observed-run-01/lambda_10 --tokenizer outputs/observed-run-01/tokenizer --empty-text skip --output outputs/selected-test.json
```

The example checkpoint path assumes lambda=10 is the desired run. Use the actual winner from `selection.json`; do not choose it using test scores.

`train_observed_legacy.json` follows the verified server script defaults: raw temperature, batch size 128, up to 50 epochs, patience 15 and no warmup. `train_hypothetical_normalized.json` is an optional **new experiment** based on unsupported manuscript settings; it is not a historical reproduction profile. It defaults to rejecting empty data. `train_random_control.json` follows the older five-epoch random-target script; a separate server script uses up to 50 epochs and validation selection. Neither establishes provenance of the draft's 51.3% result. See [reproduction](docs/reproduction.md).

## Analysis and baselines

```bash
python scripts/run.py statistics --data /path/to/original/dataset/dataset.llama3.test.clean.jsonl --scorer meta-llama/Meta-Llama-3-8B-Instruct --limit-groups 20 --device cuda --dtype bfloat16 --empty-text skip --output outputs/statistics.json
python scripts/run.py probe --data /path/to/original/dataset/dataset.llama3.test.with_variance.jsonl --checkpoint /path/to/original/saved_models_epoch50/best_defense_model_FINAL_TESTED.pt --output outputs/probe.json
```

`statistics` reports predictive entropy separately from mean token surprisal and surprisal variance. Human targets are never inferred from a valid AI temperature such as 3.0. `probe` uses a source-group holdout, so its results need not equal historical probes that used a different split.

`baseline` provides SA, Fast-DetectGPT, Binoculars and LLM-judge adapters. They are optional and are not exercised by the offline smoke test. [Baseline instructions](docs/baselines.md) explain label calibration and dependencies. API failures remain missing predictions with coverage, rather than being converted to Human. No API calls occur unless the LLM adapter is explicitly invoked.

## AutoDL preflight

From an existing AutoDL terminal, run:

```bash
python scripts/server_check.py --output /root/t-aware-preflight.json
```

The script searches common AutoDL project locations; if more than one matches, provide `--asset-root /path/to/original`. It records environment versions, input counts, checkpoint checksums, all configured evaluation populations, original-source hash comparisons and CPU test results. It does not install packages, edit source assets, start training, or read environment secrets. An existing output is never overwritten. On the inspected server, use `/root/miniconda3/envs/acl_sprint/bin/python`; a noninteractive SSH shell has no `python` command on PATH.

For a separate offline comparison with the trusted historical forward/loss implementation:

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/verify_checkpoint.py --checkpoint /path/to/original/saved_models_epoch50/best_defense_model_FINAL_TESTED.pt --model-config /path/to/local/roberta-base-snapshot --legacy-source /path/to/original/ablation_tp.py --output outputs/compatibility.json
```

Use the cached RoBERTa snapshot directory for both `--model-config` and `--tokenizer` when running `evaluate` offline. These checks have passed with Transformers 4.31.0 locally and 5.13.1 on AutoDL; optional generation and baseline adapters have not all been validated on Transformers 5.

## Data and results

- Temp-HC3 contains 418,110 records across three generators and 13 temperatures. These include repeated human anchors, not 418,110 independent prompts.
- Schema, known empty texts, dataset populations and subset caveats are documented in [data](docs/data.md).
- Metrics include `f1_ai` and `f1_macro`, with absent-class denominators reported as null. Undefined metrics must not be presented as observed zero performance.
- All result files produced here are new measurements with input/checkpoint fingerprints, not hardcoded transcriptions of paper tables.
- `scripts/plot_results.py` plots a selected protocol from a measured benchmark JSON.

Original exploratory scripts and checkpoints remain in the owner's original asset directory; duplicate notebook backups, caches, training states and redundant model definitions are excluded from this repository. The preserved attack-generation scripts are described in [migration](docs/migration.md).

## Attribution

The local Binoculars reference implementation is retained under `third_party/Binoculars/`, including its BSD 3-Clause license. Its upstream revision has not yet been recovered. No project-wide open-source license has been selected for T-Aware in this release; publishing this repository does not imply a new license grant. See [NOTICE](NOTICE.md).
