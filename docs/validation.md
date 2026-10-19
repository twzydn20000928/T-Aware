# Validation record

Release preparation date: 2026-10-19.

## Completed locally

- Offline CPU smoke: tiny RoBERTa forward/backward, AdamW update, save and strict reload; no downloads.
- 16 CPU tests, including an end-to-end tiny training run over two lambdas, scheduler, checkpoint selection, evaluate, and benchmark, plus failure/coverage handling for judge results without API calls. The training config intentionally includes a nonexistent test path: selection succeeds without opening it.
- Data checks: valid high temperature 3.0 remains AI; missing/conflicting labels fail; empty/whitespace handling is explicit; malformed JSON fails; paired source grouping and historical 13:1 human repetition are checked.
- Loss/metrics: human rows never contribute regression loss; all-human loss stays finite; raw loss matches the historical formula; normalized endpoints are checked; AI F1 and macro F1 are distinct; invalid API responses cannot become Human labels.
- Actual historical checkpoint compatibility: strict load of the available lambda=10 checkpoint (124,647,939 parameters), using its local RoBERTa config. Comparing the original `ablation_tp.py` forward function with the refactored forward on identical state and synthetic token IDs gives maximum logit difference **0.0** and loss difference **0.0**. This verifies model/loss compatibility, not dataset accuracy.
- All configured files in both evaluation matrices resolve in the inspected local asset copy: 15 dataset entries for main/domain evaluation and 15 for generator transfer.
- The actual Llama-3 training, validation and test files were read through the new checker from a different working directory: counts 112,000 / 13,650 / 13,720, with data SHA-256 recorded. This exercises migrated paths and schemas, not full-model dataset inference.
- All released Python files parse. Only source, configuration, tests, docs and attributed reference code are selected for publication; data, weights, manuscript/rebuttal files and credentials are excluded.

Local environment: Python 3.9.7, PyTorch 1.8.0 CPU, Transformers 4.31.0, NumPy 1.21.2. This is **not** a reconstruction of the historical AutoDL environment. Requirements are compatibility bounds, not a historical lockfile.

## Pending

- AutoDL SSH connectivity and environment reconciliation: the preparation session's network policy blocked the SSH destination before login.
- Full-dataset inference and comparison with all paper table values.
- GPU training, large-model generation, variance scoring, quality models and external baselines.
- Recovery of missing v2 weights and exact historical attack-generation provenance.
- CI after push. A workflow is provided, but its presence is not evidence of a completed GitHub Actions run.

No paid API requests or full research training jobs were made during release preparation.
