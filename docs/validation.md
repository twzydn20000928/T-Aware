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

- Full-dataset inference and comparison with all paper table values.
- GPU training, large-model generation, variance scoring, quality models and external baselines.
- Recovery of missing v2 weights and exact historical attack-generation provenance.
- CI after push. A workflow is provided, but its presence is not evidence of a completed GitHub Actions run.

No paid API requests or full research training jobs were made during release preparation.

## Completed on AutoDL

SSH verification succeeded on 2026-10-19. The existing `/root/autodl-tmp` research assets were left intact. The isolated `/root/t-aware-release-20261019` directory contains the release code used for checking.

- Existing `acl_sprint` environment: Python 3.10.19, PyTorch 2.9.1, Transformers 5.13.1, NumPy 2.2.6. These are the versions observed today, not a historical training lockfile. The base Conda interpreter lacks Transformers; noninteractive SSH has no `python` on PATH.
- Offline tiny CPU smoke and all 16 CPU unit/integration tests passed.
- All 30 configured dataset entries exist and were schema-checked with their declared empty-text policy; all configured checkpoints exist. Dataset statistics and SHA-256 hashes are in [preflight evidence](validation/autodl_preflight.json).
- All 8 original-source hashes in `source_migration.json` match the corresponding AutoDL files, including `ablation_tp.py` and the three preserved attack producers. Hash identity does not certify that every script was historically executed.
- All 9 main checkpoint hashes were recorded. The available final and lambda=10 files match each other and the earlier local audit; the v2 directory contains no weights.
- Actual final checkpoint strict load: **124,647,939 parameters**. The extracted original forward method and release method produce maximum absolute differences of **0.0** for logits, temperature output and loss on identical synthetic inputs. The [compatibility report](validation/checkpoint_compatibility.json) records the checkpoint, config and original-source hashes. `scripts/verify_checkpoint.py` makes this check repeatable without importing the historical training script.
- Both actual lambda=10 and lambda=0 weights completed CPU inference on the same 8 real TP records (first 4 per class), using the cached tokenizer, max length 128 and batch size 1. Outputs were finite with valid AI probabilities. This [real-data smoke report](validation/real_data_smoke.json) records sample indices and data/weight hashes; it is a pipeline check, not a performance estimate.
- Two historical benchmark logs were parsed into [aggregate evidence](validation/historical_evidence.json), retaining log/source hashes. These rows are explicitly historical transcriptions, not fresh inference. Manuscript contradictions and interpretation problems are documented in [paper alignment](paper_alignment.md).

The no-card container is limited to **2 GiB RAM and 0.5 CPU cores** (`memory.max=2147483648`, `cpu.max=50000 100000`), despite much larger host-level figures. Initial full-weight attempts were killed before the memory-mapped loader was introduced. After that change, strict-load compatibility succeeded. A full TP rerun with batch size 4 was started and then interrupted because this CPU quota is unsuitable for timely full benchmarking; it produced no complete benchmark report and is not claimed as reproduced.

Core checkpoint tests cover both modern zip and legacy serialization. Transformers 5 support is validated for the core CPU workflows, not every optional generator or baseline. CI now includes 4.31.0 and 5.13.1; remote Actions completion must be checked separately.
