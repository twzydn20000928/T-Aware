# Manuscript and AutoDL evidence alignment

Audit date: 2026-10-19. The owner supplied the EMNLP manuscript, the ICASSP draft and the complete rebuttal, and explicitly designated **AutoDL source, assets and logs as authoritative when they disagree with the drafts**. Manuscripts and review correspondence are private reference material and are not distributed here.

## Training and architecture

The discrepancies below occur in both supplied manuscripts. Settings in a script are evidence of the implementation, not proof of the exact command used to train a particular checkpoint.

| Item | Manuscript claim | Verified AutoDL implementation | Release decision |
| --- | --- | --- | --- |
| Temperature targets | normalized to [0,1] | `ablation_tp.py` passes raw `generation_T` to MSE | raw targets in historical profile |
| Epochs | 5, no early stop | default maximum 50, patience 15 | retain server defaults |
| Batch size | 16 | default 128 | retain server default |
| Warmup | 10% | AdamW without a scheduler | warmup 0 in historical profile |
| Classification | scalar sigmoid/BCE | two logits and CrossEntropyLoss | retain checkpoint architecture |
| Backbone | RoBERTa-base | RoBERTa-base, first token embedding | retained |
| Regression mask | AI only | `generation_T >= 0` | validate labels and mask AI targets |
| Balancing | repeated humans; EMNLP appendix also says ratio defaults to 0.5 | `max(1, N_ai // N_human)` | integer repetition; observed training ratio is 13 |
| Lambda | 10 | available final checkpoint is byte-identical to lambda=10 | verify hashes; full training invocation unproven |
| Historical GPU | RTX 5090, 32GB | current session is CPU-only | historical hardware unverified |
| Mistral version | v0.3 | cache and Min-P producer default use `Mistral-7B-Instruct-v0.2` | do not certify v0.3; recover per-file provenance |

`train_observed_legacy.json` is the default reproduction starting point. `train_hypothetical_normalized.json` is explicitly an optional **new experiment**, not a correction to the historical run. Its `(T-0.2)/3.3` formula is an implementation choice; neither manuscript specifies it. The old `train_paper_declared.json` filename was removed to avoid implying verified provenance.

## Results and analysis

| Claim or measurement | AutoDL evidence | Interpretation |
| --- | --- | --- |
| Main T-Aware DIPPER recall 95.1%, TP recall 83.5% | April 22 benchmark log records 95.10% / 83.54% | historical log support; label fresh measurements separately |
| Main TP ablation recall 78.3% | same log records 78.33% | compare the same HC3/ChatGPT population |
| TP transfer recall 96.93 / 90.31 / 96.32% | July 15 log, Llama/Mistral/Gemma | different populations from main table |
| F1 called macro in rebuttal | `eval/evaluate_defenses.py` and GPT judge compute AI-positive precision/recall F1 | historical T-Aware/GPT values are `f1_ai`; audit external detectors separately |
| Probe R2=0.810 vs 0.478 | both probe scripts fit RidgeCV on all features and predict those same features | in-sample fit after CV alpha selection, not independent test R2 or causal proof |
| GPT-4o-mini near-zero AI recall | old judge returns Human on API exceptions or malformed provider responses | errors can contaminate decisions; rerun with valid-response coverage |
| Random-target TP recall 51.3% | two random-target scripts exist; no corresponding weight/result found in inspected tree | unverified; 5-epoch and 50-epoch scripts are different protocols |
| Lambda curve 82.7% at 10, 76.7% at zero | current plot evaluates validation weights; backup has a different hardcoded curve | exact claimed curve not recovered; no invented measurements |
| Later validation run selects lambda=12 | April 22 training log: 99.08% clean accuracy / 71.98% attack recall; v2 directory has no weights | do not attribute these to available lambda=10 weights |
| Generator recursive files denote K=5 | filenames say k5; AI type metadata says k2 | depth remains disputed |

Aggregate historical rows and source hashes are preserved in [historical evidence](validation/historical_evidence.json). These are log transcriptions, not newly reproduced results. New benchmark outputs include checkpoint/data hashes and observed sample counts. The release outputs both AI-positive and macro F1.

The supported probe uses an 80/20 source-group holdout with scaling fitted only on training groups. Its score may differ substantially from the manuscript's in-sample number. Neither a probe nor a t-SNE plot establishes causal use of variance or disentanglement.

Some generator attack subsets contain only 36 AI passages. Do not merge populations, certify disputed recursive depths, or infer universal SOTA from aggregate logs.

## Reproduction boundary

The server files establish what is available today. Full historical training commands, missing weights, generation revisions, failed judge requests and exact draft figures cannot be recovered from filenames alone. Preserve those gaps instead of modifying code or data to match a draft. See [validation](validation.md) for completed checks and practical limits.
