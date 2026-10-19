# Historical checkpoints

Weights are not committed to Git. Put trusted local files under your external asset root.

| Role | Asset-relative path | SHA-256 verified locally and on AutoDL |
| --- | --- | --- |
| Classification-only | `saved_models_epoch50/model_lambda_0.0_best.pt` | `3ec6228b0a956785b7d5a426488c69d06d2010e862907d008eaa266d6b3ab1e5` |
| Available final | `saved_models_epoch50/best_defense_model_FINAL_TESTED.pt` | `db8ec3ea319a5390b4646e5f0674f0f7a936aa91b9ceaea63e17dd666cfd5b0a` |
| Named lambda=10 | `saved_models_epoch50/model_lambda_10.0_best.pt` | same as available final |
| Named lambda=12 | `saved_models_epoch50/model_lambda_12.0_best.pt` | `0019edf01b930841f0139241f7a8f27bda4d3aea94121a83a177171087d28ddd` |

A historical training log reports a lambda=12 winner, 99.08% standard accuracy and 71.98% attack recall saved under `saved_models_epoch50_v2/`. Both the local and AutoDL copies of that directory contain no weights. These values cannot be attributed to the available lambda=10 file without additional evidence.

New training saves `model.pt`, `config.json`, validation history and metadata inside each lambda directory. The tokenizer is stored once per run. Evaluation accepts legacy state_dict files and DataParallel `module.` prefixes with strict parameter checking.

The regression head is retained when loading a detector checkpoint; temperature is not an inference input. A raw-target regression output is in original temperature units; a normalized-target output is on the explicitly configured target scale. Do not interpret an unlabelled head output as calibrated temperature.

On PyTorch versions that support it, checkpoint loading uses `weights_only=True`. Zip-format checkpoints use memory mapping and tensor assignment when available, avoiding a duplicate resident weight copy on small CPU instances. Old serialization falls back to regular loading; both formats are covered by the round-trip test. Older environments should load only the owner's trusted checkpoints.
