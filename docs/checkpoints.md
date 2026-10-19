# Checkpoints and inference

Detector weights are not bundled with the source checkout and are not currently hosted for public download in this repository. You can obtain a compatible checkpoint by training with the provided configurations.

## Training outputs

```text
outputs/t-aware/
├── tokenizer/
├── selection.json
└── lambda_10/
    ├── model.pt
    ├── config.json
    ├── run.json
    └── history.json
```

`model.pt` stores the detector state dictionary, including the encoder, two-class classifier, and temperature regressor. `config.json` stores the backbone configuration. `selection.json` identifies the validation-selected checkpoint in the run directory.

## Classify a passage

```bash
python examples/detect.py \
  --checkpoint outputs/t-aware/lambda_10/model.pt \
  --model-config outputs/t-aware/lambda_10 \
  --tokenizer outputs/t-aware/tokenizer \
  --text "A passage to classify." \
  --device cpu
```

The result contains `label`, `score_ai`, and `score_human`. The scores are softmax outputs, not calibrated confidence estimates. The classifier selects the higher-scoring class. No generation-temperature input is needed.

## Reference checkpoint names

The benchmark configurations expect these paths under the external data root:

| Model | Path |
| --- | --- |
| T-Aware | `saved_models_epoch50/best_defense_model_FINAL_TESTED.pt` |
| Classification only | `saved_models_epoch50/model_lambda_0.0_best.pt` |

These reference state dictionaries use the `roberta-base` configuration and tokenizer. They can be used directly with `--model-config roberta-base --tokenizer roberta-base`. The loader accepts plain state dictionaries, a `state_dict` wrapper, and DataParallel `module.` prefixes; parameter loading is strict.

## Local model cache

For offline evaluation, use a downloaded RoBERTa snapshot directory for both `--model-config` and `--tokenizer`:

```bash
HF_HUB_OFFLINE=1 python examples/detect.py \
  --checkpoint /path/to/model.pt \
  --model-config /path/to/roberta-base-snapshot \
  --tokenizer /path/to/roberta-base-snapshot \
  --text "A passage to classify." \
  --device cpu
```

Modern PyTorch uses memory-mapped loading where supported. On a smaller CPU machine, run one model at a time and lower the evaluation batch size. GPU training and inference use the same model definition.
