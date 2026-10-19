# Manuscript and available-code alignment

The ICASSP source supplied by the authors is titled **T-Aware: Temperature-Supervised Learning for Robust AI-Text Detection**. The earlier EMNLP source and author rebuttal are context only and are not distributed here.

| Item | ICASSP manuscript | Available main local script | Release handling |
| --- | --- | --- | --- |
| Temperature targets | normalized to [0,1] | raw generation_T | separate observed and paper-declared profiles |
| Epochs | 5 | maximum 50, patience 15 | preserve both explicitly |
| Batch size | 16 | 128 | preserve both explicitly |
| Warmup | 10% | no scheduler | separate profiles |
| Classification head | scalar sigmoid/BCE notation | 2 logits/CrossEntropyLoss | retain 2 logits for checkpoint compatibility; softmax AI probability defines equivalent binary likelihood notation, but not an identical parameterization |
| Backbone | RoBERTa-base | RoBERTa-base | retained |
| Loss mask | AI only | generation_T >= 0 | validate labels and temperatures; mask by AI class AND available target |
| Lambda | 10 | sweep including 10 | actual final local checkpoint is identical to named lambda=10 file |
| Test data | HC3/ChatGPT main vs three-generator transfer | different datasets spread across scripts | explicit evaluation matrices |
| F1 | footnote says averaging needs verification | different scripts/figures may use different conventions | output both AI-positive and macro F1; no silent substitution |
| Validation lambda curve | 82.7% at 10; 76.7% at zero | an older hardcoded plot uses another curve | original curve provenance unresolved; no hardcoded curve shipped as a new measurement |

The precise normalization formula is not specified in the manuscript. The new normalized profile explicitly uses `(T - 0.2) / 3.3`. This is an implementation choice for new experiments, not evidence of historical use.

The two-logit model, encoder parameter names and regression head were retained to load original checkpoints. Existing state_dict files do not encode training target normalization, scheduler settings, dataset versions or a random seed. Names alone cannot establish those settings.

The observed-profile code now resets the same seed per lambda, fails on malformed inputs and records provenance. Those changes improve control and diagnosability, but can change training trajectories. Neither profile is advertised as an exact bitwise rerun of the original training job.

Before claiming full table reproduction: match each manuscript table row to a source population, valid sample count, checkpoint hash, runtime config and structured output. The original sources and results have not been rewritten to force agreement.
