# Baseline guide

The `baseline` command evaluates an external detector on JSONL records containing `text` and `label` (Human=0, AI=1). It exports metrics, per-example decisions, input fingerprints, settings, and environment versions.

## SuperAnnotate

```bash
python -m pip install -e '.[sa]'
python scripts/run.py baseline \
  --method sa \
  --model SuperAnnotate/roberta-large-llm-content-detector \
  --ai-output-label 1 \
  --data /path/to/test.jsonl \
  --empty-text skip \
  --output outputs/sa.json
```

The adapter uses the custom SuperAnnotate model architecture. `--ai-output-label` identifies which checkpoint output denotes AI; set it to match the model's label convention. The example uses AI=1.

## Fast-DetectGPT

```bash
python scripts/run.py baseline \
  --method fast-detectgpt \
  --model EleutherAI/gpt-neo-2.7B \
  --reference-stats /path/to/local_infer_ref \
  --data /path/to/test.jsonl \
  --output outputs/fast-detectgpt.json
```

This adapter uses analytic discrepancy with the selected scoring model also serving as the reference, followed by empirical calibration from reference-statistics JSON files. Supply calibration files appropriate to the model and protocol; their hashes are recorded in the result.

## Binoculars

```bash
python -m pip install -r third_party/Binoculars/requirements.txt
python scripts/run.py baseline \
  --method binoculars \
  --model vendored-defaults \
  --data /path/to/test.jsonl \
  --output outputs/binoculars.json
```

The included implementation uses Falcon-7B and Falcon-7B-Instruct, with its own threshold and device placement. Its configuration controls maximum length and devices. Run from the source checkout or an editable installation so `third_party/Binoculars/` is available. Its BSD 3-Clause license is retained.

## LLM-as-a-judge

Install the optional client and set `OPENAI_API_KEY` in your environment. `OPENAI_BASE_URL` can select a compatible provider.

```bash
python -m pip install -e '.[llm]'
python scripts/run.py baseline \
  --method llm \
  --model gpt-4o-mini \
  --data /path/to/test.jsonl \
  --attempts 3 \
  --output outputs/llm-judge.json
```

The parser accepts only `0` or `1`. Failed requests and invalid responses remain missing after bounded retries. Results include valid-request coverage and metrics on valid decisions; the command exits with an error if any record remains unresolved. Invoking this adapter sends passages to the configured API provider.

## Reporting

Use the same input population and record the model revision, input length, and threshold for each comparison. The default baseline input length is 512 tokens; T-Aware uses 128. Report AI recall alongside human recall or false-positive rate, and distinguish AI-positive F1 from macro F1.
