# Optional baselines

The core CPU tests do not download or execute these models. Every baseline reads the same detector-ready schema; pass an explicit empty-text policy and dataset. The default maximum length is 512, compared with 128 for the historical T-Aware encoder. Record any length/threshold changes.

## SuperAnnotate

Install the `sa` extra and use the retained custom model architecture from the original project, rather than assuming a generic sequence-classification loader will match it. Confirm the checkpoint's output label convention before choosing `--ai-output-label`:

```bash
python scripts/run.py baseline --method sa --model SuperAnnotate/roberta-large-llm-content-detector --ai-output-label 1 --data /path/to/test.jsonl --empty-text skip --output outputs/sa.json
```

The example uses the historical AI=1 assumption; it is not independent verification of the downloaded model's convention. Record the checkpoint revision and inspect known-label sanity samples. Batch dimension is preserved even for the last single sample.

## Fast-DetectGPT

```bash
python scripts/run.py baseline --method fast-detectgpt --model EleutherAI/gpt-neo-2.7B --reference-stats /path/to/original/eval/fast_detect_gpt/local_infer_ref --data /path/to/test.jsonl --output outputs/fast.json
```

This adapter retains the local script's analytic discrepancy with scoring=reference and empirical calibration from local JSON statistics. It is not an assertion that all upstream Fast-DetectGPT reference-model settings have been reproduced. The existing reference filenames mention GPT-J/GPT-Neo; reconcile those reference-model assumptions before claiming protocol equivalence. Reference-file hashes are saved in the output.

## Binoculars

```bash
pip install -r third_party/Binoculars/requirements.txt
python scripts/run.py baseline --method binoculars --model vendored-defaults --data /path/to/test.jsonl --output outputs/binoculars.json
```

The retained reference code uses Falcon-7B/Falcon-7B-Instruct and its own threshold/device placement. It may need substantially more memory than T-Aware; it has not been exercised on the no-card server. The `--device` and `--max-length` common flags do not override the vendored implementation's own configuration. Run from the source checkout/editable install; vendor files are not packaged in a wheel. Its BSD license is retained.

## LLM judge

Install the `llm` extra. Set `OPENAI_API_KEY` in the shell; optionally set `OPENAI_BASE_URL` for a compatible provider. Do not put credentials in scripts or Git. Invoke an exact model ID:

```bash
python scripts/run.py baseline --method llm --model gpt-4o-mini --data /path/to/test.jsonl --output outputs/judge.json
```

Only an exact `0` or `1` response is accepted. Connection failures and invalid responses are retried a bounded number of times and remain missing if unsuccessful. The output records coverage, failed count and metrics on valid decisions; the process exits nonzero if any requests fail. Historical API failures cannot be repaired by copying old aggregate scores into this output.
