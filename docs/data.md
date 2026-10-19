# Dataset layout and populations

Pass the original asset directory via `--data-root` or `T_AWARE_DATA_ROOT`. Relative asset paths in JSON evaluation/training configs are resolved against that explicit root, not against the script's directory. No research datasets are bundled.

## Temp-HC3 generated text

Generated JSONL records have `id`, `base_id`, `prompt`, `model`, `generation_T`, `generated_text`. Human anchors use temperature -1; all nonnegative temperatures represent AI. AI temperatures are 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0 and 3.5.

| Per generator | Records | AI | Human | base_id groups |
| --- | ---: | ---: | ---: | ---: |
| train | 112,000 | 104,000 | 8,000 | 8,000 |
| val.clean | 13,650 | 12,675 | 975 | 975 |
| test.clean | 13,720 | 12,740 | 980 | 980 |

Three generators produce 418,110 records in total. Human anchors are repeated across generators. Historical human repetition produces 208,000 effective training, 25,350 validation and 25,480 standard-test observations per generator. These effective observations are not independent samples.

## Attack data

Attack JSONL requires `text`, `label` (Human=0, AI=1). Some files also contain id/source metadata, `origin_text`, type and generation_T. Paired human/AI rows can intentionally share an ID; group them by source rather than splitting rows randomly.

| Population | Observed local size | Caveat |
| --- | ---: | --- |
| HC3 TP validation | 1,916 (958 per class) | separate from testing |
| HC3 TP test | 1,920 (960 per class) | original ChatGPT population |
| Llama-3 and Mistral TP | 1,960 each | different population from main HC3/ChatGPT |
| Gemma TP | 1,959 (980 Human, 979 AI) | one missing AI row |
| DIPPER, Min-P, recursive for each Temp-HC3 generator | 72 each (36 per class) | small subsets |
| variance-annotated Llama test | 280 rows, 20 base_id groups | probe subset, not full test |

Files named `dataset.recursive_k5.{llama3,mistral,gemma2}.jsonl` label their AI type as `ai_recursive_k2`. Until generation provenance is recovered, the release calls this depth disputed rather than certifying K=5.

## Data checks and empty text

The inspected files have valid JSON syntax, but contain empty texts. Generated Gemma train/val.clean/test.clean contain 165/16/20 empty records; Llama train has 4; Mistral train/val.clean/test.clean contain 14/2/2. HC3 DIPPER has 1, and Llama TP has 2. Check results may differ if whitespace-only text is counted; the new checker counts it as empty.

The loader defaults to error. Explicit `keep` preserves empty passages for historical standard-data behavior; `skip` filters them and records how many were removed. Main attack evaluation configs explicitly choose skip; standard Temp-HC3 evaluation uses keep plus historical human repetition. Modified data must receive a new version rather than overwriting source files.

Exact prompt intersections between generated train/val.clean/test.clean are zero in the inspected copy. There are nevertheless duplicate question texts inside some splits and duplicate passage texts across splits. For example, the Llama training and main HC3 TP test files share 20 distinct human text values after stripping. This warrants source-level review; it neither proves leakage by itself nor supports a claim of complete text decontamination.

HC3 clean source datasets and generated clean datasets do not have identical membership. Match records by their source lineage and prompt, not just by a count or a shared integer suffix.

The checker accepts detector-ready generated/attack schemas. Raw HC3 `question`/`human_answers` files must first be prepared into explicit source splits. This release does not silently regenerate or replace those existing splits. Recover the exact historical split/attack producers before claiming end-to-end dataset reproduction.
