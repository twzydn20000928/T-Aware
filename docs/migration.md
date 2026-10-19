# Source migration

The original asset directory was preserved. Redundant models and per-script metric implementations were consolidated into `src/t_aware/`. Exploration, notebook backups, cached packages, optimization states and all data/weights are outside the release.

| Original role | Release entry point |
| --- | --- |
| ablation_tp.py model/training | model.py, engine.py, cli.py train |
| evaluate_defenses.py / evaluate_cross_model.py | evaluate and benchmark with explicit JSON matrices |
| dataset-size and leakage scripts | check-data (counts, schema and exact text/source overlaps) |
| analyze_entropy_generic.py | statistics (explicit predictive entropy vs surprisal) |
| variance probing scripts | probe (source-group holdout) |
| train_random_temp.py | train_random_control.json |
| generate_full_dataset.py | generate (per-source/temperature resume, explicit settings) |
| baseline adapters | baseline with named method |
| calc_text_quality.py | quality |
| hardcoded result plots | plot_results.py reads structured measurements |

`scripts/data/generate_no_attack.py`, `generate_minp_attack.py` and `generate_synonym_attack.py` are retained standalone producers with overwrite guards. Original-source fingerprints and transformation notes are in `source_migration.json`. They remain optional legacy producers until server validation. Importing/running the synonym producer may initialize NLTK resources.

Historical scripts that selected a model using test scores, overwrote data in place, assumed obsolete files, or repeated the same model class are not presented as supported release entry points. Their originals remain available in the owner's original directory for provenance review.

The original Binoculars files were copied from the local reference checkout, retaining its license and README. No upstream commit was inferred from filenames or modification dates. The SA architecture/loss adapters were retained from the original project, with package-relative imports and deferred type annotations for Python 3.9 compatibility.
