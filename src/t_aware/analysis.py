"""Surprisal statistics and grouped linear probes, separate from detection."""
from collections import defaultdict
from pathlib import Path

from .data import read_records
from .io import environment, fingerprint, write_json


def statistics(args):
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import choose_device
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    rows, counts = read_records(args.data, args.empty_text)
    if args.limit_groups:
        groups = list(dict.fromkeys(r["source_id"] for r in rows))[:args.limit_groups]
        rows = [r for r in rows if r["source_id"] in set(groups)]
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer)
    dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.scorer, torch_dtype=dtype).to(device).eval()
    features, grouped = [], defaultdict(list)
    with torch.no_grad():
        for row in tqdm(rows, desc="token statistics"):
            encoded = tokenizer(row["text"], truncation=True, max_length=args.max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            if encoded["input_ids"].shape[1] < 3:
                raise ValueError(f"Record {row['id']} has fewer than two next-token observations")
            logits = model(**encoded).logits[:, :-1].float()
            log_probs = logits.log_softmax(-1)
            targets = encoded["input_ids"][:, 1:]
            surprisal = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).flatten()
            predictive_entropy = -(log_probs.exp() * log_probs).sum(-1).mean().item()
            item = {"id": row["id"], "source_id": row["source_id"], "label": row["label"],
                    "generation_T": row["temperature"], "tokens": surprisal.numel(),
                    "mean_surprisal": surprisal.mean().item(), "surprisal_variance": surprisal.var(unbiased=True).item(),
                    "mean_predictive_entropy": predictive_entropy}
            features.append(item)
            key = "human" if row["label"] == 0 else str(row["temperature"])
            grouped[key].append(item)
    summary = {k: {"n": len(values), **{field: sum(r[field] for r in values) / len(values)
                for field in ("mean_surprisal", "surprisal_variance", "mean_predictive_entropy")}}
               for k, values in grouped.items()}
    write_json(args.output, {"data": fingerprint(args.data), "scorer": args.scorer,
               "scorer_revision": getattr(model.config, "_commit_hash", None), "max_length": args.max_length,
               "variance_estimator": "unbiased sample variance, matching legacy token_losses.var()",
               "statistics": counts, "selected_records": len(rows), "summary": summary,
               "records": features, "environment": environment()})


def probe(args):
    import numpy as np
    import torch
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoTokenizer
    from .engine import choose_device
    from .model import load_checkpoint
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    rows, stats = read_records(args.data, args.empty_text)
    rows = [r for r in rows if not args.ai_only or r["label"] == 1]
    if not rows or any(r["variance"] is None or r["source_id"] is None for r in rows):
        raise ValueError("Probe inputs must contain variance and source grouping metadata")
    if any(not isinstance(r["variance"], (int, float)) or not np.isfinite(r["variance"]) for r in rows):
        raise ValueError("Variance targets must be finite numbers")
    groups = np.array([r["source_id"] for r in rows])
    if len(set(groups)) < 3:
        raise ValueError("Need at least three independent source groups")
    device = choose_device(args.device)
    model = load_checkpoint(args.checkpoint, args.model_config).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    features = []
    with torch.no_grad():
        for row in rows:
            encoded = tokenizer(row["text"], truncation=True, padding="max_length", max_length=args.max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            hidden = model.roberta(**encoded).last_hidden_state[:, 0, :]
            features.append(hidden.cpu().numpy()[0])
    features = np.asarray(features)
    targets = np.array([r["variance"] for r in rows], dtype=float)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_ids, test_ids = next(splitter.split(features, targets, groups))
    if len(test_ids) < 2:
        raise ValueError("The held-out split needs at least two observations for R2")
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    estimator.fit(features[train_ids], targets[train_ids])
    predicted = estimator.predict(features[test_ids])
    write_json(args.output, {"r2": float(r2_score(targets[test_ids], predicted)), "n_train": len(train_ids),
               "n_test": len(test_ids), "split": "80/20 by source group, StandardScaler fitted on training only",
               "seed": args.seed, "ai_only": args.ai_only, "data": fingerprint(args.data),
               "checkpoint": fingerprint(args.checkpoint), "statistics": stats, "environment": environment(),
               "status": "new grouped-probe result; do not substitute for historical paper R2 without reconciliation"})


def register_analysis(commands):
    command = commands.add_parser("statistics", help="Compute surprisal variance AND predictive entropy explicitly")
    command.add_argument("--data", required=True)
    command.add_argument("--scorer", required=True)
    command.add_argument("--max-length", type=int, default=2048)
    command.add_argument("--device", default="auto")
    command.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    command.add_argument("--limit-groups", type=int)
    command.add_argument("--empty-text", choices=["error", "keep", "skip"], default="error")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=statistics)
    command = commands.add_parser("probe", help="Probe an existing variance-annotated dataset by source groups")
    command.add_argument("--data", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--model-config", default="roberta-base")
    command.add_argument("--tokenizer", default="roberta-base")
    command.add_argument("--max-length", type=int, default=128)
    command.add_argument("--device", default="auto")
    command.add_argument("--seed", type=int, default=42)
    command.add_argument("--ai-only", action="store_true")
    command.add_argument("--empty-text", choices=["error", "keep", "skip"], default="error")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=probe)
    command = commands.add_parser("quality", help="Unrescaled BERTScore and GPT-2 perplexity on paired AI rewrites")
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="auto")
    command.add_argument("--empty-text", choices=["error", "keep", "skip"], default="error")
    command.set_defaults(handler=quality)


def quality(args):
    import math
    import torch
    from bert_score import BERTScorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import choose_device
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    rows, stats = read_records(args.data, args.empty_text)
    pairs = [r for r in rows if r["label"] == 1]
    if not pairs or any(not isinstance(r["origin_text"], str) or not r["origin_text"].strip() for r in pairs):
        raise ValueError("Each AI rewrite needs a nonempty origin_text reference")
    device = choose_device(args.device)
    scorer = BERTScorer(model_type="roberta-base", lang="en", rescale_with_baseline=False, device=device)
    _, _, f1 = scorer.score([r["text"] for r in pairs], [r["origin_text"] for r in pairs])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
    losses = []
    with torch.no_grad():
        for row in pairs:
            encoded = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=1024)
            ids = encoded["input_ids"].to(device)
            if ids.shape[1] < 2:
                raise ValueError("Perplexity requires at least two tokens per passage")
            losses.append(model(ids, labels=ids).loss.item())
    write_json(args.output, {"data": fingerprint(args.data), "statistics": stats, "ai_pairs": len(pairs),
               "bertscore_f1": f1.mean().item(), "bertscore_model": "roberta-base", "rescale_with_baseline": False,
               "perplexity": math.exp(sum(losses) / len(losses)), "ppl_model": "gpt2", "max_length": 1024,
               "ppl_aggregation": "exp(mean(per-passage mean NLL)); matches historical quality script",
               "environment": environment()})
