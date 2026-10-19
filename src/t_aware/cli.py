"""Command-line entry points. All data/model locations are explicit."""
import argparse
import json
import math
import tempfile
from pathlib import Path

from .data import assert_disjoint, balance_humans, read_records
from .io import asset_root, environment, fingerprint, input_path, read_config, write_json


def check(args):
    from collections import Counter
    import hashlib
    report, seen = [], []
    for filename in args.files:
        path = Path(filename).expanduser().resolve()
        rows, stats = read_records(path, args.empty_text)
        texts = {hashlib.sha256(r["text"].strip().encode("utf-8")).hexdigest() for r in rows if r["text"].strip()}
        sources = {r["source_id"] for r in rows if r["source_id"] is not None}
        types = Counter(str(r["type"]) for r in rows)
        report.append({"file": fingerprint(path), "statistics": stats, "types": dict(types),
                       "unique_texts": len(texts), "overlap_with_previous": [
                           {"file": old_name, "source_id_overlap": len(sources & old_sources),
                            "unique_text_overlap": len(texts & old_texts)}
                           for old_name, old_sources, old_texts in seen]})
        seen.append((path.name, sources, texts))
    result = {"normalization": "text.strip() exact equality; source IDs are not global semantic identities",
              "files": report}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _save_model(model, directory, metadata):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model.pt")
    model.config.save_pretrained(directory)
    write_json(directory / "run.json", metadata, overwrite=True)


def train(args):
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from .engine import TextDataset, choose_device, infer, seed_everything
    from .model import TemperatureAwareDetector

    cfg = read_config(args.config)
    root = asset_root(args.data_root)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run directory {output}")
    inputs = {k: input_path(root, cfg[k]) for k in ("train_file", "val_standard", "val_attack")}
    rows, stats = {}, {}
    for key, path in inputs.items():
        rows[key], stats[key] = read_records(path, cfg.get("empty_text", "error"))
    if any(r["label"] == 1 and r["temperature"] is None for r in rows["train_file"]):
        raise ValueError("AI training records must have generation_T")
    assert_disjoint(rows["train_file"], rows["val_standard"], "train/validation")
    assert_disjoint(rows["train_file"], rows["val_attack"], "train/attack-validation")
    scale = cfg.get("temperature_scale", "raw")
    if scale == "minmax_0.2_3.5" and any(r["label"] == 1 and not 0.2 <= r["temperature"] <= 3.5 for r in rows["train_file"]):
        raise ValueError("Normalized profile requires AI temperatures in [0.2, 3.5]")
    lambdas = cfg["lambdas"]
    if not lambdas or any(not math.isfinite(x) or x < 0 for x in lambdas) or len(set(lambdas)) != len(lambdas):
        raise ValueError("lambdas must be unique, finite, nonnegative values")
    if cfg["epochs"] < 1 or cfg["batch_size"] < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 <= cfg.get("warmup_ratio", 0) < 1:
        raise ValueError("warmup_ratio must be in [0, 1)")
    if cfg.get("checkpoint_policy", "best_validation") not in ("best_validation", "final"):
        raise ValueError("Unknown checkpoint policy")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    device = choose_device(args.device)
    training = balance_humans(rows["train_file"]) if cfg.get("balance_train", True) else rows["train_file"]
    validation = balance_humans(rows["val_standard"]) if cfg.get("balance_validation", True) else rows["val_standard"]
    output.mkdir(parents=True, exist_ok=False)
    tokenizer.save_pretrained(output / "tokenizer")
    provenance = {"profile": cfg, "environment": environment(), "inputs": {
        k: {"file": fingerprint(path), "statistics": stats[k]} for k, path in inputs.items()},
        "device": device, "effective_train_records": len(training), "effective_validation_records": len(validation),
        "seed_policy": "reset the same seed for each lambda", "test_used_for_selection": False}
    write_json(output / "manifest.json", provenance)
    std_loader = DataLoader(TextDataset(validation, tokenizer, cfg["max_length"]), batch_size=cfg["batch_size"])
    atk_loader = DataLoader(TextDataset(rows["val_attack"], tokenizer, cfg["max_length"]), batch_size=cfg["batch_size"])
    candidates = []
    for weight in lambdas:
        seed_everything(cfg["seed"])
        dataset = TextDataset(training, tokenizer, cfg["max_length"], scale,
                              cfg.get("random_temperature", False), cfg["seed"])
        loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
        model = TemperatureAwareDetector.pretrained(cfg["base_model"], weight).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                     weight_decay=cfg.get("weight_decay", 0.01))
        scheduler = None
        if cfg.get("warmup_ratio", 0):
            total = len(loader) * cfg["epochs"]
            warmup = max(1, int(total * cfg["warmup_ratio"]))
            def lr_multiplier(step):
                return step / warmup if step < warmup else max(0.0, (total - step) / max(1, total - warmup))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)
        run_dir = output / f"lambda_{float(weight):g}"
        best, stale, history, saved = -1.0, 0, [], None
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss = 0.0
            for batch in tqdm(loader, desc=f"lambda={weight} epoch={epoch}"):
                optimizer.zero_grad()
                batch = {k: value.to(device) for k, value in batch.items()}
                result = model(**batch)
                if not torch.isfinite(result["loss"]):
                    raise ValueError("Non-finite training loss")
                result["loss"].backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += result["loss"].item()
            standard, _ = infer(model, std_loader, device)
            attack, _ = infer(model, atk_loader, device)
            score = attack["ai_recall"]
            if score is None:
                raise ValueError("Attack validation contains no AI records")
            entry = {"epoch": epoch, "training_loss": total_loss / len(loader),
                     "validation_standard": standard, "validation_attack": attack}
            history.append(entry)
            print(json.dumps(entry))
            improved = score > best
            final_policy = cfg.get("checkpoint_policy", "best_validation") == "final"
            if improved or final_policy:
                saved = {"lambda": weight, "epoch": epoch, "validation_ai_recall": score,
                         "temperature_scale": scale, "profile_name": cfg["name"], "seed": cfg["seed"]}
                _save_model(model, run_dir, saved)
            if improved:
                best, stale = score, 0
            else:
                stale += 1
            patience = cfg.get("patience", 0)
            if patience and not final_policy and stale >= patience:
                break
        write_json(run_dir / "history.json", history)
        saved["checkpoint"] = str((run_dir / "model.pt").relative_to(output))
        saved["checkpoint_sha256"] = fingerprint(run_dir / "model.pt")["sha256"]
        candidates.append(saved)
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    winner = max(candidates, key=lambda r: r["validation_ai_recall"])
    write_json(output / "selection.json", {"selected": winner, "candidates": candidates,
               "selection_basis": "attack validation AI recall; ties choose first configured lambda",
               "test_evaluated": False})
    print(f"Training complete. Selection: {output / 'selection.json'}. Run evaluation separately.")


def evaluate(args):
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .engine import TextDataset, choose_device, infer
    from .model import load_checkpoint
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    rows, stats = read_records(args.data, args.empty_text)
    if args.balance:
        rows = balance_humans(rows)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_checkpoint(args.checkpoint, args.model_config).to(choose_device(args.device))
    loader = DataLoader(TextDataset(rows, tokenizer, args.max_length), batch_size=args.batch_size)
    metrics, predictions = infer(model, loader, choose_device(args.device))
    for row, prediction in zip(rows, predictions):
        prediction.update({"id": row["id"], "source_id": row["source_id"]})
    result = {"metrics": metrics, "data": fingerprint(args.data), "checkpoint": fingerprint(args.checkpoint),
              "statistics": stats, "effective_records": len(rows), "balance_humans": args.balance,
              "max_length": args.max_length, "environment": environment(), "predictions": predictions}
    write_json(args.output, result)
    print(json.dumps(metrics, indent=2))


def benchmark(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .engine import TextDataset, choose_device, infer
    from .model import load_checkpoint
    cfg, root = read_config(args.config), asset_root(args.data_root)
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    # Resolve every configured input before loading a model or starting inference.
    weights = [(m, input_path(root, m["checkpoint"])) for m in cfg["models"]]
    datasets = [(d, input_path(root, d["path"])) for d in cfg["datasets"]]
    reports = []
    device = choose_device(args.device)
    for spec, checkpoint in weights:
        tokenizer = AutoTokenizer.from_pretrained(spec.get("tokenizer", "roberta-base"))
        config_value = spec.get("model_config", "roberta-base")
        if config_value != "roberta-base":
            config_value = input_path(root, config_value)
        model = load_checkpoint(checkpoint, config_value).to(device)
        checkpoint_info = fingerprint(checkpoint)
        for entry, path in datasets:
            rows, stats = read_records(path, entry.get("empty_text", "error"))
            if entry.get("balance_humans", False):
                rows = balance_humans(rows)
            loader = DataLoader(TextDataset(rows, tokenizer, cfg.get("max_length", 128)),
                                batch_size=cfg.get("batch_size", 32))
            metrics, _ = infer(model, loader, device)
            report = {"model": spec["name"], "dataset": entry["name"], "protocol": entry["protocol"],
                      "checkpoint": checkpoint_info, "data": fingerprint(path), "statistics": stats,
                      "effective_records": len(rows), "metrics": metrics}
            reports.append(report)
            print(json.dumps({"model": spec["name"], "dataset": entry["name"], **metrics}))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output, {"configuration": cfg, "environment": environment(), "results": reports,
                            "status": "new evaluation; not a transcription of paper tables"})


def smoke(args):
    import torch
    from transformers import RobertaConfig
    from .model import TemperatureAwareDetector, load_checkpoint
    from .engine import seed_everything
    seed_everything(42)
    cfg = RobertaConfig(vocab_size=48, hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=32)
    model = TemperatureAwareDetector(cfg, 10.0)
    ids = torch.tensor([[0, 4, 5, 2], [0, 7, 8, 2]])
    mask = torch.ones_like(ids)
    result = model(ids, mask, torch.tensor([0, 1]), torch.tensor([-1.0, 1.2]))
    result["loss"].backward()
    torch.optim.AdamW(model.parameters(), lr=2e-5).step()
    model.eval()
    with tempfile.TemporaryDirectory() as directory:
        cfg.save_pretrained(directory)
        path = Path(directory) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = load_checkpoint(path, directory, 10.0).eval()
        with torch.no_grad():
            assert torch.allclose(model(ids, mask)["logits"], restored(ids, mask)["logits"], atol=1e-7)
    print(json.dumps({"status": "passed", "scope": "tiny random CPU model, backward, optimizer, checkpoint reload; no model downloads", "environment": environment()}, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="t-aware")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("check-data", help="Read-only schema, counts and duplicate checks")
    command.add_argument("files", nargs="+")
    command.add_argument("--empty-text", choices=["error", "keep", "skip"], default="error")
    command.add_argument("--output")
    command.set_defaults(handler=check)
    command = commands.add_parser("train", help="Train and select using validation only")
    command.add_argument("--config", required=True)
    command.add_argument("--data-root")
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="auto")
    command.set_defaults(handler=train)
    command = commands.add_parser("evaluate", help="Evaluate one trusted local checkpoint")
    command.add_argument("--data", required=True)
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--model-config", default="roberta-base")
    command.add_argument("--tokenizer", default="roberta-base")
    command.add_argument("--max-length", type=int, default=128)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--device", default="auto")
    command.add_argument("--empty-text", choices=["error", "keep", "skip"], default="error")
    command.add_argument("--balance", action="store_true", help="Apply historical human repetition for standard Temp-HC3")
    command.add_argument("--output", required=True)
    command.set_defaults(handler=evaluate)
    command = commands.add_parser("benchmark", help="Evaluate an explicit dataset/model matrix")
    command.add_argument("--config", required=True)
    command.add_argument("--data-root")
    command.add_argument("--output", required=True)
    command.add_argument("--device", default="auto")
    command.set_defaults(handler=benchmark)
    command = commands.add_parser("smoke", help="Offline CPU smoke test; no assets required")
    command.set_defaults(handler=smoke)
    from .analysis import register_analysis
    from .generation import register_generation
    from .baselines import register_baselines
    register_analysis(commands)
    register_generation(commands)
    register_baselines(commands)
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"error: {error}\n")
