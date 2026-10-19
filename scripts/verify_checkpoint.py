"""Offline strict-load and historical forward/loss compatibility check on CPU."""
import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from t_aware.io import environment, fingerprint, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", required=True, help="Local RoBERTa config directory or JSON")
    parser.add_argument("--legacy-source", required=True, help="Trusted historical ablation_tp.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    import torch
    from torch import nn
    from t_aware.model import load_checkpoint

    # Extract only the original forward method; never import the training script.
    tree = ast.parse(Path(args.legacy_source).read_text(encoding="utf-8-sig"))
    model_class = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "TemperatureAwareDetector")
    forward = next(node for node in model_class.body
                   if isinstance(node, ast.FunctionDef) and node.name == "forward")
    namespace = {"torch": torch, "nn": nn}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(args.legacy_source), "exec"), namespace)
    model = load_checkpoint(args.checkpoint, args.model_config, lambda_reg=10.0).eval()
    ids = torch.tensor([[0, 4, 5, 2], [0, 7, 8, 2]])
    mask, labels, targets = torch.ones_like(ids), torch.tensor([0, 1]), torch.tensor([-1.0, 1.2])
    with torch.no_grad():
        actual = model(ids, mask, labels, targets)
        legacy = namespace["forward"](model, ids, mask, labels, targets)
    differences = {"logits_max_abs": (actual["logits"] - legacy[1]).abs().max().item(),
                   "temperature_max_abs": (actual["temperature"] - legacy[2]).abs().max().item(),
                   "loss_abs": (actual["loss"] - legacy[0]).abs().item()}
    passed = all(value <= 1e-6 for value in differences.values())
    report = {"status": "passed" if passed else "failed", "environment": environment(),
              "checkpoint": fingerprint(args.checkpoint), "legacy_source": fingerprint(args.legacy_source),
              "model_config": fingerprint(Path(args.model_config) / "config.json"
                                          if Path(args.model_config).is_dir() else args.model_config),
              "strict_load": True, "parameters": sum(p.numel() for p in model.parameters()),
              "differences": differences,
              "scope": "Synthetic CPU inputs; verifies model/loss compatibility, not paper accuracy or training provenance"}
    write_json(args.output, report)
    print(report)
    if not passed:
        raise SystemExit("Historical forward/loss compatibility failed")


if __name__ == "__main__":
    main()
