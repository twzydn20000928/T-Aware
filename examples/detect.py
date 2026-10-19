"""Classify one passage with a trained T-Aware checkpoint."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-config", default="roberta-base")
    parser.add_argument("--tokenizer", default="roberta-base")
    parser.add_argument("--text", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args(argv)
    if not args.text.strip():
        parser.error("--text must contain a nonempty passage")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")

    import torch
    from transformers import AutoTokenizer
    from t_aware.engine import choose_device
    from t_aware.model import load_checkpoint

    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_checkpoint(args.checkpoint, args.model_config).to(device).eval()
    tokens = tokenizer(args.text, truncation=True, max_length=args.max_length, return_tensors="pt")
    with torch.no_grad():
        result = model(tokens["input_ids"].to(device), tokens["attention_mask"].to(device))
        probabilities = result["logits"].softmax(-1)[0].cpu()
    print(json.dumps({"label": "ai" if probabilities.argmax().item() == 1 else "human",
                      "score_ai": probabilities[1].item(), "score_human": probabilities[0].item()}, indent=2))


if __name__ == "__main__":
    main()
