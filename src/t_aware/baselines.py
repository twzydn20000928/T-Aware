"""Optional baseline adapters. Failures never become human predictions."""
import json
import os
import re
import time
from pathlib import Path

from .data import read_records
from .io import environment, fingerprint, write_json
from .metrics import binary_metrics


def parse_judge_label(text):
    if text is None or re.fullmatch(r"[01]", text.strip()) is None:
        raise ValueError("Judge must return exactly 0 or 1")
    return int(text.strip())


def _llm_predictions(rows, args):
    from openai import OpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY in the environment")
    client = OpenAI(base_url=os.environ.get("OPENAI_BASE_URL"), max_retries=0, timeout=30.0)
    outputs = []
    for row in rows:
        predicted = None
        error_type = None
        resolved_model = None
        for attempt in range(args.attempts):
            try:
                response = client.chat.completions.create(model=args.model,
                    messages=[{"role": "user", "content":
                        "You are an expert AI text detector. Return ONLY the digit 1 if the text is AI-generated, "
                        "or ONLY the digit 0 if the text is human-written. Do not provide any other text.\n\nText:\n" + row["text"]}],
                    temperature=0.0, max_tokens=5)
                predicted = parse_judge_label(response.choices[0].message.content)
                resolved_model = getattr(response, "model", None)
                error_type = None
                break
            except Exception as error:
                error_type = type(error).__name__
                if attempt + 1 < args.attempts:
                    time.sleep(min(2 ** attempt, 4))
        outputs.append({"id": row["id"], "label": row["label"], "prediction": predicted,
                        "error_type": error_type, "resolved_model": resolved_model})
    return outputs


def _sa_predictions(rows, args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from .engine import TextDataset, choose_device
    from .sa_model import RobertaClassifier
    if args.ai_output_label is None:
        raise ValueError("SA requires --ai-output-label 0 or 1, verified against the model's label convention")
    device = choose_device(args.device)
    model = RobertaClassifier.from_pretrained(args.model).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    loader = DataLoader(TextDataset(rows, tokenizer, args.max_length), batch_size=args.batch_size)
    predictions = []
    with torch.no_grad():
        for batch in loader:
            result = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
            logits = result[1] if isinstance(result, tuple) else result.logits
            predicted = (logits.sigmoid().squeeze(-1) > 0.5).long() if logits.shape[-1] == 1 else logits.argmax(-1)
            predictions.extend((predicted == args.ai_output_label).long().cpu().tolist())
    return [{"id": r["id"], "label": r["label"], "prediction": p} for r, p in zip(rows, predictions)]


def _fast_predictions(rows, args):
    import math
    import torch
    import numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .engine import choose_device
    if not args.reference_stats:
        raise ValueError("Fast-DetectGPT requires --reference-stats")
    real, fake = [], []
    files = sorted(Path(args.reference_stats).glob("*.json"))
    if not files:
        raise ValueError("No Fast-DetectGPT reference JSON files")
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        real.extend(data["predictions"]["real"])
        fake.extend(data["predictions"]["samples"])
    if len(real) + len(fake) < 101:
        raise ValueError("Reference calibration needs at least 101 scores")
    device = choose_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    real, fake = np.asarray(real), np.asarray(fake)
    predictions = []
    for row in rows:
        encoded = tokenizer(row["text"], truncation=True, max_length=args.max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if encoded["input_ids"].shape[1] < 2:
            raise ValueError("Fast-DetectGPT requires at least two tokens")
        with torch.no_grad():
            logits = model(**encoded).logits[:, :-1].float()
            log_probs = logits.log_softmax(-1)
            probs = log_probs.exp()
            observed = log_probs.gather(-1, encoded["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
            expected = (probs * log_probs).sum(-1)
            variance = (probs * log_probs.square()).sum(-1) - expected.square()
            discrepancy = ((observed.sum() - expected.sum()) / variance.sum().clamp_min(1e-12).sqrt()).item()
        offset = np.sort(np.abs(np.concatenate([real, fake]) - discrepancy))[100]
        n_real = np.sum((real > discrepancy - offset) & (real < discrepancy + offset))
        n_fake = np.sum((fake > discrepancy - offset) & (fake < discrepancy + offset))
        probability = float(n_fake / (n_real + n_fake)) if n_real + n_fake else 0.5
        predictions.append({"id": row["id"], "label": row["label"], "prediction": int(probability > 0.5), "score_ai": probability})
    return predictions


def _binoculars_predictions(rows, args):
    import sys
    vendor = Path(__file__).resolve().parents[2] / "third_party/Binoculars"
    if not vendor.is_dir():
        raise FileNotFoundError("Run from an editable installation or install the vendored Binoculars dependency")
    sys.path.insert(0, str(vendor))
    from binoculars import Binoculars
    if args.ai_output_label is not None:
        raise ValueError("--ai-output-label is only used for SA")
    if args.model != "vendored-defaults":
        raise ValueError("Use --model vendored-defaults: observer=falcon-7b, performer=falcon-7b-instruct")
    model = Binoculars()
    outputs = []
    for row in rows:
        label = model.predict(row["text"])
        if isinstance(label, list):
            label = label[0]
        if label not in ("Most likely human-generated", "Most likely AI-generated"):
            raise ValueError(f"Unrecognized Binoculars prediction: {label}")
        outputs.append({"id": row["id"], "label": row["label"], "prediction": int(label == "Most likely AI-generated")})
    return outputs


def baseline(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.attempts < 1:
        raise ValueError("attempts must be positive")
    rows, stats = read_records(args.data, args.empty_text)
    runners = {"llm": _llm_predictions, "sa": _sa_predictions,
               "fast-detectgpt": _fast_predictions, "binoculars": _binoculars_predictions}
    outputs = runners[args.method](rows, args)
    if len(outputs) != len(rows):
        raise ValueError("Baseline did not produce one status per input record")
    valid = [r for r in outputs if r["prediction"] is not None]
    metrics = binary_metrics([r["label"] for r in valid], [r["prediction"] for r in valid]) if valid else None
    result = {"method": args.method, "model": args.model, "data": fingerprint(args.data), "statistics": stats,
              "valid_records": len(valid), "failed_records": len(outputs) - len(valid), "coverage": len(valid) / len(outputs),
              "metrics_on_valid_records": metrics, "predictions": outputs, "environment": environment(),
              "settings": {"max_length": args.max_length, "ai_output_label": args.ai_output_label,
                           "attempts": args.attempts, "reference_files": [fingerprint(p) for p in sorted(Path(args.reference_stats).glob('*.json'))] if args.reference_stats else []}}
    write_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("predictions", "environment", "settings")}, indent=2))
    if len(valid) != len(rows):
        raise ValueError("Some judge requests failed; report saved with explicit missing predictions and coverage")


def register_baselines(commands):
    command = commands.add_parser("baseline", help="Optional external baseline adapters; no API calls unless invoked")
    command.add_argument("--method", choices=["sa", "fast-detectgpt", "binoculars", "llm"], required=True)
    command.add_argument("--model", required=True, help="Exact HF/API model ID; Binoculars uses its vendored defaults")
    command.add_argument("--data", required=True)
    command.add_argument("--output", required=True)
    command.add_argument("--reference-stats")
    command.add_argument("--ai-output-label", type=int, choices=[0, 1])
    command.add_argument("--empty-text", choices=["error", "keep", "skip"], default="error")
    command.add_argument("--device", default="auto")
    command.add_argument("--max-length", type=int, default=512)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--attempts", type=int, default=3)
    command.set_defaults(handler=baseline)
