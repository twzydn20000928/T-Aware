"""Shared tokenization, loss-target construction and deterministic inference."""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .metrics import binary_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu for inference/smoke tests")
    return value


class TextDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=128, temperature_scale="raw",
                 random_temperature=False, random_seed=42):
        if temperature_scale not in ("raw", "minmax_0.2_3.5"):
            raise ValueError("Unknown temperature target scale")
        self.rows, self.tokenizer, self.max_length = rows, tokenizer, max_length
        self.temperature_scale, self.random_temperature = temperature_scale, random_temperature
        self.random = random.Random(random_seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        target = row["temperature"]
        if row["label"] == 0 or target is None:
            target = -1.0
        else:
            if self.random_temperature:
                # Historical random-label control resampled on every access.
                target = self.random.uniform(0.2, 3.5)
            if self.temperature_scale == "minmax_0.2_3.5":
                target = (target - 0.2) / 3.3
        encoded = self.tokenizer(row["text"], truncation=True, padding="max_length",
                                 max_length=self.max_length, return_tensors="pt")
        return {"input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "labels": torch.tensor(row["label"], dtype=torch.long),
                "t_labels": torch.tensor(target, dtype=torch.float)}


def infer(model, loader, device):
    model.eval()
    predictions, labels, scores, temperatures = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            result = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            predictions.extend(result["logits"].argmax(-1).cpu().tolist())
            scores.extend(result["logits"].softmax(-1)[:, 1].cpu().tolist())
            temperatures.extend(result["temperature"].cpu().tolist())
            labels.extend(batch["labels"].tolist())
    return binary_metrics(labels, predictions), [
        {"label": y, "prediction": p, "score_ai": s, "predicted_temperature": t}
        for y, p, s, t in zip(labels, predictions, scores, temperatures)]
