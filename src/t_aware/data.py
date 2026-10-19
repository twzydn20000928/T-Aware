"""Validated adapters for generated-text JSONL and attack JSONL.

Human=0, AI=1. A missing temperature is never silently interpreted as human.
"""
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path


def source_key(item):
    if item.get("base_id"):
        return str(item["base_id"])
    if item.get("prompt"):
        return "prompt:" + hashlib.sha256(item["prompt"].strip().encode("utf-8")).hexdigest()
    key = item.get("source_id") or item.get("origin_id") or item.get("id")
    if key is None:
        return None
    return re.sub(r"_(human|attacked|Human_Anchor)$", "", str(key))


def normalize(item, location, empty_text="error"):
    if not isinstance(item, dict):
        raise ValueError(f"{location}: each record must be an object")
    text = item.get("text") or item.get("generated_text") or ""
    if not isinstance(text, str):
        raise ValueError(f"{location}: text must be a string")
    temperature = item.get("generation_T")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature):
            raise ValueError(f"{location}: generation_T must be a finite number")
        temperature = float(temperature)
    label = item.get("label")
    if label is None:
        if temperature is None:
            raise ValueError(f"{location}: expected label or generation_T")
        label = int(temperature >= 0)
    if isinstance(label, bool) or label not in (0, 1):
        raise ValueError(f"{location}: label must be 0 (human) or 1 (AI)")
    if temperature is not None and int(temperature >= 0) != label:
        raise ValueError(f"{location}: label and generation_T disagree")
    if not text.strip():
        if empty_text == "error":
            raise ValueError(f"{location}: empty text; choose --empty-text keep/skip explicitly")
        if empty_text == "skip":
            return None
    return {"text": text, "label": int(label), "temperature": temperature,
            "source_id": source_key(item), "id": str(item.get("id", location)),
            "type": item.get("type"), "origin_text": item.get("origin_text"),
            "variance": item.get("variance")}


def read_records(path, empty_text="error"):
    if empty_text not in ("error", "keep", "skip"):
        raise ValueError("Unknown empty-text policy")
    rows = []
    stats = {"input_records": 0, "empty_records": 0, "skipped_records": 0, "blank_lines": 0,
             "empty_text_policy": empty_text}
    with Path(path).open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                stats["blank_lines"] += 1
                continue
            location = f"{Path(path).name}:{number}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            stats["input_records"] += 1
            row = normalize(item, location, empty_text)
            text = (item.get("text") or item.get("generated_text") or "")
            stats["empty_records"] += int(not text.strip())
            if row is None:
                stats["skipped_records"] += 1
            else:
                rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no usable records")
    stats.update({"loaded_records": len(rows), "labels": dict(Counter(str(x["label"]) for x in rows)),
                  "source_groups": len({x["source_id"] for x in rows if x["source_id"] is not None})})
    return rows, stats


def balance_humans(rows):
    """Preserve the historical integer human-repetition rule (including order)."""
    ai = [row for row in rows if row["label"] == 1]
    human = [row for row in rows if row["label"] == 0]
    if not ai or not human:
        return list(rows)
    return ai + human * max(1, len(ai) // len(human))


def assert_disjoint(train, validation, name):
    # These datasets carry base/source IDs derived from the original prompt splits.
    left = {r["source_id"] for r in train if r["source_id"] is not None}
    right = {r["source_id"] for r in validation if r["source_id"] is not None}
    if not left or not right:
        raise ValueError(f"{name}: missing source grouping metadata")
    overlap = left & right
    if overlap:
        raise ValueError(f"{name}: {len(overlap)} shared source IDs")
