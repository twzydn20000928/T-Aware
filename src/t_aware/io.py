"""Small, explicit helpers for configurations and experiment provenance."""
import hashlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def read_config(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def asset_root(value):
    value = value or os.environ.get("T_AWARE_DATA_ROOT")
    if not value:
        raise ValueError("Pass --data-root (the original directory containing dataset/ and test_data/)")
    root = Path(value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def input_path(root, value):
    path = Path(value).expanduser()
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_json(path, value, overwrite=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def environment():
    packages = {}
    for name in ("torch", "transformers", "numpy", "tqdm", "scikit-learn", "huggingface-hub"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat()}


def fingerprint(path):
    path = Path(path)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
