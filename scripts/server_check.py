"""Read-only CPU preflight on AutoDL; does not install packages or start training."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "src"))

from t_aware.data import read_records
from t_aware.io import environment, fingerprint, read_config, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", help="Omit to discover one matching project under common AutoDL roots")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    if args.asset_root:
        root = Path(args.asset_root).expanduser().resolve()
    else:
        candidates = {Path.cwd(), Path('/root/autodl-tmp'), Path('/root/T-Aware')}
        for parent in (Path('/root/autodl-tmp'), Path('/root')):
            if parent.is_dir():
                candidates.update(p for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.'))
        matching = sorted({p.resolve() for p in candidates if (p / 'dataset/dataset.llama3.train.jsonl').is_file()})
        if len(matching) != 1:
            raise ValueError(f"Expected one project; found {[str(p) for p in matching]}. Pass --asset-root explicitly.")
        root = matching[0]
    if not root.is_dir():
        raise FileNotFoundError(root)
    limits = {}
    for name in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            limits[name] = path.read_text().strip()
    result = {"environment": environment(), "python_executable": sys.executable,
              "cgroup_memory": limits,
              "asset_root": str(root), "data": [], "weights": []}
    for rel in ["dataset/dataset.llama3.train.jsonl", "dataset/dataset.llama3.val.clean.jsonl",
                "dataset/dataset.llama3.test.clean.jsonl", "test_data/dataset.tp.hc3.val.jsonl",
                "test_data/dataset.tp.hc3.jsonl"]:
        path = root / rel
        rows, stats = read_records(path, "keep")
        result["data"].append({"path": rel, "file": fingerprint(path), "statistics": stats})
        del rows
    for folder in ["saved_models_epoch50", "saved_models_epoch50_v2"]:
        for path in sorted((root / folder).glob("*.pt")):
            result["weights"].append({"path": str(path.relative_to(root)), "file": fingerprint(path)})
    # Verify every configured population, without loading a detector or changing assets.
    result["evaluation_matrices"] = []
    for config_path in sorted((repo / "configs").glob("evaluate_*.json")):
        config = read_config(config_path)
        entries = []
        for spec in config["datasets"]:
            path = root / spec["path"]
            entry = {"name": spec["name"], "path": spec["path"], "exists": path.is_file()}
            if path.is_file():
                rows, statistics = read_records(path, spec.get("empty_text", "error"))
                entry.update({"file": fingerprint(path), "statistics": statistics})
                del rows
            entries.append(entry)
        result["evaluation_matrices"].append({"config": config_path.name, "datasets": entries,
            "checkpoints": [{"path": spec["checkpoint"], "exists": (root / spec["checkpoint"]).is_file()}
                            for spec in config["models"]]})
    result["source_comparison"] = []
    for entry in json.loads((repo / "docs/source_migration.json").read_text(encoding="utf-8")):
        path = root / entry["source"]
        observed = fingerprint(path) if path.is_file() else None
        result["source_comparison"].append({"source": entry["source"],
            "expected_sha256": entry["source_sha256"], "observed": observed,
            "matches": observed is not None and observed["sha256"] == entry["source_sha256"]})
    result["tests"] = []
    for command in [[sys.executable, "scripts/run.py", "smoke"],
                    [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]]:
        process = subprocess.run(command, cwd=repo, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result["tests"].append({"command": command, "returncode": process.returncode, "output": process.stdout})
    write_json(args.output, result)
    if any(t["returncode"] != 0 for t in result["tests"]):
        raise SystemExit("Preflight saved, but a CPU test failed. Inspect the JSON before proceeding.")
    if any(not item["exists"] for matrix in result["evaluation_matrices"]
           for item in matrix["datasets"] + matrix["checkpoints"]):
        raise SystemExit("Preflight saved, but a configured evaluation asset is missing.")
    print(f"Saved {args.output}; no training or data changes performed")


if __name__ == "__main__":
    main()
