"""Plot measured benchmark metrics; never hardcode paper scores."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", choices=["accuracy", "ai_recall", "human_recall", "f1_ai", "f1_macro"], default="ai_recall")
    parser.add_argument("--protocol", required=True, help="Exact protocol from the benchmark JSON")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    import matplotlib.pyplot as plt
    results = json.loads(Path(args.input).read_text(encoding="utf-8"))["results"]
    rows = [r for r in results if r["protocol"] == args.protocol]
    if not rows:
        raise ValueError("No results for requested protocol")
    models = list(dict.fromkeys(r["model"] for r in rows))
    conditions = list(dict.fromkeys(r["dataset"] for r in rows))
    fig, axis = plt.subplots(figsize=(max(6, len(conditions)), 4), constrained_layout=True)
    width = 0.8 / len(models)
    for number, model in enumerate(models):
        index = {r["dataset"]: r["metrics"][args.metric] for r in rows if r["model"] == model}
        values = [index.get(condition) for condition in conditions]
        if any(v is None for v in values):
            raise ValueError("Metric is missing or undefined in the requested matrix")
        axis.bar([i - 0.4 + (number + 0.5) * width for i in range(len(conditions))],
                 [100 * v for v in values], width=width, label=model)
    axis.set_xticks(range(len(conditions)))
    axis.set_xticklabels(conditions, rotation=25, ha="right")
    axis.set_ylabel(args.metric + " (%)")
    axis.set_ylim(0, 105)
    axis.legend(fontsize="small")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
