"""Plot SAE layer summaries across tasks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_rows(root: Path, relative_summary: Path) -> list[dict]:
    rows = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        path = task_dir / relative_summary
        if not path.exists():
            continue
        for row in _read_csv(path):
            rows.append({"task": task_dir.name, **row})
    return rows


def _plot(rows: list[dict], output_path: Path, title: str, metric: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    for task in sorted({row["task"] for row in rows}):
        points = sorted((int(row["layer"]), float(row[metric])) for row in rows if row["task"] == task)
        ax.plot([layer for layer, _ in points], [value for _, value in points], linewidth=1.5, alpha=0.75, label=task)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all(root: Path, output_dir: Path) -> None:
    specs = [
        ("rq1_sae", Path("plots/rq1_sae/sae_layer_summary.csv"), "RQ1 residual SAE"),
        (
            "rq21_attention_outputs_sae",
            Path("plots/rq21/attention_outputs_sae/sae_layer_summary.csv"),
            "RQ2.1 pre-o_proj attention-output SAE",
        ),
        (
            "rq21_attention_post_o_proj_outputs_sae",
            Path("plots/rq21/attention_post_o_proj_outputs_sae/sae_layer_summary.csv"),
            "RQ2.1 post-o_proj attention-output SAE",
        ),
    ]
    metrics = ("mean_top_k_feature_jaccard", "mean_feature_activation_cosine")
    for name, relative_summary, title in specs:
        rows = _task_rows(root, relative_summary)
        if not rows:
            continue
        for metric in metrics:
            _plot(rows, output_dir / f"{name}_{metric}.png", f"{title}: {metric}", metric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAE summary CSVs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_all(args.root, args.output_dir)
    print(f"Wrote SAE summary plots to {args.output_dir}")


if __name__ == "__main__":
    main()
