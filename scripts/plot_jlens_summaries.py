"""Plot J-lens layer summaries across tasks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _rows(root: Path) -> list[dict]:
    rows = []
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        path = task_dir / "plots" / "rq1_jlens" / "jlens_layer_summary.csv"
        if not path.exists():
            continue
        for row in _read_csv(path):
            rows.append({"task": task_dir.name, **row})
    return rows


def _plot(rows: list[dict], output_path: Path, metric: str, pair: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))
    filtered = [row for row in rows if row["condition_pair"] == pair]
    for task in sorted({row["task"] for row in filtered}):
        points = sorted((int(row["layer"]), float(row[metric])) for row in filtered if row["task"] == task)
        ax.plot([layer for layer, _ in points], [value for _, value in points], linewidth=1.5, alpha=0.75, label=task)
    ax.set_title(f"J-lens {pair}: {metric}")
    ax.set_xlabel("J-lens source layer")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all(root: Path, output_dir: Path) -> None:
    rows = _rows(root)
    for pair in ("base_vs_instruction_only", "base_vs_lora_only", "instruction_only_vs_lora_only"):
        for metric in ("mean_top_k_overlap", "mean_top_k_jaccard", "mean_left_target_token_rank", "mean_right_target_token_rank"):
            _plot(rows, output_dir / f"{pair}_{metric}.png", metric, pair)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot J-lens summary CSVs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_all(args.root, args.output_dir)
    print(f"Wrote J-lens summary plots to {args.output_dir}")


if __name__ == "__main__":
    main()
