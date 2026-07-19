"""Plot task acceptance metrics across all tasks in a canonical run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("experiments/task_acceptance_canonical_20260713")


def _score_metrics(summary: dict) -> list[str]:
    rows = list(summary.get("instruction_only", []))
    if isinstance(summary.get("no_instruction"), dict):
        rows.append(summary["no_instruction"])
    metrics = {
        key
        for row in rows
        for key, value in row.items()
        if key.startswith("mean_") and isinstance(value, (int, float))
    }
    return sorted(metrics)


def collect(root: Path) -> tuple[list[str], dict[str, dict[str, list[float]]], list[str]]:
    summaries = [(path.parent.name, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(root.glob("*/acceptance_summary.json"))]
    max_prompts = max((len(summary.get("instruction_only", [])) for _task, summary in summaries), default=0)
    labels = [f"prompt_{index}" for index in range(1, max_prompts + 1)] + ["no_instruction"]
    data: dict[str, dict[str, list[float]]] = {}
    metrics: set[str] = set()
    for task, summary in summaries:
        prompts = list(summary.get("instruction_only", []))
        runs = prompts + [{} for _ in range(max_prompts - len(prompts))]
        runs.append(summary["no_instruction"] if isinstance(summary.get("no_instruction"), dict) else {})
        task_metrics = _score_metrics(summary)
        metrics.update(task_metrics)
        data[task] = {
            metric: [float(run.get(metric, math.nan)) for run in runs]
            for metric in task_metrics
        }
    return sorted(metrics), data, labels


def plot_metric(metric: str, data: dict[str, dict[str, list[float]]], labels: list[str], output: Path) -> None:
    tasks = sorted(data)
    width = 0.8 / max(1, len(labels))
    xs = list(range(len(tasks)))
    fig, ax = plt.subplots(figsize=(max(12, len(tasks) * 0.7), 7))
    for index, label in enumerate(labels):
        offsets = [x + (index - (len(labels) - 1) / 2) * width for x in xs]
        values = [data[task].get(metric, [math.nan] * len(labels))[index] for task in tasks]
        ax.bar(offsets, values, width, label=label)
    ax.set(
        xticks=xs,
        xticklabels=tasks,
        ylim=(0, 1),
        ylabel=metric,
        title=f"Task acceptance {metric}",
    )
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=min(4, len(labels)))
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_plots(root: Path, output_dir: Path) -> list[Path]:
    metrics, data, labels = collect(root)
    if not data:
        raise FileNotFoundError(f"No acceptance_summary.json files found below {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for metric in metrics:
        path = output_dir / f"{metric}.png"
        plot_metric(metric, data, labels, path)
        written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot task acceptance metrics by scanning task directories.")
    parser.add_argument("root", nargs="?", type=Path, default=DEFAULT_ROOT, help="Directory containing one task subdirectory each.")
    parser.add_argument("--output-dir", type=Path, help="Where PNG files are written. Defaults to ROOT/acceptance_plots.")
    args = parser.parse_args()
    for path in write_plots(args.root, args.output_dir or args.root / "acceptance_plots"):
        print(path)


if __name__ == "__main__":
    main()
