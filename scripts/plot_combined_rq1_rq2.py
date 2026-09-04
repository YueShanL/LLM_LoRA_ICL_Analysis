"""Combined RQ1/RQ2 matplotlib plots across task run directories."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("experiments/lora_selected_tasks_instruct_rawchat_r8_20260709")
CONDITION = "instruction_only_vs_lora_only"


def read_csv(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def task_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "plots").is_dir())


def instruction_pass_rate(acceptance_root: Path, task: str) -> float:
    path = acceptance_root / task / "instruct" / "acceptance_summary.json"
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        rates = [
            float(row.get("mean_task_semantic_correct") or "nan")
            for row in summary.get("instruction_only", [])
        ]
        rates = [rate for rate in rates if not math.isnan(rate)]
        return max(rates) if rates else math.nan
    return math.nan


def _aggregate_path(aggregate_task_dir: Path, relative_path: Path) -> Path:
    return aggregate_task_dir / relative_path.with_suffix(".sample_layer_head.csv.gz")


def rows_for(task_dir: Path, aggregate_task_dir: Path, candidates: tuple[Path, ...]) -> list[dict[str, str]]:
    paths = [*(task_dir / relative_path for relative_path in candidates)]
    paths.extend(_aggregate_path(aggregate_task_dir, relative_path) for relative_path in candidates)
    for path in paths:
        if path.exists():
            rows = [row for row in read_csv(path) if row.get("condition") == CONDITION]
            if rows:
                return rows
    return []


def _metric_value(row: dict[str, str], metric: str) -> tuple[float, float] | None:
    raw_value = row.get(metric)
    if raw_value not in (None, ""):
        return float(raw_value), 1.0
    mean_value = row.get(f"{metric}_mean")
    if mean_value in (None, ""):
        return None
    count = row.get(f"{metric}_n") or row.get("source_target_token_count") or "1"
    return float(mean_value), float(count)


def layer_means(rows: list[dict[str, str]], metric: str) -> dict[int, float]:
    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        value = _metric_value(row, metric)
        if value is not None:
            grouped[int(row["layer"])].append(value)
    return {
        layer: sum(value * weight for value, weight in values) / sum(weight for _, weight in values)
        for layer, values in sorted(grouped.items())
        if values
    }


def collect(root: Path, acceptance_root: Path) -> tuple[dict[str, float], dict[str, dict[str, list[dict[str, str]]]]]:
    datasets: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(dict)
    rates: dict[str, float] = {}
    for task_dir in task_dirs(root):
        task = task_dir.name
        aggregate_task_dir = root / "analysis_aggregates" / task
        rates[task] = instruction_pass_rate(acceptance_root, task)
        datasets["rq1_similarity"][task] = rows_for(
            task_dir, aggregate_task_dir, (Path("plots/rq1/token_similarity.csv"),)
        )
        datasets["rq1_cka"][task] = datasets["rq1_similarity"][task]
        datasets["rq1_delta_subspace"][task] = read_csv(task_dir / "plots/rq1/delta_subspace_summary.csv") if (task_dir / "plots/rq1/delta_subspace_summary.csv").exists() else []
        datasets["rq2_attention_prob"][task] = rows_for(
            task_dir,
            aggregate_task_dir,
            (
                Path("plots/rq2/token_similarity.csv"),
                Path("plots/rq21/attention_probs/token_similarity.csv"),
            ),
        )
        datasets["rq2_attention_output"][task] = rows_for(
            task_dir,
            aggregate_task_dir,
            (Path("plots/rq21/attention_outputs/token_similarity.csv"),),
        )
        datasets["rq2_attention_post_o_proj_output"][task] = rows_for(
            task_dir,
            aggregate_task_dir,
            (Path("plots/rq21/attention_post_o_proj_outputs/token_similarity.csv"),),
        )
    return rates, datasets


def plot_lines(series_by_task: dict[str, dict[int, float]], title: str, ylabel: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = False
    for task, series in sorted(series_by_task.items()):
        if not series:
            continue
        layers = list(series)
        ax.plot(layers, [series[layer] for layer in layers], marker="o", linewidth=1.4, markersize=3, label=task)
        plotted = True
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_scatter(
    rows_by_task: dict[str, list[dict[str, str]]],
    rates: dict[str, float],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    xs: list[int] = []
    ys: list[float] = []
    colors: list[float] = []
    for task, rows in sorted(rows_by_task.items()):
        rate = rates.get(task, math.nan)
        for row in rows:
            metric = _metric_value(row, "cosine_similarity")
            if metric is None:
                continue
            xs.append(int(row["layer"]))
            ys.append(metric[0])
            colors.append(rate)
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(xs, ys, c=colors, cmap="viridis", s=9, alpha=0.35, edgecolors="none")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Instruction pass rate (acceptance mean semantic correctness)")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_plots(root: Path, output_dir: Path, acceptance_root: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rates, datasets = collect(root, acceptance_root)
    specs = (
        ("rq1_similarity", "cosine_similarity", "RQ1 Layer Similarity Across Tasks", "Mean cosine similarity"),
        ("rq1_cka", "cka_similarity", "RQ1 CKA Across Tasks", "Mean CKA similarity"),
        ("rq2_attention_prob", "cosine_similarity", "RQ2 Attention Probability Similarity Across Tasks", "Mean cosine similarity"),
        ("rq2_attention_output", "cosine_similarity", "RQ2 Attention Output Similarity Across Tasks", "Mean cosine similarity"),
        (
            "rq2_attention_post_o_proj_output",
            "cosine_similarity",
            "RQ2 Post-o_proj Attention Output Similarity Across Tasks",
            "Mean cosine similarity",
        ),
        ("rq1_delta_subspace", "subspace_cosine_mean", "RQ1 Delta Subspace Similarity Across Tasks", "Mean top-k subspace cosine"),
    )
    written: list[Path] = []
    for key, metric, title, ylabel in specs:
        path = output_dir / f"{key}_lines.png"
        plot_lines({task: layer_means(rows, metric) for task, rows in datasets[key].items()}, title, ylabel, path)
        written.append(path)
    for key, title in (
        ("rq1_similarity", "RQ1 Layer Similarity Scatter By Task Pass Rate"),
        ("rq2_attention_prob", "RQ2 Attention Probability Similarity Scatter By Task Pass Rate"),
        ("rq2_attention_output", "RQ2 Attention Output Similarity Scatter By Task Pass Rate"),
    ):
        path = output_dir / f"{key}_scatter_by_pass_rate.png"
        plot_scatter(datasets[key], rates, title, "Cosine similarity", path)
        written.append(path)
    final_rows = []
    for task in sorted(datasets["rq1_similarity"]):
        residual = layer_means(datasets["rq1_similarity"][task], "cosine_similarity")
        attention = layer_means(datasets["rq2_attention_post_o_proj_output"][task], "cosine_similarity")
        if residual and attention:
            final_rows.append((task, residual[max(residual)], attention[max(attention)], rates.get(task, math.nan)))
    path = output_dir / "final_attention_vs_residual_by_pass_rate.png"
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter([r[1] for r in final_rows], [r[2] for r in final_rows], c=[r[3] for r in final_rows], cmap="viridis", s=45)
    for task, x, y, _ in final_rows:
        ax.annotate(task, (x, y), fontsize=7, alpha=.75)
    ax.set(xlabel="Final residual-delta cosine", ylabel="Final post-o_proj similarity", title="Final attention vs residual alignment")
    ax.grid(alpha=.25)
    if final_rows:
        fig.colorbar(scatter, ax=ax, label="Instruction pass rate")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); written.append(path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot combined RQ1/RQ2 summaries by scanning task directories.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Directory containing one subdirectory per task.")
    parser.add_argument("--output-dir", type=Path, help="Where PNG files are written. Defaults to ROOT/combined_plots.")
    parser.add_argument(
        "--acceptance-root",
        type=Path,
        help="Directory containing TASK/instruct/acceptance_summary.json. Defaults to ROOT sibling task_acceptance_generation_screen_rerun.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / "combined_plots"
    acceptance_root = args.acceptance_root or args.root.parent / "task_acceptance_generation_screen_rerun"
    written = write_plots(args.root, output_dir, acceptance_root)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
