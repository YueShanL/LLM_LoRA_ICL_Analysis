"""Plot RQ3 correctness for every task below an experiment root."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean

import matplotlib.pyplot as plt


def collect(root: Path) -> list[dict]:
    grouped: dict[tuple, dict[str, float]] = defaultdict(dict)
    for path in root.glob("*/patches/rq3/*/metrics.jsonl"):
        task, run = path.parents[3].name, path.parent.name
        config_path = path.parent / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        patch_source = config.get("source_condition") or run.split("_to_", 1)[0]
        patch_target = config.get("target_condition") or run.split("_to_", 1)[-1].rsplit("_l", 1)[0]
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            metric = row.get("task_semantic_correct", row.get("sequence_accuracy", row.get("token_accuracy")))
            if metric is None:
                continue
            source = row.get("source_condition") or "none"
            target = row.get("target_condition") or "none"
            control = row.get("control") or ("patched" if row.get("patched") else "unpatched")
            key = (task, run, patch_source, patch_target, source, target, int(row.get("layer", -1)), control, row["sample_id"])
            grouped[key]["correct"] = float(metric)
    buckets: dict[tuple, list[float]] = defaultdict(list)
    for key, values in grouped.items():
        buckets[key[:-1]].append(values["correct"])
    return [{"task": k[0], "run": k[1], "patch_source_condition": k[2], "patch_target_condition": k[3], "source_condition": k[4], "target_condition": k[5], "layer": k[6], "control": k[7], "samples": len(v), "accuracy": fmean(v)} for k, v in sorted(buckets.items())]


def write(root: Path, output_dir: Path) -> list[Path]:
    rows = collect(root)
    if not rows:
        raise FileNotFoundError(f"No RQ3 metrics found below {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "rq3_accuracy_all_tasks.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    tasks = sorted({r["task"] for r in rows})
    controls = [control for control in ("unpatched", "base_to_target_patch", "source_to_target_patch") if any(r["control"] == control for r in rows)]
    pairs = sorted({(r["patch_source_condition"], r["patch_target_condition"]) for r in rows})
    layers = sorted({r["layer"] for r in rows})
    fig, axes = plt.subplots(len(pairs), len(controls), figsize=(max(15, len(tasks) * 1.1), 4 * len(pairs)), squeeze=False, sharex=True, sharey=True)
    xs = list(range(len(tasks)))
    for pair_index, pair in enumerate(pairs):
        for control_index, control in enumerate(controls):
            ax = axes[pair_index][control_index]
            for layer in layers:
                values = []
                for task in tasks:
                    matches = [r["accuracy"] for r in rows if (r["patch_source_condition"], r["patch_target_condition"]) == pair and r["control"] == control and r["layer"] == layer and r["task"] == task]
                    values.append(fmean(matches) if matches else float("nan"))
                ax.plot(xs, values, marker="o", linewidth=1.2, label=f"layer {layer}")
            ax.set_title(f"{pair[0]} → {pair[1]} | {control}", fontsize=10)
            ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
            if control_index == 0:
                ax.set_ylabel("Semantic correctness")
            if pair_index == len(pairs) - 1:
                ax.set_xticks(xs, tasks, rotation=45, ha="right")
            if pair_index == 0 and control_index == len(controls) - 1:
                ax.legend(fontsize=8)
    fig.suptitle("RQ3 accuracy by patching pair, control, and layer")
    fig.tight_layout(rect=(0, 0, 1, .98))
    png_path = output_dir / "rq3_accuracy_all_tasks.png"; fig.savefig(png_path, dpi=180); plt.close(fig)
    return [csv_path, png_path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    for path in write(args.root, args.output_dir or args.root / "combined_plots"):
        print(path)


if __name__ == "__main__":
    main()
