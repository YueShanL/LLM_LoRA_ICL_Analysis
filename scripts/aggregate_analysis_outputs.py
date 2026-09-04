"""Create compact sample-level summaries from token-level RQ plot tables.

The pipeline's token-level CSV files are useful for diagnosing a single token,
but the independent unit for task-level inference is the sample.  This command
keeps one row per sample, condition, layer, and (where relevant) attention head
and writes token-level mean, standard deviation, minimum, and maximum for each
metric.  It does not modify the source experiment directory.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GROUP_COLUMNS = ("sample_id", "task_id", "condition", "mode", "layer", "head")
AGGREGATION_VERSION = "sample_layer_head_v2"
METRIC_COLUMNS = frozenset(
    {
        "cosine_similarity",
        "cka_similarity",
        "logit_distribution_cosine",
        "kl_left_to_right",
        "kl_right_to_left",
        "left_entropy",
        "right_entropy",
        "left_shared_attention_mass",
        "right_shared_attention_mass",
        "ablated_cosine_to_full",
        "full_norm",
        "head_contribution_norm",
        "head_contribution_relative_norm",
    }
)


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    sum_squared_deviation: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def add(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sum_squared_deviation += delta * (value - self.mean)
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def as_row(self, name: str) -> dict[str, int | float]:
        standard_deviation = math.sqrt(self.sum_squared_deviation / (self.count - 1)) if self.count > 1 else 0.0
        return {
            f"{name}_n": self.count,
            f"{name}_mean": self.mean,
            f"{name}_std": standard_deviation,
            f"{name}_min": self.minimum,
            f"{name}_max": self.maximum,
        }


def _is_redundant_rq2_probability_csv(path: Path, experiment_root: Path) -> bool:
    """RQ2 probabilities are duplicated by RQ2.1 attention-probability output."""
    relative = path.relative_to(experiment_root)
    if relative.parts[-3:] != ("plots", "rq2", "token_similarity.csv"):
        return False
    equivalent = path.parents[1] / "rq21" / "attention_probs" / "token_similarity.csv"
    return equivalent.is_file() and equivalent.stat().st_size == path.stat().st_size


def _output_path(source: Path, experiment_root: Path, output_root: Path) -> Path:
    relative = source.relative_to(experiment_root)
    return output_root / relative.with_suffix(".sample_layer_head.csv.gz")


def _token_similarity_sources(source_root: Path) -> list[Path]:
    """Return token tables from either one task directory or an experiment root."""
    pattern = "plots/**/token_similarity.csv" if (source_root / "plots").is_dir() else "*/plots/**/token_similarity.csv"
    return sorted(source_root.glob(pattern))


def _read_manifest(output_root: Path) -> dict[str, object] | None:
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_destinations_are_complete(manifest: dict[str, object], output_root: Path) -> bool:
    processed = manifest.get("processed")
    if not isinstance(processed, list):
        return False
    for item in processed:
        if not isinstance(item, dict):
            return False
        destination = item.get("destination")
        if not isinstance(destination, str) or not (output_root / destination).is_file():
            return False
    return True


def _group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(column, "") for column in GROUP_COLUMNS)


def _metric_names(fieldnames: Iterable[str]) -> tuple[str, ...]:
    return tuple(field for field in fieldnames if field in METRIC_COLUMNS)


def _row_for_group(
    key: tuple[str, ...],
    source_token_count: int,
    stats: dict[str, RunningStats],
) -> dict[str, int | float | str]:
    row: dict[str, int | float | str] = dict(zip(GROUP_COLUMNS, key, strict=True))
    row["source_target_token_count"] = source_token_count
    for metric_name, metric_stats in stats.items():
        row.update(metric_stats.as_row(metric_name))
    return row


def aggregate_csv(source: Path, experiment_root: Path, output_root: Path) -> dict[str, object]:
    """Aggregate a token CSV streamed in its existing sample/layer/head order."""
    destination = _output_path(source, experiment_root, output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    aggregate_rows = 0

    with source.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {source}")
        missing_columns = set(GROUP_COLUMNS).difference(reader.fieldnames)
        if missing_columns:
            raise ValueError(f"CSV is missing group columns {sorted(missing_columns)}: {source}")
        metric_names = _metric_names(reader.fieldnames)
        if not metric_names:
            raise ValueError(f"CSV has no supported metric columns: {source}")

        output_fields = [*GROUP_COLUMNS, "source_target_token_count"]
        for metric_name in metric_names:
            output_fields.extend(
                (
                    f"{metric_name}_n",
                    f"{metric_name}_mean",
                    f"{metric_name}_std",
                    f"{metric_name}_min",
                    f"{metric_name}_max",
                )
            )

        with gzip.open(destination, "wt", encoding="utf-8", newline="", compresslevel=6) as destination_handle:
            writer = csv.DictWriter(destination_handle, fieldnames=output_fields)
            writer.writeheader()
            current_key: tuple[str, ...] | None = None
            current_token_count = 0
            current_stats: dict[str, RunningStats] = {}

            for row in reader:
                source_rows += 1
                key = _group_key(row)
                if current_key is not None and key != current_key:
                    writer.writerow(_row_for_group(current_key, current_token_count, current_stats))
                    aggregate_rows += 1
                    current_token_count = 0
                    current_stats = {}
                current_key = key
                current_token_count += 1
                for metric_name in metric_names:
                    raw_value = row.get(metric_name, "")
                    if raw_value in (None, ""):
                        continue
                    value = float(raw_value)
                    if math.isfinite(value):
                        current_stats.setdefault(metric_name, RunningStats()).add(value)

            if current_key is not None:
                writer.writerow(_row_for_group(current_key, current_token_count, current_stats))
                aggregate_rows += 1

    return {
        "source": source.relative_to(experiment_root).as_posix(),
        "destination": destination.relative_to(output_root).as_posix(),
        "source_rows": source_rows,
        "aggregate_rows": aggregate_rows,
        "source_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "aggregate_bytes": destination.stat().st_size,
        "metrics": metric_names,
    }


def aggregate_experiment(experiment_root: Path, output_root: Path) -> dict[str, object]:
    experiment_root = experiment_root.resolve()
    output_root = output_root.resolve()
    if not experiment_root.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_root}")
    if output_root == experiment_root:
        raise ValueError("The aggregation output directory must differ from the experiment directory.")

    processed: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for source in _token_similarity_sources(experiment_root):
        if _is_redundant_rq2_probability_csv(source, experiment_root):
            skipped.append(
                {
                    "source": source.relative_to(experiment_root).as_posix(),
                    "reason": "identical RQ2.1 attention_probs table retained instead",
                }
            )
            continue
        result = aggregate_csv(source, experiment_root, output_root)
        processed.append(result)
        print(
            f"[done] {result['source']}: {result['source_rows']} rows -> {result['aggregate_rows']} groups",
            flush=True,
        )

    manifest = {
        "aggregation": AGGREGATION_VERSION,
        "group_columns": GROUP_COLUMNS,
        "source_root": str(experiment_root),
        "source_tables_pruned": False,
        "processed": processed,
        "skipped": skipped,
        "source_bytes": sum(int(item["source_bytes"]) for item in processed),
        "aggregate_bytes": sum(int(item["aggregate_bytes"]) for item in processed),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def aggregation_is_current(experiment_root: Path, output_root: Path) -> bool:
    """Check a task or experiment aggregation without re-reading its CSV contents."""
    manifest = _read_manifest(output_root)
    if manifest is None:
        return False
    if manifest.get("aggregation") != AGGREGATION_VERSION:
        return False
    if not _manifest_destinations_are_complete(manifest, output_root):
        return False

    experiment_root = experiment_root.resolve()
    sources = _token_similarity_sources(experiment_root)
    if not sources:
        return bool(manifest.get("source_tables_pruned", False))
    expected_processed = {
        source.relative_to(experiment_root).as_posix(): source
        for source in sources
        if not _is_redundant_rq2_probability_csv(source, experiment_root)
    }
    expected_skipped = {
        source.relative_to(experiment_root).as_posix()
        for source in sources
        if _is_redundant_rq2_probability_csv(source, experiment_root)
    }
    processed = manifest.get("processed")
    skipped = manifest.get("skipped")
    if not isinstance(processed, list) or not isinstance(skipped, list):
        return False
    processed_by_source = {item.get("source"): item for item in processed if isinstance(item, dict)}
    if set(processed_by_source) != set(expected_processed):
        return False
    if {item.get("source") for item in skipped if isinstance(item, dict)} != expected_skipped:
        return False
    for source_name, source in expected_processed.items():
        item = processed_by_source[source_name]
        stat = source.stat()
        if item.get("source_bytes") != stat.st_size or item.get("source_mtime_ns") != stat.st_mtime_ns:
            return False
    return True


def prune_aggregated_source_tables(experiment_root: Path, output_root: Path) -> list[Path]:
    """Remove source token tables only after their aggregate manifest is complete."""
    experiment_root = experiment_root.resolve()
    output_root = output_root.resolve()
    manifest = _read_manifest(output_root)
    if manifest is None or manifest.get("aggregation") != AGGREGATION_VERSION:
        raise ValueError(f"No current aggregation manifest in {output_root}")
    if not _manifest_destinations_are_complete(manifest, output_root):
        raise ValueError(f"Aggregation output is incomplete in {output_root}")

    records = [*manifest.get("processed", []), *manifest.get("skipped", [])]
    removed: list[Path] = []
    removed_renderings: list[Path] = []
    for item in records:
        if not isinstance(item, dict) or not isinstance(item.get("source"), str):
            raise ValueError(f"Aggregation manifest has an invalid source entry in {output_root}")
        source = (experiment_root / Path(item["source"])).resolve()
        if not source.is_relative_to(experiment_root):
            raise ValueError(f"Aggregation manifest source escapes task root: {item['source']}")
        if source.is_file():
            source.unlink()
            removed.append(source)
        interactive_plot = source.with_suffix(".html")
        if interactive_plot.is_file():
            interactive_plot.unlink()
            removed_renderings.append(interactive_plot)

    manifest["source_tables_pruned"] = True
    manifest["pruned_source_tables"] = [str(path.relative_to(experiment_root).as_posix()) for path in removed]
    manifest["pruned_token_renderings"] = [str(path.relative_to(experiment_root).as_posix()) for path in removed_renderings]
    manifest_path = output_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate token-level RQ CSVs into compressed sample-level analysis tables.")
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Defaults to EXPERIMENT_ROOT/analysis_aggregates; source files are never modified.",
    )
    args = parser.parse_args()
    output_root = args.output_root or args.experiment_root / "analysis_aggregates"
    manifest = aggregate_experiment(args.experiment_root, output_root)
    print(
        "[summary] "
        f"{len(manifest['processed'])} tables, "
        f"{manifest['source_bytes'] / 1_000_000_000:.3f} GB -> {manifest['aggregate_bytes'] / 1_000_000_000:.3f} GB",
        flush=True,
    )


if __name__ == "__main__":
    main()
