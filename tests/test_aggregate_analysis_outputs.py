from __future__ import annotations

import csv
import gzip
import math
from pathlib import Path

from scripts.aggregate_analysis_outputs import (
    aggregate_csv,
    aggregate_experiment,
    aggregation_is_current,
    prune_aggregated_source_tables,
)


def test_aggregate_csv_collapses_target_tokens_per_sample_layer_head(tmp_path: Path):
    experiment_root = tmp_path / "experiment"
    source = experiment_root / "task_a" / "plots" / "rq21" / "attention_outputs" / "token_similarity.csv"
    source.parent.mkdir(parents=True)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "task_id",
                "condition",
                "mode",
                "layer",
                "head",
                "token_index",
                "cosine_similarity",
            ),
        )
        writer.writeheader()
        writer.writerows(
            (
                {"sample_id": "a", "task_id": "task_a", "condition": "left_vs_right", "mode": "attention_output", "layer": "0", "head": "1", "token_index": "0", "cosine_similarity": "0.2"},
                {"sample_id": "a", "task_id": "task_a", "condition": "left_vs_right", "mode": "attention_output", "layer": "0", "head": "1", "token_index": "1", "cosine_similarity": "0.6"},
                {"sample_id": "b", "task_id": "task_a", "condition": "left_vs_right", "mode": "attention_output", "layer": "0", "head": "1", "token_index": "0", "cosine_similarity": "0.9"},
            )
        )

    output_root = tmp_path / "aggregates"
    result = aggregate_csv(source, experiment_root, output_root)

    with gzip.open(output_root / result["destination"], "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert result["source_rows"] == 3
    assert result["aggregate_rows"] == 2
    assert rows[0]["source_target_token_count"] == "2"
    assert float(rows[0]["cosine_similarity_mean"]) == 0.4
    assert math.isclose(float(rows[0]["cosine_similarity_std"]), 0.282842712474619, rel_tol=1e-12)
    assert rows[1]["source_target_token_count"] == "1"
    assert float(rows[1]["cosine_similarity_mean"]) == 0.9


def test_aggregation_current_check_detects_source_changes(tmp_path: Path):
    task_root = tmp_path / "task_a"
    source = task_root / "plots" / "rq21" / "attention_outputs" / "token_similarity.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "sample_id,task_id,condition,mode,layer,head,token_index,cosine_similarity\n"
        "sample_a,task_a,left_vs_right,attention_output,0,1,0,0.5\n",
        encoding="utf-8",
    )
    output_root = tmp_path / "analysis_aggregates" / "task_a"

    aggregate_experiment(task_root, output_root)

    assert aggregation_is_current(task_root, output_root)
    with source.open("a", encoding="utf-8") as handle:
        handle.write("sample_a,task_a,left_vs_right,attention_output,0,1,1,0.7\n")
    assert not aggregation_is_current(task_root, output_root)


def test_task_aggregation_skips_redundant_rq2_probability_table(tmp_path: Path):
    task_root = tmp_path / "task_a"
    header = "sample_id,task_id,condition,mode,layer,head,token_index,cosine_similarity\n"
    row = "sample_a,task_a,left_vs_right,attention_pattern,0,1,0,0.5\n"
    for relative in (
        "plots/rq2/token_similarity.csv",
        "plots/rq21/attention_probs/token_similarity.csv",
    ):
        path = task_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + row, encoding="utf-8")

    output_root = tmp_path / "analysis_aggregates" / "task_a"
    manifest = aggregate_experiment(task_root, output_root)

    assert [item["source"] for item in manifest["processed"]] == ["plots/rq21/attention_probs/token_similarity.csv"]
    assert [item["source"] for item in manifest["skipped"]] == ["plots/rq2/token_similarity.csv"]


def test_pruned_task_aggregation_is_current_without_source_tables(tmp_path: Path):
    task_root = tmp_path / "task_a"
    source = task_root / "plots" / "rq1" / "token_similarity.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "sample_id,task_id,condition,mode,layer,head,token_index,cosine_similarity\n"
        "sample_a,task_a,left_vs_right,residual,0,,0,0.5\n",
        encoding="utf-8",
    )
    interactive_plot = source.with_suffix(".html")
    interactive_plot.write_text("<html></html>", encoding="utf-8")
    output_root = tmp_path / "analysis_aggregates" / "task_a"

    aggregate_experiment(task_root, output_root)
    removed = prune_aggregated_source_tables(task_root, output_root)

    assert removed == [source]
    assert not source.exists()
    assert not interactive_plot.exists()
    assert aggregation_is_current(task_root, output_root)
