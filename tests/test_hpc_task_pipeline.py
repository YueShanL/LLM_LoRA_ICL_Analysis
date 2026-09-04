from pathlib import Path
from unittest.mock import patch

from scripts.hpc_task_pipeline import (
    _cleanup_collected_states,
    _done_compacted_task,
    _done_rq1,
    _done_rq2,
    _done_rq21,
    _promote_task_results,
    run_task_analysis_aggregation,
)


def test_cleanup_collected_states_keeps_only_plot_prompt_eval_and_rq3_plot_data(tmp_path: Path):
    run_dir = tmp_path / "task"
    tensors = run_dir / "states" / "rq1" / "tensors"
    tensors.mkdir(parents=True)
    (tensors / "sample__base.pt").write_bytes(b"state")
    (run_dir / "states" / "rq1" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "dataset").mkdir()
    (run_dir / "dataset" / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "adapters").mkdir()
    (run_dir / "adapters" / "adapter.bin").write_bytes(b"adapter")
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    (run_dir / "plots" / "rq1").mkdir(parents=True)
    (run_dir / "plots" / "rq1" / "token_similarity.csv").write_text("layer,value\n", encoding="utf-8")
    (run_dir / "plots" / "rq1" / "similarity.png").write_bytes(b"png")
    (run_dir / "prompt_eval" / "instruction_only").mkdir(parents=True)
    (run_dir / "prompt_eval" / "instruction_only" / "summary.json").write_text("{}", encoding="utf-8")
    patch_run = run_dir / "patches" / "rq3" / "base_to_lora_l2_text"
    patch_run.mkdir(parents=True)
    (patch_run / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (patch_run / "config.json").write_text("{}", encoding="utf-8")
    (patch_run / "generations.jsonl").write_text("{}\n", encoding="utf-8")

    removed = _cleanup_collected_states(run_dir)

    assert removed
    assert not (run_dir / "states").exists()
    assert not (run_dir / "dataset").exists()
    assert not (run_dir / "adapters").exists()
    assert (run_dir / "plots" / "rq1" / "token_similarity.csv").exists()
    assert (run_dir / "plots" / "rq1" / "similarity.png").exists()
    assert (run_dir / "prompt_eval" / "instruction_only" / "summary.json").exists()
    assert (patch_run / "metrics.jsonl").exists()
    assert (patch_run / "config.json").exists()
    assert not (patch_run / "generations.jsonl").exists()
    assert (run_dir / "collected_states_cleanup.json").exists()
    assert _done_compacted_task(run_dir)


def test_promote_task_results_moves_outputs_and_rewrites_paths(tmp_path: Path):
    work_root = tmp_path / "tmp"
    final_root = tmp_path / "experiments"
    work_dir = work_root / "run" / "task"
    final_dir = final_root / "run" / "task"
    work_dir.mkdir(parents=True)
    (work_dir / "config.json").write_text(
        f'{{"output_root": "{work_root / "run"}", "run_dir": "{work_dir}"}}',
        encoding="utf-8",
    )

    _promote_task_results(work_dir, final_dir, work_root / "run", final_root / "run")

    assert not work_dir.exists()
    text = (final_dir / "config.json").read_text(encoding="utf-8")
    assert str(work_root) not in text
    assert str(final_root) in text


def test_promote_task_results_can_replace_stale_final_task(tmp_path: Path):
    work_root = tmp_path / "tmp" / "run"
    final_root = tmp_path / "experiments" / "run"
    work_dir = work_root / "task"
    final_dir = final_root / "task"
    work_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    (work_dir / "config.json").write_text("{}", encoding="utf-8")
    (final_dir / "stale_adapter.bin").write_bytes(b"stale")

    _promote_task_results(work_dir, final_dir, work_root, final_root, replace_existing=True)

    assert (final_dir / "config.json").exists()
    assert not (final_dir / "stale_adapter.bin").exists()


def test_task_aggregation_uses_final_output_and_does_not_require_tmp_dir(tmp_path: Path):
    final_root = tmp_path / "experiments" / "run"
    task_root = final_root / "task_a"
    task_root.mkdir(parents=True)
    config = {
        "output_root": str(tmp_path / "experiments"),
        "run_group_id": "run",
        "cleanup_collected_states": True,
        "aggregate_analysis_outputs": True,
    }
    with patch("scripts.hpc_task_pipeline.aggregation_is_current", return_value=False) as current, patch(
        "scripts.hpc_task_pipeline.aggregate_experiment", return_value={"processed": [], "source_bytes": 0, "aggregate_bytes": 0}
    ) as aggregate, patch("scripts.hpc_task_pipeline.prune_aggregated_source_tables", return_value=[]) as prune:
        run_task_analysis_aggregation(config, "task_a")

    current.assert_called_once_with(task_root, final_root / "analysis_aggregates" / "task_a")
    aggregate.assert_called_once_with(task_root, final_root / "analysis_aggregates" / "task_a")
    prune.assert_called_once_with(task_root, final_root / "analysis_aggregates" / "task_a")


def test_rq_completion_accepts_pruned_token_tables_when_aggregation_is_enabled(tmp_path: Path):
    run_dir = tmp_path / "run" / "task_a"
    aggregate_dir = run_dir.parent / "analysis_aggregates" / run_dir.name
    for stage in ("rq1", "rq2", "rq21"):
        (run_dir / "states" / stage).mkdir(parents=True, exist_ok=True)
        (run_dir / "states" / stage / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
        (run_dir / f"{stage}_config.json").write_text("{}", encoding="utf-8")
    for relative in (
        "plots/rq1/token_similarity.sample_layer_head.csv.gz",
        "plots/rq21/attention_probs/token_similarity.sample_layer_head.csv.gz",
        "plots/rq21/attention_outputs/token_similarity.sample_layer_head.csv.gz",
    ):
        path = aggregate_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"aggregate")

    assert _done_rq1(run_dir, allow_aggregated=True)
    assert _done_rq2(run_dir, allow_aggregated=True)
    assert _done_rq21(run_dir, allow_aggregated=True)
