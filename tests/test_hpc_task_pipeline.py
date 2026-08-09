from pathlib import Path

from scripts.hpc_task_pipeline import _cleanup_collected_states, _promote_task_results


def test_cleanup_collected_states_removes_raw_tensors_and_keeps_plot_data(tmp_path: Path):
    run_dir = tmp_path / "task"
    tensors = run_dir / "states" / "rq1" / "tensors"
    tensors.mkdir(parents=True)
    (tensors / "sample__base.pt").write_bytes(b"state")
    (run_dir / "states" / "rq1" / "metrics.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "plots" / "rq1").mkdir(parents=True)
    (run_dir / "plots" / "rq1" / "token_similarity.csv").write_text("layer,value\n", encoding="utf-8")

    removed = _cleanup_collected_states(run_dir)

    assert removed == [tensors]
    assert not tensors.exists()
    assert (run_dir / "states" / "rq1" / "metrics.jsonl").exists()
    assert (run_dir / "plots" / "rq1" / "token_similarity.csv").exists()
    assert (run_dir / "collected_states_cleanup.json").exists()


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
