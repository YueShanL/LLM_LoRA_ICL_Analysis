from pathlib import Path

from scripts.hpc_task_pipeline import _cleanup_collected_states


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
