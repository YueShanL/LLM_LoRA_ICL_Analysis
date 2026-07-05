from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.model.task_acceptance import TaskAcceptanceConfig, passes_gate
import lora_instruction_analysis.model.task_acceptance as task_acceptance


def test_passes_gate():
    config = TaskAcceptanceConfig(
        model_name="dummy",
        dataset_path=Path("dummy"),
        output_dir=Path("dummy"),
        min_instruction_token_accuracy=0.8,
        max_no_instruction_token_accuracy=0.3,
    )
    high = {"teacher_forced": {"mean_token_accuracy": 0.9}}
    low = {"teacher_forced": {"mean_token_accuracy": 0.2}}
    assert passes_gate(high, low, config)
    assert not passes_gate(low, low, config)
    assert not passes_gate(high, high, config)


def test_task_arg_builds_quick_dataset(tmp_path):
    args = type(
        "Args",
        (),
        {
            "dataset_path": None,
            "task": "last_word",
            "output_dir": tmp_path,
            "max_samples": 2,
        },
    )()
    dataset_path = task_acceptance._dataset_path(args)
    assert dataset_path == tmp_path / "dataset"
    assert (dataset_path / "test.jsonl").exists()
