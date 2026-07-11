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
        min_instruction_semantic_accuracy=0.8,
        max_no_instruction_semantic_accuracy=0.3,
    )
    high = {"autoregressive": {"mean_task_semantic_correct": 0.9}}
    low = {"autoregressive": {"mean_task_semantic_correct": 0.2}}
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


def test_validate_task_passes_custom_validator_to_prompt_eval(tmp_path, monkeypatch):
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "test.jsonl").write_text(
        '{"sample_id":"s1","task_id":"word_count","input_text":"hello world","target_text":"2","instruction_text":"count words"}\n',
        encoding="utf-8",
    )
    seen_validators = []

    def accepts_anything(_input_text: str, _pred_text: str, _expected_text: str) -> bool:
        return True

    def fake_evaluate_prompt(config):
        seen_validators.append(config.validator)
        return {
            "teacher_forced": None,
            "autoregressive": {
                "samples": 1,
                "mean_token_accuracy": 1.0,
                "mean_sequence_accuracy": 1.0,
                "mean_task_semantic_correct": 1.0 if config.include_instruction else 0.0,
            },
        }

    monkeypatch.setattr(task_acceptance, "evaluate_prompt", fake_evaluate_prompt)
    summary = task_acceptance.validate_task(
        TaskAcceptanceConfig(
            model_name="dummy",
            dataset_path=dataset_path,
            output_dir=tmp_path / "out",
            validator=accepts_anything,
        )
    )
    assert summary["accepted"]
    assert seen_validators == [accepts_anything, accepts_anything]
    assert summary["config"]["validator"] == "accepts_anything"
