import json
from pathlib import Path

import torch

from lora_instruction_analysis.model.train_lora import (
    TrainConfig,
    _parameter_nonfinite_issue,
    _write_training_failure,
)


def test_parameter_nonfinite_issue_reports_gradient_and_parameter():
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.full_like(model.weight, float("nan"))

    assert "non-finite gradient" in _parameter_nonfinite_issue(model)

    model.weight.grad = None
    model.weight.data[0, 0] = float("inf")
    assert "non-finite value" in _parameter_nonfinite_issue(model, include_gradients=False)


def test_training_failure_removes_adapter_and_writes_marker(tmp_path: Path):
    output_dir = tmp_path / "adapter"
    output_dir.mkdir()
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
        (output_dir / name).write_bytes(b"invalid")

    config = TrainConfig(
        model_name="test/model",
        dataset_path=tmp_path / "dataset",
        output_dir=output_dir,
    )
    failure_path = _write_training_failure(config, FloatingPointError("step=3: NaN loss"))

    assert failure_path.exists()
    assert not (output_dir / "adapter_model.safetensors").exists()
    assert not (output_dir / "adapter_model.bin").exists()
    assert not (output_dir / "adapter_config.json").exists()
    assert json.loads(failure_path.read_text(encoding="utf-8"))["status"] == "failed_nonfinite"
