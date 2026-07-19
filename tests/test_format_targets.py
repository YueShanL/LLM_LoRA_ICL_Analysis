from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lora_instruction_analysis.model.format_targets as format_targets
from lora_instruction_analysis.model.format_targets import FormatTargetConfig, generate_format_targets


OUTPUTS = {
    "fixed_three_bullets": "- one\n- two\n- three",
    "include_fixed_keywords": "aurora and harbor",
    "exclude_fixed_words": "safe response",
    "json_answer_schema": '{"answer": "ok", "confidence": 1}',
}
STATES = {
    "fixed_three_bullets": {"bullet_marker": "- ", "bullet_count": 3},
    "include_fixed_keywords": {"required_keywords": ["aurora", "harbor"]},
    "exclude_fixed_words": {"forbidden_words": ["the", "and"]},
    "json_answer_schema": {"json_keys": ["answer", "confidence"]},
}


def test_format_target_module_writes_attempt_used_accepted_and_splits(tmp_path, monkeypatch):
    sources = tmp_path / "sources.jsonl"
    sources.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": f"source-{task_type}",
                    "format_task_type": task_type,
                    "input_text": "answer this request",
                    "instruction_text": "follow the fixed format",
                    "target_state": STATES[task_type],
                }
            ) + "\n"
            for task_type in OUTPUTS
        ),
        encoding="utf-8",
    )

    def fake_outputs(_config, records):
        for record in records:
            yield record, OUTPUTS[record["format_task_type"]]

    monkeypatch.setattr(format_targets, "_generate_outputs", fake_outputs)
    output = tmp_path / "targets"
    generate_format_targets(
        FormatTargetConfig(
            sources=sources,
            output_dir=output,
            model_name="dummy",
            train_size=1,
            validation_size=0,
            test_size=0,
        )
    )

    for task_type in OUTPUTS:
        task_dir = output / task_type
        assert (task_dir / "generation_attempts.jsonl").exists()
        assert (task_dir / "used_sources.jsonl").exists()
        assert (task_dir / "accepted.jsonl").exists()
        manifest = json.loads((task_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["complete"]
        assert manifest["accepted"] == 1
