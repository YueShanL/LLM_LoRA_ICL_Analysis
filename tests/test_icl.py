from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.data.icl import attach_icl_examples


def test_attach_icl_examples_uses_input_output_pairs_only():
    records = [{"sample_id": "t1", "task_id": "last_word", "input_text": "new input", "target_text": "input"}]
    examples = [
        {"sample_id": "e1", "task_id": "last_word", "input_text": "one two", "target_text": "two"},
        {"sample_id": "e2", "task_id": "last_word", "input_text": "red blue", "target_text": "blue"},
    ]

    [record] = attach_icl_examples(records, examples, 2)

    assert record["prompt_preamble"] == (
        "Examples:\n\n"
        "Example 1:\nInput:\none two\nOutput:\ntwo\n\n"
        "Example 2:\nInput:\nred blue\nOutput:\nblue"
    )
    assert "instruction" not in record["prompt_preamble"].lower()
