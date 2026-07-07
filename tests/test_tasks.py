from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.data.tasks import evaluate_output, get_task


def test_candidate_tasks():
    text = "Hello tiny world."
    assert get_task("last_word").transform(text) == "world"
    assert get_task("word_count").transform(text) == "3"
    assert get_task("uppercase_last_word").transform(text) == "WORLD"
    assert get_task("at_operator_mod_minus_left").transform("17@5=?") == "-15"


def test_evaluate_output_uses_task_semantics_with_whitespace_normalization():
    row = evaluate_output("word_count", "Hello tiny world.", "  3\n", "3")
    assert row["task_expected_source"] == "task_transform"
    assert row["task_expected_text"] == "3"
    assert row["task_semantic_correct"] == 1.0
