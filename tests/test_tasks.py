from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.data.tasks import (
    evaluate_output,
    get_task,
    list_tasks,
    task_default_prompt,
    task_default_validator_name,
    task_default_prompt_variant,
    task_validation_kind,
    validate_generated_output,
)


def test_candidate_tasks():
    text = "Hello tiny world."
    assert get_task("last_word").transform(text) == "world"
    assert get_task("word_count").transform(text) == "3"
    assert get_task("uppercase_last_word").transform(text) == "WORLD"
    assert get_task("at_operator_mod_minus_left").transform("17@5=?") == "-15"
    assert get_task("second_letter").transform(text) == "e"
    assert get_task("char_count_no_space").transform(text) == "15"
    assert get_task("vowel_count").transform(text) == "4"
    assert get_task("words_starting_with_letter").transform("many small maps") == "many maps"
    assert get_task("list_letters_space_separated").transform("Hello tiny") == "H e l l o"
    assert get_task("sum_two_numbers").transform("17+5=?") == "22"
    assert get_task("exact_three_word_prefix").transform("one two three four") == "one two three"
    assert get_task("extract_items_from_set").transform("dax moon wug") == "dax wug"
    assert get_task("has_repeated_word").transform("red blue red") == "YES"
    assert get_task("words_containing_bigram_qu").transform("quick stone quartz") == "quick quartz"
    assert get_task("formal_language_a_n_b_n").transform("aaabbb") == "YES"


def test_evaluate_output_uses_task_semantics_with_whitespace_normalization():
    row = evaluate_output("word_count", "Hello tiny world.", "  3\n", "3")
    assert row["task_expected_source"] == "task_constraint"
    assert row["task_expected_text"] == "3"
    assert row["task_validation_kind"] == "constraint"
    assert row["task_validator"] == "constraint_word_count"
    assert row["task_semantic_correct"] == 1.0


def test_fixed_target_tasks_use_provided_target_text():
    row = evaluate_output("add_zxq_after_t_or_l", "cat", "CUSTOM", "CUSTOM")

    assert row["task_expected_source"] == "target_text"
    assert row["task_expected_text"] == "CUSTOM"
    assert row["task_validation_kind"] == "fixed_target"
    assert row["task_validator"] == "fixed_exact"
    assert row["task_semantic_correct"] == 1.0


def test_constraint_tasks_ignore_wrong_target_text():
    row = evaluate_output("word_count", "Hello tiny world.", "3", "999")

    assert row["task_expected_source"] == "task_constraint"
    assert row["task_expected_text"] == "3"
    assert row["task_validation_kind"] == "constraint"
    assert row["task_validator"] == "constraint_word_count"
    assert row["task_semantic_correct"] == 1.0


def test_all_tasks_have_matching_default_validation_kind():
    for task in list_tasks():
        validator = task_default_validator_name(task.task_id)
        assert validator.startswith("constraint_") if task.validation_kind == "constraint" else validator.startswith("fixed_")
        assert task_validation_kind(task.task_id) == task.validation_kind


def test_all_tasks_have_collected_default_prompt():
    for task in list_tasks():
        assert task_default_prompt_variant(task.task_id) in (1, 2, 3)
        assert task.natural_language_instruction in task_default_prompt(task.task_id)
    assert task_default_prompt_variant("words_starting_with_letter") == 2
    assert task_default_prompt_variant("exact_three_word_prefix") == 2
    assert task_default_prompt_variant("words_containing_bigram_qu") == 3


def test_at_operator_uses_long_generation_budget():
    assert get_task("at_operator_mod_minus_left").max_generate_tokens == 128


def test_task_validators_tolerate_generation_tail_text():
    assert validate_generated_output("last_word", "Hello tiny world.", "world\nDone.", "world")
    assert validate_generated_output("word_count", "Hello tiny world.", "The answer is 3.", "3")
    assert validate_generated_output("has_repeated_word", "red blue red", "yes", "YES")
    assert validate_generated_output("formal_language_a_n_b_n", "aab", "No.", "NO")
    assert validate_generated_output("at_operator_mod_minus_left", "17@5=?", "The answer is -15.", "-15")
    assert validate_generated_output("at_operator_mod_minus_left", "3@22=?", "3@22 = 3 % 22 - 3\n= 3 - 3\n= 0", "0")
    assert not validate_generated_output("at_operator_mod_minus_left", "17@5=?", "a = 17\nb = 5", "-15")
    assert not validate_generated_output("at_operator_mod_minus_left", "54@19=?", "54 % 19 = 15\n15 - 54 = -39", "-38")


def test_validation_can_be_overridden_by_name_or_callable():
    assert not validate_generated_output("word_count", "Hello tiny world.", "The answer is 3.", "3", validator="exact")

    def accepts_anything(_input_text: str, _pred_text: str, _expected_text: str) -> bool:
        return True

    row = evaluate_output("word_count", "Hello tiny world.", "wrong", "3", validator=accepts_anything)
    assert row["task_validator"] == "accepts_anything"
    assert row["task_semantic_correct"] == 1.0
