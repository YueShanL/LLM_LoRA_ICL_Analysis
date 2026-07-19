from pathlib import Path
import sys
import json
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lora_instruction_analysis.data.builder as builder
from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset, write_dataset
from lora_instruction_analysis.data.validation import DatasetValidationError, validate_dataset


def test_declared_source_route_fails_when_it_has_too_few_rows():
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: iter(())
    try:
        with pytest.raises(ValueError, match="Declared source route 'wikitext'"):
            build_dataset(DatasetBuildConfig(task_id="last_word", train_size=1, validation_size=1, test_size=1))
    finally:
        builder.iter_public_texts = original


def test_at_operator_task_generates_unique_unseen_test_inputs():
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("public source used")
    )
    try:
        splits = build_dataset(
            DatasetBuildConfig(
                task_id="at_operator_mod_minus_left",
                train_size=3,
                validation_size=2,
                test_size=2,
            )
        )
    finally:
        builder.iter_public_texts = original

    train_inputs = {record["input_text"] for record in splits["train"]}
    validation_inputs = {record["input_text"] for record in splits["validation"]}
    test_inputs = {record["input_text"] for record in splits["test"]}
    assert len(train_inputs | validation_inputs | test_inputs) == 7
    assert test_inputs.isdisjoint(train_inputs | validation_inputs)

    sample = splits["train"][0]
    a_text, b_text = sample["input_text"][:-2].split("@")
    assert sample["target_text"] == str(int(a_text) % int(b_text) - int(a_text))


def test_mechanism_candidate_tasks_build_validation_rows():
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: iter(
        f"wiki corpus row number {index} with bright dax words" for index in range(32)
    )
    try:
        for task_id in (
            "list_letters_space_separated",
            "sum_two_numbers",
            "exact_three_word_prefix",
            "extract_items_from_set",
            "has_repeated_word",
            "words_containing_bigram_qu",
            "formal_language_a_n_b_n",
        ):
            splits = build_dataset(
                DatasetBuildConfig(task_id=task_id, train_size=0, validation_size=0, test_size=16)
            )
            assert len(splits["test"]) == 16
            assert all(record["target_text"] for record in splits["test"])
    finally:
        builder.iter_public_texts = original


def test_formal_language_uses_three_seeded_random_values_and_both_labels():
    config = DatasetBuildConfig(
        task_id="formal_language_a_n_b_n",
        train_size=0,
        validation_size=0,
        test_size=100,
    )
    first = build_dataset(config)["test"]
    second = build_dataset(config)["test"]
    assert [row["input_text"] for row in first] == [row["input_text"] for row in second]
    assert len({row["input_text"] for row in first}) == 100
    assert {row["target_text"] for row in first} == {"YES", "NO"}


def test_qu_task_seededly_inserts_qu_into_random_words():
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: iter(
        f"wiki corpus sentence number {index} contains several ordinary words" for index in range(100)
    )
    try:
        rows = build_dataset(
            DatasetBuildConfig(
                task_id="words_containing_bigram_qu",
                train_size=0,
                validation_size=0,
                test_size=40,
            )
        )["test"]
    finally:
        builder.iter_public_texts = original
    positives = [row for row in rows if row["target_text"] != "NONE"]
    negatives = [row for row in rows if row["target_text"] == "NONE"]
    assert positives and negatives
    assert all("qu" in row["input_text"].lower() for row in positives)
    assert all("qu" not in row["input_text"].lower() for row in negatives)


def test_manifest_and_validation_record_route_validator_and_tokenization(tmp_path):
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: iter(("one two three", "four five six", "seven eight nine"))
    try:
        config = DatasetBuildConfig(
            task_id="reverse_words",
            output_dir=tmp_path,
            train_size=1,
            validation_size=1,
            test_size=1,
            write_csv=False,
            write_hf_dataset=False,
            model_name="model",
            tokenizer_name="tokenizer",
            prompt_template="template-v1",
        )
        write_dataset(config, build_dataset(config))
    finally:
        builder.iter_public_texts = original
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"]["validator"] == "fixed_exact"
    assert manifest["data_route"]["source_id"] == "wikitext"
    assert manifest["target_tokenization"] == {
        "model_name": "model", "tokenizer": "tokenizer", "prompt_template": "template-v1"
    }
    assert validate_dataset(tmp_path)["valid"]


def test_fixed_alignment_smoke_fixture_is_valid():
    path = Path(__file__).parent / "fixtures" / "alignment_smoke"
    assert validate_dataset(path)["splits"] == {"train": 1, "validation": 1, "test": 1}
