from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lora_instruction_analysis.data.builder as builder
from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset


def test_builtin_fallback_handles_too_few_source_rows():
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: iter(())
    try:
        splits = build_dataset(
            DatasetBuildConfig(
                task_id="last_word",
                train_size=1,
                validation_size=1,
                test_size=1,
                max_source_rows=1,
                allow_builtin_fallback=True,
            )
        )
    finally:
        builder.iter_public_texts = original
    assert [len(splits[name]) for name in ("train", "validation", "test")] == [1, 1, 1]


def test_zero_source_rows_uses_builtin_fallback_without_public_source():
    original = builder.iter_public_texts
    builder.iter_public_texts = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("public source used"))
    try:
        splits = build_dataset(
            DatasetBuildConfig(
                task_id="last_word",
                train_size=0,
                validation_size=0,
                test_size=2,
                max_source_rows=0,
                allow_builtin_fallback=True,
                write_hf_dataset=False,
            )
        )
    finally:
        builder.iter_public_texts = original
    assert len(splits["test"]) == 2


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
