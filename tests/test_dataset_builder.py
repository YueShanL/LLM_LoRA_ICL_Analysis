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
