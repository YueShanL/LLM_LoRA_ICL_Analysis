"""Few-shot prompt preprocessing from task input-output pairs."""

from __future__ import annotations

import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_icl_examples(examples: list[dict]) -> str:
    blocks = [
        f"Example {index}:\nInput:\n{row['input_text']}\nOutput:\n{row['target_text']}"
        for index, row in enumerate(examples, start=1)
    ]
    return "Examples:\n\n" + "\n\n".join(blocks)


def attach_icl_examples(records: list[dict], examples: list[dict], example_count: int) -> list[dict]:
    if example_count <= 0:
        return records
    if len(examples) < example_count:
        raise ValueError(f"Need {example_count} ICL examples, but only {len(examples)} are available.")
    task_ids = {row.get("task_id") for row in [*records, *examples[:example_count]]}
    if len(task_ids) != 1:
        raise ValueError(f"ICL examples must use exactly one task_id, found {sorted(task_ids)}")
    preamble = format_icl_examples(examples[:example_count])
    return [{**record, "prompt_preamble": preamble} for record in records]


def attach_dataset_icl_examples(
    records: list[dict],
    dataset_path: Path,
    *,
    example_count: int,
    split: str = "train",
) -> list[dict]:
    if example_count <= 0:
        return records
    path = dataset_path / f"{split}.jsonl" if dataset_path.is_dir() else dataset_path.with_name(f"{split}.jsonl")
    return attach_icl_examples(records, read_jsonl(path), example_count)
