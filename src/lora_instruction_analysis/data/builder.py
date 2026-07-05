"""Build HF/PEFT-compatible datasets for synthetic transformation tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
from typing import Iterable

from .sources import get_source, iter_builtin_fallback_texts, iter_public_texts
from .tasks import get_task


@dataclass(frozen=True)
class DatasetBuildConfig:
    task_id: str
    source_id: str = "wikitext"
    output_dir: Path = Path("data/generated")
    train_size: int = 800
    validation_size: int = 100
    test_size: int = 100
    max_source_rows: int = 5000
    min_words: int = 4
    max_words: int = 32
    seed: int = 13
    condition: str = "lora_training"
    include_instruction_in_prompt: bool = False
    write_csv: bool = True
    write_hf_dataset: bool = True
    allow_builtin_fallback: bool = False
    streaming: bool = False

    @property
    def total_size(self) -> int:
        return self.train_size + self.validation_size + self.test_size


def make_prompt(input_text: str, instruction_text: str, *, include_instruction: bool) -> str:
    if include_instruction:
        return (
            "Instruction:\n"
            f"{instruction_text}\n\n"
            "Input:\n"
            f"{input_text}\n\n"
            "Output:\n"
        )
    return f"Input:\n{input_text}\n\nOutput:\n"


def make_record(
    *,
    sample_id: str,
    task_id: str,
    input_text: str,
    instruction_text: str,
    target_text: str,
    condition: str,
    include_instruction_in_prompt: bool,
) -> dict:
    prompt = make_prompt(
        input_text,
        instruction_text,
        include_instruction=include_instruction_in_prompt,
    )
    return {
        "sample_id": sample_id,
        "task_id": task_id,
        "input_text": input_text,
        "instruction_text": instruction_text,
        "target_text": target_text,
        "condition": condition,
        "instruction": instruction_text,
        "input": input_text,
        "output": target_text,
        "prompt": prompt,
        "response": target_text,
        "text": f"{prompt}{target_text}",
        "messages": [
            {"role": "user", "content": f"{instruction_text}\n\nInput:\n{input_text}"},
            {"role": "assistant", "content": target_text},
        ],
    }


def _collect_inputs(config: DatasetBuildConfig) -> list[str]:
    if config.max_source_rows == 0 and config.allow_builtin_fallback:
        texts = list(iter_builtin_fallback_texts())
    else:
        source = get_source(config.source_id)
        try:
            texts = list(
                iter_public_texts(
                    source,
                    max_source_rows=config.max_source_rows,
                    min_words=config.min_words,
                    max_words=config.max_words,
                    streaming=config.streaming,
                )
            )
        except Exception:
            if not config.allow_builtin_fallback:
                raise
            texts = list(iter_builtin_fallback_texts())

    unique_texts = list(dict.fromkeys(texts))
    if len(unique_texts) < config.total_size and config.allow_builtin_fallback:
        unique_texts = list(dict.fromkeys([*unique_texts, *iter_builtin_fallback_texts()]))
    if len(unique_texts) < config.total_size:
        raise ValueError(
            f"Only collected {len(unique_texts)} usable source texts, "
            f"but {config.total_size} are required. Increase --max-source-rows, "
            "choose another source, reduce split sizes, or pass --allow-builtin-fallback."
        )
    random.Random(config.seed).shuffle(unique_texts)
    return unique_texts


def _split_records(records: list[dict], config: DatasetBuildConfig) -> dict[str, list[dict]]:
    train_end = config.train_size
    validation_end = train_end + config.validation_size
    return {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }


def build_dataset(config: DatasetBuildConfig) -> dict[str, list[dict]]:
    task = get_task(config.task_id)
    inputs = _collect_inputs(config)

    records = []
    for idx, input_text in enumerate(inputs):
        target_text = task.transform(input_text)
        if not target_text:
            continue
        records.append(
            make_record(
                sample_id=f"{config.task_id}-{idx:06d}",
                task_id=config.task_id,
                input_text=input_text,
                instruction_text=task.natural_language_instruction,
                target_text=target_text,
                condition=config.condition,
                include_instruction_in_prompt=config.include_instruction_in_prompt,
            )
        )
        if len(records) == config.total_size:
            break
    if len(records) < config.total_size:
        raise ValueError(f"Only built {len(records)} non-empty records, but {config.total_size} are required.")
    return _split_records(records, config)


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    csv_records = []
    for record in records:
        flat = dict(record)
        flat["messages"] = json.dumps(flat["messages"], ensure_ascii=False)
        csv_records.append(flat)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_records[0]))
        writer.writeheader()
        writer.writerows(csv_records)


def write_hf_dataset(output_dir: Path, splits: dict[str, list[dict]]) -> None:
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError("The datasets package is required for HF save_to_disk output.") from exc

    dataset_dict = DatasetDict(
        {split_name: Dataset.from_list(records) for split_name, records in splits.items()}
    )
    dataset_dict.save_to_disk(str(output_dir))


def write_dataset(config: DatasetBuildConfig, splits: dict[str, list[dict]]) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    source = get_source(config.source_id)
    task = get_task(config.task_id)
    manifest = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "source": asdict(source),
        "task": {
            "task_id": task.task_id,
            "natural_language_instruction": task.natural_language_instruction,
            "allowed_output_format": task.allowed_output_format,
        },
        "splits": {name: len(records) for name, records in splits.items()},
        "format_notes": {
            "text": "Prompt concatenated with target_text; compatible with SFTTrainer dataset_text_field='text'.",
            "messages": "Chat-style user/assistant messages for tokenizer chat templates.",
        },
    }

    (config.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for split_name, records in splits.items():
        write_jsonl(config.output_dir / f"{split_name}.jsonl", records)
        if config.write_csv:
            write_csv(config.output_dir / f"{split_name}.csv", records)
    if config.write_hf_dataset:
        write_hf_dataset(config.output_dir / "hf_dataset", splits)
