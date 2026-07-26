"""Build HF/PEFT-compatible datasets for synthetic transformation tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
import random
import re
from typing import Iterable

from .sources import get_source, iter_public_texts
from .tasks import (
    get_task,
    task_default_prompt,
    task_default_prompt_variant_name,
    task_default_validator_name,
)

ARITHMETIC_TASK_IDS = {"at_operator_mod_minus_left", "sum_two_numbers"}
PROMPT_TEMPLATE_VERSION = "input_output_v1"


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
    streaming: bool = False
    model_name: str | None = None
    tokenizer_name: str | None = None
    prompt_template: str = PROMPT_TEMPLATE_VERSION

    @property
    def total_size(self) -> int:
        return self.train_size + self.validation_size + self.test_size


def make_prompt(
    input_text: str,
    instruction_text: str,
    *,
    include_instruction: bool,
    preamble: str | None = None,
) -> str:
    if preamble is not None:
        return f"{preamble}\n\nInput:\n{input_text}\n\nOutput:\n"
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
    if config.task_id == "at_operator_mod_minus_left":
        pairs = [(a, b) for a in range(100) for b in range(1, 100)]
        if len(pairs) < config.total_size:
            raise ValueError(
                f"Only {len(pairs)} unique a@b examples are available, but {config.total_size} are required."
            )
        random.Random(config.seed).shuffle(pairs)
        return [f"{a}@{b}=?" for a, b in pairs[: config.total_size]]
    if config.task_id == "sum_two_numbers":
        pairs = [(a, b) for a in range(100) for b in range(100)]
        if len(pairs) < config.total_size:
            raise ValueError(
                f"Only {len(pairs)} unique a+b examples are available, but {config.total_size} are required."
            )
        random.Random(config.seed).shuffle(pairs)
        return [f"{a}+{b}=?" for a, b in pairs[: config.total_size]]
    if config.task_id == "formal_language_a_n_b_n":
        rng = random.Random(config.seed)
        max_length = max(64, config.total_size)
        inputs: list[str] = []
        seen = set()
        while len(inputs) < config.total_size:
            a_length = rng.randint(1, max_length)
            same_length = rng.randint(0, 1)
            random_b_length = rng.randint(1, max_length)
            if not same_length and random_b_length == a_length:
                random_b_length = random_b_length % max_length + 1
            b_length = a_length if same_length else random_b_length
            text = "a" * a_length + "b" * b_length
            if text not in seen:
                seen.add(text)
                inputs.append(text)
        return inputs
    source = get_source(config.source_id)
    texts = list(
        iter_public_texts(
            source,
            max_source_rows=config.max_source_rows,
            min_words=config.min_words,
            max_words=config.max_words,
            streaming=config.streaming,
        )
    )

    unique_texts = list(dict.fromkeys(texts))
    if len(unique_texts) < config.total_size:
        raise ValueError(
            f"Declared source route {config.source_id!r} produced {len(unique_texts)} unique rows, "
            f"but {config.total_size} are required. Increase --max-source-rows, choose another "
            "declared source route, or reduce split sizes."
        )
    random.Random(config.seed).shuffle(unique_texts)
    if config.task_id == "words_containing_bigram_qu":
        rng = random.Random(config.seed)
        augmented = []
        for text in unique_texts:
            if "qu" in text.lower():
                continue
            words = text.split()
            eligible = [index for index, word in enumerate(words) if re.search(r"[A-Za-z]", word)]
            if rng.randint(0, 1):
                word_index = rng.choice(eligible)
                match = re.search(r"[A-Za-z]+", words[word_index])
                assert match is not None
                insertion = rng.randint(match.start(), match.end())
                word = words[word_index]
                words[word_index] = f"{word[:insertion]}qu{word[insertion:]}"
            augmented.append(" ".join(words))
            if len(augmented) == config.total_size:
                return augmented
        raise ValueError(
            f"Declared source route {config.source_id!r} produced only {len(augmented)} qu-ready rows, "
            f"but {config.total_size} are required."
        )
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
                instruction_text=task_default_prompt(config.task_id),
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
    from .validation import validate_splits

    validate_splits(
        splits,
        {"train": config.train_size, "validation": config.validation_size, "test": config.test_size},
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    task = get_task(config.task_id)
    arithmetic = config.task_id in ARITHMETIC_TASK_IDS
    source = (
        {"source_id": f"{config.task_id}_grid", "route": "deterministic_arithmetic_grid"}
        if arithmetic
        else {
            "source_id": "three_random_lengths",
            "route": "seeded_a_length_equal_flag_b_length",
        }
        if config.task_id == "formal_language_a_n_b_n"
        else asdict(get_source(config.source_id))
    )
    route_kind = (
        "deterministic_arithmetic"
        if arithmetic
        else "seeded_synthetic"
        if config.task_id == "formal_language_a_n_b_n"
        else "seeded_text_augmentation"
        if config.task_id == "words_containing_bigram_qu"
        else "text_source"
    )
    manifest = {
        "config": {**asdict(config), "output_dir": str(config.output_dir)},
        "data_route": {
            "kind": route_kind,
            "source_id": source["source_id"],
            "seed": config.seed,
            "max_source_rows": config.max_source_rows,
            "streaming": config.streaming,
            "generator": (
                "a_length/equal_flag/b_length"
                if config.task_id == "formal_language_a_n_b_n"
                else "random_qu_insertion"
                if config.task_id == "words_containing_bigram_qu"
                else None
            ),
        },
        "source": source,
        "task": {
            "task_id": task.task_id,
            "natural_language_instruction": task.natural_language_instruction,
            "default_instruction_prompt": task_default_prompt(config.task_id),
            "default_prompt_variant": task_default_prompt_variant_name(config.task_id),
            "validator": task_default_validator_name(config.task_id),
            "allowed_output_format": task.allowed_output_format,
        },
        "target_tokenization": {
            "model_name": config.model_name,
            "tokenizer": config.tokenizer_name or config.model_name,
            "prompt_template": config.prompt_template,
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
