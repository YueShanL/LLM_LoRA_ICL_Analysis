"""Generate and retain auditable targets for fixed-state format tasks."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Iterable

from lora_instruction_analysis.data.builder import make_record, write_jsonl


FORMAT_TASK_TYPES = (
    "fixed_three_bullets",
    "include_fixed_keywords",
    "exclude_fixed_words",
    "json_answer_schema",
)


@dataclass(frozen=True)
class FormatTargetConfig:
    sources: Path
    output_dir: Path
    model_name: str
    task_types: tuple[str, ...] = FORMAT_TASK_TYPES
    train_size: int = 800
    validation_size: int = 100
    test_size: int = 100
    seed: int = 13
    max_attempts_per_task: int = 10000
    max_new_tokens: int = 160
    prompt_format: str = "chat_template"
    dtype: str = "auto"

    @property
    def target_total(self) -> int:
        return self.train_size + self.validation_size + self.test_size


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def format_check(record: dict, output: str) -> dict:
    task_type = record["format_task_type"]
    state = record.get("target_state") or {}
    if task_type == "fixed_three_bullets":
        marker = str(state.get("bullet_marker", "- "))
        actual = len(re.findall(rf"(?m)^\s*{re.escape(marker)}\S", output))
        expected = int(state.get("bullet_count", 3))
        return {"format_pass": actual == expected, "expected_bullets": expected, "actual_bullets": actual}
    if task_type == "include_fixed_keywords":
        keywords = list(state.get("required_keywords", []))
        lower = output.lower()
        return {"format_pass": bool(keywords) and all(word.lower() in lower for word in keywords), "required_keywords": keywords}
    if task_type == "exclude_fixed_words":
        forbidden = list(state.get("forbidden_words", []))
        words = set(re.findall(r"\b[\w-]+\b", output.lower()))
        return {"format_pass": bool(forbidden) and not any(word.lower() in words for word in forbidden), "forbidden_words": forbidden}
    if task_type == "json_answer_schema":
        keys = list(state.get("json_keys", []))
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            value = None
        return {"format_pass": isinstance(value, dict) and set(value) == set(keys), "json_keys": keys}
    raise KeyError(f"Unknown format_task_type {task_type!r}")


def _generate_outputs(config: FormatTargetConfig, records: Iterable[dict]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_instruction_analysis.model.collect import _torch_dtype
    from lora_instruction_analysis.model.formatting import ensure_chat_template

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ensure_chat_template(tokenizer, config.model_name, config.prompt_format)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=_torch_dtype(torch, config.dtype))
    model.to(device).eval()
    for record in records:
        if config.prompt_format == "chat_template":
            prompt_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": record["instruction_text"]}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids
            prompt_ids = prompt_ids.to(device)
        else:
            prompt_ids = tokenizer(record["instruction_text"], return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                do_sample=False,
                max_new_tokens=config.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        yield record, tokenizer.decode(generated[0, prompt_ids.shape[1]:], skip_special_tokens=True).strip()


def _candidates(config: FormatTargetConfig, pools: dict[str, list[dict]], accepted: dict[str, list[dict]]):
    attempts = {task_type: 0 for task_type in config.task_types}
    rng = random.Random(config.seed)
    shuffled = {task_type: rows[:] for task_type, rows in pools.items()}
    for rows in shuffled.values():
        rng.shuffle(rows)
    while True:
        active = [
            task_type for task_type in config.task_types
            if len(accepted[task_type]) < config.target_total and attempts[task_type] < config.max_attempts_per_task
        ]
        if not active:
            return
        for task_type in active:
            rows = shuffled[task_type]
            if not rows:
                raise ValueError(f"No declared source rows for {task_type}")
            index = attempts[task_type]
            attempts[task_type] += 1
            yield {**rows[index % len(rows)], "generation_attempt": index + 1}


def _write_task(config: FormatTargetConfig, task_type: str, attempts: list[dict], accepted: list[dict]) -> None:
    output = config.output_dir / task_type
    output.mkdir(parents=True, exist_ok=True)
    splits = {
        "train": accepted[:config.train_size],
        "validation": accepted[config.train_size:config.train_size + config.validation_size],
        "test": accepted[config.train_size + config.validation_size:config.target_total],
    }
    write_jsonl(output / "generation_attempts.jsonl", attempts)
    write_jsonl(output / "used_sources.jsonl", [row["source_record"] for row in attempts])
    write_jsonl(output / "accepted.jsonl", accepted)
    for split, rows in splits.items():
        write_jsonl(output / f"{split}.jsonl", rows)
    manifest = {
        "task_id": task_type,
        "target_state": attempts[0]["source_record"].get("target_state") if attempts else None,
        "model_name": config.model_name,
        "tokenizer": config.model_name,
        "prompt_template": config.prompt_format,
        "sources": str(config.sources),
        "data_route": "declared_format_source_pool",
        "attempts": len(attempts),
        "accepted": len(accepted),
        "target_accepted": config.target_total,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "complete": len(accepted) == config.target_total,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if not manifest["complete"]:
        raise RuntimeError(
            f"{task_type}: accepted {len(accepted)}/{config.target_total} after {len(attempts)} attempts"
        )


def generate_format_targets(config: FormatTargetConfig) -> None:
    sources = _read_jsonl(config.sources)
    pools = {task_type: [row for row in sources if row.get("format_task_type") == task_type] for task_type in config.task_types}
    for task_type, rows in pools.items():
        states = {json.dumps(row.get("target_state"), sort_keys=True) for row in rows}
        if len(states) != 1:
            raise ValueError(f"{task_type} must use exactly one fixed target_state, found {len(states)}")
    attempts = {task_type: [] for task_type in config.task_types}
    accepted = {task_type: [] for task_type in config.task_types}
    accepted_sources = {task_type: set() for task_type in config.task_types}
    for source, target_text in _generate_outputs(config, _candidates(config, pools, accepted)):
        task_type = source["format_task_type"]
        checks = format_check(source, target_text)
        attempt = {
            "attempt_id": f"{task_type}-attempt-{len(attempts[task_type]):06d}",
            "source_sample_id": source["sample_id"],
            "format_task_type": task_type,
            "generated_text": target_text,
            "format_checks": checks,
            "format_pass": checks["format_pass"],
            "source_record": source,
        }
        attempts[task_type].append(attempt)
        if (
            checks["format_pass"]
            and source["sample_id"] not in accepted_sources[task_type]
            and len(accepted[task_type]) < config.target_total
        ):
            accepted_sources[task_type].add(source["sample_id"])
            accepted[task_type].append(
                {
                    **make_record(
                        sample_id=f"{task_type}-{len(accepted[task_type]):06d}",
                        task_id=task_type,
                        input_text=source["input_text"],
                        instruction_text=source["instruction_text"],
                        target_text=target_text,
                        condition="lora_training",
                        include_instruction_in_prompt=False,
                    ),
                    "target_state": source.get("target_state"),
                    "format_checks": checks,
                    "source_sample_id": source["sample_id"],
                }
            )
    for task_type in config.task_types:
        _write_task(config, task_type, attempts[task_type], accepted[task_type])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--task-type", choices=FORMAT_TASK_TYPES, action="append")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-attempts-per-task", type=int, default=10000)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--prompt-format", choices=("raw", "chat_template"), default="chat_template")
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_format_targets(
        FormatTargetConfig(
            sources=args.sources,
            output_dir=args.output_dir,
            model_name=args.model_name,
            task_types=tuple(args.task_type or FORMAT_TASK_TYPES),
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            seed=args.seed,
            max_attempts_per_task=args.max_attempts_per_task,
            max_new_tokens=args.max_new_tokens,
            prompt_format=args.prompt_format,
            dtype=args.dtype,
        )
    )


if __name__ == "__main__":
    main()
