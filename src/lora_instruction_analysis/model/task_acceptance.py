"""Gate a transformation task with prompt_eval instruction/no-instruction runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset, write_dataset
from lora_instruction_analysis.data.tasks import (
    PROMPT_VARIANT_NAMES,
    ValidationSelector,
    get_task,
    instruction_prompt_variants,
    list_tasks,
    resolved_validator_name,
)
from lora_instruction_analysis.model.collect import _dataset_file, _read_jsonl
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS
from lora_instruction_analysis.model.prompt_eval import PromptEvalConfig, evaluate_prompt


@dataclass(frozen=True)
class TaskAcceptanceConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    instruction: str | None = None
    split: str = "test"
    max_samples: int | None = None
    seed: int = 13
    max_length: int = 512
    generation_extra_tokens: int = 128
    run_teacher_forced: bool = False
    min_instruction_semantic_accuracy: float = 0.8
    max_no_instruction_semantic_accuracy: float = 0.3
    dtype: str = "auto"
    device: str = "auto"
    prompt_format: str = "raw"
    append_eos: bool = True
    validator: ValidationSelector = None


def _generation_accuracy(summary: dict) -> float:
    return float(summary["autoregressive"]["mean_task_semantic_correct"])


def passes_gate(instruction_summary: dict, no_instruction_summary: dict, config: TaskAcceptanceConfig) -> bool:
    return (
        _generation_accuracy(instruction_summary) >= config.min_instruction_semantic_accuracy
        and _generation_accuracy(no_instruction_summary) <= config.max_no_instruction_semantic_accuracy
    )


def _default_instruction_variants(instruction: str) -> list[str]:
    return instruction_prompt_variants(instruction)


def validate_task(config: TaskAcceptanceConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_jsonl(_dataset_file(config.dataset_path, config.split))
    task_ids = {record["task_id"] for record in records}
    if len(task_ids) != 1:
        raise ValueError(f"Task acceptance requires exactly one task_id, found {sorted(task_ids)}")
    task_id = next(iter(task_ids))
    resolved_validator = resolved_validator_name(task_id, config.validator)
    base_instruction = config.instruction or get_task(task_id).natural_language_instruction
    common = {
        "model_name": config.model_name,
        "dataset_path": config.dataset_path,
        "split": config.split,
        "max_samples": config.max_samples,
        "seed": config.seed,
        "max_length": config.max_length,
        "generation_extra_tokens": config.generation_extra_tokens,
        "run_teacher_forced": config.run_teacher_forced,
        "run_autoregressive": True,
        "dtype": config.dtype,
        "device": config.device,
        "prompt_format": config.prompt_format,
        "append_eos": config.append_eos,
        "validator": resolved_validator,
    }
    instruction_variants = _default_instruction_variants(base_instruction)
    instruction_summaries = []
    for index, instruction in enumerate(instruction_variants, start=1):
        instruction_summaries.append(
            evaluate_prompt(
                PromptEvalConfig(
                    **common,
                    output_dir=config.output_dir / f"instruction_only_prompt_{index}",
                    instruction=instruction,
                    include_instruction=True,
                )
            )
        )
    no_instruction_summary = evaluate_prompt(
        PromptEvalConfig(
            **common,
            output_dir=config.output_dir / "no_instruction",
            include_instruction=False,
        )
    )
    scores = [_generation_accuracy(run) for run in instruction_summaries]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    accepted = passes_gate(instruction_summaries[best_index], no_instruction_summary, config)
    manifest_path = config.dataset_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    summary = {
        "accepted": accepted,
        "instruction_only": [run["autoregressive"] for run in instruction_summaries],
        "no_instruction": no_instruction_summary["autoregressive"],
        "selected_prompt": {
            "index": best_index + 1,
            "name": PROMPT_VARIANT_NAMES[best_index],
            "instruction": instruction_variants[best_index],
            "mean_task_semantic_correct": scores[best_index],
        },
        "teacher_forced_reference": {
            "instruction_only": [run["teacher_forced"] for run in instruction_summaries],
            "no_instruction": no_instruction_summary["teacher_forced"],
        } if config.run_teacher_forced else None,
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
            "instruction_variants": instruction_variants,
            "task_id": task_id,
            "validator": resolved_validator,
            "data_route": manifest.get("data_route"),
        },
    }
    (config.output_dir / "acceptance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether a task is suitable for mechanism comparison.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--task", choices=[task.task_id for task in list_tasks()], help="Build a quick test split from this registered transformation task.")
    parser.add_argument("--source", default="wikitext", help="Single declared data route used with --task.")
    parser.add_argument("--max-source-rows", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", help="Instruction prompt. Defaults to each row's instruction_text.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--generation-extra-tokens", type=int, default=128)
    parser.add_argument("--run-teacher-forced", action="store_true")
    parser.add_argument("--min-instruction-semantic-accuracy", type=float, default=0.8)
    parser.add_argument("--max-no-instruction-semantic-accuracy", type=float, default=0.3)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--validator", default=None, help="Override task validation: task_default, exact, single_token, integer, or yes_no.")
    return parser.parse_args()


def _dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path:
        return args.dataset_path
    if not args.task:
        raise SystemExit("Use --dataset-path or --task.")

    dataset_path = args.output_dir / "dataset"
    size = args.max_samples or 16
    config = DatasetBuildConfig(
        task_id=args.task,
        source_id=args.source,
        output_dir=dataset_path,
        train_size=0,
        validation_size=0,
        test_size=size,
        max_source_rows=args.max_source_rows,
        write_csv=False,
        write_hf_dataset=False,
        model_name=args.model_name,
        tokenizer_name=args.model_name,
        prompt_template=f"{args.prompt_format}_target_v1",
    )
    splits = build_dataset(config)
    write_dataset(config, splits)
    return dataset_path


def main() -> None:
    args = parse_args()
    dataset_path = _dataset_path(args)
    summary = validate_task(
        TaskAcceptanceConfig(
            model_name=args.model_name,
            dataset_path=dataset_path,
            output_dir=args.output_dir,
            instruction=args.instruction,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            max_length=args.max_length,
            generation_extra_tokens=args.generation_extra_tokens,
            run_teacher_forced=args.run_teacher_forced,
            min_instruction_semantic_accuracy=args.min_instruction_semantic_accuracy,
            max_no_instruction_semantic_accuracy=args.max_no_instruction_semantic_accuracy,
            dtype=args.dtype,
            device=args.device,
            prompt_format=args.prompt_format,
            append_eos=not args.no_append_eos,
            validator=args.validator,
        )
    )
    print(json.dumps({key: summary[key] for key in ("accepted", "instruction_only", "no_instruction")}, indent=2))
    if not summary["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
