"""Gate a transformation task with prompt_eval instruction/no-instruction runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset, write_dataset
from lora_instruction_analysis.data.tasks import list_tasks
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
    min_instruction_token_accuracy: float = 0.8
    max_no_instruction_token_accuracy: float = 0.3
    dtype: str = "auto"
    device: str = "auto"
    prompt_format: str = "raw"
    append_eos: bool = True


def _token_accuracy(summary: dict) -> float:
    return float(summary["teacher_forced"]["mean_token_accuracy"])


def passes_gate(instruction_summary: dict, no_instruction_summary: dict, config: TaskAcceptanceConfig) -> bool:
    return (
        _token_accuracy(instruction_summary) >= config.min_instruction_token_accuracy
        and _token_accuracy(no_instruction_summary) <= config.max_no_instruction_token_accuracy
    )


def validate_task(config: TaskAcceptanceConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "model_name": config.model_name,
        "dataset_path": config.dataset_path,
        "split": config.split,
        "max_samples": config.max_samples,
        "seed": config.seed,
        "max_length": config.max_length,
        "run_autoregressive": False,
        "dtype": config.dtype,
        "device": config.device,
        "prompt_format": config.prompt_format,
        "append_eos": config.append_eos,
    }
    instruction_summary = evaluate_prompt(
        PromptEvalConfig(
            **common,
            output_dir=config.output_dir / "instruction_only",
            instruction=config.instruction,
            include_instruction=True,
        )
    )
    no_instruction_summary = evaluate_prompt(
        PromptEvalConfig(
            **common,
            output_dir=config.output_dir / "no_instruction",
            include_instruction=False,
        )
    )
    summary = {
        "accepted": passes_gate(instruction_summary, no_instruction_summary, config),
        "instruction_only": instruction_summary["teacher_forced"],
        "no_instruction": no_instruction_summary["teacher_forced"],
        "config": {
            **asdict(config),
            "dataset_path": str(config.dataset_path),
            "output_dir": str(config.output_dir),
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--instruction", help="Instruction prompt. Defaults to each row's instruction_text.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-instruction-token-accuracy", type=float, default=0.8)
    parser.add_argument("--max-no-instruction-token-accuracy", type=float, default=0.3)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    return parser.parse_args()


def _dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path:
        return args.dataset_path
    if not args.task:
        raise SystemExit("Use --dataset-path or --task.")

    dataset_path = args.output_dir / "dataset"
    size = min(args.max_samples or 5, 5)
    config = DatasetBuildConfig(
        task_id=args.task,
        output_dir=dataset_path,
        train_size=0,
        validation_size=0,
        test_size=size,
        max_source_rows=0,
        allow_builtin_fallback=True,
        write_hf_dataset=False,
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
            min_instruction_token_accuracy=args.min_instruction_token_accuracy,
            max_no_instruction_token_accuracy=args.max_no_instruction_token_accuracy,
            dtype=args.dtype,
            device=args.device,
            prompt_format=args.prompt_format,
            append_eos=not args.no_append_eos,
        )
    )
    print(json.dumps({key: summary[key] for key in ("accepted", "instruction_only", "no_instruction")}, indent=2))
    if not summary["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
