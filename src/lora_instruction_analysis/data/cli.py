"""Command line entry point for dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .builder import DatasetBuildConfig, build_dataset, write_dataset
from .sources import list_sources
from .tasks import list_tasks


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic transformation datasets.")
    parser.add_argument("--list-tasks", action="store_true", help="Print available transformation tasks and exit.")
    parser.add_argument("--list-sources", action="store_true", help="Print available public text sources and exit.")
    parser.add_argument("--task", choices=[task.task_id for task in list_tasks()])
    parser.add_argument("--source", choices=[source.source_id for source in list_sources()], default="wikitext")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--train-size", type=positive_int, default=800)
    parser.add_argument("--validation-size", type=positive_int, default=100)
    parser.add_argument("--test-size", type=positive_int, default=100)
    parser.add_argument("--max-source-rows", type=positive_int, default=5000)
    parser.add_argument("--min-words", type=positive_int, default=4)
    parser.add_argument("--max-words", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--condition", default="lora_training")
    parser.add_argument(
        "--include-instruction-in-prompt",
        action="store_true",
        help="Build text prompts with the natural-language instruction included.",
    )
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-hf-dataset", action="store_true")
    parser.add_argument("--allow-builtin-fallback", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        for task in list_tasks():
            print(f"{task.task_id}: {task.natural_language_instruction}")
        return
    if args.list_sources:
        for source in list_sources():
            print(f"{source.source_id}: {source.hf_path}/{source.hf_name or ''} split={source.split}")
        return
    if args.task is None:
        raise SystemExit("--task is required unless --list-tasks or --list-sources is used.")
    if args.output_dir is None:
        raise SystemExit("--output-dir is required unless --list-tasks or --list-sources is used.")
    config = DatasetBuildConfig(
        task_id=args.task,
        source_id=args.source,
        output_dir=args.output_dir,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_source_rows=args.max_source_rows,
        min_words=args.min_words,
        max_words=args.max_words,
        seed=args.seed,
        condition=args.condition,
        include_instruction_in_prompt=args.include_instruction_in_prompt,
        write_csv=not args.no_csv,
        write_hf_dataset=not args.no_hf_dataset,
        allow_builtin_fallback=args.allow_builtin_fallback,
        streaming=args.streaming,
    )
    splits = build_dataset(config)
    write_dataset(config, splits)

    split_summary = ", ".join(f"{name}={len(records)}" for name, records in splits.items())
    print(f"Wrote dataset to {config.output_dir} ({split_summary})")


if __name__ == "__main__":
    main()
