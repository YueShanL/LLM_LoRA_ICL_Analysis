"""End-to-end dataset generation and LoRA training runner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path

from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset, write_dataset
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS
from lora_instruction_analysis.model.train_lora import TrainConfig, train_lora


DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"


@dataclass(frozen=True)
class ExperimentConfig:
    run_id: str
    output_root: Path
    model_name: str
    task_id: str
    source_id: str
    seed: int
    rank: int
    train_size: int
    validation_size: int
    test_size: int
    max_source_rows: int
    epochs: float
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_length: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]
    fp16: bool
    bf16: bool
    qlora: bool
    device_map: str | None
    include_instruction_in_prompt: bool
    streaming: bool
    prompt_format: str
    append_eos: bool

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_id

    @property
    def dataset_dir(self) -> Path:
        return self.run_dir / "dataset"

    @property
    def adapter_dir(self) -> Path:
        return self.run_dir / "adapters" / f"r{self.rank}"


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one module name")
    return items


def _default_run_id(task_id: str, rank: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{task_id}_llama32_3b_r{rank}_{stamp}"


def _jsonable(config: ExperimentConfig) -> dict:
    data = asdict(config)
    for key in ("output_root",):
        data[key] = str(data[key])
    data["run_dir"] = str(config.run_dir)
    data["dataset_dir"] = str(config.dataset_dir)
    data["adapter_dir"] = str(config.adapter_dir)
    return data


def run_experiment(config: ExperimentConfig) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=False)
    (config.run_dir / "config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    dataset_config = DatasetBuildConfig(
        task_id=config.task_id,
        source_id=config.source_id,
        output_dir=config.dataset_dir,
        train_size=config.train_size,
        validation_size=config.validation_size,
        test_size=config.test_size,
        max_source_rows=config.max_source_rows,
        seed=config.seed,
        condition="lora_training",
        include_instruction_in_prompt=config.include_instruction_in_prompt,
        streaming=config.streaming,
        model_name=config.model_name,
        tokenizer_name=config.model_name,
        prompt_template=f"{config.prompt_format}_target_v1",
    )
    write_dataset(dataset_config, build_dataset(dataset_config))

    train_lora(
        TrainConfig(
            model_name=config.model_name,
            dataset_path=config.dataset_dir,
            output_dir=config.adapter_dir,
            max_length=config.max_length,
            seed=config.seed,
            rank=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            learning_rate=config.learning_rate,
            epochs=config.epochs,
            train_batch_size=config.train_batch_size,
            eval_batch_size=config.eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            fp16=config.fp16,
            bf16=config.bf16,
            qlora=config.qlora,
            device_map=config.device_map,
            prompt_format=config.prompt_format,
            append_eos=config.append_eos,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a task dataset, train a LoRA adapter, and save the run.")
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("experiments"))
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--task", dest="task_id", default="last_word")
    parser.add_argument("--source", dest="source_id", default="wikitext")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--max-source-rows", type=int, default=5000)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=_csv, default=("auto",), help="Comma-separated names or auto.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--include-instruction-in-prompt", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        run_id=args.run_id or _default_run_id(args.task_id, args.rank),
        output_root=args.output_root,
        model_name=args.model_name,
        task_id=args.task_id,
        source_id=args.source_id,
        seed=args.seed,
        rank=args.rank,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        max_source_rows=args.max_source_rows,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        fp16=args.fp16,
        bf16=args.bf16,
        qlora=args.qlora,
        device_map=args.device_map or None,
        include_instruction_in_prompt=args.include_instruction_in_prompt,
        streaming=args.streaming,
        prompt_format=args.prompt_format,
        append_eos=not args.no_append_eos,
    )
    run_experiment(config)
    print(f"Wrote run to {config.run_dir}")
    print(f"Wrote adapter to {config.adapter_dir}")


if __name__ == "__main__":
    main()
