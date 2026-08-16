"""Train a PEFT LoRA adapter on one transformation task."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import fnmatch
import inspect
import json
import math
import numbers
from pathlib import Path
import shutil

from lora_instruction_analysis.model.formatting import PROMPT_FORMATS, encode_record, ensure_chat_template


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    train_split: str = "train"
    eval_split: str = "validation"
    max_length: int = 512
    seed: int = 13
    rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("auto",)
    learning_rate: float = 2e-4
    epochs: float = 3.0
    train_batch_size: int = 2
    eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    save_steps: int = 0
    fp16: bool = False
    bf16: bool = False
    qlora: bool = False
    device_map: str | None = "auto"
    prompt_format: str = "raw"
    append_eos: bool = True
    monitor_nonfinite: bool = True
    max_grad_norm: float = 1.0
    skip_nonfinite_loss: bool = True
    max_nonfinite_loss_skips: int = 8


class _NonFiniteTrainingError(FloatingPointError):
    """Raised when LoRA training produces NaN or Inf values."""


def _parameter_nonfinite_issue(model, *, include_gradients: bool = True) -> str | None:
    """Return the first non-finite trainable parameter or gradient, if any."""
    import torch

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        if include_gradients and parameter.grad is not None and not torch.isfinite(parameter.grad.detach()).all():
            return f"non-finite gradient in trainable parameter {name}"
        if not torch.isfinite(parameter.detach()).all():
            return f"non-finite value in trainable parameter {name}"
    return None


def _log_nonfinite_issue(logs: dict | None) -> str | None:
    if not logs:
        return None
    invalid = []
    for name, value in logs.items():
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            try:
                if not math.isfinite(float(value)):
                    invalid.append(name)
            except (TypeError, ValueError, OverflowError):
                invalid.append(name)
    return f"non-finite training log values: {', '.join(invalid)}" if invalid else None


class _NonFiniteTrainingCallbackMixin:
    """Checks the small trainable LoRA parameter set at Trainer lifecycle hooks."""

    @staticmethod
    def _raise_if_invalid(model, step: int | None, *, include_gradients: bool = True) -> None:
        if model is None:
            return
        issue = _parameter_nonfinite_issue(model, include_gradients=include_gradients)
        if issue:
            raise _NonFiniteTrainingError(f"step={step}: {issue}")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self._raise_if_invalid(kwargs.get("model"), getattr(state, "global_step", None))
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._raise_if_invalid(kwargs.get("model"), getattr(state, "global_step", None))
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        issue = _log_nonfinite_issue(logs)
        if issue:
            raise _NonFiniteTrainingError(f"step={getattr(state, 'global_step', None)}: {issue}")
        return control


def _make_nonfinite_callback():
    from transformers import TrainerCallback

    class NonFiniteTrainingCallback(_NonFiniteTrainingCallbackMixin, TrainerCallback):
        pass

    return NonFiniteTrainingCallback()


def _make_finite_trainer(base_trainer, *, skip_nonfinite_loss: bool, max_nonfinite_loss_skips: int):
    """Check loss immediately and optionally turn bad microbatches into zero-gradient steps."""

    class FiniteTrainer(base_trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._nonfinite_loss_skips = 0

        @staticmethod
        def _zero_backward_loss(model):
            zero = None
            for parameter in model.parameters():
                if parameter.requires_grad and parameter.is_floating_point():
                    term = parameter.float().sum() * 0.0
                    zero = term if zero is None else zero + term
            if zero is None:
                raise RuntimeError("Cannot skip a non-finite loss: no floating-point trainable parameters found.")
            return zero

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            try:
                result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            except TypeError as exc:
                if "num_items_in_batch" not in kwargs or "num_items_in_batch" not in str(exc):
                    raise
                result = super().compute_loss(model, inputs, return_outputs=return_outputs)
            loss = result[0] if isinstance(result, tuple) else result
            import torch

            if not torch.isfinite(loss.detach()).all():
                step = getattr(self.state, "global_step", None)
                if not model.training or not skip_nonfinite_loss:
                    raise _NonFiniteTrainingError(f"step={step}: non-finite loss returned by model.compute_loss")
                self._nonfinite_loss_skips += 1
                if self._nonfinite_loss_skips > max_nonfinite_loss_skips:
                    raise _NonFiniteTrainingError(
                        f"step={step}: exceeded max_nonfinite_loss_skips={max_nonfinite_loss_skips}"
                    )
                print(
                    f"[warn] skipped non-finite loss before backward: step={step}, "
                    f"count={self._nonfinite_loss_skips}/{max_nonfinite_loss_skips}",
                    flush=True,
                )
                zero_loss = self._zero_backward_loss(model)
                return (zero_loss, result[1]) if return_outputs and isinstance(result, tuple) else zero_loss
            return result

    return FiniteTrainer


def _upcast_trainable_parameters(model) -> None:
    """Keep the small LoRA parameter set in fp32 when the base model uses bf16/fp16."""
    import torch

    for parameter in model.parameters():
        if parameter.requires_grad and parameter.is_floating_point() and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


def _write_training_failure(config: TrainConfig, error: BaseException) -> Path:
    """Remove invalid adapter artifacts and persist a failure marker for resume logic."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    failure_marker = config.output_dir / "training_failed.json"
    if failure_marker.exists():
        failure_marker.unlink()
    removed = []
    for name in ("adapter_model.safetensors", "adapter_model.bin", "adapter_config.json"):
        path = config.output_dir / name
        if path.exists():
            path.unlink()
            removed.append(name)
    failure_path = config.output_dir / "training_failed.json"
    failure_path.write_text(
        json.dumps(
            {
                "status": "failed_nonfinite",
                "error_type": type(error).__name__,
                "error": str(error),
                "model_name": config.model_name,
                "output_dir": str(config.output_dir),
                "removed_adapter_artifacts": removed,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return failure_path


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_path(dataset_path: Path, split: str) -> Path:
    return dataset_path / f"{split}.jsonl" if dataset_path.is_dir() else dataset_path


def _load_split(dataset_path: Path, split: str):
    from datasets import Dataset

    path = _split_path(dataset_path, split)
    if not path.exists():
        return None
    rows = _read_jsonl(path)
    return Dataset.from_list(rows) if rows else None


def _torch_dtype(torch, fp16: bool, bf16: bool):
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return "auto"


def _quant_compute_dtype(torch, bf16: bool):
    return torch.bfloat16 if bf16 else torch.float16


def _infer_target_modules(model) -> tuple[str, ...]:
    model_type = getattr(model.config, "model_type", "")
    if model_type in {"gpt2", "gpt_bigcode"}:
        return ("c_attn",)
    if model_type in {"bloom", "gpt_neox", "falcon"}:
        return ("query_key_value",)
    return ("q_proj", "k_proj", "v_proj")


def _resolve_target_modules(model, target_modules: tuple[str, ...]) -> tuple[str, ...]:
    if target_modules == ("auto",):
        return _infer_target_modules(model)
    patterns = [item for item in target_modules if "*" in item or "?" in item]
    if not patterns:
        return target_modules
    resolved = [name for name, _module in model.named_modules() if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
    resolved.extend(item for item in target_modules if item not in patterns)
    if not resolved:
        raise ValueError(f"No LoRA target modules matched patterns: {', '.join(patterns)}")
    return tuple(dict.fromkeys(resolved))


def _load_model_and_tokenizer(config: TrainConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("LoRA training requires torch and transformers. Install .[train].") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ensure_chat_template(tokenizer, config.model_name, config.prompt_format)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": _torch_dtype(torch, config.fp16, config.bf16)}
    if config.device_map:
        kwargs["device_map"] = config.device_map
    if config.qlora:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("QLoRA requires a transformers build with BitsAndBytesConfig.") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_quant_compute_dtype(torch, config.bf16),
        )

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    if config.qlora:
        try:
            from peft import prepare_model_for_kbit_training
        except ImportError as exc:
            raise RuntimeError("QLoRA requires peft. Install .[train].") from exc
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    return model, tokenizer


def _encode(tokenizer, record: dict, max_length: int, prompt_format: str, append_eos: bool) -> dict:
    encoded = encode_record(
        tokenizer,
        record,
        include_instruction=False,
        prompt_format=prompt_format,
        append_eos=append_eos,
        max_length=max_length,
    )
    if not encoded["target_ids"]:
        raise ValueError(f"Training example {record.get('sample_id', '<unknown>')} has no target tokens after encoding.")
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": [1] * len(encoded["input_ids"]),
        "labels": encoded["labels"],
    }


def _preprocess(dataset, tokenizer, max_length: int, prompt_format: str, append_eos: bool):
    return dataset.map(
        lambda row: _encode(tokenizer, row, max_length, prompt_format, append_eos),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )


def train_lora(config: TrainConfig) -> None:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    except ImportError as exc:
        raise RuntimeError("LoRA training requires transformers and peft. Install .[train].") from exc

    set_seed(config.seed)
    model, tokenizer = _load_model_and_tokenizer(config)
    target_modules = _resolve_target_modules(model, config.target_modules)
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=list(target_modules),
        ),
    )
    _upcast_trainable_parameters(model)
    model.print_trainable_parameters()

    train_dataset = _load_split(config.dataset_path, config.train_split)
    eval_dataset = (
        _load_split(config.dataset_path, config.eval_split)
        if config.dataset_path.is_dir() or config.eval_split == config.train_split
        else None
    )
    if train_dataset is None:
        raise FileNotFoundError(_split_path(config.dataset_path, config.train_split))

    train_dataset = _preprocess(train_dataset, tokenizer, config.max_length, config.prompt_format, config.append_eos)
    eval_dataset = _preprocess(eval_dataset, tokenizer, config.max_length, config.prompt_format, config.append_eos) if eval_dataset else None

    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "train_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_path": str(config.dataset_path),
                "output_dir": str(config.output_dir),
                "target_modules": list(target_modules),
                "prompt_format": config.prompt_format,
                "append_eos": config.append_eos,
                "prompt_template": "input_text_only_no_instruction" if config.prompt_format == "raw" else "tokenizer_chat_template",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    manifest = config.dataset_path / "manifest.json"
    if manifest.exists():
        shutil.copy2(manifest, config.output_dir / "dataset_manifest.json")

    training_args = {
        "output_dir": str(config.output_dir),
        "seed": config.seed,
        "learning_rate": config.learning_rate,
        "num_train_epochs": config.epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_ratio": config.warmup_ratio,
        "weight_decay": config.weight_decay,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps or 500,
        "save_strategy": "steps" if config.save_steps else "no",
        "eval_steps": config.logging_steps if eval_dataset is not None else None,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "max_grad_norm": config.max_grad_norm,
        "report_to": [],
        "remove_unused_columns": False,
    }
    strategy_name = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters
        else "evaluation_strategy"
    )
    training_args[strategy_name] = "steps" if eval_dataset is not None else "no"
    args = TrainingArguments(**training_args)
    trainer_args = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
    }
    trainer_tokenizer_name = (
        "processing_class" if "processing_class" in inspect.signature(Trainer.__init__).parameters else "tokenizer"
    )
    trainer_args[trainer_tokenizer_name] = tokenizer
    if config.monitor_nonfinite:
        trainer_args["callbacks"] = [_make_nonfinite_callback()]
    trainer_class = (
        _make_finite_trainer(
            Trainer,
            skip_nonfinite_loss=config.skip_nonfinite_loss,
            max_nonfinite_loss_skips=config.max_nonfinite_loss_skips,
        )
        if config.monitor_nonfinite
        else Trainer
    )
    trainer = trainer_class(**trainer_args)
    try:
        train_output = trainer.train()
        issue = _log_nonfinite_issue(getattr(train_output, "metrics", None))
        if issue:
            raise _NonFiniteTrainingError(f"post-training result: {issue}")
        if eval_dataset is not None:
            eval_metrics = trainer.evaluate()
            issue = _log_nonfinite_issue(eval_metrics)
            if issue:
                raise _NonFiniteTrainingError(f"post-training evaluation: {issue}")
        if config.monitor_nonfinite:
            issue = _parameter_nonfinite_issue(model, include_gradients=True)
            if issue:
                raise _NonFiniteTrainingError(f"post-training validation: {issue}")
    except _NonFiniteTrainingError as exc:
        failure_path = _write_training_failure(config, exc)
        raise RuntimeError(
            f"LoRA training aborted because of non-finite values: {exc}. "
            f"Invalid adapter artifacts were removed; see {failure_path}."
        ) from exc
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))


def _csv(value: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one module name")
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a PEFT LoRA adapter without instruction text.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", type=_csv, default=("auto",), help="Comma-separated names or auto.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-monitor-nonfinite", action="store_true")
    parser.add_argument("--no-skip-nonfinite-loss", action="store_true")
    parser.add_argument("--max-nonfinite-loss-skips", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_lora(
        TrainConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            train_split=args.train_split,
            eval_split=args.eval_split,
            max_length=args.max_length,
            seed=args.seed,
            rank=args.rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            fp16=args.fp16,
            bf16=args.bf16,
            qlora=args.qlora,
            device_map=args.device_map or None,
            prompt_format=args.prompt_format,
            append_eos=not args.no_append_eos,
            monitor_nonfinite=not args.no_monitor_nonfinite,
            max_grad_norm=args.max_grad_norm,
            skip_nonfinite_loss=not args.no_skip_nonfinite_loss,
            max_nonfinite_loss_skips=args.max_nonfinite_loss_skips,
        )
    )
    print(f"Wrote LoRA adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
