"""Evaluate an instruction prompt against a generated dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import os
from pathlib import Path
import random
from statistics import mean

from lora_instruction_analysis.data.icl import attach_dataset_icl_examples
from lora_instruction_analysis.data.tasks import ValidationSelector, evaluate_output, get_task, resolved_validator_name
from lora_instruction_analysis.model.collect import _accuracy, _dataset_file, _read_jsonl, _torch_dtype, _write_jsonl
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS, encode_record, ensure_chat_template


os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass(frozen=True)
class PromptEvalConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    instruction: str | None = None
    split: str = "test"
    max_samples: int | None = None
    seed: int = 13
    max_length: int = 512
    generation_extra_tokens: int = 128
    run_teacher_forced: bool = True
    run_autoregressive: bool = True
    include_instruction: bool = True
    dtype: str = "auto"
    device: str = "auto"
    prompt_format: str = "raw"
    append_eos: bool = True
    validator: ValidationSelector = None
    adapter_path: Path | None = None
    icl_examples: int = 0
    icl_split: str = "train"


def _select_records(config: PromptEvalConfig) -> list[dict]:
    records = _read_jsonl(_dataset_file(config.dataset_path, config.split))
    if config.max_samples is None:
        return attach_dataset_icl_examples(
            records, config.dataset_path, example_count=config.icl_examples, split=config.icl_split
        )
    rng = random.Random(config.seed)
    chosen = records[:]
    rng.shuffle(chosen)
    return attach_dataset_icl_examples(
        chosen[: config.max_samples], config.dataset_path, example_count=config.icl_examples, split=config.icl_split
    )


def _load_model(config: PromptEvalConfig):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Prompt evaluation requires torch and transformers. Install .[train].") from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ensure_chat_template(tokenizer, config.model_name, config.prompt_format)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    kwargs = {"torch_dtype": _torch_dtype(torch, config.dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    if config.adapter_path is not None:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("LoRA evaluation requires peft. Install .[train].") from exc
        model = PeftModel.from_pretrained(model, str(config.adapter_path))
    if device != "auto":
        model.to(device)
    model.eval()
    return torch, tokenizer, model


def _encode(
    tokenizer,
    record: dict,
    instruction: str | None,
    max_length: int,
    include_instruction: bool = True,
    prompt_format: str = "raw",
    append_eos: bool = True,
    require_target: bool = True,
) -> dict:
    encoded = encode_record(
        tokenizer,
        record,
        include_instruction=include_instruction,
        prompt_format=prompt_format,
        append_eos=append_eos,
        instruction=instruction,
        max_length=max_length,
    )
    if require_target and not encoded["target_ids"]:
        raise ValueError(f"max_length={max_length} truncates away all target tokens for {record['sample_id']}.")
    return encoded


def _token_accuracy(pred_ids: list[int], target_ids: list[int]) -> dict:
    denom = max(len(pred_ids), len(target_ids))
    correct = sum(int(pred == target) for pred, target in zip(pred_ids, target_ids))
    return {
        "token_accuracy": correct / denom if denom else 0.0,
        "sequence_accuracy": float(bool(target_ids) and pred_ids == target_ids),
    }


def _max_new_tokens(record: dict, target_ids: list[int], generation_extra_tokens: int) -> int:
    task_limit = get_task(record["task_id"]).max_generate_tokens
    return task_limit if task_limit is not None else len(target_ids) + generation_extra_tokens


def _run_teacher_forced(
    torch,
    tokenizer,
    model,
    record: dict,
    instruction: str | None,
    max_length: int,
    include_instruction: bool,
    prompt_format: str,
    append_eos: bool,
    validator: ValidationSelector,
) -> dict:
    encoded = _encode(tokenizer, record, instruction, max_length, include_instruction, prompt_format, append_eos)
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    prediction_positions = torch.tensor(encoded["prediction_positions"], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels, use_cache=False)

    logits = outputs.logits[0, prediction_positions, :]
    pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    metrics = _accuracy(pred_ids, encoded["target_ids"])
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "loss": float(outputs.loss.detach().cpu()),
        "prompt_tokens": encoded["prompt_length"],
        "target_tokens": len(encoded["target_ids"]),
        "pred_text": pred_text,
        "target_text": record["target_text"],
        **metrics,
        **evaluate_output(record["task_id"], record["input_text"], pred_text, record["target_text"], validator),
    }


def _run_autoregressive(
    torch,
    tokenizer,
    model,
    record: dict,
    instruction: str | None,
    max_length: int,
    generation_extra_tokens: int,
    include_instruction: bool,
    prompt_format: str,
    append_eos: bool,
    validator: ValidationSelector,
) -> dict:
    encoded = _encode(
        tokenizer,
        record,
        instruction,
        max_length,
        include_instruction,
        prompt_format,
        append_eos,
        require_target=False,
    )
    target_ids = encoded["target_ids"]
    prompt_ids = encoded["input_ids"][: encoded["prompt_length"]]
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        max_new_tokens = _max_new_tokens(record, target_ids, generation_extra_tokens)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    pred_ids = generated[0, input_ids.shape[1]:].detach().cpu().tolist()
    metrics = _token_accuracy(pred_ids, target_ids)
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "prompt_tokens": len(prompt_ids),
        "target_tokens": len(target_ids),
        "max_new_tokens": max_new_tokens,
        "generated_tokens": len(pred_ids),
        "pred_text": pred_text,
        "target_text": record.get("target_text"),
        **metrics,
        **evaluate_output(record["task_id"], record["input_text"], pred_text, record.get("target_text"), validator),
    }


def _summary(rows: list[dict], *, include_loss: bool) -> dict:
    summary = {
        "samples": len(rows),
        "mean_token_accuracy": mean(row["token_accuracy"] for row in rows) if rows else 0.0,
        "mean_sequence_accuracy": mean(row["sequence_accuracy"] for row in rows) if rows else 0.0,
        "mean_task_semantic_correct": mean(row["task_semantic_correct"] for row in rows) if rows else 0.0,
    }
    if include_loss:
        summary["mean_loss"] = mean(row["loss"] for row in rows) if rows else 0.0
    return summary


def _write_report(path: Path, summary: dict) -> None:
    teacher = summary["teacher_forced"]
    autoreg = summary.get("autoregressive")
    lines = [
        "# Prompt Evaluation Report",
    ]
    if teacher is not None:
        lines.extend(
            [
                "",
                "## Teacher-Forced",
                "",
                f"- samples: {teacher['samples']}",
                f"- mean_loss: {teacher['mean_loss']:.6f}",
                f"- mean_token_accuracy: {teacher['mean_token_accuracy']:.6f}",
                f"- mean_sequence_accuracy: {teacher['mean_sequence_accuracy']:.6f}",
                f"- mean_task_semantic_correct: {teacher['mean_task_semantic_correct']:.6f}",
            ]
        )
    if autoreg is not None:
        lines.extend(
            [
                "",
                "## Autoregressive",
                "",
                f"- samples: {autoreg['samples']}",
                f"- mean_token_accuracy: {autoreg['mean_token_accuracy']:.6f}",
                f"- mean_sequence_accuracy: {autoreg['mean_sequence_accuracy']:.6f}",
                f"- mean_task_semantic_correct: {autoreg['mean_task_semantic_correct']:.6f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_prompt(config: PromptEvalConfig) -> dict:
    records = _select_records(config)
    task_ids = {record["task_id"] for record in records}
    if len(task_ids) != 1:
        raise ValueError(f"Prompt evaluation requires exactly one task_id, found {sorted(task_ids)}")
    resolved_validator = resolved_validator_name(next(iter(task_ids)), config.validator)
    torch, tokenizer, model = _load_model(config)
    try:
        teacher_rows = [
            _run_teacher_forced(
                torch,
                tokenizer,
                model,
                record,
                config.instruction,
                config.max_length,
                config.include_instruction,
                config.prompt_format,
                config.append_eos,
                resolved_validator,
            )
            for record in records
        ] if config.run_teacher_forced else []
        autoreg_rows = [
            _run_autoregressive(
                torch,
                tokenizer,
                model,
                record,
                config.instruction,
                config.max_length,
                config.generation_extra_tokens,
                config.include_instruction,
                config.prompt_format,
                config.append_eos,
                resolved_validator,
            )
            for record in records
        ] if config.run_autoregressive else []

        config.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "teacher_forced": _summary(teacher_rows, include_loss=True) if config.run_teacher_forced else None,
            "autoregressive": _summary(autoreg_rows, include_loss=False) if config.run_autoregressive else None,
            "config": {
                **asdict(config),
                "dataset_path": str(config.dataset_path),
                "output_dir": str(config.output_dir),
                "adapter_path": str(config.adapter_path) if config.adapter_path else None,
                "instruction_source": "none"
                if not config.include_instruction
                else "icl_examples"
                if config.icl_examples
                else "cli"
                if config.instruction is not None
                else "dataset_instruction_text",
                "prompt_format": config.prompt_format,
                "append_eos": config.append_eos,
                "validator": resolved_validator,
            },
        }
        if config.run_teacher_forced:
            _write_jsonl(config.output_dir / "metrics.jsonl", teacher_rows)
        if config.run_autoregressive:
            _write_jsonl(config.output_dir / "autoregressive_metrics.jsonl", autoreg_rows)
        (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_report(config.output_dir / "report.md", summary)
        return summary
    finally:
        del model
        gc.collect()
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an instruction prompt on a DatasetModule dataset.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, help="Optional LoRA adapter to evaluate with the same task validator.")
    parser.add_argument("--instruction", help="Instruction prompt. Defaults to each row's instruction_text.")
    parser.add_argument("--instruction-file", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--generation-extra-tokens", type=int, default=128)
    parser.add_argument("--skip-teacher-forced", action="store_true")
    parser.add_argument("--skip-autoregressive", action="store_true")
    parser.add_argument("--no-instruction", action="store_true")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--validator", default=None, help="Override task validation by registered validator name; default is task-specific.")
    parser.add_argument("--icl-examples", type=int, default=0, help="Use N train input-output pairs in place of instruction text.")
    parser.add_argument("--icl-split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.instruction and args.instruction_file:
        raise SystemExit("Use only one of --instruction or --instruction-file.")
    instruction = args.instruction
    if args.instruction_file:
        instruction = args.instruction_file.read_text(encoding="utf-8").strip()
    summary = evaluate_prompt(
        PromptEvalConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            instruction=instruction,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            max_length=args.max_length,
            generation_extra_tokens=args.generation_extra_tokens,
            run_teacher_forced=not args.skip_teacher_forced,
            run_autoregressive=not args.skip_autoregressive,
            include_instruction=not args.no_instruction,
            dtype=args.dtype,
            device=args.device,
            prompt_format=args.prompt_format,
            append_eos=not args.no_append_eos,
            validator=args.validator,
            adapter_path=args.adapter_path,
            icl_examples=args.icl_examples,
            icl_split=args.icl_split,
        )
    )
    print(json.dumps({key: summary[key] for key in ("teacher_forced", "autoregressive")}, indent=2))


if __name__ == "__main__":
    main()
