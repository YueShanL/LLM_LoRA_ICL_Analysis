"""Collect teacher-forced model states for base, instruction, and LoRA runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from lora_instruction_analysis.model.formatting import PROMPT_FORMATS, encode_record, ensure_chat_template


CONDITIONS = ("base", "instruction_only", "lora_only")


@dataclass(frozen=True)
class CollectConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    adapter_path: Path | None = None
    split: str = "test"
    max_samples: int | None = None
    seed: int = 13
    conditions: tuple[str, ...] = CONDITIONS
    dtype: str = "auto"
    device: str = "auto"
    collect_attention_outputs: bool = False
    prompt_format: str = "raw"
    append_eos: bool = True


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_file(dataset_path: Path, split: str) -> Path:
    return dataset_path / f"{split}.jsonl" if dataset_path.is_dir() else dataset_path


def _select_records(config: CollectConfig) -> list[dict]:
    records = _read_jsonl(_dataset_file(config.dataset_path, config.split))
    if config.max_samples is None:
        return records
    rng = random.Random(config.seed)
    chosen = records[:]
    rng.shuffle(chosen)
    return chosen[: config.max_samples]


def _torch_dtype(torch, dtype: str):
    if dtype == "auto":
        return "auto"
    try:
        return getattr(torch, dtype)
    except AttributeError as exc:
        raise ValueError(f"Unknown torch dtype {dtype!r}, e.g. float32, float16, bfloat16.") from exc


def _load_model(config: CollectConfig, *, use_lora: bool):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Model collection requires torch and transformers. Install the train extras in venv first."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    ensure_chat_template(tokenizer, config.model_name, config.prompt_format)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {
        "torch_dtype": _torch_dtype(torch, config.dtype),
        "output_hidden_states": True,
        "output_attentions": True,
        "attn_implementation": "eager",
    }
    if config.device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)

    if use_lora:
        if config.adapter_path is None:
            raise ValueError("--adapter-path is required for lora_only collection.")
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("LoRA collection requires peft. Install the train extras in venv first.") from exc
        model = PeftModel.from_pretrained(model, str(config.adapter_path))

    if config.device != "auto":
        model.to(config.device)
    model.eval()
    return torch, tokenizer, model


def _encode(
    tokenizer,
    record: dict,
    *,
    include_instruction: bool,
    prompt_format: str = "raw",
    append_eos: bool = True,
) -> dict:
    encoded = encode_record(
        tokenizer,
        record,
        include_instruction=include_instruction,
        prompt_format=prompt_format,
        append_eos=append_eos,
    )
    return {
        "prompt": encoded["prompt"],
        "target": encoded["target"],
        "input_ids": encoded["input_ids"],
        "labels": encoded["labels"],
        "prompt_length": encoded["prompt_length"],
        "target_positions": encoded["target_positions"],
        "prediction_positions": encoded["prediction_positions"],
        "target_alignment": [
            {
                "alignment_key": f"target:{index}:{token_id}",
                "token_index": index,
                "token_id": token_id,
                "target_position": encoded["target_positions"][index],
                "prediction_position": encoded["prediction_positions"][index],
            }
            for index, token_id in enumerate(encoded["target_ids"])
        ],
        "source_alignment": encoded["source_alignment"],
    }


def _accuracy(pred_ids: list[int], target_ids: list[int]) -> dict:
    pairs = list(zip(pred_ids, target_ids))
    token_correct = sum(int(pred == target) for pred, target in pairs)
    return {
        "token_accuracy": token_correct / len(pairs) if pairs else 0.0,
        "sequence_accuracy": float(bool(pairs) and token_correct == len(pairs)),
    }


def _attention_output_hooks(model, target_positions):
    captures = []
    modules = [
        (name, module)
        for name, module in model.named_modules()
        if name.endswith(".self_attn") and hasattr(module, "o_proj")
    ]
    if not modules:
        raise RuntimeError("Could not find self_attn modules with o_proj for attention output collection.")

    def hook_for(name, module):
        module_config = getattr(module, "config", None)
        head_count = getattr(module, "num_heads", None) or getattr(module_config, "num_attention_heads", None)
        head_dim = getattr(module, "head_dim", None)
        if head_count is None or head_dim is None:
            raise RuntimeError(f"Cannot infer real head layout for {name}; refusing attention output fallback.")

        def save_input(_module, inputs):
            value = inputs[0]
            if value.shape[-1] != int(head_count) * int(head_dim):
                raise RuntimeError(
                    f"Attention output width mismatch in {name}: got {value.shape[-1]}, "
                    f"expected {int(head_count) * int(head_dim)}."
                )
            per_head = value[0, target_positions, :].view(len(target_positions), int(head_count), int(head_dim))
            captures.append(per_head.permute(1, 0, 2).detach().cpu())

        return save_input

    handles = [module.o_proj.register_forward_pre_hook(hook_for(name, module)) for name, module in modules]
    return captures, handles, len(modules)


def _stack_attention_outputs(torch, captures: list, expected_layers: int) -> object:
    if len(captures) != expected_layers:
        raise RuntimeError(f"Captured {len(captures)} attention output layers, expected {expected_layers}.")
    return torch.stack(captures).contiguous()


def _run_one(
    torch,
    tokenizer,
    model,
    record: dict,
    condition: str,
    tensor_path: Path,
    *,
    collect_attention_outputs: bool = False,
    prompt_format: str = "raw",
    append_eos: bool = True,
) -> dict:
    encoded = _encode(
        tokenizer,
        record,
        include_instruction=condition == "instruction_only",
        prompt_format=prompt_format,
        append_eos=append_eos,
    )
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    target_positions = torch.tensor(encoded["target_positions"], dtype=torch.long, device=device)
    prediction_positions = torch.tensor(encoded["prediction_positions"], dtype=torch.long, device=device)
    attention_output_captures, handles, expected_attention_output_layers = (
        _attention_output_hooks(model, target_positions) if collect_attention_outputs else ([], [], 0)
    )

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=labels,
                use_cache=False,
                output_hidden_states=True,
                output_attentions=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    target_logits = outputs.logits[0, prediction_positions, :].detach().cpu()
    pred_ids = target_logits.argmax(dim=-1).tolist()
    target_ids = [token_id for token_id in encoded["labels"] if token_id != -100]

    hidden_states = torch.stack(
        [state[0, target_positions, :].detach().cpu() for state in outputs.hidden_states]
    )
    attentions = torch.stack(
        [attn[0, :, target_positions, :].detach().cpu() for attn in outputs.attentions]
    )
    tensor = {
            "sample_id": record["sample_id"],
            "task_id": record["task_id"],
            "condition": condition,
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "labels": torch.tensor(encoded["labels"], dtype=torch.long),
            "target_positions": torch.tensor(encoded["target_positions"], dtype=torch.long),
            "prediction_positions": torch.tensor(encoded["prediction_positions"], dtype=torch.long),
            "target_alignment": encoded["target_alignment"],
            "source_alignment": encoded["source_alignment"],
            "state_position_semantics": "hidden_states use target_positions; target_logits use prediction_positions",
            "attention_output_semantics": "attention_outputs are pre-o_proj per-head outputs at target_positions",
            "target_logits": target_logits,
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
    if collect_attention_outputs:
        tensor["attention_outputs"] = _stack_attention_outputs(
            torch, attention_output_captures, expected_attention_output_layers
        )
    torch.save(tensor, tensor_path)

    metrics = _accuracy(pred_ids, target_ids)
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "condition": condition,
        "tensor_path": str(tensor_path),
        "loss": float(outputs.loss.detach().cpu()),
        "prompt_tokens": encoded["prompt_length"],
        "target_tokens": len(target_ids),
        "pred_text": tokenizer.decode(pred_ids, skip_special_tokens=True),
        "target_text": record["target_text"],
        **metrics,
    }


def collect(config: CollectConfig) -> None:
    invalid = sorted(set(config.conditions) - set(CONDITIONS))
    if invalid:
        raise ValueError(f"Unknown conditions: {', '.join(invalid)}")

    records = _select_records(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    tensor_dir = config.output_dir / "tensors"
    tensor_dir.mkdir(exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps({**asdict(config), "adapter_path": str(config.adapter_path) if config.adapter_path else None,
                    "dataset_path": str(config.dataset_path), "output_dir": str(config.output_dir)},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_jsonl(config.output_dir / "dataset_snapshot.jsonl", records)

    rows = []
    for condition in config.conditions:
        torch, tokenizer, model = _load_model(config, use_lora=condition == "lora_only")
        for record in records:
            tensor_path = tensor_dir / f"{record['sample_id']}__{condition}.pt"
            rows.append(
                _run_one(
                    torch,
                    tokenizer,
                    model,
                    record,
                    condition,
                    tensor_path,
                    collect_attention_outputs=config.collect_attention_outputs,
                    prompt_format=config.prompt_format,
                    append_eos=config.append_eos,
                )
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(config.output_dir / "metrics.jsonl", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect model states for LoRA/instruction analysis.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True, help="Dataset directory or JSONL file.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--condition", action="append", choices=CONDITIONS, dest="conditions")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--collect-attention-outputs", action="store_true")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        CollectConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            adapter_path=args.adapter_path,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            conditions=tuple(args.conditions or CONDITIONS),
            dtype=args.dtype,
            device=args.device,
            collect_attention_outputs=args.collect_attention_outputs,
            prompt_format=args.prompt_format,
            append_eos=not args.no_append_eos,
        )
    )
    print(f"Wrote model state run to {args.output_dir}")


if __name__ == "__main__":
    main()
