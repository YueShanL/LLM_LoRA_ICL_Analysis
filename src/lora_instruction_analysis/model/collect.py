"""Collect teacher-forced model states for base, instruction, and LoRA runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable

from lora_instruction_analysis.data.icl import attach_dataset_icl_examples
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
    icl_examples: int = 0
    icl_split: str = "train"


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
        return attach_dataset_icl_examples(
            records, config.dataset_path, example_count=config.icl_examples, split=config.icl_split
        )
    rng = random.Random(config.seed)
    chosen = records[:]
    rng.shuffle(chosen)
    return attach_dataset_icl_examples(
        chosen[: config.max_samples], config.dataset_path, example_count=config.icl_examples, split=config.icl_split
    )


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
    import torch

    pre_o_proj_captures = []
    post_o_proj_captures = []
    head_ablation_captures = []
    executed_layer_names = []
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
        current_per_head = None

        def save_input(_module, inputs):
            nonlocal current_per_head
            value = inputs[0]
            if value.shape[-1] != int(head_count) * int(head_dim):
                raise RuntimeError(
                    f"Attention output width mismatch in {name}: got {value.shape[-1]}, "
                    f"expected {int(head_count) * int(head_dim)}."
                )
            per_head = value[0, target_positions, :].view(len(target_positions), int(head_count), int(head_dim))
            current_per_head = per_head
            executed_layer_names.append(name)
            pre_o_proj_captures.append(per_head.permute(1, 0, 2).detach().cpu())

        def save_output(o_proj, _inputs, output):
            post_output = output[0, target_positions, :]
            post_o_proj_captures.append(post_output.detach().cpu())
            if current_per_head is None or not hasattr(o_proj, "weight"):
                return
            impacts = []
            full_norm = post_output.float().norm(dim=-1).clamp_min(1e-12)
            for head in range(int(head_count)):
                start = head * int(head_dim)
                end = start + int(head_dim)
                weight = o_proj.weight[:, start:end].to(device=post_output.device)
                contribution = current_per_head[:, head, :].to(weight.dtype) @ weight.T
                contribution = contribution.to(post_output.dtype)
                contribution_norm = contribution.float().norm(dim=-1)
                ablated = post_output - contribution
                denom = full_norm * ablated.float().norm(dim=-1).clamp_min(1e-12)
                cosine = (post_output.float() * ablated.float()).sum(dim=-1) / denom
                impacts.append(
                    torch.stack(
                        [
                            full_norm,
                            contribution_norm,
                            contribution_norm / full_norm,
                            cosine,
                        ],
                        dim=-1,
                    )
                )
            head_ablation_captures.append(torch.stack(impacts).detach().cpu())

        return save_input, save_output

    handles = []
    for name, module in modules:
        save_input, save_output = hook_for(name, module)
        handles.append(module.o_proj.register_forward_pre_hook(save_input))
        handles.append(module.o_proj.register_forward_hook(save_output))
    return pre_o_proj_captures, post_o_proj_captures, head_ablation_captures, executed_layer_names, handles, len(modules)


def _stack_attention_outputs(torch, captures: list, expected_layers: int, *, pad: bool = False) -> object:
    if len(captures) != expected_layers:
        raise RuntimeError(f"Captured {len(captures)} attention output layers, expected {expected_layers}.")
    if pad and captures:
        shapes = [tuple(capture.shape) for capture in captures]
        if len(set(shapes)) > 1:
            max_shape = tuple(max(shape[dim] for shape in shapes) for dim in range(len(shapes[0])))
            padded = []
            for capture in captures:
                output = capture.new_zeros(max_shape)
                output[tuple(slice(0, size) for size in capture.shape)] = capture
                padded.append(output)
            captures = padded
    return torch.stack(captures).contiguous()


def _normalize_attention_layer_name(name: str) -> str:
    while name.startswith("base_model.model."):
        name = name.removeprefix("base_model.model.")
    return name


def _validate_attention_layer_names(expected: list[str] | None, current: list[str], sample_id: str, condition: str) -> list[str]:
    current = [_normalize_attention_layer_name(name) for name in current]
    if expected is None:
        return current
    if current != expected:
        raise RuntimeError(
            "Attention output layer path changed within collect run for "
            f"{sample_id} / {condition}: expected {expected}, got {current}."
        )
    return expected


def _validate_attention_layer_shapes(expected: list[list[int]] | None, current: list[list[int]], sample_id: str, condition: str) -> list[list[int]]:
    current_layout = [[shape[0], shape[-1]] for shape in current]
    if expected is None:
        return current_layout
    if current_layout != expected:
        raise RuntimeError(
            "Attention output layer shapes changed within collect run for "
            f"{sample_id} / {condition}: expected {expected}, got {current_layout}."
        )
    return expected


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
    pre_o_proj_captures, post_o_proj_captures, head_ablation_captures, attention_output_layer_names, handles, _registered_attention_output_layers = (
        _attention_output_hooks(model, target_positions) if collect_attention_outputs else ([], [], [], [], [], 0)
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
            "attention_output_semantics": (
                "attention_outputs are pre-o_proj per-head outputs at target_positions; "
                "attention_post_o_proj_outputs are post-o_proj attention block outputs at target_positions; "
                "attention_head_ablation_impacts are scalar post-o_proj head removal impacts"
            ),
            "target_logits": target_logits,
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
    if collect_attention_outputs:
        expected_attention_output_layers = len(pre_o_proj_captures)
        if expected_attention_output_layers == 0:
            raise RuntimeError("Captured no attention output layers.")
        if len(post_o_proj_captures) != expected_attention_output_layers or len(head_ablation_captures) != expected_attention_output_layers:
            raise RuntimeError(
                "Captured inconsistent attention output layers: "
                f"pre={len(pre_o_proj_captures)}, post={len(post_o_proj_captures)}, "
                f"head_ablation={len(head_ablation_captures)}."
            )
        attention_output_layer_names = [_normalize_attention_layer_name(name) for name in attention_output_layer_names]
        attention_output_layer_shapes = [list(capture.shape) for capture in pre_o_proj_captures]
        tensor["attention_output_layer_names"] = list(attention_output_layer_names)
        tensor["attention_output_layer_shapes"] = attention_output_layer_shapes
        tensor["attention_outputs"] = _stack_attention_outputs(
            torch, pre_o_proj_captures, expected_attention_output_layers, pad=True
        )
        tensor["attention_post_o_proj_outputs"] = _stack_attention_outputs(
            torch, post_o_proj_captures, expected_attention_output_layers, pad=True
        )
        tensor["attention_head_ablation_impact_names"] = [
            "full_norm",
            "head_contribution_norm",
            "head_contribution_relative_norm",
            "ablated_cosine_to_full",
        ]
        tensor["attention_head_ablation_impacts"] = _stack_attention_outputs(
            torch, head_ablation_captures, expected_attention_output_layers, pad=True
        )
    torch.save(tensor, tensor_path)

    metrics = _accuracy(pred_ids, target_ids)
    metrics_row = {
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
    if collect_attention_outputs:
        metrics_row["attention_output_layer_names"] = list(attention_output_layer_names)
        metrics_row["attention_output_layer_shapes"] = attention_output_layer_shapes
    return metrics_row


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
    expected_attention_output_layer_names = None
    expected_attention_output_layer_shapes = None
    for condition in config.conditions:
        torch, tokenizer, model = _load_model(config, use_lora=condition == "lora_only")
        for record in records:
            tensor_path = tensor_dir / f"{record['sample_id']}__{condition}.pt"
            row = _run_one(
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
            if config.collect_attention_outputs:
                layer_names = row["attention_output_layer_names"]
                expected_attention_output_layer_names = _validate_attention_layer_names(
                    expected_attention_output_layer_names, layer_names, record["sample_id"], condition
                )
                expected_attention_output_layer_shapes = _validate_attention_layer_shapes(
                    expected_attention_output_layer_shapes,
                    row["attention_output_layer_shapes"],
                    record["sample_id"],
                    condition,
                )
            rows.append(row)
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
    parser.add_argument("--icl-examples", type=int, default=0, help="Use N train input-output pairs in place of instruction text.")
    parser.add_argument("--icl-split", default="train")
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
            icl_examples=args.icl_examples,
            icl_split=args.icl_split,
        )
    )
    print(f"Wrote model state run to {args.output_dir}")


if __name__ == "__main__":
    main()
