"""Activation patching for teacher-forced LoRA/instruction comparisons."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random

from lora_instruction_analysis.data.tasks import ValidationSelector, evaluate_output, resolved_validator_name
from lora_instruction_analysis.model.collect import (
    CONDITIONS,
    CollectConfig,
    _accuracy,
    _dataset_file,
    _encode,
    _load_model,
    _read_jsonl,
    _write_jsonl,
)
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS


PATCH_CONDITIONS = CONDITIONS


@dataclass(frozen=True)
class PatchConfig:
    model_name: str
    dataset_path: Path
    output_dir: Path
    adapter_path: Path | None = None
    source_condition: str = "instruction_only"
    target_condition: str = "lora_only"
    layer: int = 0
    split: str = "test"
    max_samples: int | None = None
    seed: int = 13
    max_new_tokens: int = 20
    patch_span: str = "text"
    dtype: str = "auto"
    device: str = "auto"
    prompt_format: str = "raw"
    append_eos: bool = True
    validator: ValidationSelector = None


def _select_records(config: PatchConfig) -> list[dict]:
    records = _read_jsonl(_dataset_file(config.dataset_path, config.split))
    if config.max_samples is None:
        return records
    rng = random.Random(config.seed)
    chosen = records[:]
    rng.shuffle(chosen)
    return chosen[: config.max_samples]


def _model_config(config: PatchConfig) -> CollectConfig:
    return CollectConfig(
        model_name=config.model_name,
        dataset_path=config.dataset_path,
        output_dir=config.output_dir,
        adapter_path=config.adapter_path,
        split=config.split,
        max_samples=config.max_samples,
        seed=config.seed,
        dtype=config.dtype,
        device=config.device,
        prompt_format=config.prompt_format,
        append_eos=config.append_eos,
    )


def _load_condition(config: PatchConfig, condition: str):
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    return _load_model(_model_config(config), use_lora=condition == "lora_only")


def _control_sources(config: PatchConfig) -> list[tuple[str, str]]:
    controls = [
        ("base_to_target_patch", "base"),
        ("source_to_target_patch", config.source_condition),
    ]
    seen = set()
    result = []
    for control_name, condition in controls:
        if control_name in seen:
            continue
        seen.add(control_name)
        result.append((control_name, condition))
    return result


def _block(model, layer: int):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
        module = base
        try:
            for part in path.split("."):
                module = getattr(module, part)
            return module[layer]
        except (AttributeError, IndexError):
            continue
    raise RuntimeError("Could not find transformer blocks. Pass support for this model type before running RQ3.")


def _with_patched_slice(torch, output, target_positions, patch):
    hidden = output[0] if isinstance(output, tuple) else output
    if patch.ndim != 2:
        raise ValueError(f"Patch must be rank-2 [positions, hidden], got shape {tuple(patch.shape)}.")
    if patch.shape[0] != len(target_positions):
        raise ValueError(
            f"Patch position mismatch: patch has {patch.shape[0]} rows for {len(target_positions)} target positions."
        )
    if patch.shape[1] != hidden.shape[-1]:
        raise ValueError(f"Patch hidden width mismatch: patch={patch.shape[1]}, model={hidden.shape[-1]}.")
    patched = hidden.clone()
    patched[0, target_positions, :] = patch.to(device=hidden.device, dtype=hidden.dtype)
    return (patched, *output[1:]) if isinstance(output, tuple) else patched


def _positions(encoded: dict, patch_span: str) -> list[int]:
    if patch_span == "target":
        return encoded["prediction_positions"]
    if patch_span == "text":
        positions = [row["position"] for row in encoded["source_alignment"] if row["span"] == "input"]
        if not positions:
            raise ValueError("Could not find input text token positions for patch_span='text'.")
        return positions
    raise ValueError(f"Unknown patch span {patch_span!r}; use target or text.")


def _input_start(encoded: dict) -> int:
    positions = [row["position"] for row in encoded["source_alignment"] if row["span"] == "input"]
    if not positions:
        raise ValueError("Could not find input text token positions for RQ3 alignment.")
    return min(positions)


def _pad_token_id(tokenizer) -> int:
    token_id = getattr(tokenizer, "pad_token_id", None)
    if token_id is None:
        token_id = getattr(tokenizer, "eos_token_id", None)
    if token_id is None:
        raise ValueError("Tokenizer must define pad_token_id or eos_token_id for RQ3 position alignment.")
    return token_id


def _left_pad_encoded(encoded: dict, pad_count: int, pad_token_id: int) -> dict:
    if pad_count <= 0:
        return encoded
    padded = dict(encoded)
    padded["input_ids"] = [pad_token_id] * pad_count + encoded["input_ids"]
    padded["labels"] = [-100] * pad_count + encoded["labels"]
    padded["prompt_length"] = encoded["prompt_length"] + pad_count
    padded["target_positions"] = [position + pad_count for position in encoded["target_positions"]]
    padded["prediction_positions"] = [position + pad_count for position in encoded["prediction_positions"]]
    padded["target_alignment"] = [
        {
            **row,
            "target_position": row["target_position"] + pad_count,
            "prediction_position": row["prediction_position"] + pad_count,
        }
        for row in encoded["target_alignment"]
    ]
    pad_rows = [
        {"position": position, "span": "prompt", "alignment_key": f"prompt:{position}:{pad_token_id}", "token_id": pad_token_id}
        for position in range(pad_count)
    ]
    shifted_rows = []
    for row in encoded["source_alignment"]:
        shifted = {**row, "position": row["position"] + pad_count}
        if shifted["span"] == "prompt":
            shifted["alignment_key"] = f"prompt:{shifted['position']}:{shifted['token_id']}"
        shifted_rows.append(shifted)
    padded["source_alignment"] = pad_rows + shifted_rows
    return padded


def _align_encoded_to_input_start(encoded: dict, target_start: int, pad_token_id: int) -> dict:
    current_start = _input_start(encoded)
    if current_start > target_start:
        raise ValueError(
            f"Cannot align encoded input text by padding: current start {current_start} is after target start {target_start}."
        )
    return _left_pad_encoded(encoded, target_start - current_start, pad_token_id)


def _patch_mapping(source_encoded: dict, target_encoded: dict, patch_span: str) -> tuple[list[int], list[int]]:
    if patch_span == "target":
        source_rows = source_encoded["target_alignment"]
        target_rows = target_encoded["target_alignment"]
        source_positions = {row["alignment_key"]: index for index, row in enumerate(source_rows)}
        missing = [row["alignment_key"] for row in target_rows if row["alignment_key"] not in source_positions]
        if missing:
            raise ValueError(f"Missing source target activations for: {', '.join(missing)}")
        return [source_positions[row["alignment_key"]] for row in target_rows], [
            row["prediction_position"] for row in target_rows
        ]

    source_rows = [row for row in source_encoded["source_alignment"] if row["span"] == "input"]
    target_rows = [row for row in target_encoded["source_alignment"] if row["span"] == "input"]
    source_positions = {row["alignment_key"]: index for index, row in enumerate(source_rows)}
    missing = [row["alignment_key"] for row in target_rows if row["alignment_key"] not in source_positions]
    if missing:
        raise ValueError(f"Missing source text activations for: {', '.join(missing)}")
    return [source_positions[row["alignment_key"]] for row in target_rows], [row["position"] for row in target_rows]


def _capture_activation(
    torch,
    tokenizer,
    model,
    record: dict,
    condition: str,
    layer: int,
    patch_span: str,
    prompt_format: str,
    append_eos: bool,
    align_input_start: int | None = None,
):
    encoded = _encode(
        tokenizer,
        record,
        include_instruction=condition == "instruction_only",
        prompt_format=prompt_format,
        append_eos=append_eos,
    )
    if align_input_start is not None:
        encoded = _align_encoded_to_input_start(encoded, align_input_start, _pad_token_id(tokenizer))
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    patch_positions = torch.tensor(_positions(encoded, patch_span), dtype=torch.long, device=device)
    captured = {}

    def save(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["activation"] = hidden[0, patch_positions, :].detach().cpu()

    handle = _block(model, layer).register_forward_hook(save)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return captured["activation"], encoded


def _run_target(
    torch,
    tokenizer,
    model,
    record: dict,
    condition: str,
    layer: int,
    patch_span: str,
    prompt_format: str,
    append_eos: bool,
    validator: ValidationSelector,
    patch=None,
    source_encoded: dict | None = None,
) -> dict:
    encoded = _encode(
        tokenizer,
        record,
        include_instruction=condition == "instruction_only",
        prompt_format=prompt_format,
        append_eos=append_eos,
    )
    if source_encoded is not None:
        encoded = _align_encoded_to_input_start(encoded, _input_start(source_encoded), _pad_token_id(tokenizer))
    device = next(model.parameters()).device
    input_ids = torch.tensor([encoded["input_ids"]], dtype=torch.long, device=device)
    labels = torch.tensor([encoded["labels"]], dtype=torch.long, device=device)
    prediction_positions = torch.tensor(encoded["prediction_positions"], dtype=torch.long, device=device)
    patch_position_list = _positions(encoded, patch_span)

    handle = None
    if patch is not None:
        if source_encoded is None:
            raise ValueError("source_encoded is required when patching.")
        source_indices, patch_position_list = _patch_mapping(source_encoded, encoded, patch_span)
        patch = patch[source_indices]
        patch_positions = torch.tensor(patch_position_list, dtype=torch.long, device=device)

        def apply_patch(_module, _inputs, output):
            return _with_patched_slice(torch, output, patch_positions, patch)

        handle = _block(model, layer).register_forward_hook(apply_patch)
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels, use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    logits = outputs.logits[0, prediction_positions, :]
    pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    target_ids = [token_id for token_id in encoded["labels"] if token_id != -100]
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "source_condition": None,
        "target_condition": condition,
        "control": "source_to_target_patch" if patch is not None else "unpatched",
        "patched": patch is not None,
        "layer": layer,
        "patch_span": patch_span,
        "loss": float(outputs.loss.detach().cpu()),
        "prompt_tokens": encoded["prompt_length"],
        "target_tokens": len(target_ids),
        "pred_text": pred_text,
        "target_text": record["target_text"],
        **evaluate_output(record["task_id"], record["input_text"], pred_text, record["target_text"], validator),
        **_accuracy(pred_ids, target_ids),
    }


def _target_loss_ids(target_ids: list[int], token_index: int) -> int | None:
    return target_ids[token_index] if token_index < len(target_ids) else None


def _visible_patch(torch, output, patch_position_list: list[int], patch):
    hidden = output[0] if isinstance(output, tuple) else output
    visible = [(index, position) for index, position in enumerate(patch_position_list) if position < hidden.shape[1]]
    if not visible:
        return output
    patch_indices = [index for index, _position in visible]
    patch_positions = torch.tensor([position for _index, position in visible], dtype=torch.long, device=hidden.device)
    return _with_patched_slice(torch, output, patch_positions, patch[patch_indices])


def _generate_target(
    torch,
    tokenizer,
    model,
    record: dict,
    condition: str,
    layer: int,
    max_new_tokens: int,
    patch_span: str,
    prompt_format: str,
    append_eos: bool,
    validator: ValidationSelector,
    patch=None,
    source_encoded: dict | None = None,
) -> dict:
    encoded = _encode(
        tokenizer,
        record,
        include_instruction=condition == "instruction_only",
        prompt_format=prompt_format,
        append_eos=append_eos,
    )
    if source_encoded is not None:
        encoded = _align_encoded_to_input_start(encoded, _input_start(source_encoded), _pad_token_id(tokenizer))
    prompt_len = encoded["prompt_length"]
    prompt_ids = encoded["input_ids"][:prompt_len]
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    target_ids = [token_id for token_id in encoded["labels"] if token_id != -100]
    handle = None
    if patch is not None:
        if source_encoded is None:
            raise ValueError("source_encoded is required when patching generation.")
        source_indices, patch_position_list = _patch_mapping(source_encoded, encoded, patch_span)
        patch = patch[source_indices]

        def apply_patch(_module, _inputs, output):
            return _visible_patch(torch, output, patch_position_list, patch)

        handle = _block(model, layer).register_forward_hook(apply_patch)
    pred_ids = []
    loss_target_ids = []
    token_losses = []
    stopped_on_eos = False
    try:
        with torch.no_grad():
            for token_index in range(max_new_tokens):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                logits = outputs.logits[:, -1, :]
                target_id = _target_loss_ids(target_ids, token_index)
                if target_id is not None:
                    target_tensor = torch.tensor([target_id], dtype=torch.long, device=device)
                    token_losses.append(
                        float(torch.nn.functional.cross_entropy(logits.float(), target_tensor).detach().cpu())
                    )
                    loss_target_ids.append(target_id)
                pred_id = int(logits.argmax(dim=-1)[0].detach().cpu())
                pred_ids.append(pred_id)
                if tokenizer.eos_token_id is not None and pred_id == tokenizer.eos_token_id:
                    stopped_on_eos = True
                    break
                next_id = torch.tensor([[pred_id]], dtype=torch.long, device=device)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                attention_mask = torch.ones_like(input_ids)
    finally:
        if handle is not None:
            handle.remove()

    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    return {
        "sample_id": record["sample_id"],
        "task_id": record["task_id"],
        "source_condition": None,
        "target_condition": condition,
        "control": "source_to_target_patch" if patch is not None else "unpatched",
        "patched": patch is not None,
        "layer": layer,
        "patch_span": patch_span,
        "generated_tokens": len(pred_ids),
        "pred_token_ids": pred_ids,
        "target_token_ids": target_ids,
        "loss_target_token_ids": loss_target_ids,
        "target_strategy": "target_text_tokens",
        "generation_patch_strategy": "persistent_visible_positions" if patch is not None else "unpatched",
        "token_losses": token_losses,
        "eos_token_id": tokenizer.eos_token_id,
        "stopped_on_eos": stopped_on_eos,
        "pred_text": pred_text,
        "target_text": record["target_text"],
        **evaluate_output(record["task_id"], record["input_text"], pred_text, record["target_text"], validator),
        **_accuracy(pred_ids[: len(target_ids)], target_ids),
    }


def _capture_patches_for_condition(torch, tokenizer, model, records: list[dict], config: PatchConfig, condition: str):
    patches = []
    for record in records:
        source_encoded = _encode(
            tokenizer,
            record,
            include_instruction=condition == "instruction_only",
            prompt_format=config.prompt_format,
            append_eos=config.append_eos,
        )
        target_encoded = _encode(
            tokenizer,
            record,
            include_instruction=config.target_condition == "instruction_only",
            prompt_format=config.prompt_format,
            append_eos=config.append_eos,
        )
        align_input_start = max(_input_start(source_encoded), _input_start(target_encoded))
        patch, encoded = _capture_activation(
            torch,
            tokenizer,
            model,
            record,
            condition,
            config.layer,
            config.patch_span,
            config.prompt_format,
            config.append_eos,
            align_input_start,
        )
        patches.append((patch, encoded))
    return patches


def _confusion_matrix(rows: list[dict], metric: str) -> list[dict]:
    by_sample: dict[str, dict[str, bool]] = {}
    for row in rows:
        sample = by_sample.setdefault(row["sample_id"], {})
        sample[row["control"]] = bool(row.get(metric, 0.0))
    counts = Counter(
        (
            sample.get("unpatched", False),
            sample.get("source_to_target_patch", False),
        )
        for sample in by_sample.values()
        if "unpatched" in sample and "source_to_target_patch" in sample
    )
    return [
        {
            "metric": metric,
            "unpatched_correct": unpatched,
            "source_to_target_correct": patched,
            "samples": count,
        }
        for (unpatched, patched), count in sorted(counts.items())
    ]


def run_activation_patching(config: PatchConfig) -> None:
    invalid = {config.source_condition, config.target_condition} - set(PATCH_CONDITIONS)
    if invalid:
        raise ValueError(f"Unknown conditions: {', '.join(sorted(invalid))}")
    if config.patch_span not in {"target", "text"}:
        raise ValueError("--patch-span must be target or text.")

    records = _select_records(config)
    task_ids = {record["task_id"] for record in records}
    if len(task_ids) != 1:
        raise ValueError(f"Activation patching requires exactly one task_id, found {sorted(task_ids)}")
    resolved_validator = resolved_validator_name(next(iter(task_ids)), config.validator)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_path": str(config.dataset_path),
                "adapter_path": str(config.adapter_path) if config.adapter_path else None,
                "output_dir": str(config.output_dir),
                "validator": resolved_validator,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _write_jsonl(config.output_dir / "dataset_snapshot.jsonl", records)

    patches_by_control = {}
    for control_name, source_condition in _control_sources(config):
        torch, source_tokenizer, source_model = _load_condition(config, source_condition)
        try:
            patches_by_control[control_name] = _capture_patches_for_condition(
                torch, source_tokenizer, source_model, records, config, source_condition
            )
        finally:
            del source_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    _, target_tokenizer, target_model = _load_condition(config, config.target_condition)
    rows = []
    generation_rows = []
    outcome_rows = []
    try:
        for record_index, record in enumerate(records):
            base_row = _run_target(
                torch,
                target_tokenizer,
                target_model,
                record,
                config.target_condition,
                config.layer,
                config.patch_span,
                config.prompt_format,
                config.append_eos,
                resolved_validator,
            )
            rows.append(base_row)
            base_gen = _generate_target(
                torch,
                target_tokenizer,
                target_model,
                record,
                config.target_condition,
                config.layer,
                config.max_new_tokens,
                config.patch_span,
                config.prompt_format,
                config.append_eos,
                resolved_validator,
            )
            generation_rows.append(base_gen)
            outcome_rows.append(
                {
                    "sample_id": record["sample_id"],
                    "task_id": record["task_id"],
                    "control": "unpatched",
                    "source_condition": None,
                    "target_condition": config.target_condition,
                    "teacher_forced_sequence_accuracy": base_row["sequence_accuracy"],
                    "generation_sequence_accuracy": base_gen["sequence_accuracy"],
                    "generation_task_semantic_correct": base_gen["task_semantic_correct"],
                }
            )
            for control_name, source_condition in _control_sources(config):
                patch, source_encoded = patches_by_control[control_name][record_index]
                row = _run_target(
                    torch,
                    target_tokenizer,
                    target_model,
                    record,
                    config.target_condition,
                    config.layer,
                    config.patch_span,
                    config.prompt_format,
                    config.append_eos,
                    resolved_validator,
                    patch,
                    source_encoded,
                )
                row["control"] = control_name
                row["source_condition"] = source_condition
                rows.append(row)
                patch_gen = _generate_target(
                    torch,
                    target_tokenizer,
                    target_model,
                    record,
                    config.target_condition,
                    config.layer,
                    config.max_new_tokens,
                    config.patch_span,
                    config.prompt_format,
                    config.append_eos,
                    resolved_validator,
                    patch,
                    source_encoded,
                )
                patch_gen["control"] = control_name
                patch_gen["source_condition"] = source_condition
                generation_rows.append(patch_gen)
                outcome_rows.append(
                    {
                        "sample_id": record["sample_id"],
                        "task_id": record["task_id"],
                        "control": control_name,
                        "source_condition": source_condition,
                        "target_condition": config.target_condition,
                        "teacher_forced_sequence_accuracy": row["sequence_accuracy"],
                        "generation_sequence_accuracy": patch_gen["sequence_accuracy"],
                        "generation_task_semantic_correct": patch_gen["task_semantic_correct"],
                    }
                )
    finally:
        del target_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _write_jsonl(config.output_dir / "metrics.jsonl", rows)
    _write_jsonl(config.output_dir / "generations.jsonl", generation_rows)
    _write_jsonl(config.output_dir / "outcomes.jsonl", outcome_rows)
    (config.output_dir / "confusion_matrix.json").write_text(
        json.dumps(
            {
                "semantic_generation": _confusion_matrix(generation_rows, "task_semantic_correct"),
                "teacher_forced_sequence": _confusion_matrix(rows, "sequence_accuracy"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run teacher-forced activation patching.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--source-condition", choices=PATCH_CONDITIONS, default="instruction_only")
    parser.add_argument("--target-condition", choices=PATCH_CONDITIONS, default="lora_only")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--patch-span", choices=("target", "text"), default="text")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS, default="raw")
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--validator", default=None, help="Override task validation: task_default, exact, single_token, integer, or yes_no.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_activation_patching(
        PatchConfig(
            model_name=args.model_name,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            adapter_path=args.adapter_path,
            source_condition=args.source_condition,
            target_condition=args.target_condition,
            layer=args.layer,
            split=args.split,
            max_samples=args.max_samples,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            patch_span=args.patch_span,
            dtype=args.dtype,
            device=args.device,
            prompt_format=args.prompt_format,
            append_eos=not args.no_append_eos,
            validator=args.validator,
        )
    )
    print(f"Wrote activation patching metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
