"""Sparse-feature comparisons for saved residual or attention deltas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from lora_instruction_analysis.model.visualize import (
    _cosine,
    _load_tensor,
    _metrics_by_key,
    _same_prefix_shape,
    _same_target_alignment,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_sae(torch, path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing SAE path: {path}")
    sae = torch.load(path, map_location="cpu") if path.is_file() else torch.load(path / "sae.pt", map_location="cpu")
    if not isinstance(sae, dict) or "encoder_weight" not in sae:
        raise TypeError("SAE artifact must be a torch-saved dict with encoder_weight and optional encoder_bias.")
    return sae


def _encode(torch, sae: dict, vector):
    if "input_mean" in sae:
        vector = vector.float() - sae["input_mean"].float()
    if "input_scale" in sae:
        vector = vector.float() / sae["input_scale"].float().clamp_min(1e-6)
    weight = sae["encoder_weight"].float()
    bias = sae.get("encoder_bias")
    acts = vector.float() @ weight.T
    if bias is not None:
        acts = acts + bias.float()
    return torch.relu(acts)


def _feature_ids(torch, acts, top_k: int) -> list[int]:
    if acts.numel() == 0:
        return []
    values, indices = torch.topk(acts, min(top_k, acts.numel()))
    return [int(index) for value, index in zip(values.tolist(), indices.tolist()) if float(value) > 0.0]


def _jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 0.0


def _success(row: dict) -> float:
    for key in ("task_semantic_correct", "sequence_accuracy", "token_accuracy"):
        if key in row and row[key] != "":
            return float(row[key])
    return 0.0


def _tensor_values(tensor: dict, key: str):
    if key not in tensor:
        raise ValueError(f"Missing {key} in {tensor.get('sample_id')} {tensor.get('condition')}; rerun collect with required outputs.")
    return tensor[key]


def _residual_rows(torch, run_dir: Path, sae: dict, top_k: int) -> list[dict]:
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    rows = []
    for sample_id in sample_ids:
        tensors = {condition: _load_tensor(torch, metrics[(sample_id, condition)], run_dir) for condition in ("base", "instruction_only", "lora_only")}
        alignment = _same_target_alignment(*(tensors[condition] for condition in ("base", "instruction_only", "lora_only")))
        layer_count = _same_prefix_shape("hidden_states", *(tensors[condition]["hidden_states"] for condition in tensors))
        for layer in range(layer_count):
            for token_index, target in enumerate(alignment):
                inst_acts = _encode(torch, sae, tensors["instruction_only"]["hidden_states"][layer, token_index] - tensors["base"]["hidden_states"][layer, token_index])
                lora_acts = _encode(torch, sae, tensors["lora_only"]["hidden_states"][layer, token_index] - tensors["base"]["hidden_states"][layer, token_index])
                inst_ids, lora_ids = _feature_ids(torch, inst_acts, top_k), _feature_ids(torch, lora_acts, top_k)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "task_id": tensors["base"].get("task_id", ""),
                        "mode": "residual_sae_delta",
                        "layer": layer,
                        "token_index": token_index,
                        "alignment_key": target["alignment_key"],
                        "top_k_feature_overlap": len(set(inst_ids) & set(lora_ids)),
                        "top_k_feature_jaccard": _jaccard(inst_ids, lora_ids),
                        "feature_activation_cosine": _cosine(torch, inst_acts, lora_acts),
                        "instruction_feature_ids": " ".join(map(str, inst_ids)),
                        "lora_feature_ids": " ".join(map(str, lora_ids)),
                        "instruction_success": _success(metrics[(sample_id, "instruction_only")]),
                        "lora_success": _success(metrics[(sample_id, "lora_only")]),
                    }
                )
    return rows


def _attention_rows(torch, run_dir: Path, sae: dict, top_k: int, tensor_key: str) -> list[dict]:
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    rows = []
    for sample_id in sample_ids:
        tensors = {condition: _load_tensor(torch, metrics[(sample_id, condition)], run_dir) for condition in ("base", "instruction_only", "lora_only")}
        alignment = _same_target_alignment(*(tensors[condition] for condition in ("base", "instruction_only", "lora_only")))
        values = {condition: _tensor_values(tensor, tensor_key) for condition, tensor in tensors.items()}
        layer_count = _same_prefix_shape(tensor_key, *(values[condition] for condition in values))
        has_heads = values["base"].ndim == 4
        for layer in range(layer_count):
            heads = range(values["base"].shape[1]) if has_heads else (None,)
            for head in heads:
                for token_index, target in enumerate(alignment):
                    index = (layer, head, token_index) if has_heads else (layer, token_index)
                    base_vec = values["base"][index]
                    inst_acts = _encode(torch, sae, values["instruction_only"][index] - base_vec)
                    lora_acts = _encode(torch, sae, values["lora_only"][index] - base_vec)
                    inst_ids, lora_ids = _feature_ids(torch, inst_acts, top_k), _feature_ids(torch, lora_acts, top_k)
                    row = {
                        "sample_id": sample_id,
                        "task_id": tensors["base"].get("task_id", ""),
                        "mode": f"{tensor_key}_sae_delta",
                        "layer": layer,
                        "token_index": token_index,
                        "alignment_key": target["alignment_key"],
                        "top_k_feature_overlap": len(set(inst_ids) & set(lora_ids)),
                        "top_k_feature_jaccard": _jaccard(inst_ids, lora_ids),
                        "feature_activation_cosine": _cosine(torch, inst_acts, lora_acts),
                        "instruction_feature_ids": " ".join(map(str, inst_ids)),
                        "lora_feature_ids": " ".join(map(str, lora_ids)),
                        "instruction_success": _success(metrics[(sample_id, "instruction_only")]),
                        "lora_success": _success(metrics[(sample_id, "lora_only")]),
                    }
                    if head is not None:
                        row["head"] = head
                    rows.append(row)
    return rows


def summary_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], int(row["layer"]))].append(row)
    return [
        {
            "mode": mode,
            "layer": layer,
            "count": len(values),
            "mean_top_k_feature_overlap": mean(float(row["top_k_feature_overlap"]) for row in values),
            "mean_top_k_feature_jaccard": mean(float(row["top_k_feature_jaccard"]) for row in values),
            "mean_feature_activation_cosine": mean(float(row["feature_activation_cosine"]) for row in values),
            "mean_instruction_success": mean(float(row["instruction_success"]) for row in values),
            "mean_lora_success": mean(float(row["lora_success"]) for row in values),
        }
        for (mode, layer), values in sorted(grouped.items())
    ]


def run_sae_analysis(run_dir: Path, sae_path: Path, output_dir: Path, *, mode: str, top_k: int = 20) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SAE analysis requires torch because collect.py stores .pt tensors.") from exc
    sae = _load_sae(torch, sae_path)
    if mode == "residual":
        rows = _residual_rows(torch, run_dir, sae, top_k)
    elif mode == "attention_outputs":
        rows = _attention_rows(torch, run_dir, sae, top_k, "attention_outputs")
    elif mode == "attention_post_o_proj_outputs":
        rows = _attention_rows(torch, run_dir, sae, top_k, "attention_post_o_proj_outputs")
    else:
        raise ValueError(f"Unknown SAE mode {mode!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "sae_feature_rows.csv", rows)
    _write_csv(output_dir / "sae_layer_summary.csv", summary_rows(rows))
    (output_dir / "sae_config.json").write_text(
        json.dumps({"run_dir": str(run_dir), "sae_path": str(sae_path), "mode": mode, "top_k": top_k}, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sparse SAE features for saved condition-minus-base deltas.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--sae-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("residual", "attention_outputs", "attention_post_o_proj_outputs"), default="residual")
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_sae_analysis(args.run, args.sae_path, args.output_dir, mode=args.mode, top_k=args.top_k)
    print(f"Wrote SAE analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
