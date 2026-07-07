"""Visualize token-level state similarity from collect.py outputs."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean


DEFAULT_MODEL = "meta-llama/Llama-3.2-3B"
ATTENTION_ALIGNMENT_STRATEGY = (
    "Compare only shared source_alignment keys from input:* and already-emitted target:* tokens; "
    "instruction prefix tokens are excluded."
)
ATTENTION_PATTERN_DEFINITION = (
    "cosine/entropy/KL over aligned attention probabilities for the row's condition pair"
)
ATTENTION_OUTPUT_DEFINITION = (
    "cosine over per-head attention outputs for the row's condition pair"
)
RQ3_CHART_SPECS = (
    ("teacher_forced", "loss", "Teacher-Forced Loss Mean By Layer"),
    ("teacher_forced", "sequence_accuracy", "Teacher-Forced Sequence Accuracy Mean By Layer"),
    ("generation", "generation_loss", "Generation Loss Mean By Layer"),
    ("generation", "task_semantic_correct", "Generation Semantic Accuracy Mean By Layer"),
)


@dataclass(frozen=True)
class VisualizeConfig:
    run: Path | None
    left_run: Path | None
    right_run: Path | None
    output_dir: Path
    left_model: str = DEFAULT_MODEL
    right_model: str = DEFAULT_MODEL
    mode: str = "residual"
    conditions: tuple[str, ...] = ()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics_by_key(run_dir: Path) -> dict[tuple[str, str], dict]:
    return {(row["sample_id"], row["condition"]): row for row in _read_jsonl(run_dir / "metrics.jsonl")}


def _load_tensor(torch, row: dict, run_dir: Path) -> dict:
    path = Path(row["tensor_path"])
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([run_dir / path, run_dir / "tensors" / path.name])
    for candidate in candidates:
        if candidate.exists():
            return torch.load(candidate, map_location="cpu")
    raise FileNotFoundError(f"Missing tensor file for {row['sample_id']} {row['condition']}: {path}")


def _cosine(torch, left, right) -> float:
    left = left.float()
    right = right.float()
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denom)


def _entropy(torch, probs) -> float:
    safe = probs.float().clamp_min(1e-12)
    return float(-(safe * safe.log()).sum())


def _kl(torch, left, right) -> float:
    left_safe = left.float().clamp_min(1e-12)
    right_safe = right.float().clamp_min(1e-12)
    return float((left_safe * (left_safe.log() - right_safe.log())).sum())


def _linear_cka(torch, left, right) -> float:
    left = left.float()
    right = right.float()
    left = left - left.mean(dim=0, keepdim=True)
    right = right - right.mean(dim=0, keepdim=True)
    hsic = torch.linalg.matrix_norm(left.T @ right) ** 2
    denom = torch.linalg.matrix_norm(left.T @ left) * torch.linalg.matrix_norm(right.T @ right)
    if float(denom) == 0.0:
        return 0.0
    return float(hsic / denom)


def _probability_cosine(torch, left_logits, right_logits) -> float:
    return _cosine(torch, left_logits.float().softmax(dim=-1), right_logits.float().softmax(dim=-1))


def _target_token_ids(tensor: dict) -> list[int]:
    labels = tensor["labels"].tolist()
    return [token_id for token_id in labels if token_id != -100]


def _target_alignment(tensor: dict) -> list[dict]:
    if "target_alignment" not in tensor:
        raise ValueError(
            f"Missing target_alignment in {tensor.get('sample_id', '<unknown>')} "
            f"{tensor.get('condition', '<unknown>')}; rerun collect.py."
        )
    rows = list(tensor["target_alignment"])
    if len(rows) != tensor["hidden_states"].shape[1]:
        raise ValueError(
            f"target_alignment length does not match hidden_states for "
            f"{tensor.get('sample_id')} {tensor.get('condition')}."
        )
    return rows


def _same_target_alignment(*tensors: dict) -> list[dict]:
    alignments = [_target_alignment(tensor) for tensor in tensors]
    keys = [[row["alignment_key"] for row in alignment] for alignment in alignments]
    if any(key != keys[0] for key in keys[1:]):
        sample = tensors[0].get("sample_id", "<unknown>")
        raise ValueError(f"Target alignment mismatch for {sample}; refusing token_index truncation.")
    return alignments[0]


def _same_prefix_shape(name: str, *values) -> int:
    counts = [value.shape[0] for value in values]
    if any(count != counts[0] for count in counts[1:]):
        raise ValueError(f"{name} layer count mismatch: {counts}.")
    return counts[0]


def _source_index_by_key(tensor: dict, seq_len: int) -> dict[str, int]:
    if "source_alignment" not in tensor:
        raise ValueError(
            f"Missing source_alignment in {tensor.get('sample_id', '<unknown>')} "
            f"{tensor.get('condition', '<unknown>')}; rerun collect.py."
        )
    result = {}
    for row in tensor["source_alignment"]:
        position = int(row["position"])
        if position < seq_len and row["span"] in {"input", "target"}:
            result[row["alignment_key"]] = position
    return result


def _row_base(sample_id: str, tensor: dict, condition: str, mode: str, layer: int, token_index: int, alignment: dict) -> dict:
    return {
        "sample_id": sample_id,
        "task_id": tensor.get("task_id", ""),
        "condition": condition,
        "mode": mode,
        "layer": layer,
        "token_index": token_index,
        "alignment_key": alignment["alignment_key"],
        "left_token_id": alignment["token_id"],
        "right_token_id": alignment["token_id"],
    }


def _residual_rows(torch, run_dir: Path) -> list[dict]:
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    rows = []
    bundles = []
    for sample_id in sample_ids:
        base_tensor = _load_tensor(torch, metrics[(sample_id, "base")], run_dir)
        instruction = _load_tensor(torch, metrics[(sample_id, "instruction_only")], run_dir)
        lora = _load_tensor(torch, metrics[(sample_id, "lora_only")], run_dir)
        alignment = _same_target_alignment(base_tensor, instruction, lora)
        base = base_tensor["hidden_states"]
        instruction_hidden = instruction["hidden_states"]
        lora_hidden = lora["hidden_states"]
        layer_count = _same_prefix_shape("hidden_states", base, instruction_hidden, lora_hidden)
        bundles.append((sample_id, base_tensor, instruction, lora, alignment, base, instruction_hidden, lora_hidden))

    cka_by_layer = {}
    for layer in range(layer_count if bundles else 0):
        instruction_delta = torch.cat(
            [instruction_hidden[layer] - base[layer] for _, _, _, _, _, base, instruction_hidden, _ in bundles]
        )
        lora_delta = torch.cat(
            [lora_hidden[layer] - base[layer] for _, _, _, _, _, base, _, lora_hidden in bundles]
        )
        cka_by_layer[layer] = _linear_cka(torch, instruction_delta, lora_delta)

    for sample_id, base_tensor, instruction, lora, alignment, base, instruction_hidden, lora_hidden in bundles:
        for layer in range(layer_count):
            for token_index, target in enumerate(alignment):
                row = _row_base(
                    sample_id, instruction, "instruction_only_vs_lora_only", "residual_perturbation", layer, token_index, target
                )
                row["cosine_similarity"] = _cosine(
                    torch,
                    instruction_hidden[layer, token_index] - base[layer, token_index],
                    lora_hidden[layer, token_index] - base[layer, token_index],
                )
                row["cka_similarity"] = cka_by_layer[layer]
                row["logit_distribution_cosine"] = _probability_cosine(
                    torch,
                    instruction["target_logits"][token_index],
                    lora["target_logits"][token_index],
                )
                rows.append(row)
    return rows


def _attention_rows(torch, run_dir: Path) -> list[dict]:
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    rows = []
    for sample_id in sample_ids:
        base = _load_tensor(torch, metrics[(sample_id, "base")], run_dir)
        instruction = _load_tensor(torch, metrics[(sample_id, "instruction_only")], run_dir)
        lora = _load_tensor(torch, metrics[(sample_id, "lora_only")], run_dir)
        base_attention = base["attentions"]
        instruction_attention = instruction["attentions"]
        lora_attention = lora["attentions"]
        alignment = _same_target_alignment(base, instruction, lora)
        layer_count = _same_prefix_shape("attention", base_attention, instruction_attention, lora_attention)
        if len({base_attention.shape[1], instruction_attention.shape[1], lora_attention.shape[1]}) != 1:
            raise ValueError(f"attention head count mismatch for {sample_id}.")
        head_count = instruction_attention.shape[1]
        pairs = [
            ("instruction_only_vs_lora_only", instruction, instruction_attention, lora, lora_attention),
            ("lora_only_vs_base", lora, lora_attention, base, base_attention),
        ]
        for condition, left_tensor, left_attention, right_tensor, right_attention in pairs:
            left_sources = _source_index_by_key(left_tensor, left_attention.shape[3])
            right_sources = _source_index_by_key(right_tensor, right_attention.shape[3])
            for layer in range(layer_count):
                for head in range(head_count):
                    for token_index, target in enumerate(alignment):
                        keys = [
                            key
                            for key in left_sources.keys() & right_sources.keys()
                            if key.startswith("input:") or int(key.split(":", 2)[1]) <= token_index
                        ]
                        if not keys:
                            raise ValueError(f"No comparable attention source keys for {sample_id} token {token_index}.")
                        left = left_attention[layer, head, token_index, [left_sources[key] for key in keys]]
                        right = right_attention[layer, head, token_index, [right_sources[key] for key in keys]]
                        row = _row_base(sample_id, left_tensor, condition, "attention_pattern", layer, token_index, target)
                        row.update(
                            {
                                "head": head,
                                "source_keys": len(keys),
                                "metric_definition": ATTENTION_PATTERN_DEFINITION,
                                "alignment_strategy": ATTENTION_ALIGNMENT_STRATEGY,
                                "cosine_similarity": _cosine(torch, left, right),
                                "left_entropy": _entropy(torch, left),
                                "right_entropy": _entropy(torch, right),
                                "kl_left_to_right": _kl(torch, left, right),
                                "kl_right_to_left": _kl(torch, right, left),
                            }
                        )
                        rows.append(row)
    return rows


def _attention_output_rows(torch, run_dir: Path) -> list[dict]:
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    rows = []
    for sample_id in sample_ids:
        base = _load_tensor(torch, metrics[(sample_id, "base")], run_dir)
        instruction = _load_tensor(torch, metrics[(sample_id, "instruction_only")], run_dir)
        lora = _load_tensor(torch, metrics[(sample_id, "lora_only")], run_dir)
        base_outputs = base["attention_outputs"]
        instruction_outputs = instruction["attention_outputs"]
        lora_outputs = lora["attention_outputs"]
        alignment = _same_target_alignment(base, instruction, lora)
        layer_count = _same_prefix_shape("attention_outputs", base_outputs, instruction_outputs, lora_outputs)
        if len({base_outputs.shape[1], instruction_outputs.shape[1], lora_outputs.shape[1]}) != 1:
            raise ValueError(f"attention output head count mismatch for {sample_id}.")
        head_count = instruction_outputs.shape[1]
        pairs = [
            ("instruction_only_vs_lora_only", instruction, instruction_outputs, lora_outputs),
            ("lora_only_vs_base", lora, lora_outputs, base_outputs),
        ]
        for condition, left_tensor, left_outputs, right_outputs in pairs:
            for layer in range(layer_count):
                for head in range(head_count):
                    for token_index, target in enumerate(alignment):
                        row = _row_base(sample_id, left_tensor, condition, "attention_output", layer, token_index, target)
                        row["head"] = head
                        row["metric_definition"] = ATTENTION_OUTPUT_DEFINITION
                        row["alignment_strategy"] = "Target tokens are matched by target_alignment; source keys are not decomposed."
                        row["cosine_similarity"] = _cosine(
                            torch,
                            left_outputs[layer, head, token_index],
                            right_outputs[layer, head, token_index],
                        )
                        rows.append(row)
    return rows


def _base_comparison_rows(torch, run_dir: Path) -> list[dict]:
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    rows = []
    for sample_id in sample_ids:
        base = _load_tensor(torch, metrics[(sample_id, "base")], run_dir)
        base_hidden = base["hidden_states"]
        for condition in ("instruction_only", "lora_only"):
            tensor = _load_tensor(torch, metrics[(sample_id, condition)], run_dir)
            alignment = _same_target_alignment(base, tensor)
            hidden = tensor["hidden_states"]
            layer_count = _same_prefix_shape("hidden_states", base_hidden, hidden)
            for layer in range(layer_count):
                for token_index, target in enumerate(alignment):
                    row = _row_base(sample_id, tensor, f"{condition}_vs_base", "state_vs_base", layer, token_index, target)
                    row["cosine_similarity"] = _cosine(
                        torch,
                        hidden[layer, token_index],
                        base_hidden[layer, token_index],
                    )
                    rows.append(row)
    return rows


def _state_rows(torch, config: VisualizeConfig) -> list[dict]:
    if config.left_run is None or config.right_run is None:
        raise ValueError("--left-run and --right-run are required for state mode.")
    left_metrics = _metrics_by_key(config.left_run)
    right_metrics = _metrics_by_key(config.right_run)
    wanted = set(config.conditions)
    rows = []
    for sample_id, condition in sorted(set(left_metrics) & set(right_metrics)):
        if wanted and condition not in wanted:
            continue
        left = _load_tensor(torch, left_metrics[(sample_id, condition)], config.left_run)
        right = _load_tensor(torch, right_metrics[(sample_id, condition)], config.right_run)
        alignment = _same_target_alignment(left, right)
        left_hidden = left["hidden_states"]
        right_hidden = right["hidden_states"]
        layer_count = _same_prefix_shape("hidden_states", left_hidden, right_hidden)
        for layer in range(layer_count):
            for token_index, target in enumerate(alignment):
                row = _row_base(sample_id, left, condition, "state", layer, token_index, target)
                row["cosine_similarity"] = _cosine(
                    torch, left_hidden[layer, token_index], right_hidden[layer, token_index]
                )
                rows.append(row)
    return rows


def _delta_rows(torch, config: VisualizeConfig) -> list[dict]:
    if config.left_run is None or config.right_run is None:
        raise ValueError("--left-run and --right-run are required for delta mode.")
    left_metrics = _metrics_by_key(config.left_run)
    right_metrics = _metrics_by_key(config.right_run)
    sample_ids = {sample_id for sample_id, condition in left_metrics if condition == "base"}
    sample_ids &= {sample_id for sample_id, condition in right_metrics if condition == "base"}
    wanted = set(config.conditions)
    rows = []
    for sample_id in sorted(sample_ids):
        left_base_tensor = _load_tensor(torch, left_metrics[(sample_id, "base")], config.left_run)
        right_base_tensor = _load_tensor(torch, right_metrics[(sample_id, "base")], config.right_run)
        left_base = left_base_tensor["hidden_states"]
        right_base = right_base_tensor["hidden_states"]
        conditions = sorted(
            condition
            for sid, condition in set(left_metrics) & set(right_metrics)
            if sid == sample_id and condition != "base" and (not wanted or condition in wanted)
        )
        for condition in conditions:
            left = _load_tensor(torch, left_metrics[(sample_id, condition)], config.left_run)
            right = _load_tensor(torch, right_metrics[(sample_id, condition)], config.right_run)
            alignment = _same_target_alignment(left_base_tensor, right_base_tensor, left, right)
            left_hidden = left["hidden_states"]
            right_hidden = right["hidden_states"]
            layer_count = _same_prefix_shape("hidden_states", left_hidden, right_hidden, left_base, right_base)
            for layer in range(layer_count):
                for token_index, target in enumerate(alignment):
                    row = _row_base(sample_id, left, condition, "delta_vs_base", layer, token_index, target)
                    row["cosine_similarity"] = _cosine(
                        torch,
                        left_hidden[layer, token_index] - left_base[layer, token_index],
                        right_hidden[layer, token_index] - right_base[layer, token_index],
                    )
                    rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    base_fields = [
        "sample_id",
        "task_id",
        "condition",
        "mode",
        "layer",
        "head",
        "token_index",
        "left_token_id",
        "right_token_id",
        "cosine_similarity",
    ]
    fields = base_fields + sorted({key for row in rows for key in row} - set(base_fields))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plain_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_rows(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        value = float(row["cosine_similarity"])
        groups[("layer", str(row["layer"]))].append(value)
        if "head" in row:
            groups[("head", str(row["head"]))].append(value)
            groups[("layer_head", f"{row['layer']}/{row['head']}")].append(value)
        groups[("token_index", str(row["token_index"]))].append(value)
        groups[("sample_id", str(row["sample_id"]))].append(value)
        groups[("task_id", str(row["task_id"]))].append(value)
    return [
        {
            "aggregate": aggregate,
            "key": key,
            "count": len(values),
            "mean_cosine_similarity": mean(values),
            "min_cosine_similarity": min(values),
            "max_cosine_similarity": max(values),
        }
        for (aggregate, key), values in sorted(groups.items())
    ]


def _write_aggregate_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "aggregate",
        "key",
        "count",
        "mean_cosine_similarity",
        "min_cosine_similarity",
        "max_cosine_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quality_rows(run_dir: Path, label: str) -> list[dict]:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _read_jsonl(metrics_path):
        grouped[row["condition"]].append(row)
    rows = []
    for condition, values in sorted(grouped.items()):
        rows.append(
            {
                "run": label,
                "condition": condition,
                "samples": len(values),
                "mean_loss": mean(float(row["loss"]) for row in values),
                "mean_token_accuracy": mean(float(row["token_accuracy"]) for row in values),
                "mean_sequence_accuracy": mean(float(row["sequence_accuracy"]) for row in values),
                "mean_target_tokens": mean(float(row["target_tokens"]) for row in values),
            }
        )
    return rows


def _write_quality_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "run",
        "condition",
        "samples",
        "mean_loss",
        "mean_token_accuracy",
        "mean_sequence_accuracy",
        "mean_target_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _layer_distribution_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row["layer"]))].append(float(row["cosine_similarity"]))
    return [
        {
            "condition": condition,
            "layer": layer,
            "count": len(values),
            "min_cosine_similarity": min(values),
            "q1_cosine_similarity": _quantile(values, 0.25),
            "q2_cosine_similarity": _quantile(values, 0.50),
            "q3_cosine_similarity": _quantile(values, 0.75),
            "max_cosine_similarity": max(values),
            "mean_cosine_similarity": mean(values),
        }
        for (condition, layer), values in sorted(grouped.items())
    ]


def _metric_box_rows(rows: list[dict], metric: str) -> list[dict]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if metric in row:
            grouped[(str(row["condition"]), int(row["layer"]))].append(float(row[metric]))
    return [
        {
            "metric": metric,
            "condition": condition,
            "layer": layer,
            "count": len(values),
            "min": min(values),
            "q1": _quantile(values, 0.25),
            "q2": _quantile(values, 0.50),
            "q3": _quantile(values, 0.75),
            "max": max(values),
            "mean": mean(values),
        }
        for (condition, layer), values in sorted(grouped.items())
    ]


def _cell(value: float) -> str:
    clamped = max(-1.0, min(1.0, value))
    hue = 8 + (clamped + 1.0) * 58
    return (
        f'<td style="background:hsl({hue:.1f} 78% 48%);color:#111" '
        f'title="{value:.4f}">{value:.2f}</td>'
    )


def _layer_distribution_chart(rows: list[dict]) -> str:
    if not rows:
        return ""
    width, height = 820, 320
    left, top, right, bottom = 54, 28, 180, 38
    chart_width = width - left - right
    chart_height = height - top - bottom
    layers = sorted({int(row["layer"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    values = [
        float(row[key])
        for row in rows
        for key in ("min_cosine_similarity", "max_cosine_similarity")
    ]
    min_layer, max_layer = min(layers), max(layers)
    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    value_pad = (max_value - min_value) * 0.08
    min_value -= value_pad
    max_value += value_pad
    group_width = chart_width / max(1, len(layers))
    bar_width = max(3.0, min(14.0, group_width / max(1, len(conditions)) * 0.7))

    def x(layer: int, condition_index: int) -> float:
        layer_index = layers.index(layer)
        start = left + group_width * layer_index + group_width / 2
        offset = (condition_index - (len(conditions) - 1) / 2) * (bar_width + 2)
        return start + offset

    def y(value: float) -> float:
        return top + chart_height * (1.0 - ((value - min_value) / (max_value - min_value)))

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    bars = []
    for row in rows:
        condition = str(row["condition"])
        color = colors[conditions.index(condition) % len(colors)]
        center = x(int(row["layer"]), conditions.index(condition))
        y_min = y(float(row["min_cosine_similarity"]))
        y_q1 = y(float(row["q1_cosine_similarity"]))
        y_q2 = y(float(row["q2_cosine_similarity"]))
        y_q3 = y(float(row["q3_cosine_similarity"]))
        y_max = y(float(row["max_cosine_similarity"]))
        title = (
            f"{condition} layer {row['layer']}: "
            f"min {row['min_cosine_similarity']:.4f}, q1 {row['q1_cosine_similarity']:.4f}, "
            f"q2 {row['q2_cosine_similarity']:.4f}, q3 {row['q3_cosine_similarity']:.4f}, "
            f"max {row['max_cosine_similarity']:.4f}"
        )
        bars.append(
            f'<g><title>{html.escape(title)}</title>'
            f'<line x1="{center:.1f}" y1="{y_max:.1f}" x2="{center:.1f}" y2="{y_min:.1f}" stroke="{color}" stroke-width="1.5"/>'
            f'<rect x="{center - bar_width / 2:.1f}" y="{min(y_q1, y_q3):.1f}" width="{bar_width:.1f}" height="{abs(y_q3 - y_q1):.1f}" fill="{color}" opacity="0.35"/>'
            f'<line x1="{center - bar_width / 2:.1f}" y1="{y_q2:.1f}" x2="{center + bar_width / 2:.1f}" y2="{y_q2:.1f}" stroke="{color}" stroke-width="2.5"/>'
            "</g>"
        )
    legend = []
    for index, condition in enumerate(conditions):
        color = colors[index % len(colors)]
        legend_y = top + 18 * index
        legend.append(
            f'<rect x="{width - right + 22}" y="{legend_y - 8}" width="22" height="10" fill="{color}" opacity="0.35"/>'
            f'<text x="{width - right + 50}" y="{legend_y + 2}" font-size="11">{html.escape(condition)}</text>'
        )
    tick_step = max(1, len(layers) // 10)
    x_ticks = "\n".join(
        f'<text x="{x(layer, 0):.1f}" y="{height - 12}" font-size="10" text-anchor="middle">{layer}</text>'
        for layer in layers[::tick_step]
    )
    return f"""
<h2>Layer Distribution</h2>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Layer cosine similarity min quartile median and max chart">
  <rect width="{width}" height="{height}" fill="white"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#555"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#555"/>
  <text x="8" y="{top + 4}" font-size="10">{max_value:.2f}</text>
  <text x="8" y="{height - bottom}" font-size="10">{min_value:.2f}</text>
  {''.join(bars)}
  {''.join(legend)}
  {x_ticks}
</svg>
"""


def _metric_box_chart(rows: list[dict], title: str) -> str:
    if not rows:
        return ""
    width, height = 820, 300
    left, top, right, bottom = 54, 28, 180, 38
    chart_width = width - left - right
    chart_height = height - top - bottom
    layers = sorted({int(row["layer"]) for row in rows})
    conditions = sorted({str(row["condition"]) for row in rows})
    values = [float(row[key]) for row in rows for key in ("min", "max")]
    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    value_pad = (max_value - min_value) * 0.08
    min_value -= value_pad
    max_value += value_pad
    group_width = chart_width / max(1, len(layers))
    bar_width = max(3.0, min(14.0, group_width / max(1, len(conditions)) * 0.7))

    def x(layer: int, condition_index: int) -> float:
        layer_index = layers.index(layer)
        start = left + group_width * layer_index + group_width / 2
        offset = (condition_index - (len(conditions) - 1) / 2) * (bar_width + 2)
        return start + offset

    def y(value: float) -> float:
        return top + chart_height * (1.0 - ((value - min_value) / (max_value - min_value)))

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    boxes = []
    for row in rows:
        condition = str(row["condition"])
        color = colors[conditions.index(condition) % len(colors)]
        center = x(int(row["layer"]), conditions.index(condition))
        y_min = y(float(row["min"]))
        y_q1 = y(float(row["q1"]))
        y_q2 = y(float(row["q2"]))
        y_q3 = y(float(row["q3"]))
        y_max = y(float(row["max"]))
        hover = (
            f"{row['metric']} {condition} layer {row['layer']}: "
            f"min {row['min']:.4f}, q1 {row['q1']:.4f}, q2 {row['q2']:.4f}, "
            f"q3 {row['q3']:.4f}, max {row['max']:.4f}"
        )
        boxes.append(
            f'<g><title>{html.escape(hover)}</title>'
            f'<line x1="{center:.1f}" y1="{y_max:.1f}" x2="{center:.1f}" y2="{y_min:.1f}" stroke="{color}" stroke-width="1.5"/>'
            f'<rect x="{center - bar_width / 2:.1f}" y="{min(y_q1, y_q3):.1f}" width="{bar_width:.1f}" height="{max(1.0, abs(y_q3 - y_q1)):.1f}" fill="{color}" opacity="0.35"/>'
            f'<line x1="{center - bar_width / 2:.1f}" y1="{y_q2:.1f}" x2="{center + bar_width / 2:.1f}" y2="{y_q2:.1f}" stroke="{color}" stroke-width="2.5"/>'
            "</g>"
        )
    legend = []
    for index, condition in enumerate(conditions):
        color = colors[index % len(colors)]
        legend_y = top + 18 * index
        legend.append(
            f'<rect x="{width - right + 22}" y="{legend_y - 8}" width="22" height="10" fill="{color}" opacity="0.35"/>'
            f'<text x="{width - right + 50}" y="{legend_y + 2}" font-size="11">{html.escape(condition)}</text>'
        )
    tick_step = max(1, len(layers) // 10)
    x_ticks = "\n".join(
        f'<text x="{x(layer, 0):.1f}" y="{height - 12}" font-size="10" text-anchor="middle">{layer}</text>'
        for layer in layers[::tick_step]
    )
    return f"""
<h2>{html.escape(title)}</h2>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)} box plot">
  <rect width="{width}" height="{height}" fill="white"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#555"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#555"/>
  <text x="8" y="{top + 4}" font-size="10">{max_value:.2f}</text>
  <text x="8" y="{height - bottom}" font-size="10">{min_value:.2f}</text>
  {''.join(boxes)}
  {''.join(legend)}
  {x_ticks}
</svg>
"""


def _layer_mean_line_chart(rows: list[dict]) -> str:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), int(row["layer"]))].append(float(row["cosine_similarity"]))
    if not grouped:
        return ""
    width, height = 820, 300
    left, top, right, bottom = 54, 28, 180, 38
    chart_width = width - left - right
    chart_height = height - top - bottom
    series = {
        condition: [(layer, mean(values)) for (cond, layer), values in grouped.items() if cond == condition]
        for condition in sorted({condition for condition, _ in grouped})
    }
    layers = [layer for points in series.values() for layer, _ in points]
    values = [value for points in series.values() for _, value in points]
    min_layer, max_layer = min(layers), max(layers)
    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    value_pad = (max_value - min_value) * 0.08
    min_value -= value_pad
    max_value += value_pad

    def x(layer: int) -> float:
        denom = max(1, max_layer - min_layer)
        return left + chart_width * ((layer - min_layer) / denom)

    def y(value: float) -> float:
        return top + chart_height * (1.0 - ((value - min_value) / (max_value - min_value)))

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed"]
    lines = []
    legend = []
    for index, (condition, points) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = sorted(points)
        polyline = " ".join(f"{x(layer):.1f},{y(value):.1f}" for layer, value in points)
        markers = "\n".join(
            f'<circle cx="{x(layer):.1f}" cy="{y(value):.1f}" r="3"><title>{html.escape(condition)} layer {layer}: {value:.4f}</title></circle>'
            for layer, value in points
        )
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/><g fill="{color}">{markers}</g>')
        legend_y = top + 18 * index
        legend.append(
            f'<line x1="{width - right + 22}" y1="{legend_y}" x2="{width - right + 44}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>'
            f'<text x="{width - right + 50}" y="{legend_y + 4}" font-size="11">{html.escape(condition)}</text>'
        )
    tick_layers = sorted(set(layers))
    tick_step = max(1, len(tick_layers) // 10)
    x_ticks = "\n".join(
        f'<text x="{x(layer):.1f}" y="{height - 12}" font-size="10" text-anchor="middle">{layer}</text>'
        for layer in tick_layers[::tick_step]
    )
    return f"""
<h2>Layer Mean Similarity</h2>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="Layer mean cosine similarity line chart">
  <rect width="{width}" height="{height}" fill="white"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#555"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#555"/>
  <text x="8" y="{top + 4}" font-size="10">{max_value:.2f}</text>
  <text x="8" y="{height - bottom}" font-size="10">{min_value:.2f}</text>
  {''.join(lines)}
  {''.join(legend)}
  {x_ticks}
</svg>
"""


def _summary_table(title: str, rows: list[dict], fields: list[str]) -> str:
    if not rows:
        return f"<h2>{html.escape(title)}</h2><p>No rows.</p>"
    header = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row[field]
            if isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<h2>{html.escape(title)}</h2><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _run_model_name(run_dir: Path | None) -> str | None:
    if run_dir is None:
        return None
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    return json.loads(config_path.read_text(encoding="utf-8")).get("model_name")


def _load_tokenizer(model_name: str | None):
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    return AutoTokenizer.from_pretrained(model_name)


def _token_text(tokenizer, token_id) -> str:
    if tokenizer is None or token_id == "":
        return ""
    try:
        return tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
    except Exception:
        return ""


def _sample_sections(rows: list[dict], tokenizer, title: str = "Per-Sample Matrices") -> str:
    grouped: dict[tuple[str, str, int, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["condition"]), str(row["sample_id"]), int(row["layer"]), int(row["token_index"]))].append(row)
    sections = []
    for condition in sorted({key[0] for key in grouped}):
        for sample_id in sorted({sample for cond, sample, _, _ in grouped if cond == condition}):
            layers = sorted({layer for cond, sample, layer, _ in grouped if cond == condition and sample == sample_id})
            tokens = sorted({token for cond, sample, _, token in grouped if cond == condition and sample == sample_id})
            token_headers = []
            for token in tokens:
                first = grouped[(condition, sample_id, layers[0], token)][0]
                token_id = first["left_token_id"] or first["right_token_id"]
                label = _token_text(tokenizer, token_id)
                token_headers.append(
                    "<th>"
                    f"tok {token}<br>"
                    f"<code>{html.escape(str(token_id))}</code><br>"
                    f"<span>{html.escape(repr(label))}</span>"
                    "</th>"
                )
            body = []
            for layer in layers:
                cells = "".join(
                    _cell(mean(float(row["cosine_similarity"]) for row in grouped[(condition, sample_id, layer, token)]))
                    for token in tokens
                )
                body.append(f"<tr><th>layer {layer}</th>{cells}</tr>")
            sections.append(
                f"<h3>{html.escape(condition)} / {html.escape(sample_id)}</h3>"
                f"<table><thead><tr><th></th>{''.join(token_headers)}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table>"
            )
    return f"<h2>{html.escape(title)}</h2>" + "".join(sections) if sections else ""


def _write_html(
    path: Path,
    config: VisualizeConfig,
    rows: list[dict],
    aggregate_rows: list[dict],
    layer_distribution_rows: list[dict],
    cka_box_rows: list[dict],
    logit_box_rows: list[dict],
    quality_rows: list[dict],
    base_rows: list[dict] | None = None,
) -> None:
    grouped: dict[tuple[str, int, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], int(row["layer"]), int(row["token_index"]))].append(
            float(row["cosine_similarity"])
        )
    tokenizer = _load_tokenizer(_run_model_name(config.run) or config.left_model)
    notes = {
        "attention": (ATTENTION_PATTERN_DEFINITION, ATTENTION_ALIGNMENT_STRATEGY),
        "attention_output": (
            ATTENTION_OUTPUT_DEFINITION,
            "Target tokens are matched by target_alignment; source keys are not decomposed.",
        ),
    }.get(config.mode)
    conditions = sorted({key[0] for key in grouped})
    sections = []
    for condition in conditions:
        layers = sorted({layer for cond, layer, _ in grouped if cond == condition})
        tokens = sorted({token for cond, _, token in grouped if cond == condition})
        header = "".join(f"<th>tok {token}</th>" for token in tokens)
        body = []
        for layer in layers:
            cells = "".join(_cell(mean(grouped[(condition, layer, token)])) for token in tokens)
            body.append(f"<tr><th>layer {layer}</th>{cells}</tr>")
        sections.append(
            f"<h2>{html.escape(condition)}</h2>"
            f"<table><thead><tr><th></th>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
        )
    page = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Token State Similarity</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; color: #202124; }}
table {{ border-collapse: collapse; margin: 12px 0 28px; }}
th, td {{ border: 1px solid #d7d7d7; padding: 6px 8px; text-align: right; font-size: 12px; }}
th {{ background: #f4f4f4; font-weight: 600; }}
.meta {{ color: #555; line-height: 1.5; }}
</style>
<h1>Token State Similarity</h1>
<div class="meta">
run: {html.escape(str(config.run or ""))}<br>
left: {html.escape(config.left_model)} ({html.escape(str(config.left_run or ""))})<br>
right: {html.escape(config.right_model)} ({html.escape(str(config.right_run or ""))})<br>
mode: {html.escape(config.mode)}; rows: {len(rows)}
{('<br>metric: ' + html.escape(notes[0]) + '<br>alignment: ' + html.escape(notes[1])) if notes else ''}
</div>
{_layer_distribution_chart(layer_distribution_rows)}
{_metric_box_chart(cka_box_rows, "CKA Similarity Distribution")}
{_metric_box_chart(logit_box_rows, "Logit Distribution Cosine Distribution")}
{_layer_mean_line_chart(rows)}
{''.join(sections) if sections else '<p>No common tensor rows found.</p>'}
{_sample_sections(rows, tokenizer)}
{_sample_sections(base_rows or [], tokenizer, "Instruction/Base And LoRA/Base Matrices")}
{_summary_table("Layer Distribution", layer_distribution_rows, ["condition", "layer", "count", "min_cosine_similarity", "q1_cosine_similarity", "q2_cosine_similarity", "q3_cosine_similarity", "max_cosine_similarity", "mean_cosine_similarity"])}
{_summary_table("CKA Similarity Distribution", cka_box_rows, ["metric", "condition", "layer", "count", "min", "q1", "q2", "q3", "max", "mean"])}
{_summary_table("Logit Distribution Cosine Distribution", logit_box_rows, ["metric", "condition", "layer", "count", "min", "q1", "q2", "q3", "max", "mean"])}
{_summary_table("Aggregates", aggregate_rows, ["aggregate", "key", "count", "mean_cosine_similarity", "min_cosine_similarity", "max_cosine_similarity"])}
{_summary_table("Adapter Quality", quality_rows, ["run", "condition", "samples", "mean_loss", "mean_token_accuracy", "mean_sequence_accuracy", "mean_target_tokens"])}
</html>
"""
    path.write_text(page, encoding="utf-8")


def _patch_run_dirs(path: Path) -> list[Path]:
    if (path / "generations.jsonl").exists():
        return [path]
    return sorted(child for child in path.iterdir() if (child / "generations.jsonl").exists())


def _patch_label(run_dir: Path) -> str:
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        source = config.get("source_condition", "source")
        target = config.get("target_condition", "target")
        layer = config.get("layer", run_dir.name)
        span = config.get("patch_span", "span")
        return f"{source} -> {target}, layer {layer}, {span}"
    return run_dir.name


def _patch_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    return json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}


def _generation_losses(row: dict) -> list[float]:
    if "token_losses" not in row:
        raise ValueError("generations.jsonl is missing token_losses; rerun RQ3 with the current patching code.")
    if "loss_target_token_ids" not in row:
        raise ValueError("generations.jsonl is missing loss_target_token_ids; rerun RQ3 with the current loss target.")
    return [float(value) for value in row["token_losses"]]


def _patch_loss_rows(run_dir: Path) -> list[dict]:
    rows = []
    for row in _read_jsonl(run_dir / "generations.jsonl"):
        for token_index, loss in enumerate(_generation_losses(row)):
            pred_ids = row.get("pred_token_ids") or []
            target_ids = row["loss_target_token_ids"]
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "task_id": row["task_id"],
                    "layer": row.get("layer", ""),
                    "control": row.get("control", "source_to_target_patch" if row["patched"] else "unpatched"),
                    "source_condition": row.get("source_condition"),
                    "target_condition": row.get("target_condition"),
                    "patched": row["patched"],
                    "token_index": token_index,
                    "loss": loss,
                    "pred_token_id": pred_ids[token_index] if token_index < len(pred_ids) else "",
                    "target_token_id": target_ids[token_index] if token_index < len(target_ids) else "",
                    "stopped_on_eos": row.get("stopped_on_eos", False),
                    "target_text": row["target_text"],
                    "pred_text": row["pred_text"],
                }
            )
    return rows


def _patch_loss_aggregate(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["control"]), int(row["token_index"]))].append(float(row["loss"]))
    return [
        {
            "control": control,
            "token_index": token_index,
            "samples": len(values),
            "mean_loss": mean(values),
        }
        for (control, token_index), values in sorted(grouped.items())
    ]


def _row_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _rq3_metric_rows(run_dir: Path) -> list[dict]:
    config = _patch_config(run_dir)
    base = {
        "run": run_dir.name,
        "source_condition": config.get("source_condition", ""),
        "target_condition": config.get("target_condition", ""),
        "layer": config.get("layer", ""),
        "patch_span": config.get("patch_span", ""),
    }
    rows = []
    for row in _read_jsonl(run_dir / "metrics.jsonl") if (run_dir / "metrics.jsonl").exists() else []:
        control = row.get("control", "source_to_target_patch" if row.get("patched") else "unpatched")
        for metric in ("loss", "token_accuracy", "sequence_accuracy", "task_semantic_correct"):
            if metric in row:
                rows.append(
                    {
                        **base,
                        "sample_id": row["sample_id"],
                        "task_id": row["task_id"],
                        "control": control,
                        "scope": "teacher_forced",
                        "metric": metric,
                        "value": float(row[metric]),
                    }
                )
    for row in _read_jsonl(run_dir / "generations.jsonl"):
        control = row.get("control", "source_to_target_patch" if row.get("patched") else "unpatched")
        generated_loss = _row_mean([float(value) for value in row.get("token_losses", [])])
        metrics = {
            "generation_loss": generated_loss,
            "token_accuracy": row.get("token_accuracy"),
            "sequence_accuracy": row.get("sequence_accuracy"),
            "task_semantic_correct": row.get("task_semantic_correct"),
        }
        for metric, value in metrics.items():
            if value is not None:
                rows.append(
                    {
                        **base,
                        "sample_id": row["sample_id"],
                        "task_id": row["task_id"],
                        "control": control,
                        "scope": "generation",
                        "metric": metric,
                        "value": float(value),
                    }
                )
    return rows


def _rq3_summary_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    meta = {}
    for row in rows:
        key = (
            row["run"],
            row["source_condition"],
            row["target_condition"],
            row["layer"],
            row["patch_span"],
            row["control"],
            row["scope"],
            row["metric"],
        )
        grouped[key].append(float(row["value"]))
        meta[key] = row
    result = []
    for key, values in sorted(grouped.items()):
        row = meta[key]
        result.append(
            {
                "run": row["run"],
                "source_condition": row["source_condition"],
                "target_condition": row["target_condition"],
                "layer": row["layer"],
                "patch_span": row["patch_span"],
                "control": row["control"],
                "scope": row["scope"],
                "metric": row["metric"],
                "samples": len(values),
                "mean": mean(values),
                "min": min(values),
                "q1": _quantile(values, 0.25),
                "q2": _quantile(values, 0.50),
                "q3": _quantile(values, 0.75),
                "max": max(values),
            }
        )
    return result


def _rq3_confusion_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "confusion_matrix.json"
    if not path.exists():
        return []
    config = _patch_config(run_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for scope, values in data.items():
        for row in values:
            rows.append(
                {
                    "run": run_dir.name,
                    "source_condition": config.get("source_condition", ""),
                    "target_condition": config.get("target_condition", ""),
                    "layer": config.get("layer", ""),
                    "patch_span": config.get("patch_span", ""),
                    "scope": scope,
                    **row,
                }
            )
    return rows


def _rq3_box_rows(summary_rows: list[dict], scope: str, metric: str) -> list[dict]:
    return [
        {
            "metric": metric,
            "condition": row["control"],
            "layer": row["layer"],
            "count": row["samples"],
            "min": row["min"],
            "q1": row["q1"],
            "q2": row["q2"],
            "q3": row["q3"],
            "max": row["max"],
            "mean": row["mean"],
        }
        for row in summary_rows
        if row["scope"] == scope and row["metric"] == metric and str(row["layer"]) != ""
    ]


def _rq3_chart_filename(scope: str, metric: str) -> str:
    return f"rq3_{scope}_{metric}_by_pair.png"


def _rq3_chart_points(rows: list[dict], scope: str, metric: str) -> list[dict]:
    return [
        row
        for row in rows
        if row["scope"] == scope and row["metric"] == metric and str(row["layer"]) != ""
    ]


def _rq3_line_chart(rows: list[dict], scope: str, metric: str, title: str) -> str:
    points = _rq3_chart_points(rows, scope, metric)
    if not points:
        return ""
    width = 1120
    left, top, right, bottom = 64, 30, 420, 38
    series_labels = sorted(
        {
            f"{row['source_condition']} -> {row['target_condition']} | {row['control']}"
            for row in points
        }
    )
    height = max(320, top + bottom + 18 * len(series_labels) + 24)
    chart_width = width - left - right
    chart_height = height - top - bottom
    layers = sorted({int(row["layer"]) for row in points})
    values = [float(row["mean"]) for row in points]
    min_value, max_value = min(values), max(values)
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    value_pad = (max_value - min_value) * 0.08
    min_value -= value_pad
    max_value += value_pad
    min_layer, max_layer = min(layers), max(layers)
    span = max(1, max_layer - min_layer)

    def x(layer: int) -> float:
        return left + chart_width * ((layer - min_layer) / span)

    def y(value: float) -> float:
        return top + chart_height * (1.0 - ((value - min_value) / (max_value - min_value)))

    tick_count = 6
    y_ticks = [min_value + (max_value - min_value) * index / (tick_count - 1) for index in range(tick_count)]
    y_tick_marks = "".join(
        f'<line x1="{left - 4}" y1="{y(value):.1f}" x2="{width - right}" y2="{y(value):.1f}" stroke="#e5e7eb"/>'
        f'<text x="{left - 8}" y="{y(value) + 3:.1f}" font-size="10" text-anchor="end">{value:.2f}</text>'
        for value in y_ticks
    )
    colors = [
        "#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2",
        "#be123c", "#4f46e5", "#15803d", "#a16207", "#0f766e", "#7e22ce",
        "#ea580c", "#0369a1", "#65a30d", "#b91c1c",
    ]
    lines = []
    legend = []
    for index, label in enumerate(series_labels):
        color = colors[index % len(colors)]
        control_points = sorted(
            (
                (int(row["layer"]), float(row["mean"]))
                for row in points
                if f"{row['source_condition']} -> {row['target_condition']} | {row['control']}" == label
            )
        )
        polyline = _polyline([(x(layer), y(value)) for layer, value in control_points])
        markers = "".join(
            f'<circle cx="{x(layer):.1f}" cy="{y(value):.1f}" r="3"><title>{html.escape(label)} layer {layer}: {value:.4f}</title></circle>'
            for layer, value in control_points
        )
        lines.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/><g fill="{color}">{markers}</g>')
        legend_y = top + 18 * index
        legend.append(
            f'<line x1="{width - right + 22}" y1="{legend_y}" x2="{width - right + 44}" y2="{legend_y}" stroke="{color}" stroke-width="2.5"/>'
            f'<text x="{width - right + 50}" y="{legend_y + 4}" font-size="10">{html.escape(label)}</text>'
        )
    x_ticks = "".join(
        f'<text x="{x(layer):.1f}" y="{height - 12}" font-size="10" text-anchor="middle">{layer}</text>'
        for layer in layers
    )
    return f"""
<h2>{html.escape(title)}</h2>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)} line chart">
  <rect width="{width}" height="{height}" fill="white"/>
  {y_tick_marks}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#555"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#555"/>
  {''.join(lines)}
  {''.join(legend)}
  {x_ticks}
</svg>
"""


def _rq3_matplotlib_charts(summary_rows: list[dict], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    written = []
    for scope, metric, title in RQ3_CHART_SPECS:
        points = _rq3_chart_points(summary_rows, scope, metric)
        if not points:
            continue
        pairs = sorted({f"{row['source_condition']} -> {row['target_condition']}" for row in points})
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), squeeze=False)
        axes_flat = list(axes.ravel())
        for ax, pair in zip(axes_flat, pairs):
            pair_points = [
                row
                for row in points
                if f"{row['source_condition']} -> {row['target_condition']}" == pair
            ]
            controls = sorted({str(row["control"]) for row in pair_points})
            local_values = [float(row["mean"]) for row in pair_points]
            layers = sorted({int(row["layer"]) for row in pair_points})
            for control in controls:
                control_points = sorted(
                    (
                        (int(row["layer"]), float(row["mean"]))
                        for row in pair_points
                        if str(row["control"]) == control
                    )
                )
                ax.plot(
                    [layer for layer, _ in control_points],
                    [value for _, value in control_points],
                    marker="o",
                    linewidth=1.8,
                    label=control,
                )
            low, high = min(local_values), max(local_values)
            pad = 0.5 if low == high else (high - low) * 0.18
            ax.set_ylim(low - pad, high + pad)
            ax.set_title(pair)
            ax.set_xticks(layers)
            ax.set_xlabel("layer")
            ax.set_ylabel(metric)
            ax.grid(axis="y", alpha=0.35)
            ax.legend(fontsize=8)
        for ax in axes_flat[len(pairs):]:
            ax.axis("off")
        fig.suptitle(f"{title} (pair-local y scale)")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = output_dir / _rq3_chart_filename(scope, metric)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)
    return written


def _rq3_summary_html(summary_rows: list[dict], confusion_rows: list[dict]) -> str:
    sections = []
    for scope, metric, title in RQ3_CHART_SPECS:
        image_name = _rq3_chart_filename(scope, metric)
        sections.append(
            f'<h2>{html.escape(title)}</h2>\n'
            f'<img src="{html.escape(image_name)}" alt="{html.escape(title)}" class="rq3-chart">'
        )
        sections.append(_metric_box_chart(_rq3_box_rows(summary_rows, scope, metric), title.replace("Mean By Layer", "Distribution")))
    return (
        """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>RQ3 Summary</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#202124} svg{display:block;margin:12px 0 28px}.rq3-chart{display:block;max-width:100%;margin:12px 0 28px} table{border-collapse:collapse} th,td{border:1px solid #ddd;padding:4px 6px;font-size:12px}</style>
<h1>RQ3 Summary</h1>
<p>Lines show per-layer means. Boxplots show per-sample distributions grouped by control.</p>
"""
        + "\n".join(sections)
        + _summary_table("Confusion Matrix", confusion_rows, list(confusion_rows[0]) if confusion_rows else [])
        + "\n</html>\n"
    )


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def _patch_loss_svg(title: str, rows: list[dict]) -> str:
    width, height = 760, 300
    left, top, right, bottom = 48, 28, 16, 36
    chart_width = width - left - right
    chart_height = height - top - bottom
    tokens = sorted({int(row["token_index"]) for row in rows})
    if not tokens:
        return ""
    max_token = max(tokens) or 1
    max_loss = max(1.0, max(float(row["mean_loss"]) for row in rows))
    controls = sorted({str(row["control"]) for row in rows})
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f97316", "#0891b2"]

    def points(line_rows: list[dict]) -> list[tuple[float, float]]:
        by_token = {int(row["token_index"]): float(row["mean_loss"]) for row in line_rows}
        return [
            (
                left + chart_width * (token / max_token),
                top + chart_height * (1.0 - (by_token.get(token, 0.0) / max_loss)),
            )
            for token in tokens
        ]

    polylines = []
    legend = []
    for index, control in enumerate(controls):
        color = colors[index % len(colors)]
        line_rows = [row for row in rows if str(row["control"]) == control]
        polylines.append(f'<polyline points="{_polyline(points(line_rows))}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend.append(
            f'<text x="{width - 240}" y="{20 + 16 * index}" fill="{color}" font-size="11">{html.escape(control)}</text>'
        )
    x_ticks = "".join(
        f'<text x="{left + chart_width * (token / max_token):.1f}" y="{height - 10}" font-size="10" text-anchor="middle">{token}</text>'
        for token in tokens[:: max(1, len(tokens) // 10)]
    )
    return f"""
<h2>{html.escape(title)}</h2>
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img" aria-label="{html.escape(title)} token loss line chart">
  <rect width="{width}" height="{height}" fill="white"/>
  <text x="{left}" y="18" font-size="14" font-weight="600">{html.escape(title)}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#555"/>
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#555"/>
  <text x="8" y="{top + 4}" font-size="10">{max_loss:.2f}</text>
  <text x="8" y="{height - bottom}" font-size="10">0</text>
  {''.join(polylines)}
  {''.join(legend)}
  {x_ticks}
</svg>
"""


def visualize_patch_losses(config: VisualizeConfig) -> None:
    if config.run is None:
        raise ValueError("--run is required for patch_loss mode.")
    run_dirs = _patch_run_dirs(config.run)
    if not run_dirs:
        raise FileNotFoundError(f"No generations.jsonl found under {config.run}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    sections = []
    metric_rows = []
    confusion_rows = []
    for run_dir in run_dirs:
        rows = _patch_loss_rows(run_dir)
        aggregate = _patch_loss_aggregate(rows)
        label = _patch_label(run_dir)
        _write_plain_csv(config.output_dir / f"{run_dir.name}_token_loss.csv", rows)
        _write_plain_csv(config.output_dir / f"{run_dir.name}_token_loss_aggregate.csv", aggregate)
        sections.append(_patch_loss_svg(label, aggregate))
        metric_rows.extend(_rq3_metric_rows(run_dir))
        confusion_rows.extend(_rq3_confusion_rows(run_dir))
    summary_rows = _rq3_summary_rows(metric_rows)
    _write_plain_csv(config.output_dir / "rq3_metric_distribution.csv", metric_rows)
    _write_plain_csv(config.output_dir / "rq3_summary.csv", summary_rows)
    _write_plain_csv(config.output_dir / "rq3_confusion_matrix.csv", confusion_rows)
    _rq3_matplotlib_charts(summary_rows, config.output_dir)
    (config.output_dir / "rq3_summary.html").write_text(
        _rq3_summary_html(summary_rows, confusion_rows),
        encoding="utf-8",
    )
    html_path = config.output_dir / "patch_token_loss.html"
    html_path.write_text(
        """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>Patch Token Loss</title>
<style>body{font-family:system-ui,sans-serif;margin:24px;color:#202124} svg{display:block;margin:12px 0 28px}</style>
<h1>Patch Token Loss</h1>
<p>Loss is per-step cross entropy against target_text tokens. Short generations stop at EOS.</p>
"""
        + "\n".join(sections)
        + "\n</html>\n",
        encoding="utf-8",
    )


def visualize(config: VisualizeConfig) -> None:
    if config.mode == "patch_loss":
        visualize_patch_losses(config)
        return

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Visualization requires torch because collect.py stores .pt tensors.") from exc

    runs = [run for run in (config.run, config.left_run, config.right_run) if run is not None]
    for run in runs:
        if not (run / "metrics.jsonl").exists():
            raise FileNotFoundError(f"Missing {run / 'metrics.jsonl'}")

    if config.mode == "residual":
        if config.run is None:
            raise ValueError("--run is required for residual mode.")
        rows = _residual_rows(torch, config.run)
        base_rows = _base_comparison_rows(torch, config.run)
        if not rows:
            raise ValueError("Residual mode requires common base, instruction_only, and lora_only tensors.")
    elif config.mode == "attention":
        if config.run is None:
            raise ValueError("--run is required for attention mode.")
        rows = _attention_rows(torch, config.run)
        base_rows = []
        if not rows:
            raise ValueError("Attention mode requires common instruction_only and lora_only tensors.")
    elif config.mode == "attention_output":
        if config.run is None:
            raise ValueError("--run is required for attention_output mode.")
        rows = _attention_output_rows(torch, config.run)
        base_rows = []
        if not rows:
            raise ValueError("Attention output mode requires tensors collected with attention_outputs.")
    elif config.mode == "delta":
        rows = _delta_rows(torch, config)
        base_rows = []
    else:
        rows = _state_rows(torch, config)
        base_rows = []
    aggregate_rows = _aggregate_rows(rows)
    layer_distribution_rows = _layer_distribution_rows(rows)
    cka_box_rows = _metric_box_rows(rows, "cka_similarity")
    logit_box_rows = _metric_box_rows(rows, "logit_distribution_cosine")
    quality_rows = []
    if config.run is not None:
        quality_rows.extend(_quality_rows(config.run, "run"))
    if config.left_run is not None:
        quality_rows.extend(_quality_rows(config.left_run, "left"))
    if config.right_run is not None:
        quality_rows.extend(_quality_rows(config.right_run, "right"))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(config.output_dir / "token_similarity.csv", rows)
    if base_rows:
        _write_csv(config.output_dir / "base_comparison_similarity.csv", base_rows)
    _write_plain_csv(config.output_dir / "layer_distribution.csv", layer_distribution_rows)
    _write_plain_csv(config.output_dir / "cka_similarity_distribution.csv", cka_box_rows)
    _write_plain_csv(config.output_dir / "logit_distribution_cosine_distribution.csv", logit_box_rows)
    _write_aggregate_csv(config.output_dir / "aggregate_similarity.csv", aggregate_rows)
    _write_quality_csv(config.output_dir / "quality_summary.csv", quality_rows)
    _write_html(
        config.output_dir / "token_similarity.html",
        config,
        rows,
        aggregate_rows,
        layer_distribution_rows,
        cka_box_rows,
        logit_box_rows,
        quality_rows,
        base_rows,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize token-level state similarity from collect outputs.")
    parser.add_argument("--run", type=Path, help="One collect.py output directory for residual perturbation analysis.")
    parser.add_argument("--left-run", type=Path, help="First collect.py output directory for two-run comparisons.")
    parser.add_argument("--right-run", type=Path, help="Second collect.py output directory for two-run comparisons.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--left-model", default=DEFAULT_MODEL)
    parser.add_argument("--right-model", default=DEFAULT_MODEL)
    parser.add_argument("--mode", choices=("residual", "attention", "attention_output", "state", "delta", "patch_loss"), default=None)
    parser.add_argument("--condition", action="append", choices=("base", "instruction_only", "lora_only"), dest="conditions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualize(
        VisualizeConfig(
            run=args.run,
            left_run=args.left_run,
            right_run=args.right_run,
            output_dir=args.output_dir,
            left_model=args.left_model,
            right_model=args.right_model,
            mode=args.mode or ("residual" if args.run else "state"),
            conditions=tuple(args.conditions or ()),
        )
    )
    print(f"Wrote visualization to {args.output_dir}")


if __name__ == "__main__":
    main()
