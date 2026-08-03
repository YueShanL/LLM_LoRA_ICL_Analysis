"""Optional Jacobian-lens readouts for saved RQ1 residual states."""

from __future__ import annotations

import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from lora_instruction_analysis.model.collect import _torch_dtype
from lora_instruction_analysis.model.visualize import (
    _load_tensor,
    _metrics_by_key,
    _same_prefix_shape,
    _same_target_alignment,
)

INSTALL_GUIDANCE = (
    "J-lens readout requires a saved lens and the optional jlens dependency. "
    "Install with `pip install -e .[jlens]` after adding the jacobian-lens package."
)
PAIRS = (
    ("base", "instruction_only"),
    ("base", "lora_only"),
    ("instruction_only", "lora_only"),
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, quoting=csv.QUOTE_MINIMAL, escapechar="\\")
        writer.writeheader()
        writer.writerows(rows)


def _load_lens(torch, lens_path: Path):
    if not lens_path.exists():
        raise FileNotFoundError(f"Missing J-lens path: {lens_path}")
    try:
        import jlens  # type: ignore
    except ImportError:
        if not lens_path.is_file():
            raise RuntimeError(INSTALL_GUIDANCE)
        jlens = None
    if lens_path.is_file():
        raw = torch.load(lens_path, map_location="cpu", weights_only=True)
        if isinstance(raw, dict) and "J" in raw and jlens is not None:
            return jlens.JacobianLens.load(str(lens_path))
        return raw
    for name in ("lens.pt", "jlens.pt", "model.pt"):
        candidate = lens_path / name
        if candidate.exists():
            return _load_lens(torch, candidate)
    raise FileNotFoundError(f"No loadable lens file found under {lens_path}; expected lens.pt, jlens.pt, or model.pt.")


def _decode_token(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return str(token_id)
    return tokenizer.decode([int(token_id)])


def _load_tokenizer(model_name: str | None):
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    return AutoTokenizer.from_pretrained(model_name)


def _load_lens_model(torch, model_name: str | None, *, dtype: str = "bfloat16", device: str = "cuda"):
    if not model_name:
        return None
    try:
        import jlens  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(INSTALL_GUIDANCE) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    kwargs = {"torch_dtype": _torch_dtype(torch, dtype)}
    if device == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device != "auto":
        model.to(device)
    return jlens.HFLensModel(model, tokenizer)


def _lens_scores(torch, lens, hidden, *, layer: int | None = None, lens_model=None):
    if hasattr(lens, "transport"):
        if layer is None or lens_model is None:
            raise TypeError("JacobianLens readout requires layer and model_name.")
        return lens_model.unembed(lens.transport(hidden.float(), layer)).detach().cpu().float()
    if hasattr(lens, "readout"):
        result = lens.readout(hidden)
    elif callable(lens):
        result = lens(hidden)
    elif isinstance(lens, dict) and "unembed" in lens:
        result = hidden.float() @ lens["unembed"].float().T
    else:
        raise TypeError("Unsupported J-lens object; expected callable, .readout(), or {'unembed': tensor}.")
    if isinstance(result, dict):
        result = result["scores"] if "scores" in result else result.get("logits")
    if isinstance(result, tuple):
        result = result[-1]
    return result.detach().cpu().float()


def _topk(torch, lens, hidden, k: int, *, layer: int | None = None, lens_model=None) -> tuple[list[int], list[float]]:
    if hasattr(lens, "topk"):
        result = lens.topk(hidden, k)
        if isinstance(result, dict):
            return list(map(int, result["token_ids"])), list(map(float, result["scores"]))
        token_ids, scores = result
        return list(map(int, token_ids)), list(map(float, scores))
    scores = _lens_scores(torch, lens, hidden, layer=layer, lens_model=lens_model).flatten()
    values, indices = torch.topk(scores, min(k, scores.numel()))
    return [int(index) for index in indices.tolist()], [float(value) for value in values.tolist()]


def _jaccard(left: list[int], right: list[int]) -> float:
    left_set, right_set = set(left), set(right)
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _target_rank(token_ids: list[int], target_id: int) -> int:
    try:
        return token_ids.index(int(target_id)) + 1
    except ValueError:
        return 0


def _readout_layers(lens, layer_count: int) -> list[tuple[int, int]]:
    if hasattr(lens, "source_layers"):
        # collect.py stores hidden_states[0] as embeddings, then block outputs.
        return [(int(layer), int(layer) + 1) for layer in lens.source_layers if int(layer) + 1 < layer_count]
    return [(layer, layer) for layer in range(layer_count)]


def readout_rows(torch, run_dir: Path, lens, *, top_k: int, tokenizer=None, lens_model=None) -> list[dict]:
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
        layer_pairs = _readout_layers(lens, layer_count)
        for condition, tensor in tensors.items():
            hidden_states = tensor["hidden_states"]
            for lens_layer, hidden_layer in layer_pairs:
                batch_topk = None
                if hasattr(lens, "transport"):
                    logits = _lens_scores(
                        torch,
                        lens,
                        hidden_states[hidden_layer].float(),
                        layer=lens_layer,
                        lens_model=lens_model,
                    )
                    values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
                    batch_topk = (indices.tolist(), values.tolist())
                for token_index, target in enumerate(alignment):
                    if batch_topk is None:
                        token_ids, scores = _topk(
                            torch,
                            lens,
                            hidden_states[hidden_layer, token_index],
                            top_k,
                            layer=lens_layer,
                            lens_model=lens_model,
                        )
                    else:
                        token_ids = [int(token_id) for token_id in batch_topk[0][token_index]]
                        scores = [float(score) for score in batch_topk[1][token_index]]
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "task_id": tensor.get("task_id", ""),
                            "condition": condition,
                            "layer": lens_layer,
                            "hidden_state_layer": hidden_layer,
                            "token_index": token_index,
                            "alignment_key": target["alignment_key"],
                            "target_token_id": target["token_id"],
                            "target_token_text": _decode_token(tokenizer, target["token_id"]),
                            "top_k_token_ids": " ".join(map(str, token_ids)),
                            "top_k_token_texts": " || ".join(_decode_token(tokenizer, token_id) for token_id in token_ids),
                            "top_k_scores": " ".join(f"{score:.8g}" for score in scores),
                            "target_token_rank": _target_rank(token_ids, target["token_id"]),
                        }
                    )
    return rows


def pair_overlap_rows(rows: list[dict]) -> list[dict]:
    by_key = {(r["sample_id"], r["layer"], r["token_index"], r["condition"]): r for r in rows}
    output = []
    positions = sorted({(sample_id, layer, token_index) for sample_id, layer, token_index, _ in by_key})
    for sample_id, layer, token_index in positions:
        for left, right in PAIRS:
            left_row = by_key.get((sample_id, layer, token_index, left))
            right_row = by_key.get((sample_id, layer, token_index, right))
            if not left_row or not right_row:
                continue
            left_ids = [int(value) for value in str(left_row["top_k_token_ids"]).split()]
            right_ids = [int(value) for value in str(right_row["top_k_token_ids"]).split()]
            output.append(
                {
                    "sample_id": sample_id,
                    "task_id": left_row["task_id"],
                    "condition_pair": f"{left}_vs_{right}",
                    "layer": layer,
                    "token_index": token_index,
                    "alignment_key": left_row["alignment_key"],
                    "top_k_overlap": len(set(left_ids) & set(right_ids)),
                    "top_k_jaccard": _jaccard(left_ids, right_ids),
                    "left_target_token_rank": left_row["target_token_rank"],
                    "right_target_token_rank": right_row["target_token_rank"],
                }
            )
    return output


def layer_summary_rows(overlap_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in overlap_rows:
        grouped[(row["condition_pair"], int(row["layer"]))].append(row)
    return [
        {
            "condition_pair": pair,
            "layer": layer,
            "count": len(values),
            "mean_top_k_overlap": mean(float(row["top_k_overlap"]) for row in values),
            "mean_top_k_jaccard": mean(float(row["top_k_jaccard"]) for row in values),
            "mean_left_target_token_rank": mean(float(row["left_target_token_rank"]) for row in values),
            "mean_right_target_token_rank": mean(float(row["right_target_token_rank"]) for row in values),
        }
        for (pair, layer), values in sorted(grouped.items())
    ]


def write_html(path: Path, summary_rows: list[dict]) -> None:
    fields = list(summary_rows[0]) if summary_rows else []
    table = ""
    if fields:
        head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(row[field]))}</td>" for field in fields) + "</tr>"
            for row in summary_rows
        )
        table = f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
    path.write_text(
        "<!doctype html><meta charset=\"utf-8\"><title>J-lens Readouts</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:24px}table{border-collapse:collapse}"
        "td,th{border:1px solid #ddd;padding:4px 6px;font-size:12px}</style>"
        "<h1>J-lens Readouts</h1>" + table,
        encoding="utf-8",
    )


def run_jlens_readout(
    run_dir: Path,
    lens_path: Path,
    output_dir: Path,
    *,
    top_k: int = 20,
    model_name: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("J-lens readout requires torch because collect.py stores .pt tensors.") from exc
    lens = _load_lens(torch, lens_path)
    tokenizer = _load_tokenizer(model_name)
    lens_model = _load_lens_model(torch, model_name, dtype=dtype, device=device) if hasattr(lens, "transport") else None
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = readout_rows(torch, run_dir, lens, top_k=top_k, tokenizer=tokenizer, lens_model=lens_model)
    overlaps = pair_overlap_rows(rows)
    summary = layer_summary_rows(overlaps)
    _write_csv(output_dir / "jlens_readouts.csv", rows)
    _write_csv(output_dir / "jlens_pair_overlap.csv", overlaps)
    _write_csv(output_dir / "jlens_layer_summary.csv", summary)
    write_html(output_dir / "jlens_readouts.html", summary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional J-lens readouts on existing RQ1 collect outputs.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--jlens-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--model-name")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_jlens_readout(
        args.run,
        args.jlens_path,
        args.output_dir,
        top_k=args.top_k,
        model_name=args.model_name,
        dtype=args.dtype,
        device=args.device,
    )
    print(f"Wrote J-lens readouts to {args.output_dir}")


if __name__ == "__main__":
    main()
