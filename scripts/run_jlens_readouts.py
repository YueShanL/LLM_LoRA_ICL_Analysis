"""Run J-lens readouts for every task under one experiment root."""

from __future__ import annotations

import argparse
from pathlib import Path

from lora_instruction_analysis.model.jlens_readout import (
    _load_lens,
    _load_lens_model,
    _load_tokenizer,
    _write_csv,
    layer_summary_rows,
    pair_overlap_rows,
    readout_rows,
    write_html,
)


def run_all(root: Path, lens_path: Path, model_name: str, *, top_k: int, dtype: str, device: str) -> None:
    import torch

    lens = _load_lens(torch, lens_path)
    tokenizer = _load_tokenizer(model_name)
    lens_model = _load_lens_model(torch, model_name, dtype=dtype, device=device) if hasattr(lens, "transport") else None
    for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        run_dir = task_dir / "states" / "rq1"
        if not (run_dir / "metrics.jsonl").exists():
            continue
        output_dir = task_dir / "plots" / "rq1_jlens"
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = readout_rows(torch, run_dir, lens, top_k=top_k, tokenizer=tokenizer, lens_model=lens_model)
        overlaps = pair_overlap_rows(rows)
        summary = layer_summary_rows(overlaps)
        _write_csv(output_dir / "jlens_readouts.csv", rows)
        _write_csv(output_dir / "jlens_pair_overlap.csv", overlaps)
        _write_csv(output_dir / "jlens_layer_summary.csv", summary)
        write_html(output_dir / "jlens_readouts.html", summary)
        print(f"Wrote J-lens readouts to {output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run J-lens readouts for all task RQ1 state directories.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jlens-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all(args.root, args.jlens_path, args.model_name, top_k=args.top_k, dtype=args.dtype, device=args.device)


if __name__ == "__main__":
    main()
