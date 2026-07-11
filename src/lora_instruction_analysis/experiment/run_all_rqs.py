"""Run RQ1, RQ2, RQ2.1, and RQ3 for one LoRA run."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from lora_instruction_analysis.experiment import run_rq1, run_rq2, run_rq3
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all RQ analyses for a LoRA run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="LoRA run directory with config.json.")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--source-condition")
    parser.add_argument("--target-condition")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--patch-span", choices=("target", "text"), default="text")
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS)
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--validator", default=None, help="Override RQ3 task validation: task_default, exact, single_token, integer, or yes_no.")
    parser.set_defaults(states_dir=None, plots_dir=None, output_dir=None)
    return parser.parse_args()


def run_all_rqs(args: argparse.Namespace) -> None:
    rq1_config = run_rq1._infer_config(args)
    run_rq1.run_rq1(rq1_config)

    rq2_config = run_rq2._infer_config(args)
    run_rq2.run_rq2(rq2_config)

    rq21_config = replace(run_rq2._infer_config(args, rq_name="rq21"), collect_attention_outputs=True)
    run_rq2.run_rq2(rq21_config)

    rq3_config = run_rq3._infer_config(args)
    run_rq3.run_rq3(rq3_config)


def main() -> None:
    args = parse_args()
    run_all_rqs(args)
    print(f"Wrote RQ1 states to {args.run_dir / 'states' / 'rq1'}")
    print(f"Wrote RQ2 states to {args.run_dir / 'states' / 'rq2'}")
    print(f"Wrote RQ2.1 states to {args.run_dir / 'states' / 'rq21'}")
    print(f"Wrote RQ3 patching metrics to {args.run_dir / 'patches' / 'rq3'}")
    print(f"Wrote RQ3 patching plots to {args.run_dir / 'plots' / 'patch_loss' / 'rq3'}")


if __name__ == "__main__":
    main()
