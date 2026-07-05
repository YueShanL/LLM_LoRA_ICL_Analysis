"""Run RQ2.1 attention-probability and head-output analysis."""

from __future__ import annotations

from dataclasses import replace

from lora_instruction_analysis.experiment.run_rq2 import _infer_config, parse_args, run_rq2


def main() -> None:
    config = replace(_infer_config(parse_args(), rq_name="rq21"), collect_attention_outputs=True)
    run_rq2(config)
    print(f"Wrote RQ2.1 states to {config.resolved_states_dir}")
    print(f"Wrote RQ2.1 plots to {config.resolved_plots_dir}")


if __name__ == "__main__":
    main()
