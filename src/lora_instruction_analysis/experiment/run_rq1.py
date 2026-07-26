"""Run RQ1 residual perturbation collection and analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lora_instruction_analysis.model.collect import CONDITIONS, CollectConfig, collect
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS
from lora_instruction_analysis.model.jlens_readout import run_jlens_readout
from lora_instruction_analysis.model.sae_analysis import run_sae_analysis
from lora_instruction_analysis.model.visualize import VisualizeConfig, visualize


@dataclass(frozen=True)
class RQ1Config:
    run_dir: Path
    model_name: str
    dataset_path: Path
    adapter_path: Path
    split: str = "test"
    max_samples: int | None = None
    seed: int = 13
    dtype: str = "auto"
    device: str = "auto"
    states_dir: Path | None = None
    plots_dir: Path | None = None
    prompt_format: str = "raw"
    append_eos: bool = True
    icl_examples: int = 0
    icl_split: str = "train"
    run_jlens_readout: bool = False
    jlens_path: Path | None = None
    jlens_top_k: int = 20
    run_sae_analysis: bool = False
    sae_path: Path | None = None
    sae_top_k: int = 20

    @property
    def resolved_states_dir(self) -> Path:
        return self.states_dir or self.run_dir / "states" / "rq1"

    @property
    def resolved_plots_dir(self) -> Path:
        return self.plots_dir or self.run_dir / "plots" / "rq1"


def _read_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _infer_config(args: argparse.Namespace) -> RQ1Config:
    run_config = _read_run_config(args.run_dir)
    return RQ1Config(
        run_dir=args.run_dir,
        model_name=args.model_name or run_config["model_name"],
        dataset_path=args.dataset_path or Path(run_config["dataset_dir"]),
        adapter_path=args.adapter_path or Path(run_config["adapter_dir"]),
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed if args.seed is not None else int(run_config.get("seed", 13)),
        dtype=args.dtype,
        device=args.device,
        states_dir=args.states_dir,
        plots_dir=args.plots_dir,
        prompt_format=getattr(args, "prompt_format", None) or run_config.get("prompt_format", "raw"),
        append_eos=False if getattr(args, "no_append_eos", False) else bool(run_config.get("append_eos", True)),
        icl_examples=int(getattr(args, "icl_examples", 0) or 0),
        icl_split=getattr(args, "icl_split", "train"),
        run_jlens_readout=bool(getattr(args, "run_jlens_readout", False)),
        jlens_path=getattr(args, "jlens_path", None),
        jlens_top_k=int(getattr(args, "jlens_top_k", 20)),
        run_sae_analysis=bool(getattr(args, "run_sae_analysis", False)),
        sae_path=getattr(args, "sae_path", None),
        sae_top_k=int(getattr(args, "sae_top_k", 20)),
    )


def _jsonable(config: RQ1Config) -> dict:
    data = asdict(config)
    for key in ("run_dir", "dataset_path", "adapter_path", "states_dir", "plots_dir", "jlens_path", "sae_path"):
        if data[key] is not None:
            data[key] = str(data[key])
    data["resolved_states_dir"] = str(config.resolved_states_dir)
    data["resolved_plots_dir"] = str(config.resolved_plots_dir)
    data["conditions"] = list(CONDITIONS)
    data["comparison"] = "cosine(hidden_instruction_only - hidden_base, hidden_lora_only - hidden_base)"
    return data


def run_rq1(config: RQ1Config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "rq1_config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    collect(
        CollectConfig(
            model_name=config.model_name,
            dataset_path=config.dataset_path,
            output_dir=config.resolved_states_dir,
            adapter_path=config.adapter_path,
            split=config.split,
            max_samples=config.max_samples,
            seed=config.seed,
            conditions=CONDITIONS,
            dtype=config.dtype,
            device=config.device,
            prompt_format=config.prompt_format,
            append_eos=config.append_eos,
            icl_examples=config.icl_examples,
            icl_split=config.icl_split,
        )
    )
    visualize(
        VisualizeConfig(
            run=config.resolved_states_dir,
            left_run=None,
            right_run=None,
            output_dir=config.resolved_plots_dir,
            left_model=config.model_name,
            right_model=config.model_name,
            mode="residual",
        )
    )
    if config.run_jlens_readout:
        if config.jlens_path is None:
            raise ValueError("--jlens-path is required when --run-jlens-readout is set.")
        run_jlens_readout(
            config.resolved_states_dir,
            config.jlens_path,
            config.run_dir / "plots" / "rq1_jlens",
            top_k=config.jlens_top_k,
            model_name=config.model_name,
        )
    if config.run_sae_analysis:
        if config.sae_path is None:
            raise ValueError("--sae-path is required when --run-sae-analysis is set.")
        run_sae_analysis(
            config.resolved_states_dir,
            config.sae_path,
            config.run_dir / "plots" / "rq1_sae",
            mode="residual",
            top_k=config.sae_top_k,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ1 residual perturbation analysis for a LoRA run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="LoRA run directory with config.json.")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--states-dir", type=Path)
    parser.add_argument("--plots-dir", type=Path)
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS)
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--icl-examples", type=int, default=0)
    parser.add_argument("--icl-split", default="train")
    parser.add_argument("--run-jlens-readout", action="store_true")
    parser.add_argument("--jlens-path", type=Path)
    parser.add_argument("--jlens-top-k", type=int, default=20)
    parser.add_argument("--run-sae-analysis", action="store_true")
    parser.add_argument("--sae-path", type=Path)
    parser.add_argument("--sae-top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    config = _infer_config(parse_args())
    run_rq1(config)
    print(f"Wrote RQ1 states to {config.resolved_states_dir}")
    print(f"Wrote RQ1 plots to {config.resolved_plots_dir}")


if __name__ == "__main__":
    main()
