"""Run RQ2 attention-pattern collection and analysis."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lora_instruction_analysis.model.collect import CONDITIONS, CollectConfig, collect
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS
from lora_instruction_analysis.model.visualize import (
    ATTENTION_ALIGNMENT_STRATEGY,
    ATTENTION_OUTPUT_DEFINITION,
    ATTENTION_PATTERN_DEFINITION,
    VisualizeConfig,
    visualize,
)


@dataclass(frozen=True)
class RQ2Config:
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
    rq_name: str = "rq2"
    collect_attention_outputs: bool = False
    prompt_format: str = "raw"
    append_eos: bool = True

    @property
    def resolved_states_dir(self) -> Path:
        return self.states_dir or self.run_dir / "states" / self.rq_name

    @property
    def resolved_plots_dir(self) -> Path:
        return self.plots_dir or self.run_dir / "plots" / self.rq_name


def _read_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _infer_config(args: argparse.Namespace, *, rq_name: str = "rq2") -> RQ2Config:
    run_config = _read_run_config(args.run_dir)
    return RQ2Config(
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
        rq_name=rq_name,
        prompt_format=getattr(args, "prompt_format", None) or run_config.get("prompt_format", "raw"),
        append_eos=False if getattr(args, "no_append_eos", False) else bool(run_config.get("append_eos", True)),
    )


def _jsonable(config: RQ2Config) -> dict:
    data = asdict(config)
    for key in ("run_dir", "dataset_path", "adapter_path", "states_dir", "plots_dir"):
        if data[key] is not None:
            data[key] = str(data[key])
    data["resolved_states_dir"] = str(config.resolved_states_dir)
    data["resolved_plots_dir"] = str(config.resolved_plots_dir)
    data["conditions"] = list(CONDITIONS)
    data["comparison"] = (
        f"{ATTENTION_PATTERN_DEFINITION}; {ATTENTION_OUTPUT_DEFINITION}"
        if config.collect_attention_outputs
        else ATTENTION_PATTERN_DEFINITION
    )
    data["alignment_strategy"] = ATTENTION_ALIGNMENT_STRATEGY
    data["attention_output_status"] = "enabled" if config.collect_attention_outputs else "disabled"
    return data


def run_rq2(config: RQ2Config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / f"{config.rq_name}_config.json").write_text(
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
            collect_attention_outputs=config.collect_attention_outputs,
            prompt_format=config.prompt_format,
            append_eos=config.append_eos,
        )
    )
    required_keys = ("source_alignment",) + (("attention_outputs",) if config.collect_attention_outputs else ())
    _require_tensor_keys(config.resolved_states_dir, required_keys)
    attention_plot_dir = config.resolved_plots_dir / "attention_probs" if config.collect_attention_outputs else config.resolved_plots_dir
    visualize(
        VisualizeConfig(
            run=config.resolved_states_dir,
            left_run=None,
            right_run=None,
            output_dir=attention_plot_dir,
            left_model=config.model_name,
            right_model=config.model_name,
            mode="attention",
        )
    )
    if config.collect_attention_outputs:
        visualize(
            VisualizeConfig(
                run=config.resolved_states_dir,
                left_run=None,
                right_run=None,
                output_dir=config.resolved_plots_dir / "attention_outputs",
                left_model=config.model_name,
                right_model=config.model_name,
                mode="attention_output",
            )
        )


def _require_tensor_keys(states_dir: Path, required_keys: tuple[str, ...]) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("RQ2 tensor validation requires torch because collect.py stores .pt tensors.") from exc

    metrics = [json.loads(line) for line in (states_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    for row in metrics:
        path = Path(row["tensor_path"])
        tensor_path = path if path.is_absolute() else states_dir / "tensors" / path.name
        tensor = torch.load(tensor_path, map_location="cpu")
        missing = [key for key in required_keys if key not in tensor]
        if missing:
            raise ValueError(f"Missing {', '.join(missing)} in {tensor_path}; delete old states and rerun collect.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ2 attention-pattern analysis for a LoRA run.")
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
    return parser.parse_args()


def main() -> None:
    config = _infer_config(parse_args())
    run_rq2(config)
    print(f"Wrote RQ2 states to {config.resolved_states_dir}")
    print(f"Wrote RQ2 plots to {config.resolved_plots_dir}")


if __name__ == "__main__":
    main()
