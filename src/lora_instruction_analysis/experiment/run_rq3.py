"""Run RQ3 activation patching."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from lora_instruction_analysis.data.tasks import ValidationSelector, validator_name
from lora_instruction_analysis.model.formatting import PROMPT_FORMATS
from lora_instruction_analysis.model.patch import PatchConfig, run_activation_patching
from lora_instruction_analysis.model.visualize import VisualizeConfig, visualize


DEFAULT_PATCH_PAIRS = (
    ("lora_only", "instruction_only"),
    ("lora_only", "base"),
    ("instruction_only", "lora_only"),
    ("base", "lora_only"),
)
DEFAULT_LAYERS = (1, 21, 27)
RQ3_CONTROLS = (
    "unpatched",
    "base_to_target_patch",
    "source_to_target_patch",
)


@dataclass(frozen=True)
class RQ3Config:
    run_dir: Path
    model_name: str
    dataset_path: Path
    adapter_path: Path
    source_condition: str | None = None
    target_condition: str | None = None
    layer: int | None = None
    split: str = "test"
    max_samples: int | None = None
    seed: int = 13
    max_new_tokens: int = 20
    patch_span: str = "text"
    dtype: str = "auto"
    device: str = "auto"
    output_dir: Path | None = None
    plots_dir: Path | None = None
    prompt_format: str = "raw"
    append_eos: bool = True
    validator: ValidationSelector = None

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or self.run_dir / "patches" / "rq3"

    @property
    def resolved_plots_dir(self) -> Path:
        return self.plots_dir or self.run_dir / "plots" / "patch_loss" / "rq3"


def _patch_specs(config: RQ3Config) -> list[tuple[str, str, int]]:
    if (config.source_condition is None) != (config.target_condition is None):
        raise ValueError("source_condition and target_condition must be provided together.")
    pairs = (
        [(config.source_condition, config.target_condition)]
        if config.source_condition is not None and config.target_condition is not None
        else list(DEFAULT_PATCH_PAIRS)
    )
    layers = [config.layer] if config.layer is not None else list(DEFAULT_LAYERS)
    return [(source, target, layer) for source, target in pairs for layer in layers]


def _run_name(source_condition: str, target_condition: str, layer: int, patch_span: str) -> str:
    return f"{source_condition}_to_{target_condition}_l{layer}_{patch_span}"


def _run_controls(source_condition: str, target_condition: str) -> list[dict]:
    return [
        {"control": "unpatched", "source_condition": None, "target_condition": target_condition},
        {"control": "base_to_target_patch", "source_condition": "base", "target_condition": target_condition},
        {"control": "source_to_target_patch", "source_condition": source_condition, "target_condition": target_condition},
    ]


def _read_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def _infer_config(args: argparse.Namespace) -> RQ3Config:
    run_config = _read_run_config(args.run_dir)
    return RQ3Config(
        run_dir=args.run_dir,
        model_name=args.model_name or run_config["model_name"],
        dataset_path=args.dataset_path or Path(run_config["dataset_dir"]),
        adapter_path=args.adapter_path or Path(run_config["adapter_dir"]),
        source_condition=args.source_condition,
        target_condition=args.target_condition,
        layer=args.layer,
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed if args.seed is not None else int(run_config.get("seed", 13)),
        max_new_tokens=args.max_new_tokens,
        patch_span=args.patch_span,
        dtype=args.dtype,
        device=args.device,
        output_dir=args.output_dir,
        plots_dir=getattr(args, "plots_dir", None),
        prompt_format=getattr(args, "prompt_format", None) or run_config.get("prompt_format", "raw"),
        append_eos=False if getattr(args, "no_append_eos", False) else bool(run_config.get("append_eos", True)),
        validator=getattr(args, "validator", None),
    )


def _jsonable(config: RQ3Config) -> dict:
    data = asdict(config)
    for key in ("run_dir", "dataset_path", "adapter_path", "output_dir", "plots_dir"):
        if data[key] is not None:
            data[key] = str(data[key])
    data["resolved_output_dir"] = str(config.resolved_output_dir)
    data["resolved_plots_dir"] = str(config.resolved_plots_dir)
    data["comparison"] = f"teacher-forced and autoregressive block-output activation patching over {config.patch_span} span"
    data["validator"] = validator_name(config.validator)
    data["result_status"] = "partial"
    data["status_reason"] = "RQ3 has controls, generation metrics, semantic task scoring, and shape checks; activation-site sweep is still pending."
    data["default_patch_pairs"] = [list(pair) for pair in DEFAULT_PATCH_PAIRS]
    data["default_layers"] = list(DEFAULT_LAYERS)
    data["controls"] = list(RQ3_CONTROLS)
    data["patch_runs"] = [
        {
            "source_condition": source,
            "target_condition": target,
            "layer": layer,
            "patch_span": config.patch_span,
            "activation_site": "block_output",
            "controls": _run_controls(source, target),
            "output_dir": str(config.resolved_output_dir / _run_name(source, target, layer, config.patch_span)),
        }
        for source, target, layer in _patch_specs(config)
    ]
    return data


def run_rq3(config: RQ3Config) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "rq3_config.json").write_text(
        json.dumps(_jsonable(config), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (config.run_dir / "rq3_status.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "implemented": [
                    "unpatched/base-to-target/source-to-target controls",
                    "teacher-forced and autoregressive metrics",
                    "task-level semantic generation scoring",
                    "patch shape mismatch checks",
                ],
                "pending": ["activation-site sweep"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    for source_condition, target_condition, layer in _patch_specs(config):
        run_activation_patching(
            PatchConfig(
                model_name=config.model_name,
                dataset_path=config.dataset_path,
                output_dir=config.resolved_output_dir
                / _run_name(source_condition, target_condition, layer, config.patch_span),
                adapter_path=config.adapter_path,
                source_condition=source_condition,
                target_condition=target_condition,
                layer=layer,
                split=config.split,
                max_samples=config.max_samples,
                seed=config.seed,
                max_new_tokens=config.max_new_tokens,
                patch_span=config.patch_span,
                dtype=config.dtype,
                device=config.device,
                prompt_format=config.prompt_format,
                append_eos=config.append_eos,
                validator=config.validator,
            )
        )
    visualize(
        VisualizeConfig(
            run=config.resolved_output_dir,
            left_run=None,
            right_run=None,
            output_dir=config.resolved_plots_dir,
            left_model=config.model_name,
            right_model=config.model_name,
            mode="patch_loss",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ3 activation patching for a LoRA run.")
    parser.add_argument("--run-dir", type=Path, required=True, help="LoRA run directory with config.json.")
    parser.add_argument("--model-name")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--source-condition")
    parser.add_argument("--target-condition")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--patch-span", choices=("target", "text"), default="text")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plots-dir", type=Path)
    parser.add_argument("--prompt-format", choices=PROMPT_FORMATS)
    parser.add_argument("--no-append-eos", action="store_true")
    parser.add_argument("--validator", default=None, help="Override task validation: task_default, exact, single_token, integer, or yes_no.")
    return parser.parse_args()


def main() -> None:
    config = _infer_config(parse_args())
    run_rq3(config)
    print(f"Wrote RQ3 patching metrics to {config.resolved_output_dir}")
    print(f"Wrote RQ3 patching plots to {config.resolved_plots_dir}")


if __name__ == "__main__":
    main()
