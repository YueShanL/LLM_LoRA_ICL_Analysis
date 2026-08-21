"""Resumeable HPC orchestrator for data, LoRA, prompt eval, RQs, and plots."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset, write_dataset
from lora_instruction_analysis.data.tasks import task_default_prompt
from lora_instruction_analysis.experiment import run_rq1, run_rq2, run_rq3
from lora_instruction_analysis.model.jlens_fit import fit_jlens
from lora_instruction_analysis.model.prompt_eval import PromptEvalConfig, evaluate_prompt
from lora_instruction_analysis.model.train_lora import TrainConfig, train_lora
from scripts.plot_all_rq3_accuracy import write as write_rq3_plots
from scripts.plot_combined_rq1_rq2 import write_plots as write_rq12_plots


def _read_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _stage(config: dict, name: str) -> dict:
    return dict(config.get("stages", {}).get(name, {}))


def _enabled(config: dict, name: str) -> bool:
    return bool(_stage(config, name).get("enabled", True))


def _cleanup_collected_states_enabled(config: dict) -> bool:
    return bool(config.get("cleanup_collected_states", False))


def _tmp_dir(config: dict) -> Path:
    raw = config.get("TMP_DIR") or config.get("tmp_dir") or os.environ.get("TMP_DIR")
    if not raw:
        raise ValueError(
            "cleanup_collected_states=true requires TMP_DIR in the config or TMP_DIR in the environment."
        )
    return Path(raw).expanduser()


def _instruction_mode(config: dict) -> tuple[int, str]:
    mode = config.get("instruction_mode", {}).get("mode", "prompt")
    if mode == "prompt":
        return 0, "train"
    if mode != "icl":
        raise ValueError("instruction_mode.mode must be 'prompt' or 'icl'.")
    return int(config.get("instruction_mode", {}).get("icl_examples", 0) or 0), config.get("instruction_mode", {}).get("icl_split", "train")


def _run_root(config: dict) -> Path:
    return Path(config.get("output_root", "experiments")) / config["run_group_id"]


def _task_run_dir(config: dict, task_id: str) -> Path:
    return _run_root(config) / task_id


def _adapter_dir(config: dict, task_id: str) -> Path:
    lora = _stage(config, "lora")
    return _task_run_dir(config, task_id) / "adapters" / f"r{int(lora.get('rank', 8))}"


def _done_dataset(run_dir: Path) -> bool:
    dataset = run_dir / "dataset"
    return all((dataset / name).exists() for name in ("manifest.json", "train.jsonl", "validation.jsonl", "test.jsonl"))


def _done_lora(adapter_dir: Path) -> bool:
    if (adapter_dir / "training_failed.json").exists():
        return False
    return (adapter_dir / "adapter_config.json").exists() and (
        (adapter_dir / "adapter_model.safetensors").exists() or (adapter_dir / "adapter_model.bin").exists()
    )


def _icl_matches(data: dict, icl_examples: int, icl_split: str) -> bool:
    return int(data.get("icl_examples", 0) or 0) == icl_examples and data.get("icl_split", "train") == icl_split


def _done_prompt_eval(run_dir: Path, name: str, *, icl_examples: int, icl_split: str) -> bool:
    path = run_dir / "prompt_eval" / name / "summary.json"
    if not path.exists():
        return False
    return _icl_matches(_read_config(path).get("config", {}), icl_examples, icl_split)


def _done_jlens(path: Path) -> bool:
    return (path / "validation_summary.json").exists() and (path / "lens").exists()


def _done_rq1(run_dir: Path, *, needs_jlens: bool = False, icl_examples: int = 0, icl_split: str = "train") -> bool:
    done = (run_dir / "states" / "rq1" / "metrics.jsonl").exists() and (run_dir / "plots" / "rq1" / "token_similarity.csv").exists()
    if needs_jlens:
        done = done and (run_dir / "plots" / "rq1_jlens" / "jlens_readouts.csv").exists()
    return done and (run_dir / "rq1_config.json").exists() and _icl_matches(_read_config(run_dir / "rq1_config.json"), icl_examples, icl_split)


def _done_rq2(run_dir: Path, *, icl_examples: int = 0, icl_split: str = "train") -> bool:
    return (
        (run_dir / "states" / "rq2" / "metrics.jsonl").exists()
        and (run_dir / "plots" / "rq2" / "token_similarity.csv").exists()
        and (run_dir / "rq2_config.json").exists()
        and _icl_matches(_read_config(run_dir / "rq2_config.json"), icl_examples, icl_split)
    )


def _done_rq21(run_dir: Path, *, icl_examples: int = 0, icl_split: str = "train") -> bool:
    return (
        (run_dir / "states" / "rq21" / "metrics.jsonl").exists()
        and (run_dir / "plots" / "rq21" / "attention_probs" / "token_similarity.csv").exists()
        and (run_dir / "plots" / "rq21" / "attention_outputs" / "token_similarity.csv").exists()
        and (run_dir / "rq21_config.json").exists()
        and _icl_matches(_read_config(run_dir / "rq21_config.json"), icl_examples, icl_split)
    )


def _done_rq3(run_dir: Path, *, icl_examples: int = 0, icl_split: str = "train", layers: list[int] | None = None) -> bool:
    config_path = run_dir / "rq3_config.json"
    if not config_path.exists():
        return False
    config = _read_config(config_path)
    expected_layers = list(layers) if layers is not None else None
    return (
        (run_dir / "patches" / "rq3").exists()
        and (run_dir / "plots" / "patch_loss" / "rq3" / "rq3_summary.csv").exists()
        and _icl_matches(config, icl_examples, icl_split)
        and (expected_layers is None or config.get("active_layers") == expected_layers)
    )


def _done_cross_task(output_dir: Path) -> bool:
    combined = output_dir
    return (combined / "rq1_similarity_lines.png").exists() and (combined / "rq3_accuracy_all_tasks.csv").exists()


def _task_complete(config: dict, task_id: str, jlens_path: Path | None) -> bool:
    run_dir = _task_run_dir(config, task_id)
    icl_examples, icl_split = _instruction_mode(config)
    if _enabled(config, "data") and not _done_dataset(run_dir):
        return False
    if _enabled(config, "lora") and not _done_lora(_adapter_dir(config, task_id)):
        return False
    if _enabled(config, "prompt_eval"):
        for run in _stage(config, "prompt_eval").get("runs", []):
            if not _done_prompt_eval(run_dir, run["name"], icl_examples=icl_examples, icl_split=icl_split):
                return False
    if _enabled(config, "rq1"):
        needs_jlens = bool(_stage(config, "rq1").get("run_jlens_readout", False) and jlens_path)
        if not _done_rq1(run_dir, needs_jlens=needs_jlens, icl_examples=icl_examples, icl_split=icl_split):
            return False
    if _enabled(config, "rq2") and not _done_rq2(run_dir, icl_examples=icl_examples, icl_split=icl_split):
        return False
    if _enabled(config, "rq21") and not _done_rq21(run_dir, icl_examples=icl_examples, icl_split=icl_split):
        return False
    if _enabled(config, "rq3") and not _done_rq3(
        run_dir, icl_examples=icl_examples, icl_split=icl_split, layers=_stage(config, "rq3").get("layers")
    ):
        return False
    return True


def _cleanup_collected_states(run_dir: Path) -> list[Path]:
    """Remove raw collection tensors after all task-level analyses have succeeded."""
    states_root = (run_dir / "states").resolve()
    if not states_root.is_dir():
        return []

    removed: list[Path] = []
    for state_dir in sorted(states_root.iterdir()):
        tensor_dir = state_dir / "tensors"
        if not tensor_dir.is_dir() or tensor_dir.is_symlink():
            continue
        if tensor_dir.resolve().parent != states_root / state_dir.name:
            raise RuntimeError(f"Refusing to clean tensor directory outside states root: {tensor_dir}")
        shutil.rmtree(tensor_dir)
        removed.append(tensor_dir)

    report = run_dir / "collected_states_cleanup.json"
    report.write_text(
        json.dumps(
            {
                "status": "complete",
                "removed": [str(path) for path in removed],
                "retained": [
                    "states/*/metrics.jsonl",
                    "states/*/config.json",
                    "states/*/dataset_snapshot.jsonl",
                    "plots/**",
                    "patches/rq3/**/metrics.jsonl",
                    "patches/rq3/**/generations.jsonl",
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return removed


def _task_work_config(config: dict, task_id: str, jlens_path: Path | None) -> dict | None:
    if not _cleanup_collected_states_enabled(config):
        return config
    if _task_complete(config, task_id, jlens_path):
        return None

    tmp_dir = _tmp_dir(config)
    final_root = _run_root(config).resolve()
    work_root = (tmp_dir / config["run_group_id"]).resolve()
    if work_root == final_root:
        raise ValueError("TMP_DIR must resolve to a directory different from output_root/run_group_id.")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_run_dir = _task_run_dir(config, task_id)
    work_run_dir = work_root / task_id
    if not work_run_dir.exists() and final_run_dir.exists():
        shutil.copytree(final_run_dir, work_run_dir, dirs_exist_ok=True)
    return {
        **config,
        "output_root": str(tmp_dir),
        "_final_output_root": str(_run_root(config)),
    }


def _remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _merge_move(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and not source.is_symlink():
        if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
            _remove_path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for child in list(source.iterdir()):
            _merge_move(child, destination / child.name)
        source.rmdir()
        return
    if destination.exists() or destination.is_symlink():
        _remove_path(destination)
    shutil.move(str(source), str(destination))


def _rewrite_promoted_paths(run_dir: Path, replacements: list[tuple[str, str]]) -> None:
    text_suffixes = {".json", ".jsonl", ".md"}
    for path in run_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _promote_task_results(work_dir: Path, final_dir: Path, work_root: Path, final_root: Path) -> None:
    work_dir = work_dir.resolve()
    final_dir = final_dir.resolve()
    work_root = work_root.resolve()
    final_root = final_root.resolve()
    try:
        work_dir.relative_to(work_root)
        final_dir.relative_to(final_root)
    except ValueError as exc:
        raise RuntimeError("Refusing to promote task results outside their configured roots.") from exc
    if not work_dir.is_dir():
        raise FileNotFoundError(f"Temporary task output is missing: {work_dir}")
    final_dir.mkdir(parents=True, exist_ok=True)
    for child in list(work_dir.iterdir()):
        _merge_move(child, final_dir / child.name)
    work_dir.rmdir()
    _rewrite_promoted_paths(
        final_dir,
        [
            (str(work_dir), str(final_dir)),
            (str(work_root), str(final_root)),
        ],
    )


def _write_run_config(config: dict, task_id: str) -> None:
    run_dir = _task_run_dir(config, task_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    lora = _stage(config, "lora")
    data = _stage(config, "data")
    payload = {
        "run_id": task_id,
        "output_root": str(_run_root(config)),
        "model_name": config["model_name"],
        "task_id": task_id,
        "source_id": config.get("source_id", "wikitext"),
        "seed": int(config.get("seed", 13)),
        "rank": int(lora.get("rank", 8)),
        "train_size": int(data.get("train_size", 800)),
        "validation_size": int(data.get("validation_size", 100)),
        "test_size": int(data.get("test_size", 100)),
        "max_source_rows": int(data.get("max_source_rows", 5000)),
        "epochs": float(lora.get("epochs", 3.0)),
        "learning_rate": float(lora.get("learning_rate", 2e-4)),
        "warmup_ratio": float(lora.get("warmup_ratio", 0.03)),
        "train_batch_size": int(lora.get("train_batch_size", 2)),
        "eval_batch_size": int(lora.get("eval_batch_size", 2)),
        "gradient_accumulation_steps": int(lora.get("gradient_accumulation_steps", 8)),
        "max_length": int(lora.get("max_length", 512)),
        "lora_alpha": int(lora.get("lora_alpha", 16)),
        "lora_dropout": float(lora.get("lora_dropout", 0.05)),
        "target_modules": list(lora.get("target_modules", ["auto"])),
        "fp16": bool(lora.get("fp16", False)),
        "bf16": bool(lora.get("bf16", False)),
        "monitor_nonfinite": bool(lora.get("monitor_nonfinite", True)),
        "max_grad_norm": float(lora.get("max_grad_norm", 1.0)),
        "skip_nonfinite_loss": bool(lora.get("skip_nonfinite_loss", True)),
        "max_nonfinite_loss_skips": int(lora.get("max_nonfinite_loss_skips", 8)),
        "detect_anomaly": bool(lora.get("detect_anomaly", False)),
        "qlora": bool(lora.get("qlora", False)),
        "device_map": lora.get("device_map", "auto"),
        "include_instruction_in_prompt": bool(data.get("include_instruction_in_prompt", False)),
        "streaming": bool(data.get("streaming", False)),
        "prompt_format": lora.get("prompt_format", "raw"),
        "append_eos": bool(lora.get("append_eos", True)),
        "run_dir": str(run_dir),
        "dataset_dir": str(run_dir / "dataset"),
        "adapter_dir": str(_adapter_dir(config, task_id)),
    }
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def run_data(config: dict, task_id: str) -> None:
    run_dir = _task_run_dir(config, task_id)
    if _done_dataset(run_dir):
        print(f"[skip] data {task_id}", flush=True)
        return
    data = _stage(config, "data")
    dataset_config = DatasetBuildConfig(
        task_id=task_id,
        source_id=config.get("source_id", "wikitext"),
        output_dir=run_dir / "dataset",
        train_size=int(data.get("train_size", 800)),
        validation_size=int(data.get("validation_size", 100)),
        test_size=int(data.get("test_size", 100)),
        max_source_rows=int(data.get("max_source_rows", 5000)),
        min_words=int(data.get("min_words", 4)),
        max_words=int(data.get("max_words", 32)),
        seed=int(config.get("seed", 13)),
        condition="lora_training",
        include_instruction_in_prompt=bool(data.get("include_instruction_in_prompt", False)),
        write_csv=bool(data.get("write_csv", True)),
        write_hf_dataset=bool(data.get("write_hf_dataset", True)),
        streaming=bool(data.get("streaming", False)),
        model_name=config["model_name"],
        tokenizer_name=config["model_name"],
        prompt_template=f"{_stage(config, 'lora').get('prompt_format', 'raw')}_target_v1",
    )
    write_dataset(dataset_config, build_dataset(dataset_config))
    print(f"[done] data {task_id}", flush=True)


def run_lora(config: dict, task_id: str) -> None:
    adapter_dir = _adapter_dir(config, task_id)
    if _done_lora(adapter_dir):
        print(f"[skip] lora {task_id}", flush=True)
        return
    lora = _stage(config, "lora")
    try:
        train_lora(
            TrainConfig(
                model_name=config["model_name"],
                dataset_path=_task_run_dir(config, task_id) / "dataset",
                output_dir=adapter_dir,
                max_length=int(lora.get("max_length", 512)),
                seed=int(config.get("seed", 13)),
                rank=int(lora.get("rank", 8)),
                lora_alpha=int(lora.get("lora_alpha", 16)),
                lora_dropout=float(lora.get("lora_dropout", 0.05)),
                target_modules=tuple(lora.get("target_modules", ["auto"])),
                learning_rate=float(lora.get("learning_rate", 2e-4)),
                warmup_ratio=float(lora.get("warmup_ratio", 0.03)),
                epochs=float(lora.get("epochs", 3.0)),
                train_batch_size=int(lora.get("train_batch_size", 2)),
                eval_batch_size=int(lora.get("eval_batch_size", 2)),
                gradient_accumulation_steps=int(lora.get("gradient_accumulation_steps", 8)),
                fp16=bool(lora.get("fp16", False)),
                bf16=bool(lora.get("bf16", False)),
                monitor_nonfinite=bool(lora.get("monitor_nonfinite", True)),
                max_grad_norm=float(lora.get("max_grad_norm", 1.0)),
                skip_nonfinite_loss=bool(lora.get("skip_nonfinite_loss", True)),
                max_nonfinite_loss_skips=int(lora.get("max_nonfinite_loss_skips", 8)),
                detect_anomaly=bool(lora.get("detect_anomaly", False)),
                qlora=bool(lora.get("qlora", False)),
                device_map=lora.get("device_map", "auto"),
                prompt_format=lora.get("prompt_format", "raw"),
                append_eos=bool(lora.get("append_eos", True)),
            )
        )
    except Exception:
        failure_report = adapter_dir / "training_failed.json"
        final_root = config.get("_final_output_root")
        if failure_report.exists() and final_root:
            final_adapter = Path(final_root) / task_id / "adapters" / f"r{int(lora.get('rank', 8))}"
            final_adapter.mkdir(parents=True, exist_ok=True)
            shutil.copy2(failure_report, final_adapter / failure_report.name)
        raise
    print(f"[done] lora {task_id}", flush=True)


def run_prompt_eval(config: dict, task_id: str) -> None:
    prompt = _stage(config, "prompt_eval")
    icl_examples, icl_split = _instruction_mode(config)
    for run in prompt.get("runs", []):
        name = run["name"]
        run_dir = _task_run_dir(config, task_id)
        if _done_prompt_eval(run_dir, name, icl_examples=icl_examples, icl_split=icl_split):
            print(f"[skip] prompt_eval {task_id}/{name}", flush=True)
            continue
        evaluate_prompt(
            PromptEvalConfig(
                model_name=config["model_name"],
                dataset_path=run_dir / "dataset",
                output_dir=run_dir / "prompt_eval" / name,
                instruction=run.get("instruction") or task_default_prompt(task_id),
                split=prompt.get("split", "test"),
                max_samples=prompt.get("max_samples"),
                seed=int(config.get("seed", 13)),
                max_length=int(prompt.get("max_length", 512)),
                generation_extra_tokens=int(prompt.get("generation_extra_tokens", 128)),
                include_instruction=bool(run.get("include_instruction", True)),
                dtype=prompt.get("dtype", "auto"),
                device=prompt.get("device", "auto"),
                prompt_format=prompt.get("prompt_format", _stage(config, "lora").get("prompt_format", "raw")),
                append_eos=bool(prompt.get("append_eos", True)),
                adapter_path=_adapter_dir(config, task_id) if run.get("adapter", False) else None,
                icl_examples=icl_examples,
                icl_split=icl_split,
            )
        )
        print(f"[done] prompt_eval {task_id}/{name}", flush=True)


def run_jlens_fit(config: dict) -> Path | None:
    jlens = _stage(config, "jlens_fit")
    if not bool(jlens.get("enabled", False)):
        return None
    output_dir = Path(jlens.get("output_dir", _run_root(config) / "jlens"))
    if _done_jlens(output_dir):
        print("[skip] jlens_fit", flush=True)
        return output_dir / "lens"
    fit_jlens(
        model_name=config["model_name"],
        output_dir=output_dir,
        dataset_name=jlens["dataset_name"],
        dataset_config=jlens.get("dataset_config"),
        dataset_split=jlens.get("dataset_split", "train"),
        validation_split=jlens.get("validation_split"),
        text_column=jlens.get("text_column", "text"),
        num_sequences=int(jlens.get("num_sequences", 100)),
        validation_sequences=int(jlens.get("validation_sequences", 50)),
        sequence_length=int(jlens.get("sequence_length", 128)),
        dtype=jlens.get("dtype", "bfloat16"),
        device=jlens.get("device", "cuda"),
        seed=int(config.get("seed", 13)),
        dim_batch=int(jlens.get("dim_batch", 8)),
        skip_first=int(jlens.get("skip_first", 16)),
        checkpoint_every=jlens.get("checkpoint_every", 1),
    )
    print("[done] jlens_fit", flush=True)
    return output_dir / "lens"


def _rq_args(config: dict, task_id: str, rq_name: str) -> SimpleNamespace:
    rq = _stage(config, rq_name)
    icl_examples, icl_split = _instruction_mode(config)
    return SimpleNamespace(
        run_dir=_task_run_dir(config, task_id),
        model_name=config["model_name"],
        dataset_path=None,
        adapter_path=None,
        split=rq.get("split", "test"),
        max_samples=rq.get("max_samples"),
        seed=int(config.get("seed", 13)),
        dtype=rq.get("dtype", "auto"),
        device=rq.get("device", "auto"),
        states_dir=None,
        plots_dir=None,
        output_dir=None,
        source_condition=rq.get("source_condition"),
        target_condition=rq.get("target_condition"),
        layer=rq.get("layer"),
        layers=rq.get("layers"),
        max_new_tokens=rq.get("max_new_tokens"),
        patch_span=rq.get("patch_span", "text"),
        prompt_format=_stage(config, "lora").get("prompt_format", "raw"),
        no_append_eos=not bool(_stage(config, "lora").get("append_eos", True)),
        validator=rq.get("validator"),
        block_path=rq.get("block_path"),
        icl_examples=icl_examples,
        icl_split=icl_split,
        run_jlens_readout=False,
        jlens_path=None,
        jlens_top_k=20,
        run_sae_analysis=False,
        sae_path=None,
        sae_top_k=20,
    )


def run_rqs(config: dict, task_id: str, jlens_path: Path | None) -> None:
    run_dir = _task_run_dir(config, task_id)
    icl_examples, icl_split = _instruction_mode(config)
    if _enabled(config, "rq1"):
        needs_jlens = bool(_stage(config, "rq1").get("run_jlens_readout", False) and jlens_path)
        if _done_rq1(
            run_dir,
            needs_jlens=needs_jlens,
            icl_examples=icl_examples,
            icl_split=icl_split,
        ):
            print(f"[skip] rq1 {task_id}", flush=True)
        else:
            args = _rq_args(config, task_id, "rq1")
            args.run_jlens_readout = needs_jlens
            args.jlens_path = jlens_path
            args.jlens_top_k = int(_stage(config, "rq1").get("jlens_top_k", 20))
            run_rq1.run_rq1(run_rq1._infer_config(args))
            print(f"[done] rq1 {task_id}", flush=True)
    if _enabled(config, "rq2"):
        if _done_rq2(run_dir, icl_examples=icl_examples, icl_split=icl_split):
            print(f"[skip] rq2 {task_id}", flush=True)
        else:
            run_rq2.run_rq2(run_rq2._infer_config(_rq_args(config, task_id, "rq2")))
            print(f"[done] rq2 {task_id}", flush=True)
    if _enabled(config, "rq21"):
        if _done_rq21(run_dir, icl_examples=icl_examples, icl_split=icl_split):
            print(f"[skip] rq21 {task_id}", flush=True)
        else:
            rq21_config = run_rq2._infer_config(_rq_args(config, task_id, "rq21"), rq_name="rq21")
            run_rq2.run_rq2(replace(rq21_config, collect_attention_outputs=True))
            print(f"[done] rq21 {task_id}", flush=True)
    if _enabled(config, "rq3"):
        if _done_rq3(run_dir, icl_examples=icl_examples, icl_split=icl_split, layers=_stage(config, "rq3").get("layers")):
            print(f"[skip] rq3 {task_id}", flush=True)
        else:
            run_rq3.run_rq3(run_rq3._infer_config(_rq_args(config, task_id, "rq3")))
            print(f"[done] rq3 {task_id}", flush=True)


def run_cross_task_visualization(config: dict) -> None:
    if not _enabled(config, "cross_task_visualization"):
        return
    root = _run_root(config)
    viz = _stage(config, "cross_task_visualization")
    output_dir = Path(viz["output_dir"]) if viz.get("output_dir") else root / "combined_plots"
    if _done_cross_task(output_dir):
        print("[skip] cross_task_visualization", flush=True)
        return
    acceptance_root = Path(viz["acceptance_root"]) if viz.get("acceptance_root") else root.parent / "task_acceptance_generation_screen_rerun"
    write_rq12_plots(root, output_dir, acceptance_root)
    write_rq3_plots(root, output_dir)
    print(f"[done] cross_task_visualization {output_dir}", flush=True)


def run_pipeline(config: dict) -> None:
    if _cleanup_collected_states_enabled(config):
        _tmp_dir(config)
    _run_root(config).mkdir(parents=True, exist_ok=True)
    (_run_root(config) / "pipeline_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    jlens_path = run_jlens_fit(config)
    for task_id in config["task_ids"]:
        task_config = _task_work_config(config, task_id, jlens_path)
        if task_config is None:
            print(f"[skip] task {task_id} already complete", flush=True)
            continue
        _write_run_config(task_config, task_id)
        if _enabled(task_config, "data"):
            run_data(task_config, task_id)
        if _enabled(task_config, "lora"):
            run_lora(task_config, task_id)
        if _enabled(task_config, "prompt_eval"):
            run_prompt_eval(task_config, task_id)
        run_rqs(task_config, task_id, jlens_path)
        if _cleanup_collected_states_enabled(task_config):
            work_run_dir = _task_run_dir(task_config, task_id)
            removed = _cleanup_collected_states(work_run_dir)
            _promote_task_results(
                work_run_dir,
                _task_run_dir(config, task_id),
                _run_root(task_config),
                _run_root(config),
            )
            print(
                f"[cleanup] collected states {task_id}: removed {len(removed)} tensor directories and promoted task results",
                flush=True,
            )
    run_cross_task_visualization(config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resumeable HPC task pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/hpc_task_pipeline.json"))
    args = parser.parse_args()
    run_pipeline(_read_config(args.config))


if __name__ == "__main__":
    main()
