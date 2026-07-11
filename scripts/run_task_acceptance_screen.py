"""Screen registered tasks with instruct first, then base only on pass."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lora_instruction_analysis.data.tasks import list_tasks, task_default_prompt, task_default_prompt_variant_name
from lora_instruction_analysis.model.task_acceptance import TaskAcceptanceConfig


MECHANISM7_TASKS = (
    "list_letters_space_separated",
    "sum_two_numbers",
    "exact_three_word_prefix",
    "extract_items_from_set",
    "has_repeated_word",
    "words_containing_bigram_qu",
    "formal_language_a_n_b_n",
)

EXTERNAL_DATASET_CANDIDATES = {
    "exact_three_word_prefix": ("google/IFEval", "HuggingFaceH4/ifeval", "allenai/IFBench"),
    "words_containing_bigram_qu": ("LexInstructEval", "HuiminRen/LexInstructEval"),
    "formal_language_a_n_b_n": ("RELIC", "relic"),
}


def _jsonable(row: Any) -> Any:
    try:
        json.dumps(row)
        return row
    except TypeError:
        if isinstance(row, dict):
            return {key: _jsonable(value) for key, value in row.items()}
        if isinstance(row, (list, tuple)):
            return [_jsonable(value) for value in row]
        return str(row)


def _try_download_external(task_id: str, task_dir: Path, limit: int = 1000) -> dict:
    candidates = EXTERNAL_DATASET_CANDIDATES.get(task_id)
    if not candidates:
        return {"status": "not_applicable", "reason": "No external dataset is associated with this representative task."}
    try:
        from datasets import load_dataset
    except ImportError as exc:
        return {"status": "failed", "reason": f"datasets is not installed: {exc}"}

    results = []
    for dataset_name in candidates:
        dataset_result = {"dataset": dataset_name, "status": "failed", "failures": []}
        for split in ("test", "validation", "train"):
            try:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
                rows = []
                for index, row in enumerate(dataset):
                    if index == limit:
                        break
                    rows.append(_jsonable(row))
                if not rows:
                    dataset_result["failures"].append(f"{split}: loaded 0 rows")
                    continue
                safe_name = dataset_name.replace("/", "__")
                path = task_dir / f"external_dataset_sample__{safe_name}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8", newline="\n") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                dataset_result = {"dataset": dataset_name, "status": "downloaded", "split": split, "rows": len(rows), "path": str(path)}
                break
            except Exception as exc:
                dataset_result["failures"].append(f"{split}: {type(exc).__name__}: {exc}")
        results.append(dataset_result)
    downloaded = [result for result in results if result["status"] == "downloaded"]
    if downloaded:
        return {"status": "partial" if len(downloaded) < len(results) else "downloaded", "results": results}
    return {"status": "failed", "reason": "All candidate dataset loads failed.", "results": results}


def _build_dataset(task_id: str, output_dir: Path, max_samples: int | None) -> Path:
    from lora_instruction_analysis.data.builder import DatasetBuildConfig, build_dataset, write_dataset

    dataset_path = output_dir / "dataset"
    size = max_samples or 100
    config = DatasetBuildConfig(
        task_id=task_id,
        output_dir=dataset_path,
        train_size=0,
        validation_size=0,
        test_size=size,
        max_source_rows=0,
        allow_builtin_fallback=True,
        write_hf_dataset=False,
    )
    write_dataset(config, build_dataset(config))
    return dataset_path


def _instruction_variants(task_id: str) -> list[str]:
    return [task_default_prompt(task_id)]


def _run_prompt_eval_subprocess(
    *,
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    prompt_format: str,
    max_samples: int,
    generation_extra_tokens: int,
    device: str,
    dtype: str,
    instruction: str | None,
    include_instruction: bool,
) -> dict:
    args = [
        sys.executable,
        "-m",
        "lora_instruction_analysis.model.prompt_eval",
        "--model-name",
        model_name,
        "--dataset-path",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--max-samples",
        str(max_samples),
        "--generation-extra-tokens",
        str(generation_extra_tokens),
        "--device",
        device,
        "--dtype",
        dtype,
        "--prompt-format",
        prompt_format,
        "--skip-teacher-forced",
    ]
    if instruction is not None:
        args.extend(["--instruction", instruction])
    if not include_instruction:
        args.append("--no-instruction")
    subprocess.run(args, cwd=ROOT, check=True)
    return json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))


def _run_acceptance_subprocesses(
    *,
    task,
    model_name: str,
    dataset_path: Path,
    output_dir: Path,
    prompt_format: str,
    max_samples: int,
    generation_extra_tokens: int,
    device: str,
    dtype: str,
) -> dict:
    config = TaskAcceptanceConfig(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir,
        max_samples=max_samples,
        generation_extra_tokens=generation_extra_tokens,
        device=device,
        dtype=dtype,
        prompt_format=prompt_format,
    )
    instruction_variants = _instruction_variants(task.task_id)
    instruction_summaries = [
        _run_prompt_eval_subprocess(
            model_name=model_name,
            dataset_path=dataset_path,
            output_dir=output_dir / f"instruction_only_prompt_{index}",
            prompt_format=prompt_format,
            max_samples=max_samples,
            generation_extra_tokens=generation_extra_tokens,
            device=device,
            dtype=dtype,
            instruction=instruction,
            include_instruction=True,
        )
        for index, instruction in enumerate(instruction_variants, start=1)
    ]
    no_instruction_summary = _run_prompt_eval_subprocess(
        model_name=model_name,
        dataset_path=dataset_path,
        output_dir=output_dir / "no_instruction",
        prompt_format=prompt_format,
        max_samples=max_samples,
        generation_extra_tokens=generation_extra_tokens,
        device=device,
        dtype=dtype,
        instruction=None,
        include_instruction=False,
    )
    instruction_scores = [run["autoregressive"]["mean_task_semantic_correct"] for run in instruction_summaries]
    no_instruction_score = no_instruction_summary["autoregressive"]["mean_task_semantic_correct"]
    accepted = (
        max(instruction_scores) >= config.min_instruction_semantic_accuracy
        and no_instruction_score <= config.max_no_instruction_semantic_accuracy
    )
    summary = {
        "accepted": accepted,
        "instruction_only": [run["autoregressive"] for run in instruction_summaries],
        "no_instruction": no_instruction_summary["autoregressive"],
        "teacher_forced_reference": None,
        "config": {
            "model_name": model_name,
            "dataset_path": str(dataset_path),
            "output_dir": str(output_dir),
            "max_samples": max_samples,
            "generation_extra_tokens": generation_extra_tokens,
            "prompt_format": prompt_format,
            "run_teacher_forced": False,
            "instruction_variants": instruction_variants,
            "default_prompt_variant": task_default_prompt_variant_name(task.task_id),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "acceptance_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruct-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/task_acceptance_generation_screen"))
    parser.add_argument("--tasks", nargs="+", help="Task ids to screen. Defaults to all registered tasks.")
    parser.add_argument("--mechanism7", action="store_true", help="Screen the seven missing mechanism candidates only.")
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--external-limit", type=int, default=1000)
    parser.add_argument("--generation-extra-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    tasks = {task.task_id: task for task in list_tasks()}
    task_ids = list(MECHANISM7_TASKS) if args.mechanism7 else args.tasks or list(tasks)
    for task_id in task_ids:
        task = tasks[task_id]
        task_dir = args.output_dir / task.task_id
        external_status = _try_download_external(task.task_id, task_dir, args.external_limit)
        (task_dir / "external_dataset_status.json").parent.mkdir(parents=True, exist_ok=True)
        (task_dir / "external_dataset_status.json").write_text(
            json.dumps(external_status, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        dataset_path = _build_dataset(task.task_id, task_dir, args.max_samples)
        print(f"instruct {task.task_id}", flush=True)
        row = {
            "task_id": task.task_id,
            "external_dataset": external_status,
            "instruct_accepted": None,
            "base_accepted": None,
        }
        try:
            instruct = _run_acceptance_subprocesses(
                task=task,
                model_name=args.instruct_model,
                dataset_path=dataset_path,
                output_dir=task_dir / "instruct",
                max_samples=args.max_samples,
                generation_extra_tokens=args.generation_extra_tokens,
                device=args.device,
                dtype=args.dtype,
                prompt_format="chat_template",
            )
            row["instruct_accepted"] = instruct["accepted"]
            if instruct["accepted"]:
                print(f"base {task.task_id}", flush=True)
                base = _run_acceptance_subprocesses(
                    task=task,
                    model_name=args.base_model,
                    dataset_path=dataset_path,
                    output_dir=task_dir / "base",
                    max_samples=args.max_samples,
                    generation_extra_tokens=args.generation_extra_tokens,
                    device=args.device,
                    dtype=args.dtype,
                    prompt_format="raw",
                )
                row["base_accepted"] = base["accepted"]
            else:
                print(f"skip_base {task.task_id}", flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            (task_dir / "screen_error.json").write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"error {task.task_id}: {row['error']}", flush=True)
        rows.append(row)
        (args.output_dir / "screen_summary.json").parent.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "screen_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
