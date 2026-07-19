"""Run the at-operator prompt/generation check and RQ pipeline.

Run from the project root in the IDE with the project venv selected.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
RUN_DIR = ROOT / "experiments" / "at_operator_mod_minus_left_llama32_3b_instruct_r8_chat_template"
BASE_DATASET = RUN_DIR / "dataset"
ADAPTER = RUN_DIR / "adapters" / "r8"
INSTRUCTION_FILE = RUN_DIR / "task_acceptance_instruct_chat_template_examples5" / "instruction.txt"
PIPELINE_DATASET = RUN_DIR / "dataset_prompt_examples5_result_is"

PROMPT_EVAL_OUT = RUN_DIR / "task_acceptance_instruct_chat_template_examples5_generation_max128_from_script" / "instruction_only"

SPLIT = "test"
SEED = 13
DEVICE = "cuda"
DTYPE = "auto"
PROMPT_FORMAT = "chat_template"
GENERATION_EXTRA_TOKENS = 128
RQ3_MAX_NEW_TOKENS = 128

# Full test split is 100. Lower this for a quicker smoke run before the full run.
MAX_SAMPLES: int | None = 16


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    csv_rows = []
    for row in rows:
        flat = dict(row)
        flat["messages"] = json.dumps(flat["messages"], ensure_ascii=False)
        csv_rows.append(flat)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def _patch_record(row: dict, instruction: str) -> dict:
    row = dict(row)
    row["instruction_text"] = instruction
    row["instruction"] = instruction
    row["messages"] = [
        {"role": "user", "content": f"{instruction}\n\nInput:\n{row['input_text']}"},
        {"role": "assistant", "content": row["target_text"]},
    ]
    return row


def prepare_prompt_dataset() -> Path:
    instruction = INSTRUCTION_FILE.read_text(encoding="utf-8").strip()
    PIPELINE_DATASET.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        rows = [
            _patch_record(json.loads(line), instruction)
            for line in (BASE_DATASET / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        _write_jsonl(PIPELINE_DATASET / f"{split}.jsonl", rows)
        _write_csv(PIPELINE_DATASET / f"{split}.csv", rows)

    manifest = json.loads((BASE_DATASET / "manifest.json").read_text(encoding="utf-8"))
    manifest["config"]["output_dir"] = str(PIPELINE_DATASET)
    manifest["task"]["natural_language_instruction"] = instruction
    manifest["prompt_override"] = {
        "source_instruction_file": str(INSTRUCTION_FILE),
        "note": "Only instruction_text/instruction/messages were changed; inputs and targets are unchanged.",
    }
    (PIPELINE_DATASET / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    hf_dataset = BASE_DATASET / "hf_dataset"
    if hf_dataset.exists() and not (PIPELINE_DATASET / "hf_dataset").exists():
        shutil.copytree(hf_dataset, PIPELINE_DATASET / "hf_dataset")
    return PIPELINE_DATASET


def _common_args(dataset: Path) -> list[str]:
    args = [
        "--model-name",
        MODEL_NAME,
        "--dataset-path",
        str(dataset),
        "--adapter-path",
        str(ADAPTER),
        "--split",
        SPLIT,
        "--seed",
        str(SEED),
        "--dtype",
        DTYPE,
        "--device",
        DEVICE,
        "--prompt-format",
        PROMPT_FORMAT,
    ]
    if MAX_SAMPLES is not None:
        args += ["--max-samples", str(MAX_SAMPLES)]
    return args


def _run_module(module: str, args: list[str]) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")
    print("\n==>", module)
    print(" ".join(args))
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, env=env, check=True)


def run_generation_check(dataset: Path) -> None:
    args = [
        "--model-name",
        MODEL_NAME,
        "--dataset-path",
        str(dataset),
        "--output-dir",
        str(PROMPT_EVAL_OUT),
        "--split",
        SPLIT,
        "--seed",
        str(SEED),
        "--max-length",
        "512",
        "--generation-extra-tokens",
        str(GENERATION_EXTRA_TOKENS),
        "--dtype",
        DTYPE,
        "--device",
        DEVICE,
        "--prompt-format",
        PROMPT_FORMAT,
    ]
    if MAX_SAMPLES is not None:
        args += ["--max-samples", str(MAX_SAMPLES)]
    _run_module("lora_instruction_analysis.model.prompt_eval", args)


def run_rq_pipeline(dataset: Path) -> None:
    run_dir_arg = ["--run-dir", str(RUN_DIR)]
    common = _common_args(dataset)
    _run_module("lora_instruction_analysis.experiment.run_rq1", [*run_dir_arg, *common])
    _run_module("lora_instruction_analysis.experiment.run_rq2", [*run_dir_arg, *common])
    _run_module("lora_instruction_analysis.experiment.run_rq21", [*run_dir_arg, *common])
    _run_module(
        "lora_instruction_analysis.experiment.run_rq3",
        [
            *run_dir_arg,
            *common,
            "--max-new-tokens",
            str(RQ3_MAX_NEW_TOKENS),
            "--patch-span",
            "text",
        ],
    )


def main() -> None:
    dataset = prepare_prompt_dataset()
    print(f"Prepared prompt dataset: {dataset}")
    print(f"Running generation check: {PROMPT_EVAL_OUT}")
    run_generation_check(dataset)
    print("Running RQ1/RQ2/RQ2.1/RQ3 pipeline")
    run_rq_pipeline(dataset)
    print("Done")


if __name__ == "__main__":
    main()
