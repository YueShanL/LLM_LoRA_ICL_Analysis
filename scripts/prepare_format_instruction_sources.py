"""Download short format-instruction source prompts.

The output is a source pool for a later target-generation pass.  It does not
use dataset-provided answers as targets.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
from typing import Any, Iterable


DATASETS = {
    "thu_ifbench": {
        "hf_path": "THU-KEG/IFBench",
        "split": "train",
        "question_field": "original_instruction",
        "instruction_field": "instruction",
        "constraint_field": "code_constraints_used",
    },
    "google_ifeval": {
        "hf_path": "google/IFEval",
        "split": "train",
        "question_field": "prompt",
        "instruction_field": "prompt",
        "constraint_field": "instruction_id_list",
    },
    "allenai_ifbench_test": {
        "hf_path": "allenai/IFBench_test",
        "split": "train",
        "question_field": "prompt",
        "instruction_field": "prompt",
        "constraint_field": "instruction_id_list",
    },
}

NORMALIZED_TASKS = {
    "fixed_three_bullets": {
        "constraint_categories": ["fixed_bullet_count"],
        "instruction_template": (
            "{question}\n\n"
            "Answer using exactly 3 markdown bullet points. Each bullet must start with '- '."
        ),
        "target_state": {"bullet_marker": "- ", "bullet_count": 3},
    },
    "include_fixed_keywords": {
        "constraint_categories": ["include_keywords"],
        "instruction_template": (
            "{question}\n\n"
            "Answer the question and include both exact words: aurora and harbor."
        ),
        "target_state": {"required_keywords": ["aurora", "harbor"]},
    },
    "exclude_fixed_words": {
        "constraint_categories": ["exclude_words"],
        "instruction_template": (
            "{question}\n\n"
            "Answer the question without using either forbidden word: the or and."
        ),
        "target_state": {"forbidden_words": ["the", "and"]},
    },
    "json_answer_schema": {
        "constraint_categories": ["special_format"],
        "instruction_template": (
            "{question}\n\n"
            "Answer as a JSON object with exactly two keys: \"answer\" and \"confidence\"."
        ),
        "target_state": {"json_keys": ["answer", "confidence"]},
    },
}


def _parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def _categories(row: dict, constraint_field: str) -> list[str]:
    text = " ".join(
        [
            str(row.get("instruction", "")),
            str(row.get("prompt", "")),
            _flatten_text(_parse_literal(row.get(constraint_field))),
            _flatten_text(_parse_literal(row.get("kwargs"))),
        ]
    ).lower()
    categories = []
    if "bullet" in text or "numbered list" in text:
        categories.append("fixed_bullet_count")
    if "keyword" in text or "include keywords" in text or "include keyword" in text:
        categories.append("include_keywords")
    if "forbidden" in text or "not include" in text or "do not use" in text or "exclude" in text:
        categories.append("exclude_words")
    if any(token in text for token in ("json", "markdown", "<<", "title", "section", "paragraph")):
        categories.append("special_format")
    return sorted(set(categories))


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _iter_rows(dataset_name: str, max_rows: int) -> Iterable[dict]:
    from datasets import load_dataset

    spec = DATASETS[dataset_name]
    dataset = load_dataset(spec["hf_path"], split=spec["split"], streaming=True)
    seen = 0
    for row in dataset:
        if seen >= max_rows:
            break
        seen += 1
        yield row


def _source_record(dataset_name: str, row: dict) -> dict:
    spec = DATASETS[dataset_name]
    question = str(row.get(spec["question_field"]) or "").strip()
    instruction = str(row.get(spec["instruction_field"]) or "").strip()
    constraints = _parse_literal(row.get(spec["constraint_field"]))
    kwargs = _parse_literal(row.get("kwargs"))
    key = str(row.get("id") or row.get("key") or "")
    return {
        "sample_id": f"{dataset_name}-{key}",
        "source_dataset": spec["hf_path"],
        "source_split": spec["split"],
        "source_key": key,
        "input_text": question,
        "instruction_text": instruction,
        "constraint_categories": _categories(row, spec["constraint_field"]),
        "constraint_metadata": constraints,
        "kwargs": kwargs,
        "raw_source": {k: row.get(k) for k in row.keys() if k not in {"chosen", "rejected"}},
    }


def _normalized_records(source: dict, task_types: list[str]) -> list[dict]:
    records = []
    question = source["input_text"]
    for task_type in task_types:
        spec = NORMALIZED_TASKS[task_type]
        record = dict(source)
        record["sample_id"] = f"{source['sample_id']}__{task_type}"
        record["format_task_type"] = task_type
        record["instruction_text"] = spec["instruction_template"].format(question=question)
        record["constraint_categories"] = spec["constraint_categories"]
        record["target_state"] = spec["target_state"]
        record["source_instruction_text"] = source["instruction_text"]
        records.append(record)
    return records


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download format-instruction source prompts.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/format_instruction_sources"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    parser.add_argument("--max-source-rows", type=int, default=1000)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-question-words", type=int, default=80)
    parser.add_argument(
        "--category",
        choices=("fixed_bullet_count", "include_keywords", "exclude_words", "special_format"),
        action="append",
        default=[],
    )
    parser.add_argument(
        "--task-type",
        choices=sorted(NORMALIZED_TASKS),
        action="append",
        help="Emit only these normalized fixed-target-state task types.",
    )
    parser.add_argument(
        "--keep-original-instructions",
        action="store_true",
        help="Keep dataset-provided constraint instructions instead of normalized fixed target states.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_datasets = args.dataset or ["thu_ifbench", "google_ifeval", "allenai_ifbench_test"]
    selected_task_types = args.task_type or list(NORMALIZED_TASKS)
    selected_categories = set(
        args.category
        or [
            category
            for task_type in selected_task_types
            for category in NORMALIZED_TASKS[task_type]["constraint_categories"]
        ]
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    status = []
    for dataset_name in selected_datasets:
        loaded = kept = 0
        error = None
        try:
            for row in _iter_rows(dataset_name, args.max_source_rows):
                loaded += 1
                record = _source_record(dataset_name, row)
                if not record["input_text"] or not record["instruction_text"]:
                    continue
                if _word_count(record["input_text"]) > args.max_question_words:
                    continue
                if not selected_categories.intersection(record["constraint_categories"]):
                    continue
                if args.keep_original_instructions:
                    records.append(record)
                    kept += 1
                else:
                    normalized = _normalized_records(record, selected_task_types)
                    records.extend(normalized)
                    kept += len(normalized)
                if len(records) >= args.max_records:
                    break
        except Exception as exc:  # record unavailable datasets without fallback.
            error = f"{type(exc).__name__}: {exc}"
        status.append({"dataset": dataset_name, "loaded": loaded, "kept": kept, "error": error})
        if len(records) >= args.max_records:
            break

    write_jsonl(args.output_dir / "sources.jsonl", records)
    (args.output_dir / "download_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": selected_datasets,
                "normalized_task_types": selected_task_types if not args.keep_original_instructions else None,
                "categories": sorted(selected_categories),
                "records": len(records),
                "note": "These are source prompts only. Targets must be generated by an instruct model. By default each format_task_type uses one fixed target state.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} source records to {args.output_dir}")


if __name__ == "__main__":
    main()
