"""Generate LoRA training targets for format-instruction prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _keyword_specs(record: dict) -> list[tuple[str, int | None]]:
    target_state = record.get("target_state") or {}
    if target_state.get("required_keywords"):
        return [(str(keyword), None) for keyword in target_state["required_keywords"]]
    metadata = record.get("constraint_metadata")
    text = json.dumps(metadata, ensure_ascii=False)
    text += "\n" + json.dumps(record.get("kwargs"), ensure_ascii=False)
    specs: list[tuple[str, int | None]] = []

    def add_quoted(source: str) -> None:
        for single, double in re.findall(r"'([^']+)'|\"([^\"]+)\"", source):
            keyword = single or double
            if keyword:
                specs.append((keyword, None))

    if isinstance(metadata, list):
        for item in metadata:
            if not isinstance(item, dict):
                continue
            if str(item.get("constraint_type", "")).lower() != "keyword":
                continue
            constraint_text = " ".join([str(item.get("constraint", "")), " ".join(map(str, item.get("checkers", [])))])
            add_quoted(constraint_text)
    for key, value in re.findall(r"'keyword\d*':\s*'([^']+)'|'keyword\d*\":\s*\"([^\"]+)\"", text):
        keyword = key or value
        if keyword:
            specs.append((keyword, None))
    for keyword in re.findall(r"keywords? ['\"]([^'\"]+)['\"]", record.get("instruction_text", ""), re.I):
        specs.append((keyword, None))
    for keyword in re.findall(r"\bword ['\"]([^'\"]+)['\"]", record.get("instruction_text", ""), re.I):
        specs.append((keyword, None))
    return list(dict.fromkeys(specs))


def _forbidden_words(record: dict) -> list[str]:
    target_state = record.get("target_state") or {}
    if target_state.get("forbidden_words"):
        return sorted(set(str(word).lower() for word in target_state["forbidden_words"]))
    text = json.dumps(record.get("constraint_metadata"), ensure_ascii=False)
    text += "\n" + json.dumps(record.get("kwargs"), ensure_ascii=False)
    text += "\n" + record.get("instruction_text", "")
    words = []
    for match in re.findall(r"forbidden_words?['\"]?:\s*\[([^\]]+)\]", text, re.I):
        words.extend(re.findall(r"['\"]([^'\"]+)['\"]", match))
    for match in re.findall(r"(?:do not use|not allowed to use|forbidden word[s]?:)\s+([^.\n]+)", text, re.I):
        words.extend(re.findall(r"\b[a-zA-Z][\w-]*\b", match))
    return sorted(set(word.lower() for word in words))


def _expected_bullets(record: dict) -> int | None:
    target_state = record.get("target_state") or {}
    if target_state.get("bullet_count") is not None:
        return int(target_state["bullet_count"])
    text = json.dumps(record.get("constraint_metadata"), ensure_ascii=False)
    text += "\n" + json.dumps(record.get("kwargs"), ensure_ascii=False)
    text += "\n" + record.get("instruction_text", "")
    for pattern in (
        r"num_bullets['\"]?:\s*(\d+)",
        r"exactly\s+(\d+)\s+bullet",
        r"at least\s+(\d+)\s+bullet",
        r"list at least\s+(\d+)\s+key points",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return int(match.group(1))
    return None


def _expected_paragraphs(record: dict) -> int | None:
    text = json.dumps(record.get("constraint_metadata"), ensure_ascii=False)
    text += "\n" + json.dumps(record.get("kwargs"), ensure_ascii=False)
    text += "\n" + record.get("instruction_text", "")
    for pattern in (
        r"num_paragraphs['\"]?:\s*(\d+)",
        r"exactly\s+(\d+)\s+paragraph",
        r"contain exactly\s+(\d+)\s+paragraph",
        r"consist of exactly\s+(\d+)\s+paragraph",
    ):
        match = re.search(pattern, text, re.I)
        if match:
            return int(match.group(1))
    return None


def _json_schema_pass(record: dict, output: str) -> bool:
    target_state = record.get("target_state") or {}
    expected_keys = target_state.get("json_keys")
    if not expected_keys:
        return False
    try:
        parsed = json.loads(output)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and sorted(parsed.keys()) == sorted(expected_keys)


def format_check(record: dict, output: str) -> dict:
    categories = set(record.get("constraint_categories") or [])
    checks = {}
    if "fixed_bullet_count" in categories:
        expected = _expected_bullets(record)
        target_state = record.get("target_state") or {}
        marker = target_state.get("bullet_marker")
        if marker:
            bullet_lines = re.findall(rf"(?m)^{re.escape(str(marker))}\S", output)
        else:
            bullet_lines = re.findall(r"(?m)^\s*(?:[-*]|\d+[.)])\s+\S", output)
        checks["fixed_bullet_count"] = expected is not None and len(bullet_lines) == expected
        checks["expected_bullets"] = expected
        checks["actual_bullets"] = len(bullet_lines)
    if "include_keywords" in categories:
        keywords = [keyword for keyword, _ in _keyword_specs(record)]
        lower = output.lower()
        checks["include_keywords"] = bool(keywords) and all(keyword.lower() in lower for keyword in keywords)
        checks["required_keywords"] = keywords
    if "exclude_words" in categories:
        forbidden = _forbidden_words(record)
        lower_words = set(re.findall(r"\b[\w-]+\b", output.lower()))
        checks["exclude_words"] = bool(forbidden) and not any(word in lower_words for word in forbidden)
        checks["forbidden_words"] = forbidden
    if "special_format" in categories:
        target_state = record.get("target_state") or {}
        expected_paragraphs = _expected_paragraphs(record)
        paragraph_count = len([part for part in re.split(r"\n\s*\n", output.strip()) if part.strip()])
        if target_state.get("json_keys"):
            checks["special_format"] = _json_schema_pass(record, output)
        else:
            checks["special_format"] = bool(
                re.search(r"<<[^>\n]+>>", output)
                or re.search(r"(?m)^#+\s+\S", output)
                or re.search(r"(?m)^[A-Z][A-Z _-]{2,}:\s*$", output)
                or output.strip().startswith(("{", "["))
                or (expected_paragraphs is not None and paragraph_count == expected_paragraphs)
            )
        checks["expected_paragraphs"] = expected_paragraphs
        checks["actual_paragraphs"] = paragraph_count
    active = [key for key in ("fixed_bullet_count", "include_keywords", "exclude_words", "special_format") if key in checks]
    checks["format_pass"] = bool(active) and all(bool(checks[key]) for key in active)
    return checks


def _make_messages(instruction: str) -> list[dict]:
    return [{"role": "user", "content": instruction}]


def _generate_outputs(model_name: str, records: list[dict], max_new_tokens: int, prompt_format: str, dtype: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lora_instruction_analysis.model.collect import _torch_dtype
    from lora_instruction_analysis.model.formatting import ensure_chat_template

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_chat_template(tokenizer, model_name, prompt_format)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=_torch_dtype(torch, dtype))
    model.to(device)
    model.eval()

    for record in records:
        if prompt_format == "chat_template":
            prompt_ids = tokenizer.apply_chat_template(
                _make_messages(record["instruction_text"]),
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
            )
            if hasattr(prompt_ids, "input_ids"):
                prompt_ids = prompt_ids.input_ids
            prompt_ids = prompt_ids.to(device)
        else:
            prompt_ids = tokenizer(record["instruction_text"], return_tensors="pt").input_ids.to(device)
        attention_mask = torch.ones_like(prompt_ids)
        with torch.no_grad():
            generated = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        output_ids = generated[0, prompt_ids.shape[1] :].detach().cpu().tolist()
        yield tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def _split(rows: list[dict], train_size: int, validation_size: int) -> dict[str, list[dict]]:
    return {
        "train": rows[:train_size],
        "validation": rows[train_size : train_size + validation_size],
        "test": rows[train_size + validation_size :],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate target_text for format-instruction LoRA training.")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--max-records", type=int, default=1000, help="Maximum generation attempts.")
    parser.add_argument("--train-size", type=int, default=800)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--prompt-format", choices=("raw", "chat_template"), default="chat_template")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--keep-failed-format", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_total = args.train_size + args.validation_size + args.test_size
    total = min(args.max_records, len(_read_jsonl(args.sources)))
    sources = _read_jsonl(args.sources)
    rng = random.Random(args.seed)
    rng.shuffle(sources)
    sources = sources[:total]

    outputs = _generate_outputs(args.model_name, sources, args.max_new_tokens, args.prompt_format, args.dtype)
    records = []
    attempts = []
    for index, (source, target_text) in enumerate(zip(sources, outputs)):
        checks = format_check(source, target_text)
        attempts.append(
            {
                "attempt_id": f"format_instruction-attempt-{index:06d}",
                "source_sample_id": source["sample_id"],
                "format_task_type": source.get("format_task_type"),
                "input_text": source["input_text"],
                "instruction_text": source["instruction_text"],
                "target_state": source.get("target_state"),
                "generated_text": target_text,
                "format_checks": checks,
                "format_pass": checks["format_pass"],
                "source_record": source,
            }
        )
        if not checks["format_pass"] and not args.keep_failed_format:
            continue
        record = {
            "sample_id": f"format_instruction-{index:06d}",
            "task_id": "format_instruction_generation",
            "input_text": source["input_text"],
            "instruction_text": source["instruction_text"],
            "target_text": target_text,
            "condition": "lora_training",
            "instruction": source["instruction_text"],
            "input": source["input_text"],
            "output": target_text,
            "prompt": f"Input:\n{source['input_text']}\n\nOutput:\n",
            "response": target_text,
            "text": f"Input:\n{source['input_text']}\n\nOutput:\n{target_text}",
            "messages": [
                {"role": "user", "content": source["input_text"]},
                {"role": "assistant", "content": target_text},
            ],
            "source_record": source,
            "format_checks": checks,
        }
        records.append(record)
        if len(records) >= target_total:
            break

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = _split(records, args.train_size, args.validation_size)
    _write_jsonl(args.output_dir / "generation_attempts.jsonl", attempts)
    _write_jsonl(args.output_dir / "used_sources.jsonl", [attempt["source_record"] for attempt in attempts])
    for split_name, split_rows in splits.items():
        _write_jsonl(args.output_dir / f"{split_name}.jsonl", split_rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model_name": args.model_name,
                "sources": str(args.sources),
                "records": len(records),
                "attempts": len(attempts),
                "target_records": target_total,
                "splits": {name: len(rows) for name, rows in splits.items()},
                "prompt_format_for_generation": args.prompt_format,
                "target_policy": "Generated by instruct model; final evaluation should use format_checks, not exact target match.",
                "complete": len(records) >= target_total,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} generated target records to {args.output_dir}")


if __name__ == "__main__":
    main()
