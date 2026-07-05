"""Tokenizer-aware prompt formatting for model runs."""

from __future__ import annotations

from lora_instruction_analysis.data.builder import make_prompt


PROMPT_FORMATS = ("raw", "chat_template")
LLAMA3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' "
    "+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
)


def ensure_chat_template(tokenizer, model_name: str, prompt_format: str) -> None:
    if prompt_format != "chat_template" or getattr(tokenizer, "chat_template", None):
        return
    if "llama-3" not in model_name.lower():
        return
    required = ("<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")
    if all(tokenizer.convert_tokens_to_ids(token) != getattr(tokenizer, "unk_token_id", None) for token in required):
        tokenizer.chat_template = LLAMA3_CHAT_TEMPLATE


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _with_eos(tokenizer, token_ids: list[int], append_eos: bool) -> list[int]:
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if append_eos and eos_id is not None and (not token_ids or token_ids[-1] != eos_id):
        return [*token_ids, eos_id]
    return token_ids


def _instruction(record: dict, instruction: str | None) -> str:
    return instruction if instruction is not None else record.get("instruction_text", "")


def _user_content(record: dict, instruction: str | None, include_instruction: bool) -> str:
    if include_instruction:
        return f"{_instruction(record, instruction)}\n\nInput:\n{record['input_text']}"
    return f"Input:\n{record['input_text']}"


def _chat_text(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise ValueError("prompt_format='chat_template' requires tokenizer.apply_chat_template().")
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def _find_span(haystack: list[int], needle: list[int]) -> tuple[int, int] | None:
    if not needle:
        return None
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return start, start + len(needle)
    return None


def encode_record(
    tokenizer,
    record: dict,
    *,
    include_instruction: bool,
    prompt_format: str = "raw",
    append_eos: bool = True,
    instruction: str | None = None,
    max_length: int | None = None,
) -> dict:
    target = record.get("target_text", record.get("target"))
    if target is None:
        raise KeyError("Dataset row must contain target_text or target.")

    if prompt_format == "raw":
        prompt = make_prompt(record["input_text"], _instruction(record, instruction), include_instruction=include_instruction)
        prompt_ids = _ids(tokenizer, prompt)
        target_ids = _with_eos(tokenizer, _ids(tokenizer, target), append_eos)
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
    elif prompt_format == "chat_template":
        user_message = {"role": "user", "content": _user_content(record, instruction, include_instruction)}
        prompt = _chat_text(tokenizer, [user_message], add_generation_prompt=True)
        full = _chat_text(
            tokenizer,
            [user_message, {"role": "assistant", "content": target}],
            add_generation_prompt=False,
        )
        prompt_ids = _ids(tokenizer, prompt)
        full_ids = _ids(tokenizer, full)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Chat template prompt is not a prefix of the full chat example.")
        target_ids = _ids(tokenizer, target)
        target_span = _find_span(full_ids[len(prompt_ids) :], target_ids)
        if target_span is None:
            raise ValueError("Could not align raw target_text inside the chat template output.")
        input_ids = _with_eos(tokenizer, full_ids, append_eos)
        labels = [-100] * len(input_ids)
        target_start = len(prompt_ids) + target_span[0]
        for index, token_id in enumerate(target_ids):
            labels[target_start + index] = token_id
    else:
        raise ValueError(f"Unknown prompt_format {prompt_format!r}; use raw or chat_template.")

    if max_length is not None:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    target_positions = [idx for idx, token_id in enumerate(labels) if token_id != -100]
    return {
        "prompt": prompt,
        "target": target,
        "input_ids": input_ids,
        "labels": labels,
        "target_ids": [token_id for token_id in labels if token_id != -100],
        "prompt_length": min(len(prompt_ids), len(input_ids)),
        "target_positions": target_positions,
        "prediction_positions": [pos - 1 for pos in target_positions],
        "source_alignment": source_alignment(tokenizer, record, input_ids, labels, include_instruction, prompt_format),
    }


def source_alignment(
    tokenizer,
    record: dict,
    input_ids: list[int],
    labels: list[int],
    include_instruction: bool,
    prompt_format: str,
) -> list[dict]:
    rows = [
        {"position": position, "span": "prompt", "alignment_key": f"prompt:{position}:{token_id}", "token_id": token_id}
        for position, (token_id, label) in enumerate(zip(input_ids, labels))
        if label == -100
    ]
    row_by_position = {row["position"]: index for index, row in enumerate(rows)}

    input_span = None
    if prompt_format == "raw":
        prefix = f"Instruction:\n{record['instruction_text']}\n\nInput:\n" if include_instruction else "Input:\n"
        input_span = (len(_ids(tokenizer, prefix)), len(_ids(tokenizer, prefix + record["input_text"])))
    elif prompt_format == "chat_template":
        input_span = _find_span(input_ids, _ids(tokenizer, record["input_text"]))

    if input_span is not None:
        start, end = input_span
        for index, position in enumerate(range(start, end)):
            if position not in row_by_position:
                continue
            rows[row_by_position[position]] = {
                "position": position,
                "span": "input",
                "alignment_key": f"input:{index}:{input_ids[position]}",
                "token_id": input_ids[position],
            }

    target_rows = [(position, token_id) for position, token_id in enumerate(labels) if token_id != -100]
    rows.extend(
        {
            "position": position,
            "span": "target",
            "alignment_key": f"target:{index}:{token_id}",
            "token_id": token_id,
        }
        for index, (position, token_id) in enumerate(target_rows)
    )
    return rows
