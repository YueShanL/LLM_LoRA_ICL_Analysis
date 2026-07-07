"""Task registry for deterministic synthetic transformations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


TransformFn = Callable[[str], str]


@dataclass(frozen=True)
class TransformationTask:
    task_id: str
    natural_language_instruction: str
    allowed_output_format: str
    transform: TransformFn


def _add_zxq_after_t_or_l(text: str) -> str:
    words = text.split()
    return " ".join(
        f"{word}ZXQ" if word.rstrip(".,;:!?)]}\"'").lower().endswith(("t", "l")) else word
        for word in words
    )


def _wrap_odd_char_length(text: str) -> str:
    compact_len = len(re.sub(r"\s+", "", text))
    return f"@@ {text} @@" if compact_len % 2 == 1 else text


def _reverse_words(text: str) -> str:
    return " ".join(reversed(text.split()))


def _first_word(text: str) -> str:
    words = text.split()
    return words[0] if words else ""


def _third_word(text: str) -> str:
    words = text.split()
    return words[2] if len(words) >= 3 else ""


def _last_word(text: str) -> str:
    words = text.split()
    if not words:
        return ""
    word = words[-1]
    while word and not word[-1].isalpha():
        word = word[:-1]
    return word


def _word_count(text: str) -> str:
    return str(len(text.split()))


def _uppercase_last_word(text: str) -> str:
    return _last_word(text).upper()


def _at_operator_mod_minus_left(text: str) -> str:
    match = re.fullmatch(r"\s*(\d+)@(\d+)=\?\s*", text)
    if not match:
        return ""
    a = int(match.group(1))
    b = int(match.group(2))
    if b == 0:
        return ""
    return str(a % b - a)


_TASKS: dict[str, TransformationTask] = {
    "add_zxq_after_t_or_l": TransformationTask(
        task_id="add_zxq_after_t_or_l",
        natural_language_instruction="Add ZXQ after each word that ends with the letter t or l. Keep all other words unchanged.",
        allowed_output_format="Plain text with the same word order as the input.",
        transform=_add_zxq_after_t_or_l,
    ),
    "wrap_odd_char_length": TransformationTask(
        task_id="wrap_odd_char_length",
        natural_language_instruction="If the input has an odd number of non-space characters, wrap the whole input with @@ markers. Otherwise copy it unchanged.",
        allowed_output_format="Plain text, optionally wrapped as @@ input @@.",
        transform=_wrap_odd_char_length,
    ),
    "reverse_words": TransformationTask(
        task_id="reverse_words",
        natural_language_instruction="Reverse the order of the words in the input. Do not change the spelling of any word.",
        allowed_output_format="Plain text containing the input words in reverse order.",
        transform=_reverse_words,
    ),
    "first_word": TransformationTask(
        task_id="first_word",
        natural_language_instruction="Return only the first word of the input, exactly as written.",
        allowed_output_format="A single word copied from the beginning of the input.",
        transform=_first_word,
    ),
    "third_word": TransformationTask(
        task_id="third_word",
        natural_language_instruction="Return only the third word of the input, exactly as written.",
        allowed_output_format="A single word copied from the third word of the input.",
        transform=_third_word,
    ),
    "last_word": TransformationTask(
        task_id="last_word",
        natural_language_instruction="Return only the last word of the input, without trailing non-letter symbols.",
        allowed_output_format="A single word copied from the end of the input, with trailing non-letter symbols removed.",
        transform=_last_word,
    ),
    "word_count": TransformationTask(
        task_id="word_count",
        natural_language_instruction="Return only the number of words in the input.",
        allowed_output_format="A base-10 integer as plain text.",
        transform=_word_count,
    ),
    "uppercase_last_word": TransformationTask(
        task_id="uppercase_last_word",
        natural_language_instruction="Return only the last word of the input in uppercase, without trailing non-letter symbols.",
        allowed_output_format="A single uppercase word copied from the end of the input, with trailing non-letter symbols removed.",
        transform=_uppercase_last_word,
    ),
    "at_operator_mod_minus_left": TransformationTask(
        task_id="at_operator_mod_minus_left",
        natural_language_instruction="The @ operator is defined as a@b = a % b - a. Given an expression a@b=?, return only the resulting integer.",
        allowed_output_format="A base-10 integer as plain text.",
        transform=_at_operator_mod_minus_left,
    ),
}


def list_tasks() -> list[TransformationTask]:
    return list(_TASKS.values())


def get_task(task_id: str) -> TransformationTask:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown task_id {task_id!r}. Available tasks: {available}") from exc


def _normalized_output(text: str) -> str:
    return " ".join(text.strip().split())


def evaluate_output(task_id: str, input_text: str, pred_text: str, target_text: str | None = None) -> dict:
    """Score a generated string against the registered task semantics."""
    try:
        expected_text = get_task(task_id).transform(input_text)
        expected_source = "task_transform"
    except KeyError:
        if target_text is None:
            raise
        expected_text = target_text
        expected_source = "target_text"
    pred_normalized = _normalized_output(pred_text)
    expected_normalized = _normalized_output(expected_text)
    return {
        "task_expected_text": expected_text,
        "task_expected_source": expected_source,
        "task_pred_normalized": pred_normalized,
        "task_expected_normalized": expected_normalized,
        "task_semantic_correct": float(pred_normalized == expected_normalized),
    }
