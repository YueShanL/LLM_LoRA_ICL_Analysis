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
}


def list_tasks() -> list[TransformationTask]:
    return list(_TASKS.values())


def get_task(task_id: str) -> TransformationTask:
    try:
        return _TASKS[task_id]
    except KeyError as exc:
        available = ", ".join(sorted(_TASKS))
        raise KeyError(f"Unknown task_id {task_id!r}. Available tasks: {available}") from exc
