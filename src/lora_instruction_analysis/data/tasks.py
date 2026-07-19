"""Task registry for deterministic synthetic transformations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable


TransformFn = Callable[[str], str]
ValidationFn = Callable[[str, str, str], bool]
ValidationSelector = str | ValidationFn | None


PROMPT_VARIANT_NAMES = ("natural", "follow_rule_only_answer", "task_no_explanation")


@dataclass(frozen=True)
class TransformationTask:
    task_id: str
    natural_language_instruction: str
    allowed_output_format: str
    transform: TransformFn
    validation_kind: str = "fixed_target"
    max_generate_tokens: int | None = None


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


def _clean_word(word: str) -> str:
    return word.strip(".,;:!?)]}\"'")


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


def _first_letter(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    return compact[0] if compact else ""


def _second_letter(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    return compact[1] if len(compact) >= 2 else ""


def _last_letter(text: str) -> str:
    compact = re.sub(r"\s+", "", text)
    return compact[-1] if compact else ""


def _char_count_no_space(text: str) -> str:
    return str(len(re.sub(r"\s+", "", text)))


def _vowel_count(text: str) -> str:
    return str(sum(char.lower() in "aeiou" for char in text))


def _at_operator_mod_minus_left(text: str) -> str:
    match = re.fullmatch(r"\s*(\d+)@(\d+)=\?\s*", text)
    if not match:
        return ""
    a = int(match.group(1))
    b = int(match.group(2))
    if b == 0:
        return ""
    return str(a % b - a)


def _words_starting_with_letter(text: str) -> str:
    words = [_clean_word(word) for word in text.split()]
    matches = [word for word in words if word.lower().startswith("m")]
    return " ".join(matches) if matches else "NONE"


def _list_letters_space_separated(text: str) -> str:
    words = [_clean_word(word) for word in text.split()]
    word = words[0] if words else ""
    return " ".join(word)


def _sum_two_numbers(text: str) -> str:
    match = re.fullmatch(r"\s*(\d+)\+(\d+)=\?\s*", text)
    if not match:
        return ""
    return str(int(match.group(1)) + int(match.group(2)))


def _exact_three_word_prefix(text: str) -> str:
    return " ".join(text.split()[:3])


_SYNTHETIC_SET = {"dax", "wug", "blick"}


def _extract_items_from_set(text: str) -> str:
    matches = [word for word in (_clean_word(word).lower() for word in text.split()) if word in _SYNTHETIC_SET]
    return " ".join(matches) if matches else "NONE"


def _has_repeated_word(text: str) -> str:
    words = [_clean_word(word).lower() for word in text.split()]
    words = [word for word in words if word]
    return "YES" if len(words) != len(set(words)) else "NO"


def _words_containing_bigram_qu(text: str) -> str:
    matches = [word for word in (_clean_word(word) for word in text.split()) if "qu" in word.lower()]
    return " ".join(matches) if matches else "NONE"


def _formal_language_a_n_b_n(text: str) -> str:
    compact = re.sub(r"\s+", "", text.lower())
    match = re.fullmatch(r"a+b+", compact)
    if not match:
        return "NO"
    a_count = len(compact) - len(compact.lstrip("a"))
    b_count = len(compact) - a_count
    return "YES" if a_count == b_count else "NO"


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
        validation_kind="constraint",
    ),
    "third_word": TransformationTask(
        task_id="third_word",
        natural_language_instruction="Return only the third word of the input, exactly as written.",
        allowed_output_format="A single word copied from the third word of the input.",
        transform=_third_word,
        validation_kind="constraint",
    ),
    "last_word": TransformationTask(
        task_id="last_word",
        natural_language_instruction="Return only the last word of the input, without trailing non-letter symbols.",
        allowed_output_format="A single word copied from the end of the input, with trailing non-letter symbols removed.",
        transform=_last_word,
        validation_kind="constraint",
    ),
    "word_count": TransformationTask(
        task_id="word_count",
        natural_language_instruction="Return only the number of words in the input.",
        allowed_output_format="A base-10 integer as plain text.",
        transform=_word_count,
        validation_kind="constraint",
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
        validation_kind="constraint",
        max_generate_tokens=128,
    ),
    "first_letter": TransformationTask(
        task_id="first_letter",
        natural_language_instruction="Return only the first non-space character of the input.",
        allowed_output_format="A single character.",
        transform=_first_letter,
        validation_kind="constraint",
    ),
    "second_letter": TransformationTask(
        task_id="second_letter",
        natural_language_instruction="Return only the second non-space character of the input.",
        allowed_output_format="A single character.",
        transform=_second_letter,
        validation_kind="constraint",
    ),
    "last_letter": TransformationTask(
        task_id="last_letter",
        natural_language_instruction="Return only the last non-space character of the input.",
        allowed_output_format="A single character.",
        transform=_last_letter,
        validation_kind="constraint",
    ),
    "char_count_no_space": TransformationTask(
        task_id="char_count_no_space",
        natural_language_instruction="Return only the number of non-space characters in the input.",
        allowed_output_format="A base-10 integer as plain text.",
        transform=_char_count_no_space,
        validation_kind="constraint",
    ),
    "vowel_count": TransformationTask(
        task_id="vowel_count",
        natural_language_instruction="Return only the number of vowels in the input. Count a, e, i, o, and u, ignoring case.",
        allowed_output_format="A base-10 integer as plain text.",
        transform=_vowel_count,
        validation_kind="constraint",
    ),
    "words_starting_with_letter": TransformationTask(
        task_id="words_starting_with_letter",
        natural_language_instruction="Return only the words that start with the letter m, in their original order. If no words match, return NONE.",
        allowed_output_format="Matching words separated by spaces, or NONE.",
        transform=_words_starting_with_letter,
        validation_kind="constraint",
    ),
    "list_letters_space_separated": TransformationTask(
        task_id="list_letters_space_separated",
        natural_language_instruction="Return only the letters of the first word, separated by single spaces.",
        allowed_output_format="Letters separated by spaces.",
        transform=_list_letters_space_separated,
        validation_kind="constraint",
    ),
    "sum_two_numbers": TransformationTask(
        task_id="sum_two_numbers",
        natural_language_instruction="Given an expression a+b=?, return only the integer sum.",
        allowed_output_format="A base-10 integer as plain text.",
        transform=_sum_two_numbers,
        validation_kind="constraint",
    ),
    "exact_three_word_prefix": TransformationTask(
        task_id="exact_three_word_prefix",
        natural_language_instruction="Return exactly the first three words of the input and nothing else.",
        allowed_output_format="Exactly three words copied from the input.",
        transform=_exact_three_word_prefix,
        validation_kind="constraint",
    ),
    "extract_items_from_set": TransformationTask(
        task_id="extract_items_from_set",
        natural_language_instruction="The special set is {dax, wug, blick}. Return only input words that belong to that set, in order. If none match, return NONE.",
        allowed_output_format="Matching set items separated by spaces, or NONE.",
        transform=_extract_items_from_set,
        validation_kind="constraint",
    ),
    "has_repeated_word": TransformationTask(
        task_id="has_repeated_word",
        natural_language_instruction="Return YES if any word appears more than once in the input, otherwise return NO.",
        allowed_output_format="YES or NO.",
        transform=_has_repeated_word,
        validation_kind="constraint",
    ),
    "words_containing_bigram_qu": TransformationTask(
        task_id="words_containing_bigram_qu",
        natural_language_instruction="Return only the words that contain the bigram qu, in their original order. If no words match, return NONE.",
        allowed_output_format="Matching words separated by spaces, or NONE.",
        transform=_words_containing_bigram_qu,
        validation_kind="constraint",
    ),
    "formal_language_a_n_b_n": TransformationTask(
        task_id="formal_language_a_n_b_n",
        natural_language_instruction="Return YES if the input belongs to the formal language a^n b^n for n >= 1; otherwise return NO.",
        allowed_output_format="YES or NO.",
        transform=_formal_language_a_n_b_n,
        validation_kind="constraint",
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


def instruction_prompt_variants(instruction: str) -> list[str]:
    return [
        instruction,
        f"Follow this rule exactly: {instruction} Respond with only the answer.",
        f"Task: {instruction} Do not explain or add extra text.",
    ]


_TASK_DEFAULT_PROMPT_VARIANTS: dict[str, int] = {
    "add_zxq_after_t_or_l": 1,
    "wrap_odd_char_length": 3,
    "reverse_words": 3,
    "first_word": 2,
    "third_word": 3,
    "last_word": 2,
    "word_count": 1,
    "uppercase_last_word": 2,
    "at_operator_mod_minus_left": 1,
    "first_letter": 2,
    "second_letter": 3,
    "last_letter": 2,
    "char_count_no_space": 1,
    "vowel_count": 1,
    "words_starting_with_letter": 2,
    "list_letters_space_separated": 1,
    "sum_two_numbers": 2,
    "exact_three_word_prefix": 2,
    "extract_items_from_set": 2,
    "has_repeated_word": 2,
    "words_containing_bigram_qu": 3,
    "formal_language_a_n_b_n": 2,
}


def task_default_prompt_variant(task_id: str) -> int:
    return _TASK_DEFAULT_PROMPT_VARIANTS.get(task_id, 1)


def task_default_prompt_variant_name(task_id: str) -> str:
    return PROMPT_VARIANT_NAMES[task_default_prompt_variant(task_id) - 1]


def task_default_prompt(task_id: str) -> str:
    task = get_task(task_id)
    return instruction_prompt_variants(task.natural_language_instruction)[task_default_prompt_variant(task_id) - 1]


def _normalized_output(text: str) -> str:
    return " ".join(text.strip().split())


def _first_generated_line(text: str) -> str:
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _validate_exact(_input_text: str, pred_text: str, expected_text: str) -> bool:
    return _normalized_output(_first_generated_line(pred_text)) == _normalized_output(expected_text)


def _validate_single_token(_input_text: str, pred_text: str, expected_text: str) -> bool:
    first = _normalized_output(_first_generated_line(pred_text)).split()
    return bool(first) and first[0] == expected_text


def _validate_integer(_input_text: str, pred_text: str, expected_text: str) -> bool:
    match = re.search(r"-?\d+", pred_text)
    return bool(match) and match.group(0) == expected_text


def _validate_yes_no(_input_text: str, pred_text: str, expected_text: str) -> bool:
    match = re.search(r"\b(YES|NO)\b", pred_text, flags=re.IGNORECASE)
    return bool(match) and match.group(1).upper() == expected_text


def _generated_words(text: str) -> list[str]:
    return [_clean_word(word) for word in _normalized_output(_first_generated_line(text)).split()]


def _validate_first_word_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _first_word(input_text)
    words = _generated_words(pred_text)
    return words == [expected]


def _validate_third_word_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _third_word(input_text)
    words = _generated_words(pred_text)
    return words == [expected]


def _validate_last_word_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _last_word(input_text)
    words = _generated_words(pred_text)
    return words == [expected]


def _validate_first_letter_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _generated_words(pred_text)[:1] == [_first_letter(input_text)]


def _validate_second_letter_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _generated_words(pred_text)[:1] == [_second_letter(input_text)]


def _validate_last_letter_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _generated_words(pred_text)[:1] == [_last_letter(input_text)]


def _validate_word_count_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _validate_integer(input_text, pred_text, _word_count(input_text))


def _validate_char_count_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _validate_integer(input_text, pred_text, _char_count_no_space(input_text))


def _validate_vowel_count_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _validate_integer(input_text, pred_text, _vowel_count(input_text))


def _validate_sum_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _validate_integer(input_text, pred_text, _sum_two_numbers(input_text))


def _extract_generated_integer_answer(text: str) -> str | None:
    answer_patterns = (
        r"(?:answer|result|output)\s*(?:is|:|=)\s*(-?\d+)",
        r"=\s*(-?\d+)\s*(?:$|[.\n])",
        r"(?:^|\n)\s*(-?\d+)\s*(?:$|\n)",
    )
    matches = [
        match.group(1)
        for pattern in answer_patterns
        for match in re.finditer(pattern, text, flags=re.IGNORECASE)
    ]
    if matches:
        return matches[-1]
    first_line = _first_generated_line(text)
    strict_first_line_patterns = (
        r"=\s*(-?\d+)\s*(?:$|[.\n])",
    )
    for pattern in strict_first_line_patterns:
        match = re.search(pattern, first_line, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    match = re.fullmatch(r"\s*(-?\d+)\s*", first_line)
    return match.group(1) if match else None


def _validate_at_operator_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _extract_generated_integer_answer(pred_text) == _at_operator_mod_minus_left(input_text)


def _validate_exact_three_word_prefix_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _generated_words(pred_text) == [_clean_word(word) for word in input_text.split()[:3]]


def _validate_words_starting_with_m_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _words_starting_with_letter(input_text).split()
    return _generated_words(pred_text) == expected


def _validate_words_containing_qu_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _words_containing_bigram_qu(input_text).split()
    return _generated_words(pred_text) == expected


def _validate_extract_set_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _extract_items_from_set(input_text).split()
    return [word.lower() for word in _generated_words(pred_text)] == expected


def _validate_list_letters_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    expected = _list_letters_space_separated(input_text).split()
    return _generated_words(pred_text) == expected


def _validate_repeated_word_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _validate_yes_no(input_text, pred_text, _has_repeated_word(input_text))


def _validate_formal_language_constraint(input_text: str, pred_text: str, _expected_text: str) -> bool:
    return _validate_yes_no(input_text, pred_text, _formal_language_a_n_b_n(input_text))


def _validator_name_by_task() -> dict[str, str]:
    return {
        "add_zxq_after_t_or_l": "fixed_exact",
        "wrap_odd_char_length": "fixed_exact",
        "reverse_words": "fixed_exact",
        "uppercase_last_word": "fixed_single_token",
        "first_word": "constraint_first_word",
        "third_word": "constraint_third_word",
        "last_word": "constraint_last_word",
        "word_count": "constraint_word_count",
        "at_operator_mod_minus_left": "constraint_at_operator",
        "first_letter": "constraint_first_letter",
        "second_letter": "constraint_second_letter",
        "last_letter": "constraint_last_letter",
        "char_count_no_space": "constraint_char_count",
        "vowel_count": "constraint_vowel_count",
        "words_starting_with_letter": "constraint_words_starting_with_m",
        "list_letters_space_separated": "constraint_list_letters",
        "sum_two_numbers": "constraint_sum",
        "exact_three_word_prefix": "constraint_exact_three_word_prefix",
        "extract_items_from_set": "constraint_extract_set",
        "has_repeated_word": "constraint_repeated_word",
        "words_containing_bigram_qu": "constraint_words_containing_qu",
        "formal_language_a_n_b_n": "constraint_formal_language",
    }


_VALIDATORS: dict[str, ValidationFn] = {
    "add_zxq_after_t_or_l": _validate_exact,
    "wrap_odd_char_length": _validate_exact,
    "reverse_words": _validate_exact,
    "first_word": _validate_first_word_constraint,
    "third_word": _validate_third_word_constraint,
    "last_word": _validate_last_word_constraint,
    "word_count": _validate_word_count_constraint,
    "uppercase_last_word": _validate_single_token,
    "at_operator_mod_minus_left": _validate_at_operator_constraint,
    "first_letter": _validate_first_letter_constraint,
    "second_letter": _validate_second_letter_constraint,
    "last_letter": _validate_last_letter_constraint,
    "char_count_no_space": _validate_char_count_constraint,
    "vowel_count": _validate_vowel_count_constraint,
    "words_starting_with_letter": _validate_words_starting_with_m_constraint,
    "list_letters_space_separated": _validate_list_letters_constraint,
    "sum_two_numbers": _validate_sum_constraint,
    "exact_three_word_prefix": _validate_exact_three_word_prefix_constraint,
    "extract_items_from_set": _validate_extract_set_constraint,
    "has_repeated_word": _validate_repeated_word_constraint,
    "words_containing_bigram_qu": _validate_words_containing_qu_constraint,
    "formal_language_a_n_b_n": _validate_formal_language_constraint,
}

_NAMED_VALIDATORS: dict[str, ValidationFn] = {
    "exact": _validate_exact,
    "fixed_exact": _validate_exact,
    "single_token": _validate_single_token,
    "fixed_single_token": _validate_single_token,
    "integer": _validate_integer,
    "fixed_integer": _validate_integer,
    "yes_no": _validate_yes_no,
    "fixed_yes_no": _validate_yes_no,
    "constraint_first_word": _validate_first_word_constraint,
    "constraint_third_word": _validate_third_word_constraint,
    "constraint_last_word": _validate_last_word_constraint,
    "constraint_first_letter": _validate_first_letter_constraint,
    "constraint_second_letter": _validate_second_letter_constraint,
    "constraint_last_letter": _validate_last_letter_constraint,
    "constraint_word_count": _validate_word_count_constraint,
    "constraint_char_count": _validate_char_count_constraint,
    "constraint_vowel_count": _validate_vowel_count_constraint,
    "constraint_sum": _validate_sum_constraint,
    "constraint_at_operator": _validate_at_operator_constraint,
    "constraint_exact_three_word_prefix": _validate_exact_three_word_prefix_constraint,
    "constraint_words_starting_with_m": _validate_words_starting_with_m_constraint,
    "constraint_words_containing_qu": _validate_words_containing_qu_constraint,
    "constraint_extract_set": _validate_extract_set_constraint,
    "constraint_list_letters": _validate_list_letters_constraint,
    "constraint_repeated_word": _validate_repeated_word_constraint,
    "constraint_formal_language": _validate_formal_language_constraint,
}

_TASK_VALIDATOR_NAMES = _validator_name_by_task()
if set(_TASK_VALIDATOR_NAMES) != set(_TASKS):
    missing = sorted(set(_TASKS) - set(_TASK_VALIDATOR_NAMES))
    extra = sorted(set(_TASK_VALIDATOR_NAMES) - set(_TASKS))
    raise RuntimeError(f"Task validator registry mismatch: missing={missing}, extra={extra}")


def validator_name(validator: ValidationSelector) -> str:
    if validator is None:
        return "task_default"
    if isinstance(validator, str):
        return validator
    return getattr(validator, "__name__", validator.__class__.__name__)


def _resolve_validator(task_id: str, validator: ValidationSelector = None) -> ValidationFn:
    if validator is None or validator == "task_default":
        return _VALIDATORS.get(task_id, _validate_exact)
    if isinstance(validator, str):
        try:
            return _NAMED_VALIDATORS[validator]
        except KeyError as exc:
            available = ", ".join(["task_default", *sorted(_NAMED_VALIDATORS)])
            raise KeyError(f"Unknown validator {validator!r}. Available validators: {available}") from exc
    return validator


def task_default_validator_name(task_id: str) -> str:
    try:
        return _TASK_VALIDATOR_NAMES[task_id]
    except KeyError as exc:
        raise KeyError(f"No explicit validator is registered for task_id {task_id!r}.") from exc


def resolved_validator_name(task_id: str, validator: ValidationSelector = None) -> str:
    """Return the auditable validator name, never the moving alias task_default."""
    if validator in (None, "task_default"):
        return task_default_validator_name(task_id)
    selected = validator_name(validator)
    _resolve_validator(task_id, validator)
    return selected


def task_validation_kind(task_id: str, validator: ValidationSelector = None) -> str:
    selected = resolved_validator_name(task_id, validator)
    return "constraint" if selected.startswith("constraint_") else "fixed_target"


def validate_generated_output(
    task_id: str,
    input_text: str,
    pred_text: str,
    target_text: str | None = None,
    validator: ValidationSelector = None,
) -> bool:
    try:
        expected_text = get_task(task_id).transform(input_text) if target_text is None else target_text
    except KeyError:
        if target_text is None:
            raise
        expected_text = target_text
    return _resolve_validator(task_id, validator)(input_text, pred_text, expected_text)


def evaluate_output(
    task_id: str,
    input_text: str,
    pred_text: str,
    target_text: str | None = None,
    validator: ValidationSelector = None,
) -> dict:
    """Score a generated string against the registered task semantics."""
    selected_validator = resolved_validator_name(task_id, validator)
    validation_kind = task_validation_kind(task_id, validator)
    try:
        task_expected_text = get_task(task_id).transform(input_text)
        if validation_kind == "fixed_target" and target_text is not None:
            expected_text = target_text
            expected_source = "target_text"
        else:
            expected_text = task_expected_text
            expected_source = "task_constraint" if validation_kind == "constraint" else "task_transform"
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
        "task_validation_kind": validation_kind,
        "task_validator": selected_validator,
        "task_pred_normalized": pred_normalized,
        "task_expected_normalized": expected_normalized,
        "task_semantic_correct": float(validate_generated_output(task_id, input_text, pred_text, expected_text, validator)),
    }
