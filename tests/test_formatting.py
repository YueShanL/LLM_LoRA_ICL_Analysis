from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.model.formatting import encode_record


class CharTokenizer:
    eos_token_id = 3

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class ChatTokenizer(CharTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ""
        for message in messages:
            text += f"<{message['role']}>" + message["content"] + "</>"
        if add_generation_prompt:
            text += "<assistant>"
        return text


def _record():
    return {
        "sample_id": "s1",
        "task_id": "task",
        "input_text": "ab",
        "instruction_text": "copy",
        "target_text": "xy",
    }


def test_raw_format_masks_prompt_and_appends_eos():
    encoded = encode_record(CharTokenizer(), _record(), include_instruction=False)

    assert encoded["target_ids"] == [ord("x"), ord("y"), 3]
    assert encoded["labels"][-3:] == [ord("x"), ord("y"), 3]
    assert all(label == -100 for label in encoded["labels"][:-3])


def test_raw_format_can_skip_eos():
    encoded = encode_record(CharTokenizer(), _record(), include_instruction=False, append_eos=False)

    assert encoded["target_ids"] == [ord("x"), ord("y")]
    assert encoded["labels"][-2:] == [ord("x"), ord("y")]


def test_chat_template_masks_template_suffix_and_eos():
    encoded = encode_record(
        ChatTokenizer(),
        _record(),
        include_instruction=True,
        prompt_format="chat_template",
    )

    prompt_len = encoded["labels"].index(ord("x"))
    assert "".join(chr(token_id) for token_id in encoded["input_ids"][:prompt_len]).endswith("<assistant>")
    assert encoded["target_ids"] == [ord("x"), ord("y")]
    assert encoded["input_ids"][-1] == 3
    assert encoded["labels"][-1] == -100
    assert all(row["span"] == "target" for row in encoded["source_alignment"][-2:])
