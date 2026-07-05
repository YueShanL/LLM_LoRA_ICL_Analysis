from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lora_instruction_analysis.model.collect import _encode
from lora_instruction_analysis.model.visualize import _same_target_alignment


class CharTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[ord(char) for char in text])


class AlignmentTests(unittest.TestCase):
    def test_encode_target_alignment_survives_instruction_prompt_offset(self):
        record = {
            "sample_id": "s1",
            "task_id": "task",
            "input_text": "ab",
            "instruction_text": "copy",
            "target_text": "xy",
        }
        plain = _encode(CharTokenizer(), record, include_instruction=False)
        instructed = _encode(CharTokenizer(), record, include_instruction=True)

        self.assertEqual(
            [row["alignment_key"] for row in plain["target_alignment"]],
            [row["alignment_key"] for row in instructed["target_alignment"]],
        )
        self.assertNotEqual(plain["target_positions"], instructed["target_positions"])

    def test_target_alignment_mismatch_fails_instead_of_truncating(self):
        left = {
            "sample_id": "s1",
            "condition": "left",
            "hidden_states": torch.zeros(1, 1, 2),
            "target_alignment": [{"alignment_key": "target:0:1", "token_id": 1}],
        }
        right = {
            "sample_id": "s1",
            "condition": "right",
            "hidden_states": torch.zeros(1, 1, 2),
            "target_alignment": [{"alignment_key": "target:0:2", "token_id": 2}],
        }

        with self.assertRaisesRegex(ValueError, "Target alignment mismatch"):
            _same_target_alignment(left, right)


if __name__ == "__main__":
    unittest.main()
