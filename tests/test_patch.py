from pathlib import Path
import importlib
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.model import patch
from lora_instruction_analysis.model import visualize


class PatchTests(unittest.TestCase):
    def test_target_loss_ids_do_not_repeat_or_invent_targets(self):
        self.assertEqual(patch._target_loss_ids([10, 20], 0), 10)
        self.assertEqual(patch._target_loss_ids([10, 20], 1), 20)
        self.assertIsNone(patch._target_loss_ids([10, 20], 2))

    def test_patch_shape_mismatch_fails_fast(self):
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            self.skipTest("torch is not installed")
        hidden = torch.zeros(1, 2, 3)
        bad_patch = torch.zeros(1, 3)

        with self.assertRaisesRegex(ValueError, "Patch position mismatch"):
            patch._with_patched_slice(torch, hidden, torch.tensor([0, 1]), bad_patch)

    def test_visible_patch_patches_only_positions_available_in_current_forward(self):
        try:
            torch = importlib.import_module("torch")
        except ImportError:
            self.skipTest("torch is not installed")
        hidden = torch.zeros(1, 2, 3)
        patch_rows = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0],
            ]
        )

        patched = patch._visible_patch(torch, hidden, [0, 1, 2], patch_rows)

        self.assertTrue(torch.equal(patched[0, 0, :], patch_rows[0]))
        self.assertTrue(torch.equal(patched[0, 1, :], patch_rows[1]))

    def test_patch_loss_visualization_rejects_legacy_generation_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            row = {
                "sample_id": "s1",
                "task_id": "task",
                "patched": False,
                "token_losses": [0.5],
                "target_repeat_token_ids": [123],
                "pred_token_ids": [123],
                "target_text": "x",
                "pred_text": "x",
            }
            with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

            with self.assertRaisesRegex(ValueError, "loss_target_token_ids"):
                visualize._patch_loss_rows(run_dir)


if __name__ == "__main__":
    unittest.main()
