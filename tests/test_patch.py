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

    def test_patch_loss_visualization_writes_rq3_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "patches" / "source_to_target_l1_text"
            out_dir = root / "plots"
            run_dir.mkdir(parents=True)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "source_condition": "instruction_only",
                        "target_condition": "lora_only",
                        "layer": 1,
                        "patch_span": "text",
                    }
                ),
                encoding="utf-8",
            )
            metric_rows = [
                {
                    "sample_id": "s1",
                    "task_id": "word_count",
                    "control": "unpatched",
                    "patched": False,
                    "loss": 1.0,
                    "token_accuracy": 0.0,
                    "sequence_accuracy": 0.0,
                    "task_semantic_correct": 0.0,
                },
                {
                    "sample_id": "s1",
                    "task_id": "word_count",
                    "control": "source_to_target_patch",
                    "patched": True,
                    "loss": 0.2,
                    "token_accuracy": 1.0,
                    "sequence_accuracy": 1.0,
                    "task_semantic_correct": 1.0,
                },
            ]
            generation_rows = [
                {
                    "sample_id": "s1",
                    "task_id": "word_count",
                    "control": row["control"],
                    "patched": row["patched"],
                    "layer": 1,
                    "token_losses": [row["loss"]],
                    "loss_target_token_ids": [20],
                    "pred_token_ids": [20],
                    "target_text": "1",
                    "pred_text": "1",
                    "token_accuracy": row["token_accuracy"],
                    "sequence_accuracy": row["sequence_accuracy"],
                    "task_semantic_correct": row["task_semantic_correct"],
                }
                for row in metric_rows
            ]
            with (run_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in metric_rows:
                    handle.write(json.dumps(row) + "\n")
            with (run_dir / "generations.jsonl").open("w", encoding="utf-8") as handle:
                for row in generation_rows:
                    handle.write(json.dumps(row) + "\n")
            (run_dir / "confusion_matrix.json").write_text(
                json.dumps(
                    {
                        "semantic_generation": [
                            {
                                "metric": "task_semantic_correct",
                                "unpatched_correct": False,
                                "source_to_target_correct": True,
                                "samples": 1,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            visualize.visualize_patch_losses(
                visualize.VisualizeConfig(run=root / "patches", left_run=None, right_run=None, output_dir=out_dir)
            )

            summary = (out_dir / "rq3_summary.csv").read_text(encoding="utf-8")
            html = (out_dir / "rq3_summary.html").read_text(encoding="utf-8")
            self.assertIn("generation,task_semantic_correct", summary)
            self.assertIn('src="rq3_teacher_forced_loss_by_pair.png"', html)
            self.assertIn('class="rq3-chart"', html)
            self.assertTrue((out_dir / "rq3_summary.html").exists())
            self.assertTrue((out_dir / "rq3_teacher_forced_loss_by_pair.png").exists())
            self.assertTrue((out_dir / "rq3_metric_distribution.csv").exists())


if __name__ == "__main__":
    unittest.main()
