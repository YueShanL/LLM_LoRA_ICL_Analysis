from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lora_instruction_analysis.model.visualize import _layer_distribution_rows, _metric_box_rows, _residual_rows


def _tensor(sample_id, condition, hidden_states, logits):
    return {
        "sample_id": sample_id,
        "task_id": "task",
        "condition": condition,
        "labels": torch.tensor([-100, 10, 11]),
        "hidden_states": hidden_states,
        "target_logits": logits,
        "target_alignment": [
            {"alignment_key": "target:0:10", "token_id": 10},
            {"alignment_key": "target:1:11", "token_id": 11},
        ],
    }


class RQ1MetricTests(unittest.TestCase):
    def test_residual_rows_include_cka_and_logit_distribution_similarity(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tensor_dir = run / "tensors"
            tensor_dir.mkdir()
            rows = []
            base_hidden = torch.zeros(2, 2, 3)
            instruction_hidden = torch.tensor(
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                ]
            )
            lora_hidden = instruction_hidden * 2
            logits = torch.tensor([[8.0, 1.0, 0.0], [0.0, 8.0, 1.0]])

            for condition, hidden, condition_logits in (
                ("base", base_hidden, torch.zeros_like(logits)),
                ("instruction_only", instruction_hidden, logits),
                ("lora_only", lora_hidden, logits),
            ):
                path = tensor_dir / f"s1__{condition}.pt"
                torch.save(_tensor("s1", condition, hidden, condition_logits), path)
                rows.append(
                    {
                        "sample_id": "s1",
                        "condition": condition,
                        "tensor_path": str(path),
                    }
                )
            with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            metric_rows = _residual_rows(torch, run)

        self.assertEqual(len(metric_rows), 4)
        self.assertTrue(all(row["cka_similarity"] > 0.99 for row in metric_rows))
        self.assertTrue(all(row["logit_distribution_cosine"] > 0.99 for row in metric_rows))

    def test_layer_distribution_rows_include_quartiles(self):
        rows = [
            {"condition": "a", "layer": 0, "cosine_similarity": value}
            for value in (0.0, 1.0, 2.0, 3.0)
        ]

        [row] = _layer_distribution_rows(rows)

        self.assertEqual(row["min_cosine_similarity"], 0.0)
        self.assertEqual(row["q1_cosine_similarity"], 0.75)
        self.assertEqual(row["q2_cosine_similarity"], 1.5)
        self.assertEqual(row["q3_cosine_similarity"], 2.25)
        self.assertEqual(row["max_cosine_similarity"], 3.0)

    def test_metric_box_rows_use_requested_metric(self):
        rows = [
            {"condition": "a", "layer": 0, "cka_similarity": value, "cosine_similarity": 99.0}
            for value in (0.0, 1.0, 2.0, 3.0)
        ]

        [row] = _metric_box_rows(rows, "cka_similarity")

        self.assertEqual(row["metric"], "cka_similarity")
        self.assertEqual(row["min"], 0.0)
        self.assertEqual(row["q2"], 1.5)
        self.assertEqual(row["max"], 3.0)


if __name__ == "__main__":
    unittest.main()
