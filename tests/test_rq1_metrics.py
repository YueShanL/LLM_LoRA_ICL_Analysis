from pathlib import Path
import json
import math
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lora_instruction_analysis.model.visualize import (
    _attention_head_ablation_rows,
    _attention_output_delta_rows,
    _attention_rows,
    _attention_post_o_proj_output_rows,
    _layer_distribution_rows,
    _metric_box_rows,
    _residual_rows,
    _chart_images,
    _rq2_matplotlib_charts,
)


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
        "source_alignment": [
            {"position": 0, "span": "prompt", "alignment_key": "prompt:0:1", "token_id": 1},
            {"position": 1, "span": "input", "alignment_key": "input:0:2", "token_id": 2},
            {"position": 2, "span": "input", "alignment_key": "input:1:3", "token_id": 3},
            {"position": 3, "span": "target", "alignment_key": "target:0:10", "token_id": 10},
            {"position": 4, "span": "target", "alignment_key": "target:1:11", "token_id": 11},
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

    def test_post_o_proj_attention_output_rows_compare_block_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tensor_dir = run / "tensors"
            tensor_dir.mkdir()
            rows = []
            base_outputs = torch.zeros(1, 2, 3)
            instruction_outputs = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
            lora_outputs = instruction_outputs * 2

            for condition, outputs in (
                ("base", base_outputs),
                ("instruction_only", instruction_outputs),
                ("lora_only", lora_outputs),
            ):
                path = tensor_dir / f"s1__{condition}.pt"
                tensor = _tensor("s1", condition, torch.zeros(2, 2, 3), torch.zeros(2, 3))
                tensor["attention_post_o_proj_outputs"] = outputs
                torch.save(tensor, path)
                rows.append({"sample_id": "s1", "condition": condition, "tensor_path": str(path)})
            with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            metric_rows = _attention_post_o_proj_output_rows(torch, run)

        self.assertEqual(len(metric_rows), 4)
        self.assertTrue(all(row["mode"] == "attention_post_o_proj_output" for row in metric_rows))
        self.assertNotIn("head", metric_rows[0])
        instruction_vs_lora = [
            row for row in metric_rows if row["condition"] == "instruction_only_vs_lora_only"
        ]
        self.assertTrue(all(row["cosine_similarity"] > 0.99 for row in instruction_vs_lora))

    def test_attention_rows_report_source_stats_and_normalized_distribution_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tensor_dir = run / "tensors"
            tensor_dir.mkdir()
            rows = []
            attentions = {
                "base": torch.zeros(1, 1, 2, 5),
                "instruction_only": torch.tensor([[[[0.0, 0.2, 0.2, 0.2, 0.4], [0.0, 0.1, 0.1, 0.3, 0.5]]]]),
                "lora_only": torch.tensor([[[[0.0, 0.1, 0.1, 0.4, 0.4], [0.0, 0.2, 0.2, 0.2, 0.4]]]]),
            }
            for condition, attention in attentions.items():
                path = tensor_dir / f"s1__{condition}.pt"
                tensor = _tensor("s1", condition, torch.zeros(2, 2, 3), torch.zeros(2, 3))
                tensor["attentions"] = attention
                torch.save(tensor, path)
                rows.append({"sample_id": "s1", "condition": condition, "tensor_path": str(path)})
            with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            [row] = [
                row
                for row in _attention_rows(torch, run)
                if row["condition"] == "instruction_only_vs_lora_only" and row["token_index"] == 0
            ]

        self.assertAlmostEqual(row["left_shared_attention_mass"], 0.6, places=6)
        self.assertAlmostEqual(row["left_entropy"], math.log(3), places=6)
        self.assertEqual(row["shared_input_tokens"], 2)
        self.assertEqual(row["shared_target_prefix_tokens"], 1)
        self.assertEqual(row["excluded_instruction_tokens"], 2)
        self.assertEqual(row["excluded_other_tokens"], 1)

    def test_attention_output_delta_rows_compare_condition_minus_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tensor_dir = run / "tensors"
            tensor_dir.mkdir()
            rows = []
            base_outputs = torch.zeros(1, 1, 2, 3)
            instruction_outputs = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]])
            lora_outputs = instruction_outputs * 2
            for condition, outputs in (
                ("base", base_outputs),
                ("instruction_only", instruction_outputs),
                ("lora_only", lora_outputs),
            ):
                path = tensor_dir / f"s1__{condition}.pt"
                tensor = _tensor("s1", condition, torch.zeros(2, 2, 3), torch.zeros(2, 3))
                tensor["attention_outputs"] = outputs
                torch.save(tensor, path)
                rows.append({"sample_id": "s1", "condition": condition, "tensor_path": str(path)})
            with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            metric_rows = _attention_output_delta_rows(
                torch,
                run,
                tensor_key="attention_outputs",
                mode="attention_output_delta",
                metric_definition="test",
            )

        self.assertEqual(len(metric_rows), 2)
        self.assertTrue(all(row["condition"] == "instruction_delta_vs_lora_delta" for row in metric_rows))
        self.assertTrue(all(row["cosine_similarity"] > 0.99 for row in metric_rows))

    def test_attention_head_ablation_rows_use_saved_impact_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            tensor_dir = run / "tensors"
            tensor_dir.mkdir()
            rows = []
            impacts = torch.tensor([[[[2.0, 1.0, 0.5, 0.8], [3.0, 1.5, 0.5, 0.7]]]])
            for condition in ("base", "instruction_only", "lora_only"):
                path = tensor_dir / f"s1__{condition}.pt"
                tensor = _tensor("s1", condition, torch.zeros(2, 2, 3), torch.zeros(2, 3))
                tensor["attention_head_ablation_impact_names"] = [
                    "full_norm",
                    "head_contribution_norm",
                    "head_contribution_relative_norm",
                    "ablated_cosine_to_full",
                ]
                tensor["attention_head_ablation_impacts"] = impacts
                torch.save(tensor, path)
                rows.append({"sample_id": "s1", "condition": condition, "tensor_path": str(path)})
            with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

            metric_rows = _attention_head_ablation_rows(torch, run)

        self.assertEqual(len(metric_rows), 6)
        self.assertEqual(metric_rows[0]["mode"], "attention_head_ablation_impact")
        self.assertEqual(metric_rows[0]["cosine_similarity"], metric_rows[0]["ablated_cosine_to_full"])

    def test_rq2_matplotlib_charts_write_line_and_heatmap_pngs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            line_rows = [
                {
                    "condition": "instruction_delta_vs_lora_delta",
                    "layer": layer,
                    "cosine_similarity": 0.9 + layer * 0.01,
                }
                for layer in (0, 1)
            ]
            [line_path] = _rq2_matplotlib_charts(line_rows, output_dir, "attention_output_delta")
            heatmap_rows = [
                {
                    "condition": condition,
                    "layer": layer,
                    "head": head,
                    "head_contribution_relative_norm": 0.1 * (head + 1),
                }
                for condition in ("base", "instruction_only")
                for layer in (0, 1)
                for head in (0, 1)
            ]
            [heatmap_path] = _rq2_matplotlib_charts(heatmap_rows, output_dir, "attention_head_ablation")

            html = _chart_images([line_path, heatmap_path])

            self.assertEqual(line_path.name, "attention_output_delta_layer_mean.png")
            self.assertTrue(line_path.exists())
            self.assertTrue(heatmap_path.exists())
            self.assertIn("attention_output_delta_layer_mean.png", html)
            self.assertIn("head_ablation_heatmap.png", html)


if __name__ == "__main__":
    unittest.main()
