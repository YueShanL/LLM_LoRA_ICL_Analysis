from pathlib import Path
import csv
import gzip
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.plot_combined_rq1_rq2 import collect, layer_means, task_dirs, write_plots
from scripts.plot_all_rq3_accuracy import collect as collect_rq3
from scripts.plot_task_acceptance import collect as collect_acceptance, write_plots as write_acceptance_plots


def _write_token_similarity(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "layer", "cosine_similarity"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "condition": "instruction_only_vs_lora_only",
                "layer": "0",
                "cosine_similarity": "0.9",
            }
        )


def _write_aggregated_similarity(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample_id", "task_id", "condition", "mode", "layer", "head", "cosine_similarity_n", "cosine_similarity_mean"],
        )
        writer.writeheader()
        writer.writerows(
            (
                {
                    "sample_id": "sample_a",
                    "task_id": "task_a",
                    "condition": "instruction_only_vs_lora_only",
                    "mode": "residual",
                    "layer": "0",
                    "head": "",
                    "cosine_similarity_n": "2",
                    "cosine_similarity_mean": "0.6",
                },
                {
                    "sample_id": "sample_b",
                    "task_id": "task_a",
                    "condition": "instruction_only_vs_lora_only",
                    "mode": "residual",
                    "layer": "0",
                    "head": "",
                    "cosine_similarity_n": "1",
                    "cosine_similarity_mean": "0.9",
                },
            )
        )


class CombinedPlotTests(unittest.TestCase):
    def test_task_dirs_excludes_aggregation_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_a" / "plots").mkdir(parents=True)
            (root / "analysis_aggregates" / "task_a" / "plots").mkdir(parents=True)

            self.assertEqual(task_dirs(root), [root / "task_a"])

    def test_collects_aggregated_rows_and_preserves_token_weighted_layer_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "task_a" / "plots").mkdir(parents=True)
            _write_aggregated_similarity(
                root / "analysis_aggregates" / "task_a" / "plots" / "rq1" / "token_similarity.sample_layer_head.csv.gz"
            )

            _rates, datasets = collect(root, root / "acceptance")

            rows = datasets["rq1_similarity"]["task_a"]
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(layer_means(rows, "cosine_similarity")[0], 0.7)

    def test_collects_and_plots_post_o_proj_attention_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task_a"
            _write_token_similarity(task / "plots/rq1/token_similarity.csv")
            _write_token_similarity(task / "plots/rq2/token_similarity.csv")
            _write_token_similarity(task / "plots/rq21/attention_outputs/token_similarity.csv")
            _write_token_similarity(task / "plots/rq21/attention_post_o_proj_outputs/token_similarity.csv")

            _rates, datasets = collect(root, root / "acceptance")
            written = write_plots(root, root / "combined", root / "acceptance")

            self.assertEqual(len(datasets["rq2_attention_post_o_proj_output"]["task_a"]), 1)
            self.assertIn(root / "combined/rq2_attention_post_o_proj_output_lines.png", written)
            self.assertTrue((root / "combined/rq2_attention_post_o_proj_output_lines.png").exists())

    def test_collects_raw_rq3_accuracy_and_normalizes_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "task_a/patches/rq3/base_to_lora_l1_text/metrics.jsonl"
            path.parent.mkdir(parents=True)
            (path.parent / "config.json").write_text(json.dumps({"source_condition": "base", "target_condition": "lora_only"}), encoding="utf-8")
            path.write_text(json.dumps({"sample_id": "s1", "target_condition": "lora_only", "layer": 1, "patched": False, "task_semantic_correct": 1}) + "\n", encoding="utf-8")

            [row] = collect_rq3(root)

            self.assertEqual((row["source_condition"], row["control"], row["accuracy"]), ("none", "unpatched", 1.0))
            self.assertEqual((row["patch_source_condition"], row["patch_target_condition"]), ("base", "lora_only"))

    def test_plots_task_acceptance_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task_a"
            task.mkdir()
            (task / "acceptance_summary.json").write_text(
                json.dumps(
                    {
                        "instruction_only": [
                            {"samples": 2, "mean_token_accuracy": 0.5, "mean_sequence_accuracy": 0.0, "mean_task_semantic_correct": 1.0},
                            {"samples": 2, "mean_token_accuracy": 0.25, "mean_sequence_accuracy": 0.0, "mean_task_semantic_correct": 0.5},
                        ],
                        "no_instruction": {"samples": 2, "mean_token_accuracy": 0.0, "mean_sequence_accuracy": 0.0, "mean_task_semantic_correct": 0.0},
                    }
                ),
                encoding="utf-8",
            )

            metrics, data, labels = collect_acceptance(root)
            written = write_acceptance_plots(root, root / "plots")

            self.assertEqual(labels, ["prompt_1", "prompt_2", "no_instruction"])
            self.assertIn("mean_task_semantic_correct", metrics)
            self.assertEqual(data["task_a"]["mean_task_semantic_correct"], [1.0, 0.5, 0.0])
            self.assertTrue((root / "plots/mean_task_semantic_correct.png").exists())
            self.assertIn(root / "plots/mean_task_semantic_correct.png", written)


if __name__ == "__main__":
    unittest.main()
