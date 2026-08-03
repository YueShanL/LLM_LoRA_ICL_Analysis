from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from lora_instruction_analysis.experiment import run_rq1
from lora_instruction_analysis.model import jlens_fit
from lora_instruction_analysis.model.jlens_readout import _write_csv, pair_overlap_rows, run_jlens_readout
from lora_instruction_analysis.model.sae_analysis import run_sae_analysis


def _tensor(sample_id, condition, hidden):
    return {
        "sample_id": sample_id,
        "task_id": "task",
        "condition": condition,
        "labels": torch.tensor([-100, 1]),
        "hidden_states": hidden,
        "target_logits": torch.zeros(1, 4),
        "target_alignment": [{"alignment_key": "target:0:1", "token_id": 1}],
    }


def _write_run(root: Path) -> Path:
    run = root / "states"
    tensors = run / "tensors"
    tensors.mkdir(parents=True)
    rows = []
    states = {
        "base": torch.zeros(1, 1, 3),
        "instruction_only": torch.tensor([[[1.0, 0.0, 0.0]]]),
        "lora_only": torch.tensor([[[2.0, 0.0, 0.0]]]),
    }
    for condition, hidden in states.items():
        path = tensors / f"s1__{condition}.pt"
        torch.save(_tensor("s1", condition, hidden), path)
        rows.append(
            {
                "sample_id": "s1",
                "task_id": "task",
                "condition": condition,
                "tensor_path": str(path),
                "sequence_accuracy": 1.0,
            }
        )
    with (run / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return run


class FakeDataset:
    column_names = ["text"]

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class JLensSaeTests(unittest.TestCase):
    def test_write_csv_quotes_token_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            _write_csv(path, [{"token": 'a,"b"\n c', "score": 1.0}])

            text = path.read_text(encoding="utf-8")
            self.assertIn('"a,""b""\n c"', text)

    def test_jlens_readout_writes_expected_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _write_run(root)
            lens = root / "lens.pt"
            torch.save({"unembed": torch.eye(4, 3)}, lens)

            run_jlens_readout(run, lens, root / "out", top_k=2)

            readouts = (root / "out" / "jlens_readouts.csv").read_text(encoding="utf-8")
            overlap = (root / "out" / "jlens_pair_overlap.csv").read_text(encoding="utf-8")
            self.assertIn("top_k_token_ids", readouts)
            self.assertIn("instruction_only_vs_lora_only", overlap)

    def test_pair_overlap_handles_empty_partial_and_full(self):
        rows = [
            {"sample_id": "s", "task_id": "t", "condition": "base", "layer": 0, "token_index": 0, "alignment_key": "a", "top_k_token_ids": left, "target_token_rank": 0}
            for left in ("",)
        ] + [
            {"sample_id": "s", "task_id": "t", "condition": "instruction_only", "layer": 0, "token_index": 0, "alignment_key": "a", "top_k_token_ids": "1 2", "target_token_rank": 0},
            {"sample_id": "s", "task_id": "t", "condition": "lora_only", "layer": 0, "token_index": 0, "alignment_key": "a", "top_k_token_ids": "2 3", "target_token_rank": 0},
        ]

        overlaps = {row["condition_pair"]: row["top_k_jaccard"] for row in pair_overlap_rows(rows)}

        self.assertEqual(overlaps["base_vs_instruction_only"], 0.0)
        self.assertAlmostEqual(overlaps["instruction_only_vs_lora_only"], 1 / 3)

    def test_sae_residual_analysis_writes_feature_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = _write_run(root)
            sae = root / "sae.pt"
            torch.save({"encoder_weight": torch.eye(3)}, sae)

            run_sae_analysis(run, sae, root / "sae_out", mode="residual", top_k=2)

            rows = (root / "sae_out" / "sae_feature_rows.csv").read_text(encoding="utf-8")
            self.assertIn("feature_activation_cosine", rows)
            self.assertIn("residual_sae_delta", rows)

    def test_rq1_jlens_requires_path_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = run_rq1.RQ1Config(
                run_dir=root,
                model_name="model",
                dataset_path=root / "data",
                adapter_path=root / "adapter",
                run_jlens_readout=True,
            )
            with self.assertRaisesRegex(ValueError, "--jlens-path is required"):
                with unittest.mock.patch.object(run_rq1, "collect", lambda collect_config: None), unittest.mock.patch.object(run_rq1, "visualize", lambda visualize_config: None):
                    run_rq1.run_rq1(config)

    def test_jlens_fit_samples_hf_dataset_without_fit_validation_overlap(self):
        dataset = FakeDataset([{"text": f"row {index}"} for index in range(10)])
        with unittest.mock.patch.object(jlens_fit, "_load_dataset_split", lambda *_args: dataset):
            fit_rows, validation_rows = jlens_fit._sample_dataset_texts(
                dataset_name="dataset",
                dataset_config=None,
                dataset_split="train",
                validation_split=None,
                text_column="text",
                num_sequences=4,
                validation_sequences=3,
                seed=13,
            )

        fit_indices = {row["row_index"] for row in fit_rows}
        validation_indices = {row["row_index"] for row in validation_rows}
        self.assertEqual(len(fit_rows), 4)
        self.assertEqual(len(validation_rows), 3)
        self.assertFalse(fit_indices & validation_indices)


if __name__ == "__main__":
    unittest.main()
