from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.experiment import run_rq2


class RQ2RunnerTests(unittest.TestCase):
    def test_rq21_collects_and_visualizes_attention_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = run_rq2.RQ2Config(
                run_dir=root,
                model_name="model",
                dataset_path=root / "data",
                adapter_path=root / "adapter",
                rq_name="rq21",
                collect_attention_outputs=True,
            )
            seen = {"modes": [], "output_dirs": []}

            def fake_collect(collect_config):
                seen["collect_attention_outputs"] = collect_config.collect_attention_outputs
                collect_config.output_dir.mkdir(parents=True, exist_ok=True)
                (collect_config.output_dir / "metrics.jsonl").write_text("", encoding="utf-8")

            def fake_visualize(visualize_config):
                seen["modes"].append(visualize_config.mode)
                seen["output_dirs"].append(visualize_config.output_dir)

            with (
                patch.object(run_rq2, "collect", fake_collect),
                patch.object(run_rq2, "visualize", fake_visualize),
                patch.object(run_rq2, "_require_tensor_keys", lambda _states_dir, _keys: None),
            ):
                run_rq2.run_rq2(config)

            self.assertTrue(seen["collect_attention_outputs"])
            self.assertEqual(seen["modes"], ["attention", "attention_output"])
            self.assertEqual(seen["output_dirs"], [root / "plots" / "rq21" / "attention_probs", root / "plots" / "rq21" / "attention_outputs"])
            self.assertTrue((root / "rq21_config.json").exists())


if __name__ == "__main__":
    unittest.main()
