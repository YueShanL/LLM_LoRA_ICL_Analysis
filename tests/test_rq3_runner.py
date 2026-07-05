from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.experiment import run_rq3


class RQ3RunnerTests(unittest.TestCase):
    def test_rq3_writes_patch_loss_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = run_rq3.RQ3Config(
                run_dir=root,
                model_name="model",
                dataset_path=root / "data",
                adapter_path=root / "adapter",
            )
            seen = {"patches": []}

            def fake_patch(patch_config):
                seen["patches"].append(
                    (
                        patch_config.source_condition,
                        patch_config.target_condition,
                        patch_config.layer,
                        patch_config.patch_span,
                        patch_config.output_dir,
                    )
                )

            def fake_visualize(visualize_config):
                seen["plot_run"] = visualize_config.run
                seen["plot_output_dir"] = visualize_config.output_dir
                seen["mode"] = visualize_config.mode

            with (
                patch.object(run_rq3, "run_activation_patching", fake_patch),
                patch.object(run_rq3, "visualize", fake_visualize),
            ):
                run_rq3.run_rq3(config)

            expected_specs = [
                (source, target, layer, "text", root / "patches" / "rq3" / f"{source}_to_{target}_l{layer}_text")
                for source, target in run_rq3.DEFAULT_PATCH_PAIRS
                for layer in run_rq3.DEFAULT_LAYERS
            ]
            self.assertEqual(seen["patches"], expected_specs)
            self.assertEqual(seen["plot_run"], root / "patches" / "rq3")
            self.assertEqual(seen["plot_output_dir"], root / "plots" / "patch_loss" / "rq3")
            self.assertEqual(seen["mode"], "patch_loss")
            self.assertTrue((root / "rq3_config.json").exists())
            self.assertTrue((root / "rq3_status.json").exists())
            rq3_config = json.loads((root / "rq3_config.json").read_text(encoding="utf-8"))
            self.assertEqual(rq3_config["result_status"], "partial")
            self.assertEqual(
                [control["control"] for control in rq3_config["patch_runs"][0]["controls"]],
                list(run_rq3.RQ3_CONTROLS),
            )

    def test_explicit_pair_and_layer_runs_single_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = run_rq3.RQ3Config(
                run_dir=root,
                model_name="model",
                dataset_path=root / "data",
                adapter_path=root / "adapter",
                source_condition="base",
                target_condition="lora_only",
                layer=21,
            )
            seen = []

            with (
                patch.object(run_rq3, "run_activation_patching", lambda patch_config: seen.append(patch_config)),
                patch.object(run_rq3, "visualize", lambda _visualize_config: None),
            ):
                run_rq3.run_rq3(config)

            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].source_condition, "base")
            self.assertEqual(seen[0].target_condition, "lora_only")
            self.assertEqual(seen[0].layer, 21)
            self.assertEqual(seen[0].patch_span, "text")


if __name__ == "__main__":
    unittest.main()
