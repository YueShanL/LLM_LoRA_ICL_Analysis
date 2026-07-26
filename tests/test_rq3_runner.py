from argparse import Namespace
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.experiment import run_rq3


class RQ3RunnerTests(unittest.TestCase):
    def test_infers_task_generation_budget_when_not_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_name": "model",
                        "dataset_dir": str(data),
                        "adapter_dir": str(root / "adapter"),
                        "seed": 7,
                    }
                ),
                encoding="utf-8",
            )
            (data / "test.jsonl").write_text(
                '{"sample_id":"s1","task_id":"at_operator_mod_minus_left","input_text":"17@5=?","target_text":"-15","instruction_text":"compute"}\n',
                encoding="utf-8",
            )

            config = run_rq3._infer_config(
                Namespace(
                    run_dir=root,
                    model_name=None,
                    dataset_path=None,
                    adapter_path=None,
                    source_condition=None,
                    target_condition=None,
                    layer=None,
                    split="test",
                    max_samples=None,
                    seed=None,
                    max_new_tokens=None,
                    patch_span="text",
                    dtype="auto",
                    device="auto",
                    output_dir=None,
                    plots_dir=None,
                    prompt_format=None,
                    no_append_eos=False,
                    validator=None,
                )
            )

            self.assertEqual(config.max_new_tokens, 128)

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

    def test_explicit_layers_run_multiple_patches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = run_rq3.RQ3Config(
                run_dir=root,
                model_name="model",
                dataset_path=root / "data",
                adapter_path=root / "adapter",
                source_condition="base",
                target_condition="lora_only",
                layers=[2, 24, 38],
            )
            seen = []

            with (
                patch.object(run_rq3, "run_activation_patching", lambda patch_config: seen.append(patch_config)),
                patch.object(run_rq3, "visualize", lambda _visualize_config: None),
            ):
                run_rq3.run_rq3(config)

            self.assertEqual([patch.layer for patch in seen], [2, 24, 38])
            rq3_config = json.loads((root / "rq3_config.json").read_text(encoding="utf-8"))
            self.assertEqual(rq3_config["active_layers"], [2, 24, 38])


if __name__ == "__main__":
    unittest.main()
