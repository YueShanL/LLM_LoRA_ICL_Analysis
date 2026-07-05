from argparse import Namespace
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.experiment import run_all_rqs


class RunAllRQTests(unittest.TestCase):
    def test_runs_existing_rq_runners_with_original_output_locations(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "model_name": "model",
                        "dataset_dir": str(run_dir / "data"),
                        "adapter_dir": str(run_dir / "adapter"),
                        "seed": 7,
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace(
                run_dir=run_dir,
                model_name=None,
                dataset_path=None,
                adapter_path=None,
                split="test",
                max_samples=3,
                seed=None,
                dtype="auto",
                device="auto",
                states_dir=None,
                plots_dir=None,
                output_dir=None,
                source_condition="instruction_only",
                target_condition="lora_only",
                layer=4,
                max_new_tokens=20,
                patch_span="target",
                prompt_format="chat_template",
                no_append_eos=True,
            )
            seen = []

            with (
                patch.object(
                    run_all_rqs.run_rq1,
                    "run_rq1",
                    lambda config: seen.append(
                        ("rq1", config.resolved_states_dir, config.resolved_plots_dir, config.prompt_format, config.append_eos)
                    ),
                ),
                patch.object(
                    run_all_rqs.run_rq2,
                    "run_rq2",
                    lambda config: seen.append(
                        (
                            config.rq_name,
                            config.resolved_states_dir,
                            config.resolved_plots_dir,
                            config.collect_attention_outputs,
                            config.prompt_format,
                            config.append_eos,
                        )
                    ),
                ),
                patch.object(
                    run_all_rqs.run_rq3,
                    "run_rq3",
                    lambda config: seen.append(
                        ("rq3", config.resolved_output_dir, config.layer, config.prompt_format, config.append_eos)
                    ),
                ),
            ):
                run_all_rqs.run_all_rqs(args)

        self.assertEqual(
            seen,
            [
                ("rq1", run_dir / "states" / "rq1", run_dir / "plots" / "rq1", "chat_template", False),
                ("rq2", run_dir / "states" / "rq2", run_dir / "plots" / "rq2", False, "chat_template", False),
                ("rq21", run_dir / "states" / "rq21", run_dir / "plots" / "rq21", True, "chat_template", False),
                ("rq3", run_dir / "patches" / "rq3", 4, "chat_template", False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
