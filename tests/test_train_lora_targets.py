from pathlib import Path
import sys

import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lora_instruction_analysis.model.train_lora import _resolve_target_modules


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language = nn.Module()
        self.language.layers = nn.ModuleList([nn.Module(), nn.Module()])
        for layer in self.language.layers:
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Linear(2, 2)
            layer.self_attn.v_proj = nn.Linear(2, 2)
        self.vision = nn.Module()
        self.vision.q_proj = nn.Module()
        self.vision.q_proj.linear = nn.Linear(2, 2)


def test_resolve_target_modules_expands_globs_to_full_names():
    assert _resolve_target_modules(
        TinyModel(),
        (
            "language.layers.*.self_attn.q_proj",
            "language.layers.*.self_attn.v_proj",
        ),
    ) == (
        "language.layers.0.self_attn.q_proj",
        "language.layers.0.self_attn.v_proj",
        "language.layers.1.self_attn.q_proj",
        "language.layers.1.self_attn.v_proj",
    )


def test_resolve_target_modules_leaves_short_names_for_existing_models():
    assert _resolve_target_modules(TinyModel(), ("q_proj", "v_proj")) == ("q_proj", "v_proj")
