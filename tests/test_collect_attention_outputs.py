from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from lora_instruction_analysis.model.collect import _attention_output_hooks, _stack_attention_outputs


class FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.head_dim = 3
        self.o_proj = nn.Linear(6, 6, bias=False)
        self.o_proj.weight.data.copy_(torch.eye(6))

    def forward(self, x):
        return self.o_proj(x)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeSelfAttention()

    def forward(self, x):
        return self.self_attn(x)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer(), FakeLayer()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CollectAttentionOutputTests(unittest.TestCase):
    def test_collects_pre_o_proj_outputs_by_module_head_layout(self):
        model = FakeModel()
        pre_captures, post_captures, head_ablation_captures, handles, expected_layers = _attention_output_hooks(
            model, torch.tensor([1, 2])
        )
        try:
            x = torch.arange(1 * 4 * 6, dtype=torch.float32).view(1, 4, 6)
            model(x)
        finally:
            for handle in handles:
                handle.remove()

        stacked = _stack_attention_outputs(torch, pre_captures, expected_layers)
        post_stacked = _stack_attention_outputs(torch, post_captures, expected_layers)
        ablation_stacked = _stack_attention_outputs(torch, head_ablation_captures, expected_layers)

        self.assertEqual(tuple(stacked.shape), (2, 2, 2, 3))
        self.assertEqual(tuple(post_stacked.shape), (2, 2, 6))
        self.assertEqual(tuple(ablation_stacked.shape), (2, 2, 2, 4))
        self.assertTrue(torch.equal(stacked[0], x[0, [1, 2], :].view(2, 2, 3).permute(1, 0, 2)))
        self.assertTrue(torch.equal(post_stacked[0], x[0, [1, 2], :]))
        self.assertTrue(torch.all(ablation_stacked[..., 1] > 0))

    def test_missing_head_layout_fails(self):
        model = FakeModel()
        delattr(model.layers[0].self_attn, "head_dim")

        with self.assertRaisesRegex(RuntimeError, "Cannot infer real head layout"):
            _attention_output_hooks(model, torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
