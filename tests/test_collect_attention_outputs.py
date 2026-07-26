from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch import nn

from lora_instruction_analysis.model.collect import (
    _attention_output_hooks,
    _stack_attention_outputs,
    _validate_attention_layer_names,
    _validate_attention_layer_shapes,
)


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


class FakeModelWithUnusedAttention(FakeModel):
    def __init__(self):
        super().__init__()
        self.unused = FakeLayer()


class CollectAttentionOutputTests(unittest.TestCase):
    def test_collects_pre_o_proj_outputs_by_module_head_layout(self):
        model = FakeModel()
        pre_captures, post_captures, head_ablation_captures, layer_names, handles, expected_layers = _attention_output_hooks(
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

        self.assertEqual(layer_names, ["layers.0.self_attn", "layers.1.self_attn"])
        self.assertEqual(tuple(stacked.shape), (2, 2, 2, 3))
        self.assertEqual(tuple(post_stacked.shape), (2, 2, 6))
        self.assertEqual(tuple(ablation_stacked.shape), (2, 2, 2, 4))
        self.assertTrue(torch.equal(stacked[0], x[0, [1, 2], :].view(2, 2, 3).permute(1, 0, 2)))
        self.assertTrue(torch.equal(post_stacked[0], x[0, [1, 2], :]))
        self.assertTrue(torch.all(ablation_stacked[..., 1] > 0))

    def test_stacks_only_attention_modules_executed_by_forward(self):
        model = FakeModelWithUnusedAttention()
        pre_captures, _post_captures, _head_ablation_captures, layer_names, handles, registered_layers = _attention_output_hooks(
            model, torch.tensor([1, 2])
        )
        try:
            x = torch.arange(1 * 4 * 6, dtype=torch.float32).view(1, 4, 6)
            model(x)
        finally:
            for handle in handles:
                handle.remove()

        self.assertEqual(registered_layers, 3)
        self.assertEqual(len(pre_captures), 2)
        self.assertEqual(layer_names, ["layers.0.self_attn", "layers.1.self_attn"])
        stacked = _stack_attention_outputs(torch, pre_captures, len(pre_captures))
        self.assertEqual(tuple(stacked.shape), (2, 2, 2, 3))

    def test_attention_layer_name_changes_fail(self):
        expected = _validate_attention_layer_names(None, ["a.self_attn", "b.self_attn"], "s1", "base")
        self.assertEqual(expected, ["a.self_attn", "b.self_attn"])
        self.assertEqual(
            _validate_attention_layer_names(
                expected,
                ["base_model.model.a.self_attn", "base_model.model.b.self_attn"],
                "s1",
                "lora_only",
            ),
            expected,
        )
        with self.assertRaisesRegex(RuntimeError, "layer path changed"):
            _validate_attention_layer_names(expected, ["a.self_attn", "c.self_attn"], "s1", "lora_only")

    def test_attention_stack_can_pad_heterogeneous_head_dims(self):
        captures = [torch.ones(8, 2, 256), torch.ones(8, 2, 512)]

        stacked = _stack_attention_outputs(torch, captures, 2, pad=True)

        self.assertEqual(tuple(stacked.shape), (2, 8, 2, 512))
        self.assertTrue(torch.equal(stacked[0, :, :, 256:], torch.zeros(8, 2, 256)))

    def test_attention_layer_shape_changes_fail(self):
        expected = _validate_attention_layer_shapes(None, [[8, 2, 256], [8, 2, 512]], "s1", "base")
        self.assertEqual(expected, [[8, 256], [8, 512]])
        self.assertEqual(
            _validate_attention_layer_shapes(expected, [[8, 3, 256], [8, 3, 512]], "s2", "base"),
            expected,
        )
        with self.assertRaisesRegex(RuntimeError, "layer shapes changed"):
            _validate_attention_layer_shapes(expected, [[8, 2, 256], [8, 2, 256]], "s1", "lora_only")

    def test_missing_head_layout_fails(self):
        model = FakeModel()
        delattr(model.layers[0].self_attn, "head_dim")

        with self.assertRaisesRegex(RuntimeError, "Cannot infer real head layout"):
            _attention_output_hooks(model, torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
