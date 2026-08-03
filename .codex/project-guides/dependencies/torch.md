# torch Guide

Generated fallback guide from the repository venv. No dependency-owned `GUIDE.md`, `SKILL.md`, or `GUIDE_INDEX.json` was present. This is intentionally limited to the public surface needed by the standalone learnable block-attention index pipeline.

## Resolution

- Resolved version: `2.11.0+cu128` (CUDA 12.8 wheel).
- Python requirement: `>=3.10`.
- Local sources: `venv/Lib/site-packages/torch/` and `venv/Lib/site-packages/torch-2.11.0+cu128.dist-info/METADATA`.
- The project manifest permits `torch>=2.2.0`, but this guide is aligned to the installed `2.11.0+cu128` API.

## Router Model and Data

- Build query/key networks as ordinary `torch.nn.Module` objects. The small baseline surface is `nn.Linear(in_features, out_features, bias=True)`, `nn.LayerNorm(normalized_shape, eps=1e-5)`, an activation, and `torch.nn.functional.normalize(input, p=2, dim=-1, eps=1e-12)` before a cosine-compatible dot product.
- A map-style `torch.utils.data.Dataset` implements `__len__` and `__getitem__`. Batch it with `DataLoader(dataset, batch_size=..., shuffle=..., num_workers=0, collate_fn=...)`; use a custom collator for the variable number of candidate blocks and carry an explicit candidate mask.
- Train only router parameters with `torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, ...)`. Call `optimizer.zero_grad(set_to_none=True)`, `loss.backward()`, and `optimizer.step()` in the normal autograd region.
- Mixed precision is available through `torch.autocast(device_type, dtype=..., enabled=True)` and `torch.amp.GradScaler(device="cuda", enabled=True)`. Keep probability reductions and calibration metrics in float32 when numerical range matters.

## Loss and Isolation Semantics

- `torch.nn.functional.kl_div(input, target, reduction="batchmean", log_target=False)` expects `input` to contain log-probabilities. Use `log_softmax(router_scores, dim=-1)` and a normalized target distribution; `batchmean` has the mathematical KL scaling, unlike the default elementwise `mean`.
- `torch.nn.functional.cross_entropy` accepts class probabilities with the same shape as logits, but KL makes the soft-distribution contract more explicit. Use `binary_cross_entropy_with_logits` or a regression loss for the separate total historical-mass/demand head.
- Mask non-candidate padding before `softmax`/`log_softmax`; do not allow padding blocks into the conditional distribution.
- Frozen-model teacher/student collection belongs under `torch.inference_mode()` (or `no_grad()` when version counters are required), and collected residual/attention tensors should be detached before persistence. Do not wrap router training in either context.
- Keep teacher labels and student router inputs as separate tensors/artifacts. Device placement and autograd do not enforce the specification's information boundary.

## Checkpoints and Reproducibility

- Prefer saving a plain dictionary of router `state_dict`, optimizer state, resolved hyperparameters, feature dimensions, and dataset/label schema via `torch.save`.
- Load with an explicit `map_location`. In this version, `torch.load(..., weights_only=True)` is the documented default; set policy deliberately and never load an untrusted pickle payload.
- Record random seeds and the candidate-mask/pooling conventions in the run manifest. A state dict alone is insufficient to reproduce block scores.

## Pipeline Constraints

- Keep residual summaries, attention targets, and persisted metrics in explicit dtypes; do not silently inherit the frozen language model's half precision for router targets.
- Mean pooling must divide only by real block/query tokens, not batch padding.
- Compute the conditional block-distribution loss only for samples with a defined historical distribution; retain the absolute historical mass and train/report its loss separately.
- Start Windows smoke tests with `num_workers=0`; multi-process loading requires pickle-safe dataset/collator definitions and can obscure deterministic failures.
