# RQ1 Result: Residual Perturbation Similarity

## Status

This report uses the latest RQ1 rerun with explicit target-token alignment metadata.

- Date run: 2026-07-01
- Base model: `meta-llama/Llama-3.2-3B`
- Conditions:
  - `base`: no instruction, no LoRA
  - `instruction_only`: natural-language task instruction, no LoRA
  - `lora_only`: LoRA adapter, no instruction
- Metric:
  - `cosine(hidden_instruction_only - hidden_base, hidden_lora_only - hidden_base)`
  - computed only on teacher-forced target output token positions
- Alignment:
  - target tokens aligned by `target:{token_index}:{token_id}`
  - mismatched target tokenization now fails instead of truncating by length

## Runs

| Task | Run directory | Samples |
| --- | --- | ---: |
| `add_zxq_after_t_or_l` | `experiments/add_zxq_llama32_3b_r8_20260628_retry2` | 100 |
| `last_word` | `experiments/last_word_llama32_3b_r8_clean` | 100 |

## Adapter Quality

### `add_zxq_after_t_or_l`

| Condition | Mean loss | Token accuracy | Sequence accuracy | Mean target tokens |
| --- | ---: | ---: | ---: | ---: |
| `base` | 1.2884 | 0.8489 | 0.0400 | 32.08 |
| `instruction_only` | 0.6836 | 0.9044 | 0.1400 | 32.08 |
| `lora_only` | 0.0384 | 0.9868 | 0.5800 | 32.08 |

### `last_word`

| Condition | Mean loss | Token accuracy | Sequence accuracy | Mean target tokens |
| --- | ---: | ---: | ---: | ---: |
| `base` | 7.0229 | 0.1425 | 0.0100 | 1.35 |
| `instruction_only` | 6.1725 | 0.1417 | 0.0100 | 1.35 |
| `lora_only` | 0.0343 | 0.9900 | 0.9900 | 1.35 |

## Residual Similarity Summary

| Task | Token-layer rows | Overall mean cosine | Min | Max | Best layer | Best layer mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `add_zxq_after_t_or_l` | 93,032 | 0.3290 | -0.7936 | 0.8742 | 16 | 0.4318 |
| `last_word` | 3,915 | 0.3938 | -0.4426 | 0.7974 | 22 | 0.5644 |

### Best Layers

| Task | Layer | Mean cosine | Count | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| `add_zxq_after_t_or_l` | 16 | 0.4318 | 3,208 | 0.0470 | 0.8109 |
| `add_zxq_after_t_or_l` | 13 | 0.4239 | 3,208 | 0.0779 | 0.7126 |
| `add_zxq_after_t_or_l` | 17 | 0.4233 | 3,208 | 0.0269 | 0.8109 |
| `add_zxq_after_t_or_l` | 18 | 0.4207 | 3,208 | 0.0492 | 0.8033 |
| `add_zxq_after_t_or_l` | 19 | 0.4191 | 3,208 | 0.0357 | 0.8079 |
| `last_word` | 22 | 0.5644 | 135 | 0.1470 | 0.7566 |
| `last_word` | 21 | 0.5571 | 135 | 0.1120 | 0.7458 |
| `last_word` | 23 | 0.5566 | 135 | 0.1337 | 0.7608 |
| `last_word` | 24 | 0.5475 | 135 | 0.1262 | 0.7652 |
| `last_word` | 20 | 0.5393 | 135 | 0.0655 | 0.7397 |

## Interpretation

Both LoRA adapters learned their target task well. The `last_word` adapter is especially clean, with 0.99 token and sequence accuracy. The `add_zxq_after_t_or_l` adapter has high token accuracy but lower sequence accuracy because the target sequence is much longer.

Residual perturbations are moderately aligned. `last_word` reaches stronger best-layer alignment than `add_zxq_after_t_or_l`, but neither task shows near-identical perturbation directions across the full residual stream.

This supports a cautious conclusion: LoRA-only and instruction-only behavior can point residual changes in related directions, especially in middle/late layers, but RQ1 alone does not prove that LoRA reconstructs the same computation path as natural-language instruction prompting.

## Output Files

### `add_zxq_after_t_or_l`

- `experiments/add_zxq_llama32_3b_r8_20260628_retry2/states/rq1/`
- `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq1/token_similarity.html`
- `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq1/token_similarity.csv`
- `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq1/aggregate_similarity.csv`
- `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq1/quality_summary.csv`

### `last_word`

- `experiments/last_word_llama32_3b_r8_clean/states/rq1/`
- `experiments/last_word_llama32_3b_r8_clean/plots/rq1/token_similarity.html`
- `experiments/last_word_llama32_3b_r8_clean/plots/rq1/token_similarity.csv`
- `experiments/last_word_llama32_3b_r8_clean/plots/rq1/aggregate_similarity.csv`
- `experiments/last_word_llama32_3b_r8_clean/plots/rq1/quality_summary.csv`

## Limitations

- RQ1 compares residual perturbation directions, not causal interchangeability.
- RQ2/RQ3 should be rerun after their alignment-specific fixes before making claims about attention routing or causal patchability.
- `add_zxq_after_t_or_l` is harder to judge at sequence level because long targets make exact sequence accuracy strict.
