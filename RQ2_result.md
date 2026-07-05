# RQ2 / RQ2.1 Result

## Scope

This note summarizes the current attention-side experiments:

- RQ2: attention probability pattern similarity.
- RQ2.1: pre-`o_proj` per-head attention output similarity.

Both analyses now report two condition pairs:

| Pair | Purpose |
| --- | --- |
| `instruction_only_vs_lora_only` | Whether LoRA reproduces instruction-induced attention behavior |
| `lora_only_vs_base` | Whether the adapter changes attention behavior relative to the base model |

## Runs

Both runs used `meta-llama/Llama-3.2-3B`, rank-8 LoRA adapters, `test` split, `seed=13`, `dtype=bfloat16`, `device=auto`, and `max_samples=16`.

| Task | Run directory | Adapter |
| --- | --- | --- |
| `add_zxq_after_t_or_l` | `experiments/add_zxq_llama32_3b_r8_20260628_retry2` | `adapters/r8` |
| `last_word` | `experiments/last_word_llama32_3b_r8_clean` | `adapters/r8` |

The compared conditions were:

| Condition | Description |
| --- | --- |
| `base` | Base model, no task instruction, no LoRA adapter |
| `instruction_only` | Base model with task instruction, no LoRA adapter |
| `lora_only` | Base model with LoRA adapter, no task instruction |

All conditions were evaluated with teacher forcing on the same target output sequence.

## Method

RQ2 compares target-position attention probability vectors:

```text
attention_probs = softmax(QK^T)
```

For key-side alignment, the implementation compares only shared `source_alignment` keys:

```text
input:* and already-emitted target:*
```

Instruction-prefix keys are excluded. This makes the comparison span-aligned over comparable input and target-prefix tokens instead of truncating by raw sequence position.

RQ2.1 captures the input to each layer's `self_attn.o_proj` with a forward pre-hook, using the module's real `num_heads` / `head_dim` layout:

```text
layer, head, target_token, head_dim
```

This is the per-head pre-`o_proj` attention output:

```text
attention_output = attention_probs @ V
```

## Task Quality

| Task | Condition | Mean loss | Token accuracy | Sequence accuracy |
| --- | --- | ---: | ---: | ---: |
| `add_zxq_after_t_or_l` | Base | 0.9255 | 0.8719 | 0.1875 |
| `add_zxq_after_t_or_l` | Instruction-only | 0.5854 | 0.9204 | 0.2500 |
| `add_zxq_after_t_or_l` | LoRA-only | 0.0283 | 0.9905 | 0.6875 |
| `last_word` | Base | 6.3218 | 0.2135 | 0.0000 |
| `last_word` | Instruction-only | 5.6400 | 0.1979 | 0.0000 |
| `last_word` | LoRA-only | 0.0009 | 1.0000 | 1.0000 |

In both tasks, LoRA-only is the strongest behavioral condition.

## Results

### RQ2: Attention Probability Similarity

| Task | Pair | Rows | Mean | Median | Q25 | Q75 | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `add_zxq_after_t_or_l` | `instruction_only_vs_lora_only` | 358,848 | 0.7969 | 0.8511 | 0.6995 | 0.9541 | 0.0035 | 1.0000 |
| `add_zxq_after_t_or_l` | `lora_only_vs_base` | 358,848 | 0.7972 | 0.8479 | 0.6857 | 0.9630 | 0.0059 | 1.0000 |
| `last_word` | `instruction_only_vs_lora_only` | 16,800 | 0.7910 | 0.8879 | 0.6758 | 0.9776 | 0.0100 | 1.0000 |
| `last_word` | `lora_only_vs_base` | 16,800 | 0.8195 | 0.8985 | 0.7205 | 0.9874 | 0.0106 | 1.0000 |

Attention routing is highly similar in both tasks. The LoRA-vs-base line is also high, which means the adapter does not radically rewrite attention probability patterns.

### RQ2.1: Attention Output Similarity

| Task | Pair | Rows | Mean | Median | Q25 | Q75 | Min | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `add_zxq_after_t_or_l` | `instruction_only_vs_lora_only` | 358,848 | 0.5643 | 0.5635 | 0.4205 | 0.7232 | -0.6918 | 1.0000 |
| `add_zxq_after_t_or_l` | `lora_only_vs_base` | 358,848 | 0.5190 | 0.5047 | 0.3418 | 0.7152 | -0.6968 | 1.0000 |
| `last_word` | `instruction_only_vs_lora_only` | 16,800 | 0.6210 | 0.6465 | 0.4508 | 0.8200 | -0.5788 | 1.0000 |
| `last_word` | `lora_only_vs_base` | 16,800 | 0.6295 | 0.6528 | 0.4393 | 0.8600 | -0.7106 | 0.9999 |

Attention outputs are much less similar than attention probabilities. This holds both for instruction-vs-LoRA and LoRA-vs-base.

## Outputs

| Task | RQ2 attention probs | RQ2.1 attention probs | RQ2.1 attention outputs |
| --- | --- | --- | --- |
| `add_zxq_after_t_or_l` | `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq2` | `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq21/attention_probs` | `experiments/add_zxq_llama32_3b_r8_20260628_retry2/plots/rq21/attention_outputs` |
| `last_word` | `experiments/last_word_llama32_3b_r8_clean/plots/rq2` | `experiments/last_word_llama32_3b_r8_clean/plots/rq21/attention_probs` | `experiments/last_word_llama32_3b_r8_clean/plots/rq21/attention_outputs` |

Each HTML report now plots layer-mean similarity with separate colored lines for:

```text
instruction_only_vs_lora_only
lora_only_vs_base
```

## Interpretation

The main result is stable across both tasks:

```text
attention probabilities: high similarity
attention outputs: lower similarity
```

So the adapter largely preserves where attention looks, but changes what attention reads out. The `lora_only_vs_base` comparison strengthens this interpretation: LoRA remains close to base in attention routing, yet diverges more in pre-`o_proj` value-space outputs.

The conservative conclusion is:

```text
LoRA does not mainly implement these tasks by rewriting attention routing.
Its larger effect appears in the value/readout side of attention.
```

This does not isolate the final residual writeback by head, and it does not prove the change is specifically in `v_proj`. The captured tensor is `attention_probs @ V`, so differences can come from value projection weights, hidden states entering value projection, or earlier-layer changes.

## Limitations

- Attention probability KL/entropy are computed on raw probability mass over the shared support, not a renormalized distribution.
- RQ2.1 captures pre-`o_proj` per-head outputs, but does not decompose the final `o_proj` residual writeback by head.
- Results currently use 16 test samples per task.

## Conclusion

RQ2 shows high attention probability similarity for both tasks and both condition pairs. RQ2.1 shows lower pre-`o_proj` attention output similarity. The current evidence supports a routing-preserving, readout-changing view of the LoRA adapters.
