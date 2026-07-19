# RQ1 Result: Residual Perturbation Similarity

## Status

This report was updated from:

```text
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709
```

The run covers 10 selected synthetic tasks with `meta-llama/Llama-3.2-3B-Instruct`, rank-8 LoRA adapters, `test` split, `max_samples=16`, `seed=13`, `dtype=bfloat16`, `device=cuda`, `prompt_format=chat_template`, and EOS appended.

RQ1 compares residual-stream perturbation directions:

```text
cosine(hidden_instruction_only - hidden_base, hidden_lora_only - hidden_base)
```

The comparison is computed on aligned teacher-forced target-token positions.

## Tasks

| Task | Rows | Mean cosine | Min | Max | Best layer | Best layer mean | LoRA seq acc | Instruction seq acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `reverse_words` | 54,404 | 0.6488 | -0.3409 | 0.9273 | 27 | 0.7820 | 1.0000 | 0.0000 |
| `first_word` | 2,204 | 0.3025 | -0.2721 | 0.7629 | 28 | 0.5253 | 1.0000 | 0.9375 |
| `words_starting_with_letter` | 2,552 | 0.3914 | -0.1739 | 0.8295 | 23 | 0.5653 | 0.9375 | 0.7500 |
| `exact_three_word_prefix` | 5,568 | 0.3170 | -0.3816 | 0.8498 | 27 | 0.5853 | 1.0000 | 1.0000 |
| `has_repeated_word` | 1,856 | 0.2150 | -0.1977 | 0.5559 | 28 | 0.4185 | 1.0000 | 0.8750 |
| `at_operator_mod_minus_left` | 2,552 | 0.1731 | -0.2575 | 0.6297 | 25 | 0.3838 | 1.0000 | 0.0000 |
| `formal_language_a_n_b_n` | 1,856 | 0.3809 | 0.0000 | 0.6502 | 28 | 0.6059 | 1.0000 | 0.7500 |
| `extract_items_from_set` | 6,960 | 0.3112 | -0.1946 | 0.7188 | 28 | 0.5703 | 1.0000 | 0.6875 |
| `words_containing_bigram_qu` | 3,248 | 0.2572 | -0.0732 | 0.5138 | 28 | 0.4543 | 1.0000 | 0.5000 |
| `uppercase_last_word` | 3,712 | 0.4822 | -0.3678 | 0.9449 | 24 | 0.7033 | 1.0000 | 0.0625 |

Across tasks, the unweighted mean of task-level mean cosine similarity is **0.3479**. Task means range from **0.1731** (`at_operator_mod_minus_left`) to **0.6488** (`reverse_words`).

## Interpretation

The LoRA adapters are behaviorally strong on teacher-forced sequence accuracy: 9 of 10 tasks reach 1.0000, and `words_starting_with_letter` reaches 0.9375. Instruction-only prompting is much less consistent, ranging from 0.0000 to 1.0000.

Residual perturbation similarity is moderate rather than near-identical. Best layers are mostly late layers, especially layers 24-28, but overall task means remain well below a computational-equivalence result.

The current RQ1 evidence supports a cautious claim: LoRA-only and instruction-only runs often move residual activations in related directions, but the effect is task-dependent and does not show that the adapter reconstructs the same residual-stream trajectory as natural-language instruction prompting.

## Output Files

For each task:

```text
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/states/rq1/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq1/aggregate_similarity.csv
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq1/quality_summary.csv
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq1/token_similarity.csv
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq1/token_similarity.html
```

## Limitations

- Each task uses 16 test samples, so task-level estimates are still small-sample.
- RQ1 measures directional similarity of residual perturbations, not causal interchangeability.
- Instruction-only behavior is weak for several tasks, which limits direct equivalence claims.
