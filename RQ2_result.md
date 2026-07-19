# RQ2 / RQ2.1 Result

## Status

This report was updated from:

```text
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709
```

The run covers 10 selected synthetic tasks with `meta-llama/Llama-3.2-3B-Instruct`, rank-8 LoRA adapters, `test` split, `max_samples=16`, `seed=13`, `dtype=bfloat16`, `device=cuda`, `prompt_format=chat_template`, and EOS appended.

RQ2 compares attention probability patterns between `instruction_only` and `lora_only`.

RQ2.1 compares attention-side vectors, including pre-`o_proj` attention outputs and related output-delta diagnostics.

## RQ2: Attention Probability Similarity

| Task | Rows | Mean cosine | Min | Max | LoRA seq acc | Instruction seq acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `reverse_words` | 3,782,016 | 0.8356 | 0.0011 | 1.0000 | 1.0000 | 0.0000 |
| `first_word` | 153,216 | 0.8784 | 0.0096 | 1.0000 | 1.0000 | 0.9375 |
| `words_starting_with_letter` | 177,408 | 0.8846 | 0.0578 | 1.0000 | 0.9375 | 0.7500 |
| `exact_three_word_prefix` | 387,072 | 0.9274 | 0.0328 | 1.0000 | 1.0000 | 1.0000 |
| `has_repeated_word` | 129,024 | 0.9015 | 0.0558 | 1.0000 | 1.0000 | 0.8750 |
| `at_operator_mod_minus_left` | 177,408 | 0.9026 | 0.0325 | 1.0000 | 1.0000 | 0.0000 |
| `formal_language_a_n_b_n` | 129,024 | 0.9444 | 0.1443 | 1.0000 | 1.0000 | 0.7500 |
| `extract_items_from_set` | 483,840 | 0.8653 | 0.0534 | 1.0000 | 1.0000 | 0.6875 |
| `words_containing_bigram_qu` | 225,792 | 0.9139 | 0.0257 | 1.0000 | 1.0000 | 0.5000 |
| `uppercase_last_word` | 258,048 | 0.8442 | 0.0130 | 1.0000 | 1.0000 | 0.0625 |

Across tasks, the unweighted mean of task-level RQ2 mean cosine similarity is **0.8898**. Task means range from **0.8356** to **0.9444**.

## RQ2.1: Attention-Side Diagnostics

| Diagnostic | Tasks | Mean cosine | Min task mean | Max task mean |
| --- | ---: | ---: | ---: | ---: |
| `attention_probs` | 10 | 0.8898 | 0.8356 | 0.9444 |
| `attention_outputs` | 10 | 0.6316 | 0.5759 | 0.6745 |
| `attention_output_deltas` | 10 | 0.2914 | 0.1522 | 0.6044 |
| `attention_post_o_proj_outputs` | 10 | 0.5842 | 0.5163 | 0.6486 |
| `attention_post_o_proj_output_deltas` | 10 | 0.2994 | 0.1427 | 0.6330 |
| `attention_head_ablation` | 10 | 0.9723 | 0.9717 | 0.9733 |

## Interpretation

Attention probability routing is highly similar between instruction-only and LoRA-only runs across all 10 tasks. This remains true even when instruction-only prompting is behaviorally weak, so high RQ2 similarity should not be read as task success by itself.

RQ2.1 shows the important split: attention probabilities are high-similarity, but attention outputs and post-`o_proj` outputs are much lower. Output deltas are lower still. This supports the current interpretation that LoRA does not primarily implement these tasks by changing where attention looks; the larger differences appear in what attention reads out and writes onward.

The `attention_head_ablation` diagnostic is very high and tightly clustered. Treat it as a stability/control signal rather than evidence that individual heads are interchangeable.

## Output Files

For each task:

```text
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq2/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq21/attention_probs/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq21/attention_outputs/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq21/attention_output_deltas/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq21/attention_post_o_proj_outputs/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq21/attention_post_o_proj_output_deltas/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/rq21/attention_head_ablation/
```

## Limitations

- Each task uses 16 test samples.
- RQ2 is a similarity measurement over attention probability patterns, not a causal intervention.
- RQ2.1 localizes the gap to attention-side vector content more than routing, but it does not by itself isolate whether differences originate in value projections, earlier hidden states, or downstream residual writes.
