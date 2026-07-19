# RQ3 Result: Cross-Condition Activation Patching

## Status

This report was updated from:

```text
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709
```

The run covers 10 selected synthetic tasks with `meta-llama/Llama-3.2-3B-Instruct`, rank-8 LoRA adapters, `test` split, `max_samples=16`, `seed=13`, `dtype=bfloat16`, `device=cuda`, `prompt_format=chat_template`, EOS appended, and `max_new_tokens=20`.

All 10 tasks have the expected 12 default text-span patch runs:

```text
4 source/target directions x 3 layers = 12 runs per task
```

RQ3 status in the generated configs remains `partial`: controls, generation metrics, semantic task scoring, and shape checks are present; activation-site sweep is still pending.

## Method

RQ3 patches block-output activations over the raw text span:

```text
patch_span = text
activation_site = block_output
layers = 1, 21, 27
```

Default directions:

| Direction | Question |
| --- | --- |
| `lora_only -> instruction_only` | Can LoRA text activations improve or redirect the instruction-only target? |
| `lora_only -> base` | Can LoRA text activations transfer task behavior into the base target? |
| `instruction_only -> lora_only` | Can instruction text activations replace or disrupt LoRA behavior? |
| `base -> lora_only` | Control: does base activation patching disrupt LoRA behavior? |

Each patch run records three controls:

| Control | Meaning |
| --- | --- |
| `unpatched` | Target condition without patching |
| `base_to_target_patch` | Base activations patched into the target |
| `source_to_target_patch` | Requested source activations patched into the target |

## Cross-Task Summary

The table below reports unweighted averages across the 10 tasks. `sem` is teacher-forced semantic correctness. `gen_sem` is autoregressive semantic correctness from `outcomes.jsonl`.

| Direction | Layer | Control | Loss | Token acc | Seq acc | sem | gen_sem |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `lora_only -> instruction_only` | 1 | `unpatched` | 2.4451 | 0.7002 | 0.5563 | 0.5500 | 0.5500 |
| `lora_only -> instruction_only` | 1 | `source_to_target_patch` | 2.4172 | 0.6685 | 0.5312 | 0.5250 | 0.5250 |
| `lora_only -> instruction_only` | 21 | `unpatched` | 2.4451 | 0.7002 | 0.5563 | 0.5500 | 0.5500 |
| `lora_only -> instruction_only` | 21 | `source_to_target_patch` | 2.4503 | 0.6790 | 0.4938 | 0.4875 | 0.4875 |
| `lora_only -> instruction_only` | 27 | `unpatched` | 2.4451 | 0.7002 | 0.5563 | 0.5500 | 0.5500 |
| `lora_only -> instruction_only` | 27 | `source_to_target_patch` | 2.4451 | 0.7002 | 0.5563 | 0.5500 | 0.5500 |
| `lora_only -> base` | 1 | `unpatched` | 10.5255 | 0.2200 | 0.0187 | 0.0187 | 0.0000 |
| `lora_only -> base` | 1 | `source_to_target_patch` | 9.7473 | 0.2352 | 0.0250 | 0.0250 | 0.0000 |
| `lora_only -> base` | 21 | `unpatched` | 10.5255 | 0.2200 | 0.0187 | 0.0187 | 0.0000 |
| `lora_only -> base` | 21 | `source_to_target_patch` | 9.7964 | 0.2172 | 0.0187 | 0.0187 | 0.0000 |
| `lora_only -> base` | 27 | `unpatched` | 10.5255 | 0.2200 | 0.0187 | 0.0187 | 0.0000 |
| `lora_only -> base` | 27 | `source_to_target_patch` | 10.5255 | 0.2200 | 0.0187 | 0.0187 | 0.0000 |
| `instruction_only -> lora_only` | 1 | `unpatched` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `instruction_only -> lora_only` | 1 | `source_to_target_patch` | 0.0689 | 0.9812 | 0.9750 | 0.9688 | 0.5062 |
| `instruction_only -> lora_only` | 21 | `unpatched` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `instruction_only -> lora_only` | 21 | `source_to_target_patch` | 0.0206 | 0.9863 | 0.9563 | 0.9500 | 0.5125 |
| `instruction_only -> lora_only` | 27 | `unpatched` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `instruction_only -> lora_only` | 27 | `source_to_target_patch` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `base -> lora_only` | 1 | `unpatched` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `base -> lora_only` | 1 | `source_to_target_patch` | 0.0088 | 0.9969 | 0.9938 | 0.9875 | 0.5125 |
| `base -> lora_only` | 21 | `unpatched` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `base -> lora_only` | 21 | `source_to_target_patch` | 0.0222 | 0.9945 | 0.9625 | 0.9563 | 0.5437 |
| `base -> lora_only` | 27 | `unpatched` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |
| `base -> lora_only` | 27 | `source_to_target_patch` | 0.0086 | 0.9969 | 0.9938 | 0.9875 | 0.5312 |

## Interpretation

The text-span patching matrix does not support strong computational interchangeability between LoRA and natural-language instruction.

Main observations:

- LoRA-only targets are very strong under teacher forcing: unpatched `lora_only` averages 0.9938 sequence accuracy and 0.9875 semantic correctness.
- Instruction-only targets are weaker: unpatched `instruction_only` averages 0.5563 sequence accuracy and 0.5500 semantic correctness.
- Patching `lora_only` activations into `instruction_only` does not improve the instruction target on average; layer 21 is worse than unpatched.
- Patching `lora_only` activations into `base` does not transfer task behavior. Autoregressive semantic correctness stays 0.0000.
- Patching `instruction_only` or `base` into a LoRA target causes only small teacher-forced degradation at layers 1 and 21, and no visible effect at layer 27 in this matrix.

The current RQ3 evidence is therefore negative for simple interchangeability: LoRA activations over the text span are not sufficient to make weaker targets behave like the LoRA condition, and instruction activations are not a clean substitute for the LoRA path.

## Output Files

For each task:

```text
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/rq3_config.json
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/rq3_status.json
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/patches/rq3/{source}_to_{target}_l{layer}_text/
experiments/lora_selected_tasks_instruct_rawchat_r8_20260709/{task}/plots/patch_loss/rq3/
```

Each patch directory contains:

```text
config.json
confusion_matrix.json
dataset_snapshot.jsonl
generations.jsonl
metrics.jsonl
outcomes.jsonl
```

## Limitations

- RQ3 is still marked partial because activation-site sweep is pending.
- Each task uses 16 test samples.
- The patch is limited to block-output activations over text span at layers 1, 21, and 27.
- Teacher-forced metrics and autoregressive semantic correctness can diverge; both should be reported when making causal claims.
