# RQ3 Result: cross-condition activation patching

## Scope

RQ3 tests whether LoRA and natural-language instruction use interchangeable computation paths.

The experiment uses activation patching: run a source condition, capture hidden activations at one transformer block, inject those activations into a target condition, and measure whether target behavior is preserved or redirected.

The current implementation keeps model-side patching in:

```text
src/lora_instruction_analysis/model/patch.py
```

The runner is intentionally thin:

```text
src/lora_instruction_analysis/experiment/run_rq3.py
```

It only resolves `run_dir/config.json`, writes `rq3_config.json`, and calls `run_activation_patching`.

## Experimental Setup

Run directory:

```text
experiments/last_word_llama32_3b_r8_clean
```

Model and task:

| Field | Value |
| --- | --- |
| Base model | `meta-llama/Llama-3.2-3B` |
| Task | `last_word` |
| Adapter | `adapters/r8` |
| Split | `test` |
| Samples | 16 |
| Seed | 13 |
| dtype | `bfloat16` |
| Device | `auto` |
| Autoregressive max new tokens | 20 |

Conditions:

| Condition | Description |
| --- | --- |
| `base` | Base model, no instruction, no LoRA |
| `instruction_only` | Base model with the natural-language task instruction |
| `lora_only` | Base model with LoRA adapter, no instruction |

Behavioral baseline from the full RQ1 collection on this run:

| Condition | Samples | Mean loss | Token accuracy | Sequence accuracy |
| --- | ---: | ---: | ---: | ---: |
| `base` | 100 | 7.0229 | 0.1425 | 0.0100 |
| `instruction_only` | 100 | 6.1725 | 0.1417 | 0.0100 |
| `lora_only` | 100 | 0.0343 | 0.9900 | 0.9900 |

This matters for interpretation: `instruction_only` is not a strong behavioral baseline on `last_word`; the LoRA adapter is the condition that reliably learned the task.

## Runner Flow

The default RQ3 runner now executes 12 raw-text patching runs:

```text
lora_only -> instruction_only
lora_only -> base
instruction_only -> lora_only
base -> lora_only

layers: 1, 21, 27
patch_span: text
```

Each run is written under:

```text
patches/rq3/{source}_to_{target}_l{layer}_text/
```

For each run:

1. Load selected test samples.
2. Load the source condition model.
3. For each sample, run teacher forcing and capture the selected layer's activation.
4. Release the source model to avoid holding two 3B models in memory.
5. Load the target condition model.
6. For each sample, write:
   - unpatched teacher-forced metrics
   - patched teacher-forced metrics
   - unpatched greedy autoregressive output
   - patched greedy autoregressive output
7. Write:

```text
config.json
dataset_snapshot.jsonl
metrics.jsonl
generations.jsonl
```

`metrics.jsonl` contains teacher-forced loss and accuracy. `generations.jsonl` contains greedy generation outputs plus per-step token metadata.

Current `generations.jsonl` fields include:

```text
pred_token_ids
target_token_ids
loss_target_token_ids
target_strategy
generation_patch_strategy
token_losses
eos_token_id
stopped_on_eos
```

The current autoregressive token-loss perspective is:

```text
loss_t = CE(logits_t, target_text_token_t)
```

Losses are emitted only for real supervised `target_text` token positions. If generation continues beyond the target sequence, those free-running tokens are recorded without invented CE targets. Generation stops early only when EOS is produced.

The default RQ3 equivalence span is now `patch_span=text`: patch raw input-text activations and measure the target model's output and degradation. `patch_span=target` remains available as an output-position diagnostic, but it is not the default equivalence path.

## Experiment A: instruction activation into LoRA target output-position diagnostic

Direction:

```text
source = instruction_only
target = lora_only
patch_span = target
```

This run patches output-token prediction positions. It is useful as a causal diagnostic, but it should not be treated as the default raw-text equivalence experiment.

Layers are user-facing layer numbers. Internally Llama block indices are 0-based, so:

| Reported layer | Internal block index |
| ---: | ---: |
| 3 | 2 |
| 22 | 21 |
| 28 | 27 |

Output directories:

```text
experiments/last_word_llama32_3b_r8_clean/patches/rq3_l3
experiments/last_word_llama32_3b_r8_clean/patches/rq3_l22
experiments/last_word_llama32_3b_r8_clean/patches/rq3_l28
```

Teacher-forced and generation accuracy summary:

| Layer | Patched | Mean TF loss | TF token acc | TF seq acc | Gen token acc | Gen seq acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | false | 0.000868 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 3 | true | 0.001072 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 22 | false | 0.000868 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 22 | true | 3.405461 | 0.260417 | 0.125000 | 0.125000 | 0.125000 |
| 28 | false | 0.000868 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| 28 | true | 5.639999 | 0.197917 | 0.000000 | 0.000000 | 0.000000 |

Observation:

- Layer 3 patching has almost no behavioral effect.
- Layer 22 patching strongly disrupts the LoRA target run.
- Layer 28 patching is even more destructive.

The autoregressive outputs show a useful dynamic: after patching late layers, generation can initially follow an instruction-like path, but after several generated tokens the run tends to collapse back toward the LoRA task attractor, i.e. outputting/repeating the last-word answer pattern. This effect is strongest in layers 22 and 28 and weak in layer 3.

## Experiment B: LoRA text activations into instruction target

Direction:

```text
source = lora_only
target = instruction_only
patch_span = text
```

`patch_span=text` means only input-text token activations are replaced. The instruction prefix is preserved.

Output directories:

```text
experiments/last_word_llama32_3b_r8_clean/patches/rq3_lora_to_instruction_text_l2
experiments/last_word_llama32_3b_r8_clean/patches/rq3_lora_to_instruction_text_l21
experiments/last_word_llama32_3b_r8_clean/patches/rq3_lora_to_instruction_text_l27
```

Summary:

| Layer | Patched | Mean TF loss | TF token acc | TF seq acc | Gen token acc | Gen seq acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | false | 5.640001 | 0.197917 | 0.000000 | 0.000000 | 0.000000 |
| 2 | true | 5.054300 | 0.197917 | 0.000000 | 0.000000 | 0.000000 |
| 21 | false | 5.640001 | 0.197917 | 0.000000 | 0.000000 | 0.000000 |
| 21 | true | 5.016546 | 0.229167 | 0.062500 | 0.062500 | 0.062500 |
| 27 | false | 5.640001 | 0.197917 | 0.000000 | 0.000000 | 0.000000 |
| 27 | true | 5.640001 | 0.197917 | 0.000000 | 0.000000 | 0.000000 |

Observation:

- The instruction-only target is weak before patching.
- Injecting LoRA text activations gives only a small improvement in layer 21.
- Layers 2 and 27 do not materially recover the instruction target run.

This asymmetry is important. `instruction_only -> lora_only` late-layer patching can damage a strong LoRA run, but `lora_only -> instruction_only` text-span patching does not reliably make the weak instruction run behave like the LoRA run.

## Token-Loss Visualizations

Patch-loss visualizations are generated with:

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.visualize `
  --mode patch_loss `
  --run experiments\last_word_llama32_3b_r8_clean\patches\rq3_l28 `
  --output-dir experiments\last_word_llama32_3b_r8_clean\plots\patch_loss\instruction_to_lora_l28
```

Current visualization outputs:

```text
experiments/last_word_llama32_3b_r8_clean/plots/patch_loss/instruction_to_lora_l3/
experiments/last_word_llama32_3b_r8_clean/plots/patch_loss/instruction_to_lora_l22/
experiments/last_word_llama32_3b_r8_clean/plots/patch_loss/instruction_to_lora_l28/
experiments/last_word_llama32_3b_r8_clean/plots/patch_loss/lora_to_instruction_text_l2/
experiments/last_word_llama32_3b_r8_clean/plots/patch_loss/lora_to_instruction_text_l21/
experiments/last_word_llama32_3b_r8_clean/plots/patch_loss/lora_to_instruction_text_l27/
```

Each directory contains:

```text
patch_token_loss.html
*_token_loss.csv
*_token_loss_aggregate.csv
```

The newest loss perspective is implemented in the runner and used for regenerated runs: per-step CE loss against the true `target_text` token sequence, with EOS as the only early stop. The summary tables above were produced before this correction, so their generation-loss plots should be regenerated before using token-loss magnitudes as evidence.

## Interpretation

The existing RQ3 result does not support simple computational equivalence between instruction prompting and LoRA, but it mixes output-position diagnostics with text-span patching. Regenerate the default `patch_span=text` runs before treating this file as the current primary RQ3 evidence.

The main evidence:

- LoRA-only is behaviorally strong; instruction-only is weak.
- Late-layer `instruction_only -> lora_only` output-position patching disrupts LoRA behavior instead of cleanly substituting an instruction computation path.
- `lora_only -> instruction_only` text-span patching only weakly improves the instruction target, mainly at layer 21.
- Autoregressive outputs suggest a transient path effect: patching can push the next few tokens toward the source condition, but the target run's own dynamics can reassert themselves over subsequent tokens.

Conservative conclusion:

> The LoRA adapter and the natural-language instruction were not interchangeable in these historical runs. Treat this as provisional until the default raw-text patching matrix is regenerated.

## Limitations

- Only one task, one model, one LoRA rank, and 16 sampled test rows were used for RQ3.
- `instruction_only` is not behaviorally matched to `lora_only`, making causal interchangeability harder to interpret.
- Current text-span patching aligns by tokenized input text span and preserves the instruction prefix, but it does not perform a richer semantic span alignment.
- Current autoregressive patching applies the captured prompt-position patch during greedy decoding; it is a compact diagnostic rather than a full causal tracing sweep.
