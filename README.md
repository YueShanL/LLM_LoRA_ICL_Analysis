# LoRA Instruction Analysis

This repository contains the first system module from `plan_summary.md`: a reproducible dataset generator for synthetic text transformation tasks.

## Generate a Dataset

Use the project virtual environment once it is repaired or recreated:

```powershell
venv\Scripts\python.exe -m pip install -e .
venv\Scripts\python.exe -m lora_instruction_analysis.data.cli --task last_word --source wikitext --output-dir data/generated/last_word
```

For instruction-model acceptance checks, try `last_word`, `word_count`, or `uppercase_last_word`; `base + instruction`
should solve it much more reliably than the `add_zxq_after_t_or_l` toy task.

The generated JSONL rows keep the plan fields:

```text
sample_id, task_id, input_text, instruction_text, target_text, condition
```

They also include Hugging Face / PEFT-friendly fields:

```text
instruction, input, output, prompt, response, text, messages
```

`text` is ready for `trl.SFTTrainer(dataset_text_field="text")`. `messages` is an OpenAI/HF chat-style list that can be passed through tokenizer chat templates.

## Example

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.data.cli `
  --task last_word `
  --source wikitext `
  --max-source-rows 500 `
  --train-size 300 `
  --validation-size 50 `
  --test-size 50 `
  --output-dir data/generated/last_word
```

The command writes:

```text
data/generated/last_word/
  manifest.json
  train.jsonl
  validation.jsonl
  test.jsonl
  train.csv
  validation.csv
  test.csv
  hf_dataset/
```

`hf_dataset/` is written when the `datasets` package is installed.

## Train a LoRA Adapter

LoRA training uses `input_text` as the prompt input and `target_text` as the
supervised output. It does not add `instruction_text` to the raw training
prompt. The supervised target is ended with the tokenizer EOS token by default;
pass `--no-append-eos` only to reproduce old no-EOS runs.

```powershell
venv\Scripts\python.exe -m pip install -e .[train]
venv\Scripts\python.exe -m lora_instruction_analysis.model.train_lora `
  --model-name gpt2 `
  --dataset-path data/generated/add_zxq `
  --output-dir experiments/add_zxq/adapters/r8 `
  --rank 8 `
  --epochs 3
```

Use `--prompt-format chat_template` to format examples with the tokenizer's
chat template instead of raw `Input:/Output:` text. Keep the default `raw` for
base-model LoRA-only runs unless you are intentionally running a chat-format
control.

The adapter directory contains PEFT adapter weights/config, tokenizer files,
`trainer_state.json`, `train_config.json`, and a copied `dataset_manifest.json`
when the dataset has one.

## Evaluate an Instruction Prompt

Prompt evaluation loads a generated dataset, applies an instruction prompt, and
writes teacher-forced plus greedy autoregressive token/sequence accuracy metrics.

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.prompt_eval `
  --model-name gpt2 `
  --dataset-path data/generated/add_zxq `
  --split test `
  --instruction "Add ZXQ after each word that ends with the letter t or l. Keep all other words unchanged." `
  --output-dir experiments/add_zxq/prompt_eval
```

Omit `--instruction` to use each row's `instruction_text`. The command writes
`metrics.jsonl`, `autoregressive_metrics.jsonl`, `summary.json`, and `report.md`.
Pass `--skip-autoregressive` for teacher-forced-only checks.

## Validate a Task

Check whether the task is useful for mechanism comparison: instruction prompt
accuracy must be high, while no-instruction accuracy must stay low.

Quickly build test cases from a registered transformation task and validate one
instruction prompt against a chosen model:

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.task_acceptance `
  --task last_word `
  --model-name gpt2 `
  --instruction "Return only the last word of the input, without trailing non-letter symbols." `
  --output-dir experiments/last_word/task_acceptance `
  --max-samples 5
```

`--task` selects the registered transformation function used to generate
`target_text`. Use `--dataset-path` instead when the dataset already exists.

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.task_acceptance `
  --model-name gpt2 `
  --dataset-path data/generated/add_zxq `
  --output-dir experiments/add_zxq/task_acceptance `
  --max-samples 16
```

## Run Dataset Generation + LoRA Training

Use the end-to-end runner when starting a full run from `plan_summary.md`.
It generates the dataset, writes a canonical run config, trains the LoRA
adapter, and stores the adapter under the same run directory.

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.experiment.run_lora `
  --model-name meta-llama/Llama-3.2-3B `
  --task add_zxq_after_t_or_l `
  --rank 8 `
  --epochs 3 `
  --output-root experiments
```

The command writes:

```text
experiments/{run_id}/
  config.json
  dataset/
  adapters/r{rank}/
```

## Collect Model States

After installing the training extras in the project venv, run teacher-forced
collection for the three comparison conditions:

```powershell
venv\Scripts\python.exe -m pip install -e .[train]
venv\Scripts\python.exe -m lora_instruction_analysis.model.collect `
  --model-name gpt2 `
  --dataset-path data/generated/add_zxq `
  --split test `
  --max-samples 8 `
  --adapter-path experiments/add_zxq/adapters/r8 `
  --output-dir experiments/runs/add_zxq_gpt2_r8_states
```

The run writes:

```text
config.json
dataset_snapshot.jsonl
metrics.jsonl
tensors/{sample_id}__{condition}.pt
```

Each tensor file keeps target-position `hidden_states`, `attentions`,
`target_logits`, labels, and token alignment metadata for RQ1/RQ2/RQ3 analysis.
Pass `--collect-attention-outputs` to capture pre-`o_proj` per-head attention
outputs at target positions for RQ2.1.

## Run RQ1 Residual Analysis

After `lia-run-lora` has produced a run directory, use the RQ1 runner to collect
base, instruction-only, and LoRA-only teacher-forced states and compute residual
perturbation cosine similarity.

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.experiment.run_rq1 `
  --run-dir experiments/add_zxq_llama32_3b_r8_20260628_retry2 `
  --dtype bfloat16 `
  --max-samples 16
```

The command writes:

```text
experiments/{run_id}/
  rq1_config.json
  states/rq1/
  plots/rq1/
```

## Run RQ3 Activation Patching

RQ3 keeps model-side patching logic in `lora_instruction_analysis.model.patch`.
The runner only resolves run config and calls it.

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.experiment.run_rq3 `
  --run-dir experiments/add_zxq_llama32_3b_r8_20260628_retry2 `
  --dtype bfloat16 `
  --max-samples 16
```

`--patch-span text` is the default RQ3 equivalence test: it patches only the raw
input text span and preserves the instruction prefix. `--patch-span target`
patches output-token prediction positions and is an optional diagnostic, not the
default equivalence path.

By default the runner executes 12 raw-text patching runs:

```text
lora_only -> instruction_only
lora_only -> base
instruction_only -> lora_only
base -> lora_only

layers: 1, 21, 27
```

Pass `--source-condition`, `--target-condition`, and/or `--layer` to narrow the
batch.

The command writes unpatched, same-condition, base-to-target, and requested
source-to-target teacher-forced metrics plus greedy autoregressive generations
with per-step logits loss metadata:

```text
experiments/{run_id}/
  rq3_config.json
  rq3_status.json
  patches/rq3/
    {source}_to_{target}_l{layer}_text/
      config.json
      dataset_snapshot.jsonl
      metrics.jsonl
      generations.jsonl
      outcomes.jsonl
      confusion_matrix.json
```

`generations.jsonl` includes generated token ids, per-step CE losses against the
true `target_text` token sequence, patch/loss target strategy metadata, and EOS
stop metadata. It also includes task-level semantic scoring from the registered
transformation task, so generation output is scored after whitespace
normalization instead of only by token alignment. Losses are emitted only for
supervised target positions; later free-running tokens are generated but are not
assigned invented CE targets.

`rq3_status.json` is currently marked `partial` because activation-site sweeps
are still pending; the implemented patch site is block output.

Patch generation loss plots are produced with:

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.visualize `
  --mode patch_loss `
  --run experiments/{run_id}/patches/rq3 `
  --output-dir experiments/{run_id}/plots/patch_loss/rq3
```

The plot compares patched vs unpatched per-token generation loss. Current RQ3
loss uses `target_text` tokens as the next-step CE targets and stops early only
on EOS. Autoregressive patching keeps the hook active across generation steps so
the selected raw text activations keep affecting future behavior.

## Visualize Token-Level State Similarity

Analyze one `lia-collect-states` output directory and write token-level
similarity rows, aggregates, adapter quality summary, and a self-contained HTML
heatmap.

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.visualize `
  --run experiments/runs/llama32_3b_states `
  --output-dir experiments/runs/llama32_3b_similarity
```

Default `--mode residual` computes:

```text
cosine(hidden_instruction_only - hidden_base, hidden_lora_only - hidden_base)
```

Other single-run modes support attention analysis:

```text
--mode attention         compares attention probability patterns
--mode attention_output  compares pre-o_proj per-head attention outputs collected with --collect-attention-outputs
```

The command writes:

```text
token_similarity.csv
aggregate_similarity.csv
quality_summary.csv
token_similarity.html
```

For functional testing of two original `meta-llama/Llama-3.2-3B` collect runs,
the older two-run comparison is still available:

```powershell
venv\Scripts\python.exe -m lora_instruction_analysis.model.visualize `
  --left-run experiments/runs/llama32_3b_a `
  --right-run experiments/runs/llama32_3b_b `
  --mode state `
  --output-dir experiments/runs/llama32_3b_pair_similarity
```
