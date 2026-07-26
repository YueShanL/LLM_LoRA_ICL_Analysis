# J-lens Implementation Plan

## Scope

Add Jacobian Lens (J-lens) as an optional RQ1 interpretive readout for residual-stream activations.

The first implementation should only apply J-lens to existing `base`, `instruction_only`, and `lora_only` RQ1 hidden states. Do not add J-space activation patching, task-specific lens fitting, or per-adapter lens fitting in the first version.

## Research Role

Current RQ1 compares residual perturbation directions:

```text
cosine(hidden_instruction_only - hidden_base, hidden_lora_only - hidden_base)
```

J-lens should add a vocabulary/concept readout:

```text
hidden_states[layer, token_position, hidden_dim]
  -> J-lens transport
  -> final residual basis
  -> unembedding readout
  -> top-k tokens/concepts
```

Use it to ask whether `instruction_only` and `lora_only` form similar readable task-related trajectories by layer and token position.

## Data Policy

Fit the lens on random generic text, not project task examples.

Recommended split:

```text
fit corpus:
  100-1000 random generic sequences
  128 tokens each
  no task labels
  no instruction/answer pairs

validation corpus:
  50-100 separate generic sequences
  used only to check readout stability

application set:
  existing RQ1 task activations
  base / instruction_only / lora_only
```

The main report should state:

```text
The Jacobian lens was fitted on random generic text sequences and applied zero-shot to task activations. No task labels or task examples were used during lens fitting.
```

Do not fit on RQ1 test examples and then report readouts on the same examples as main evidence.

## Model Policy

Fit one lens for the fixed base checkpoint used by the run, currently:

```text
meta-llama/Llama-3.2-3B-Instruct
```

Use the same base-model lens to read all three conditions. This keeps `base`, `instruction_only`, and `lora_only` in one shared explanation coordinate system.

Do not fit one lens per LoRA adapter unless a later diagnostic specifically asks whether LoRA-specific lenses differ from the base lens.

## New Files

Add two optional modules:

```text
src/lora_instruction_analysis/model/jlens_fit.py
src/lora_instruction_analysis/model/jlens_readout.py
```

Optional console entry points:

```text
lia-fit-jlens
lia-jlens-readout
```

Keep J-lens outside the default dependency path. The normal RQ1 pipeline must still run without J-lens installed.

## Dependency

Prefer an optional dependency or an explicit runtime error with install guidance:

```toml
[project.optional-dependencies]
jlens = [
  "jlens @ git+https://github.com/anthropics/jacobian-lens",
]
```

Because `jlens` is research code, pinning should be revisited after the first working install.

## Fit Module

`jlens_fit.py` should:

- Load the base model and tokenizer.
- Read or sample generic text sequences.
- Tokenize to fixed-length windows.
- Fit the Jacobian lens.
- Save the lens and fit metadata.
- Run a small validation readout check.

Minimum CLI:

```text
--model-name
--output-dir
--dataset-name
--dataset-config
--dataset-split train
--validation-split validation
--text-column text
--num-sequences 100
--validation-sequences 50
--sequence-length 128
--dtype bfloat16
--device cuda
--seed 13
```

Output:

```text
experiments/{run_id}/jlens/base_model/
  lens/
  config.json
  validation_summary.json
```

`config.json` must record:

- model name
- checkpoint or revision if available
- tokenizer name
- Hugging Face dataset name/config
- fit and validation split names
- text column
- fit sample row indices
- validation sample row indices
- sequence length
- number of fit sequences
- number of validation sequences
- random seed
- dtype and device
- J-lens package/version or commit if available

## Readout Module

`jlens_readout.py` should read existing RQ1 collect outputs. It should not rerun the model or retokenize task examples.

Input:

```text
experiments/{run_id}/{task}/states/rq1/
  metrics.jsonl
  tensors/{sample_id}__{condition}.pt
```

Use:

```text
hidden_states[layer, token_index, hidden_dim]
target_alignment
target_logits
condition metadata
```

Output:

```text
experiments/{run_id}/{task}/plots/rq1_jlens/
  jlens_readouts.csv
  jlens_pair_overlap.csv
  jlens_layer_summary.csv
  jlens_readouts.html
```

## Metrics

Per readout row:

- `sample_id`
- `task_id`
- `condition`
- `layer`
- `token_index`
- `alignment_key`
- `target_token_id`
- `target_token_text`
- `top_k_token_ids`
- `top_k_token_texts`
- `top_k_scores`

Pairwise rows:

```text
base vs instruction_only
base vs lora_only
instruction_only vs lora_only
```

Metrics:

- `top_k_overlap`
- `top_k_jaccard`
- `rank_biased_overlap` if cheap to add
- `readout_distribution_cosine` if scores are comparable
- `target_token_rank`
- task keyword rank when a task-level keyword list exists

Layer summary:

- mean overlap by layer
- mean target-token rank by layer
- best layer by `instruction_only` vs `lora_only` overlap
- late-layer summary for layers already emphasized by RQ1

## RQ1 Runner Integration

Add optional RQ1 flags:

```text
--run-jlens-readout
--jlens-path
--jlens-top-k 20
```

Behavior:

- If `--run-jlens-readout` is absent, RQ1 behavior is unchanged.
- If `--run-jlens-readout` is present and `--jlens-path` is missing, fail clearly.
- If the J-lens dependency is missing, fail with install guidance.
- Write J-lens output under `plots/rq1_jlens/`, not mixed into existing residual-cosine CSVs.

## Tests

Keep tests small:

- Missing lens path gives a clear error.
- Missing J-lens dependency gives a clear error.
- Fake readout rows write the expected CSV schema.
- Pairwise overlap works for empty, partial, and full top-k overlap.
- Existing RQ1 run path still works without J-lens installed.

Do not add model-loading tests to the normal unit suite.

## First Milestone

Implement only:

1. Fit one base-model J-lens on 100 generic 128-token sequences.
2. Apply it to one task's existing RQ1 tensors.
3. Write `jlens_readouts.csv` and `jlens_pair_overlap.csv`.
4. Check whether `instruction_only` and `lora_only` have interpretable top-k overlap by layer.

Scale to all tasks only after this pilot produces stable and useful readouts.
