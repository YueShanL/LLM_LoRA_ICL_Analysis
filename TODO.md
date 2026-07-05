# TODO: Experiment Pipeline Bug List

This file tracks remaining pipeline gaps only. Completed RQ2/RQ2.1 alignment and attention-output runner work has been removed from this list.

## 1. DatasetModule

Boundary: owns synthetic transformation data generation, required fields, split integrity, task metadata, and prompt sample text. Does not run models or compute activation metrics.

- [ ] Add dataset validation command: required fields, split sizes, duplicate `input_text`, empty `target_text`, and one `task_id` per dataset.
- [ ] Record canonical target tokenization dependencies in manifest: `model_name`, tokenizer, and prompt template.
- [ ] Keep one fixed tiny smoke dataset for target-token alignment, attention-source alignment, and patching alignment tests.

## 2. LoRATrainModule

Boundary: trains a LoRA adapter for a task and saves adapter/config/quality metrics. Does not run RQ1/RQ2/RQ3 analysis.

- [ ] Fix default LoRA target modules if needed: planned attention projections are `q_proj`, `k_proj`, `v_proj`, `o_proj`; verify current `auto` behavior.
- [ ] Save adapter quality-gate metrics: validation loss, token accuracy, and sequence accuracy.
- [ ] Make training prompt strategy explicit: LoRA-only training must not include instruction unless the config says so, and such runs must not be mislabeled `lora_only`.

## 3. StateCollectModule

Boundary: runs teacher-forced forward passes and saves aligned raw tensors/metadata. Does not compute cross-condition similarity.

- [ ] Save complete condition position mapping: prompt span, input span, target span, prediction positions, and raw token ids.
- [ ] Add `lora_instruction` condition as an upper bound; it must be distinct from `lora_only`.
- [ ] Save prompt template metadata: instruction inclusion, template version/string, and prompt token count.

## 4. SimilarityAnalysisModule

Boundary: reads collect artifacts and computes RQ1 residual similarity. Does not retokenize, change sample selection, or run models.

- [x] Add CKA and logit-distribution similarity once target alignment artifacts are present.
- [x] Add box-plot visualization and CSV summaries for CKA and logit-distribution similarity.

## 5. AttentionAnalysisModule / RQ2

Boundary: reads collect artifacts and computes RQ2 attention pattern / head output metrics. Does not own head ablation or activation patching.

- [ ] Clarify KL / entropy semantics: current values use raw probability mass on shared support; either document this everywhere or renormalize before distribution metrics.
- [ ] Add source-span statistics to attention rows: shared input tokens, shared target-prefix tokens, excluded instruction tokens, and excluded other tokens.
- [ ] Implement head ablation impact as a separate output from attention similarity.

## 6. ActivationPatchingModule

Boundary: owns RQ3 source activation capture, target injection, and patching metrics. Does not generate generic similarity charts.

- [ ] Distinguish activation sites: block input, block output, attention output, and MLP output. Current `layer` alone is not enough.
- [ ] Add activation site sweep over block input, block output, attention output, and MLP output.
- [x] Replace the `previous_last_word` generation-loss simplification with a target from true `target_text` or the task evaluation function.
- [x] Add RQ3 controls: unpatched, same-condition patch, base-to-target patch, and source-to-target patch.
- [x] Add confusion matrix / per-sample outcomes; loss and greedy text alone are not enough.
- [x] Add task-level evaluator for autoregressive outputs so format differences are scored by task semantics, not only token alignment.
- [x] Fail fast on patch shape mismatch; do not silently truncate patch tensors.
- [x] Mark current RQ3 output as partial/debug-only until generation metrics, controls, activation sites, and shape checks are complete.

## 7. ExperimentRunnerModule

Boundary: orchestrates modules, fixes run config, and passes paths/parameters. Does not patch data or metric definitions inside runners.

- [ ] Add RQ1 required alignment metadata validation after collect.
- [x] Add default RQ3 batch runs for lora->instruction, lora->base, instruction->lora, and base->lora over layers 1, 21, and 27.
- [ ] Extend RQ3 batch runs to activation site/span sweeps and write source/target/control combinations to config.
- [ ] Add adapter quality gate: stop mechanistic interpretation when LoRA-only fails the task threshold.

## 8. ResultStoreAndReportModule

Boundary: saves configs, metrics, plots, and report summaries. Does not change metric definitions.

- [ ] Normalize output directory structure: `config`, `dataset_snapshot`, `adapters`, `metrics`, `activations`, `plots`, `reports`.
- [ ] Ensure every CSV / HTML output includes metric definition and alignment strategy where applicable.
- [ ] Reports should label results as `implemented`, `partial`, or `invalidated`.
