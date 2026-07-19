# TODO: Experiment Pipeline Bug List

This file tracks remaining pipeline gaps only. Completed RQ2/RQ2.1 alignment and attention-output runner work has been removed from this list.

## 0. Task Data Status

Current usable transformation-task data is in `experiments/lora_selected_tasks_instruct_rawchat_r8_20260709`. These tasks have generated train/validation/test datasets, default prompt variants from screening, and completed instruct-model LoRA adapters trained with raw user text to target text through the tokenizer chat template. Each task currently has `train=800`, `validation=100`, and `test=100`.

Task acceptance policy:

- [x] Replace the current mixed acceptance artifacts and runners with one canonical acceptance test. Evaluating multiple prompt variants and selecting the highest-scoring prompt is the intended behavior; keep that behavior in the single canonical path.
- [x] Bind every registered `task_id` to one explicit validation method and use that same method for acceptance, prompt evaluation, LoRA evaluation, and RQ3 semantic scoring. Save the resolved validator name in every config and summary so an old run can be audited without relying on the current registry.
- [x] Remove acceptance-time dataset fallback and exception-driven source switching. Use one configured data route per task; a run must fail clearly when that route is unavailable rather than silently changing its data to improve the chance of passing.
- [x] Keep acceptance behavior deterministic and independent of the observed score: fixed source route, split construction, seed, prompt candidates, validator, thresholds, and model formatting. Passing or failing may select/report the best prompt, but must not change how later candidates are built or evaluated.

- [x] `reverse_words` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `task_no_explanation`.
- [x] `first_word` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.
- [x] `words_starting_with_letter` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.
- [x] `exact_three_word_prefix` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.
- [x] `has_repeated_word` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.
- [x] `at_operator_mod_minus_left` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `natural`.
- [x] `formal_language_a_n_b_n` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.
- [x] `extract_items_from_set` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.
- [x] `words_containing_bigram_qu` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `task_no_explanation`.
- [x] `uppercase_last_word` - generated dataset and r=8 instruct chat-template adapter complete; default prompt variant: `follow_rule_only_answer`.

Format-generation task data status:

- [x] Source prompts collected in `data/format_instruction_sources_run1`: 3,756 records from `thu_ifbench`, `google_ifeval`, and `allenai_ifbench_test`.
- [x] Generate target/used/accepted datasets for `fixed_three_bullets`.
- [x] Generate target/used/accepted datasets for `include_fixed_keywords`.
- [x] Generate target/used/accepted datasets for `exclude_fixed_words`.
- [x] Generate target/used/accepted datasets for `json_answer_schema`.
- [x] After target generation, keep both attempted/used records and format-valid accepted records; top up to 1,000 accepted examples per fixed target state if filtering leaves fewer than 1,000.

## 1. DatasetModule

Boundary: owns synthetic transformation data generation, required fields, split integrity, task metadata, and prompt sample text. Does not run models or compute activation metrics.

- [x] Add dataset validation command: required fields, split sizes, duplicate `input_text`, empty `target_text`, and one `task_id` per dataset.
- [x] Record canonical target tokenization dependencies in manifest: `model_name`, tokenizer, and prompt template.
- [x] Keep one fixed tiny smoke dataset for target-token alignment, attention-source alignment, and patching alignment tests.
- [x] Remove builder repetition/top-up of the small hard-coded template lists. Route ordinary template-style tasks through the existing Wiki-corpus input construction, with explicit seeded custom routes for `formal_language_a_n_b_n` (three random length/control values) and `words_containing_bigram_qu` (Wiki text plus random `qu` insertion).
- [x] Remove `allow_builtin_fallback` from production dataset/acceptance pipeline invocations. Test rows do not need to be identical between acceptance and experiments, but both must come from their declared single route and record that route in the manifest.

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
- [ ] Add RQ1 J-lens readouts for residual-stream states: compare `base`, `instruction_only`, and `lora_only` top-token/concept trajectories by layer and target/input position. Treat J-lens as an RQ1 interpretive readout, not an RQ2 attention metric.
- [ ] Add RQ1 SAE feature analysis for residual-stream perturbations: encode `instruction_only - base` and `lora_only - base`, then report sparse-feature overlap, feature-activation cosine, and task-success correlations.
- [ ] Add delta-based functional subspace visualizations for LoRA-vs-instruction convergence:
  - [x] Plot layerwise principal-angle / subspace-cosine similarity between top-k PCA bases of `instruction_only - base` and `lora_only - base`.
  - [x] Plot final-layer PCA of residual deltas, colored by condition and shaped/annotated by task in combined views.
  - [ ] Plot module-boundary delta trajectories across `pre_o_proj`, `post_o_proj`, `post_attn_resid`, `ffn_out`, and `post_ffn_resid` to locate where LoRA/instruction paths diverge or partially converge.
  - [x] Add task-level scatter: final attention-output similarity vs final block/residual similarity, colored by LoRA/instruction pass rate, to show whether high-accuracy tasks occupy a shared functional-alignment region.
  - [x] Keep raw hidden-state plots separate from delta plots; functional-subspace claims are based on `condition - base`.

## 5. AttentionAnalysisModule / RQ2

Boundary: reads collect artifacts and computes RQ2 attention pattern / head output metrics. Does not own head ablation or activation patching.

- [x] Clarify KL / entropy semantics: current values use raw probability mass on shared support; either document this everywhere or renormalize before distribution metrics.
- [x] Add source-span statistics to attention rows: shared input tokens, shared target-prefix tokens, excluded instruction tokens, and excluded other tokens.
- [x] Add post-`o_proj` attention-output comparison so RQ2 pre-`o_proj` head outputs can be checked against the block-level representation used by RQ1.
- [x] Add delta-based attention-output comparison (`condition - base`) so raw activation similarity is not mixed with perturbation similarity.
- [x] Implement head ablation impact as a separate output from attention similarity.
- [ ] Add RQ2 SAE feature analysis for attention-side vectors: train or load SAEs for pre-`o_proj` head outputs and/or post-`o_proj` attention outputs, then compare `instruction_only` and `lora_only` sparse features and feature deltas. Keep this separate from J-lens, which belongs to RQ1 residual readout.

## 6. ActivationPatchingModule

Boundary: owns RQ3 source activation capture, target injection, and patching metrics. Does not generate generic similarity charts.

- [ ] Add RQ3.1 configurable linear-delta causal injection to test whether full-patching compatibility can be explained by a stable, linearly extractable task component:
  - [ ] Define a replaceable linear delta-extractor interface that is fitted only on a held-out same-task extraction subset and returns the delta to inject for an evaluation activation.
  - [ ] Use the mean direction, `H_delta = E[h_task - h_base]`, as the default baseline extractor; allow alternative linear extractors such as PCA/SVD subspaces and fitted linear projections without changing the patching or evaluation pipeline.
  - [ ] Inject the extracted delta into different, non-overlapping target instances from the same task as `h_target_patched = h_base_target + H_delta`, generalized to extractor-specific projected deltas where applicable.
  - [ ] Keep delta-extraction samples and injection/evaluation samples disjoint; no input instance may appear in both sets.
  - [ ] Run separate `H_delta` estimates for `instruction_only - base` and `lora_only - base`, then cross-inject into `base`, `instruction_only`, and `lora_only` targets where shape/alignment permits.
  - [ ] Report unpatched, same-condition delta injection, cross-condition delta injection, and random/permuted delta controls.
  - [ ] Score both teacher-forced metrics and autoregressive task-semantic correctness, because token accuracy alone can miss task-level failures.
  - [ ] Treat RQ3.1 as causal evidence for a shared linearly extractable functional component only if a delta fitted without evaluation instances transfers to unseen same-task instances and changes behavior in the predicted direction; report conclusions separately by extractor.
- [ ] Distinguish activation sites: block input, block output, attention output, and MLP output. Current `layer` alone is not enough.
- [ ] Add activation site sweep over block input, block output, attention output, and MLP output.
- [x] Replace the `previous_last_word` generation-loss simplification with a target from true `target_text` or the task evaluation function.
- [ ] Keep the same-condition patch call path available as an opt-in positive control, but disable it in the default RQ3 matrix to save compute. The config must state whether it was enabled; do not claim the control was run when it was only callable.
- [x] Keep the default per-run controls limited to unpatched, base-to-target patch, and source-to-target patch.
- [x] Add confusion matrix / per-sample outcomes; loss and greedy text alone are not enough.
- [x] Add task-level evaluator for autoregressive outputs so format differences are scored by task semantics, not only token alignment.
- [x] Fail fast on patch shape mismatch; do not silently truncate patch tensors.
- [x] Mark current RQ3 output as partial/debug-only until generation metrics, controls, activation sites, and shape checks are complete.
- [ ] Remove autoregressive `sequence_accuracy` and token-prefix accuracy from RQ3 outputs, summaries, plots, and outcome tables. Use the task's single resolved semantic validator and report one task pass rate; generation length or extra text must be handled only by that validator.
- [ ] Diagnose and document RQ3 autoregressive patch sensitivity with logit deltas/top-1 margins, not only greedy `pred_text`. Existing layer 1/21 artifacts show changed losses and occasional changed tokens, so unchanged text there is usually stable argmax/model calibration rather than a dead hook.
- [ ] Treat final-layer text-span block-output patching as structurally ineffective unless the injection site changes: replacing past input-position outputs after the last block has no downstream cross-position operation that can affect the current next-token logit. Add a fast causal-path check and either move this case to block input/attention output or skip it with an explicit reason.

## 7. ExperimentRunnerModule

Boundary: orchestrates modules, fixes run config, and passes paths/parameters. Does not patch data or metric definitions inside runners.

- [ ] Add RQ1 required alignment metadata validation after collect.
- [ ] Remove `base -> lora` from the default RQ3 batch matrix because its source-to-target patch duplicates the base-to-target control. Keep default runs for `lora -> instruction`, `lora -> base`, and `instruction -> lora`; layers must also pass the causal-path check above.
- [ ] Add RQ3.1 runner config for the extractor name and hyperparameters plus per-task split assignment: `delta_extract_sample_ids` and `injection_eval_sample_ids` must be same-task, disjoint, recorded in config, and saved with the dataset snapshot.
- [ ] Extend RQ3 batch runs to activation site/span sweeps and write source/target/control combinations to config.
- [ ] Add adapter quality gate: stop mechanistic interpretation when LoRA-only fails the task threshold.

## 8. ResultStoreAndReportModule

Boundary: saves configs, metrics, plots, and report summaries. Does not change metric definitions.

- [ ] Normalize output directory structure: `config`, `dataset_snapshot`, `adapters`, `metrics`, `activations`, `plots`, `reports`.
- [ ] Ensure every CSV / HTML output includes metric definition and alignment strategy where applicable.
- [ ] Reports should label results as `implemented`, `partial`, or `invalidated`.
