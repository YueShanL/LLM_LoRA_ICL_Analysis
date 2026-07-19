# Technical Report Draft: LoRA Instruction-Conditioned Computation Analysis

## 1. Research Goal

- This project studies whether LoRA adapters reconstruct internal Transformer computations induced by natural-language instructions.
- The main comparison is between three aligned model conditions:
  - `base`: original base model, no task instruction, no LoRA adapter.
  - `instruction_only`: base model with natural-language task instruction, no LoRA adapter.
  - `lora_only`: base model with LoRA adapter, no natural-language task instruction.
- Optional or planned comparison conditions:
  - `lora_instruction`: LoRA adapter plus natural-language instruction, used as an upper-bound or combined-control condition.
  - Base activation patching: negative control for activation patching.
  - Same-condition patching: positive/control path to verify that patching itself is not destructive.
- Core hypothesis:
  - If LoRA and natural-language instructions induce similar hidden-state perturbations, attention patterns, and causal computation paths, LoRA may be interpreted as a parameterized form of instruction control.
  - If output behavior is similar but internal computation differs, then the same task may be solved through multiple alternative computational paths.

## 2. Current Experiment Setup

- Main experiment directory:
  - `experiments/lora_selected_tasks_instruct_rawchat_r8_20260709`
- Model and run configuration:
  - Base model: `meta-llama/Llama-3.2-3B-Instruct`
  - LoRA rank: `r=8`
  - Data split: `test`
  - Samples per task for current RQ analysis: `max_samples=16`
  - Seed: `13`
  - dtype: `bfloat16`
  - device: `cuda`
  - prompt format: `chat_template`
  - EOS appended
- Current task set:
  - `reverse_words`
  - `first_word`
  - `words_starting_with_letter`
  - `exact_three_word_prefix`
  - `has_repeated_word`
  - `at_operator_mod_minus_left`
  - `formal_language_a_n_b_n`
  - `extract_items_from_set`
  - `words_containing_bigram_qu`
  - `uppercase_last_word`
- Current task data status:
  - Each task has generated train, validation, and test datasets.
  - Current split sizes are `train=800`, `validation=100`, and `test=100`.
  - Each task has a selected default prompt variant from screening.
  - Each task has a completed r=8 instruct-model LoRA adapter trained with raw user text to target text through the tokenizer chat template.

## 3. System and Pipeline Summary

- Dataset module:
  - Generates synthetic transformation tasks.
  - Keeps one deterministic transformation per task.
  - Stores instruction text, input text, target text, task metadata, and split information.
  - Current acceptance flow is canonicalized:
    - Fixed source route.
    - Explicit validator per `task_id`.
    - Deterministic prompt screening.
    - No silent fallback to alternative data routes.

- LoRA training module:
  - Trains one adapter per task.
  - Saves adapter/config artifacts.
  - Current r=8 adapters are complete for the 10 selected tasks.
  - Remaining gaps:
    - Verify whether current `auto` target-module behavior matches planned attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
    - Save adapter quality-gate metrics, including validation loss, token accuracy, and sequence accuracy.
    - Make training prompt strategy explicit so `lora_only` runs are not mislabeled if instruction text is present during training.

- State collection module:
  - Runs teacher-forced forward passes under aligned conditions.
  - Collects residual stream activations, attention probabilities, attention outputs, logits, and alignment metadata.
  - Remaining gaps:
    - Save complete condition position mappings:
      - prompt span
      - input span
      - target span
      - prediction positions
      - raw token ids
    - Add a distinct `lora_instruction` upper-bound condition.
    - Save prompt template metadata and prompt token counts.

- RQ1 similarity analysis:
  - Compares residual-stream perturbations:
    - `delta_instruction = hidden_instruction_only - hidden_base`
    - `delta_lora = hidden_lora_only - hidden_base`
  - Main metric:
    - `cosine(delta_instruction, delta_lora)`
  - Additional implemented metrics include CKA, logit-distribution similarity, subspace similarity, and related plots.

- RQ2 / RQ2.1 attention analysis:
  - RQ2 compares attention probability patterns between `instruction_only` and `lora_only`.
  - RQ2.1 compares attention-side vectors:
    - pre-`o_proj` attention outputs
    - attention output deltas
    - post-`o_proj` outputs
    - post-`o_proj` output deltas
    - head-ablation diagnostics

- RQ3 activation patching:
  - Current patching configuration:
    - `patch_span = text`
    - `activation_site = block_output`
    - layers: `1`, `21`, `27`
    - `max_new_tokens=20`
  - Current default directions:
    - `lora_only -> instruction_only`
    - `lora_only -> base`
    - `instruction_only -> lora_only`
    - `base -> lora_only`
  - Each task has the expected 12 default text-span patch runs:
    - 4 source/target directions x 3 layers
  - Current RQ3 status remains `partial` because activation-site sweep is still pending.

## 4. Current Results

### RQ1: Residual Perturbation Similarity

- LoRA-only behavior is strong under teacher-forced sequence accuracy:
  - 9 of 10 tasks reach `1.0000`.
  - `words_starting_with_letter` reaches `0.9375`.
- Instruction-only behavior is much less consistent:
  - Task sequence accuracy ranges from `0.0000` to `1.0000`.
- Across tasks, the unweighted mean task-level cosine similarity is `0.3479`.
- Task-level mean cosine range:
  - Lowest: `at_operator_mod_minus_left`, `0.1731`
  - Highest: `reverse_words`, `0.6488`
- Best layers are mostly late layers, especially layers `24-28`.
- Interpretation:
  - LoRA-only and instruction-only often move residual activations in related directions.
  - Similarity is moderate rather than near-identical.
  - Current RQ1 evidence does not show that LoRA reconstructs the same residual-stream trajectory as natural-language instruction prompting.

### RQ2: Attention Probability Similarity

- Attention probability routing is highly similar across all 10 tasks.
- Across tasks, the unweighted mean task-level attention probability cosine similarity is `0.8898`.
- Task means range from `0.8356` to `0.9444`.
- High attention probability similarity also appears when instruction-only prompting is behaviorally weak.
- Interpretation:
  - LoRA and instruction-only conditions often attend to similar token regions.
  - Attention routing similarity alone is not evidence of task success or computational equivalence.

### RQ2.1: Attention-Side Diagnostics

- Cross-task diagnostic means:
  - `attention_probs`: `0.8898`
  - `attention_outputs`: `0.6316`
  - `attention_output_deltas`: `0.2914`
  - `attention_post_o_proj_outputs`: `0.5842`
  - `attention_post_o_proj_output_deltas`: `0.2994`
  - `attention_head_ablation`: `0.9723`
- Interpretation:
  - Attention probabilities are highly similar.
  - Attention outputs and post-`o_proj` outputs are substantially less similar.
  - Output deltas are lower still.
  - This supports the current hypothesis that LoRA does not primarily implement these tasks by changing where attention looks.
  - Larger differences appear in what attention reads out and writes onward.

### RQ3: Cross-Condition Activation Patching

- Current RQ3 evidence is negative for simple computational interchangeability.
- Unpatched `lora_only` target performance is strong:
  - Average sequence accuracy: `0.9938`
  - Average semantic correctness: `0.9875`
- Unpatched `instruction_only` target performance is weaker:
  - Average sequence accuracy: `0.5563`
  - Average semantic correctness: `0.5500`
- `lora_only -> instruction_only`:
  - Does not improve the instruction target on average.
  - Layer 21 is worse than unpatched.
- `lora_only -> base`:
  - Does not transfer task behavior into the base target.
  - Autoregressive semantic correctness remains `0.0000`.
- `instruction_only -> lora_only`:
  - Causes only small teacher-forced degradation at layers 1 and 21.
  - Shows no visible effect at layer 27 in the current matrix.
- `base -> lora_only`:
  - Mostly acts as a disruption/control comparison.
  - The default matrix should likely be simplified because this direction duplicates the base-to-target control.
- Interpretation:
  - LoRA text-span activations are not sufficient to make weaker targets behave like the LoRA condition.
  - Instruction activations are not a clean substitute for the LoRA path.
  - The current result does not support a simple claim that LoRA and natural-language instructions use interchangeable computation paths.

## 5. Collected Data and Artifacts

- RQ1 outputs include:
  - `states/rq1/`
  - `plots/rq1/aggregate_similarity.csv`
  - `plots/rq1/quality_summary.csv`
  - `plots/rq1/token_similarity.csv`
  - `plots/rq1/token_similarity.html`
- RQ2 / RQ2.1 outputs include:
  - attention probability plots
  - attention output plots
  - attention output delta plots
  - post-`o_proj` output plots
  - post-`o_proj` delta plots
  - head-ablation diagnostics
- RQ3 outputs include:
  - `rq3_config.json`
  - `rq3_status.json`
  - per-direction patch directories under `patches/rq3/`
  - patch-loss plots under `plots/patch_loss/rq3/`
- Each RQ3 patch directory contains:
  - `config.json`
  - `confusion_matrix.json`
  - `dataset_snapshot.jsonl`
  - `generations.jsonl`
  - `metrics.jsonl`
  - `outcomes.jsonl`
- Additional format-generation task data:
  - Source prompts collected in `data/format_instruction_sources_run1`
  - 3,756 records from:
    - `thu_ifbench`
    - `google_ifeval`
    - `allenai_ifbench_test`
  - Accepted datasets exist for:
    - `fixed_three_bullets`
    - `include_fixed_keywords`
    - `exclude_fixed_words`
    - `json_answer_schema`

## 6. Current Limitations

- Each RQ analysis currently uses only 16 test samples per task.
- Instruction-only behavior is weak for several tasks, which limits direct equivalence claims.
- RQ1 and RQ2 are similarity measurements, not causal tests.
- RQ3 is still partial:
  - Only text-span block-output patching has been run.
  - Only layers `1`, `21`, and `27` are included.
  - Activation-site sweep is pending.
- Teacher-forced metrics and autoregressive semantic correctness can diverge, so both must be reported.
- Final-layer text-span block-output patching may be structurally ineffective:
  - Replacing past input-position outputs after the last block may have no downstream cross-position operation that can affect the current next-token logit.
- Current RQ3 output still contains some metrics that should be removed or de-emphasized:
  - autoregressive `sequence_accuracy`
  - token-prefix accuracy
  - non-semantic generation metrics

## 7. Recommended Next Steps

- Complete RQ3 activation-site sweep:
  - block input
  - block output
  - attention output
  - MLP output
- Add causal-path checks:
  - Skip patch configurations that cannot structurally affect downstream logits.
  - Explicitly document skipped layer/site combinations.
- Simplify the default RQ3 batch matrix:
  - Remove default `base -> lora_only` source-to-target patch because it duplicates the base-to-target control.
  - Keep default directions:
    - `lora -> instruction`
    - `lora -> base`
    - `instruction -> lora`
- Add RQ3.1 linear-delta causal injection:
  - Fit `H_delta = E[h_task - h_base]` on held-out same-task extraction samples.
  - Keep extraction and evaluation samples disjoint.
  - Test both `instruction_only - base` and `lora_only - base` deltas.
  - Cross-inject into `base`, `instruction_only`, and `lora_only` targets where alignment permits.
  - Include random or permuted delta controls.
- Run matched-behavior RQ3:
  - Only interpret interchangeability on tasks where both instruction-only and LoRA-only reach task thresholds.
- Test the value/readout hypothesis:
  - Compare or patch `v_proj` output, `o_proj` output, and MLP output.
  - Determine whether divergence appears in value extraction, attention writeback, or later MLP computation.
- Run a small LoRA rank sweep:
  - Prioritize `r=1`, `r=4`, `r=8`, and `r=16`.
  - Track residual similarity, attention-output similarity, and RQ3 patch effects.
- Improve statistical reliability:
  - Increase sample count beyond 16 test examples per task.
  - Add bootstrap confidence intervals for layer-level means.
  - Repeat key runs across multiple random seeds.
- Improve reporting:
  - Label every result as `implemented`, `partial`, or `invalidated`.
  - Ensure every CSV and HTML report includes metric definitions and alignment strategy.
  - Normalize output directory structure:
    - `config`
    - `dataset_snapshot`
    - `adapters`
    - `metrics`
    - `activations`
    - `plots`
    - `reports`

## 8. Current Working Conclusion

- The current pipeline is sufficient for an initial mechanistic comparison between natural-language instruction control and LoRA-based parameter control.
- LoRA adapters are behaviorally strong on the selected synthetic tasks.
- Instruction-only prompting is weaker and less stable, making direct equivalence claims difficult.
- RQ1 suggests related but not identical residual perturbation directions.
- RQ2 shows very high attention routing similarity.
- RQ2.1 indicates that the important differences likely lie in attention value/readout/writeback rather than attention probability routing.
- RQ3 does not support simple activation interchangeability under the current text-span block-output patching setup.
- The next stage should focus on stronger causal localization:
  - activation-site sweep
  - linear-delta injection
  - matched-behavior patching
  - value/readout path analysis
  - rank, seed, and sample-size expansion
