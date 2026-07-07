# LoRA/Instruction Mechanism Task Background Search

Date: 2026-07-07

## Goal

This note collects candidate task families for comparing natural-language instruction control with LoRA parameter control. The main acceptance criterion is not task variety by itself. A task is useful only if it can first demonstrate that the instruction is causally important:

```text
instruction-only accuracy high
no-instruction/base accuracy low
LoRA-only accuracy high after adapter training
```

Only tasks passing that gate should be used for RQ1/RQ2/RQ3 mechanism comparison.

## Background Sources

- Instruction Induction introduced 24 executable tasks where the task can be described by a natural language instruction and scored by executing that instruction on held-out examples. The task list includes spelling, word extraction, pluralization, arithmetic, membership, style, and semantic tasks. Source: https://arxiv.org/abs/2205.10782
- Super-NaturalInstructions gives a larger instruction-centered benchmark design: each task has a natural language definition, input-output instances, and task-type metadata. It is useful as a source of task categories, but many tasks depend on external NLP labels or world knowledge. Source: https://arxiv.org/abs/2204.07705
- IFEval focuses on verifiable instruction following, with constraints such as keyword count and output length. It is useful because scoring is programmatic and objective, but many prompts are open-ended generation tasks and need simplification before mechanistic comparison. Source: https://arxiv.org/abs/2311.07911
- BIG-bench contains many diverse tasks, including algorithmic and symbolic tasks. It is useful for ideas, but many tasks were designed to be hard or multi-step, so they need filtering for stable instruction-only performance. Source: https://arxiv.org/abs/2206.04615
- Recent LoRA representational work motivates using delta activations and CKA/cosine-style comparisons for adapter-induced internal changes. Source: https://arxiv.org/abs/2605.28896

## Task Selection Criteria

Use these as hard gates before spending GPU time on LoRA training or patching:

1. Programmatic label function: `target_text = f(input_text)` must be deterministic.
2. Instruction contrast: the same `input_text` should be ambiguous without the instruction.
3. Low base shortcut risk: raw input should not contain a marker that trivially reveals the output rule.
4. Output length control: prefer one-token or short outputs for early RQ runs; allow longer sequence tasks only after alignment is robust.
5. Instruction-only pass: target model must pass prompt-eval with instruction.
6. No-instruction fail: same model and same inputs must fail or stay near chance without instruction.
7. LoRA learnability: LoRA-only must learn from input-output pairs without seeing instruction text.
8. Mechanistic alignment: output positions must be alignable across base, instruction-only, and LoRA-only.

## Highest Priority Tasks

These should be added or retained first because they best isolate instruction importance while keeping scoring and alignment simple.

| Priority | Task family | Candidate task | Example instruction | Output | Why useful |
| --- | --- | --- | --- | --- | --- |
| P0 | Positional extraction | `first_word`, `last_word`, `third_word`, `nth_word_k` | Return only the third word. | single copied word | Clean instruction contrast: same input supports many possible extractions. Already partly implemented. |
| P0 | Character extraction | `first_letter`, `second_letter`, `last_letter`, `nth_char_k` | Return only the second letter of the input word. | single character | From Instruction Induction. Very short targets and easy scoring. Good for early residual/patching tests. |
| P0 | Count/extract property | `word_count`, `char_count_no_space`, `vowel_count` | Return only the number of vowels. | integer | Deterministic and instruction-sensitive. Need guard against base models guessing common counts. |
| P0 | Conditional copy/marker | `wrap_odd_char_length`, `wrap_if_contains_letter`, `copy_if_even_else_marker` | If the number of non-space characters is odd, wrap with @@. | copied text or marker | Tests conditional control. Existing `wrap_odd_char_length` fits, but long copied outputs are less ideal for first RQ3 runs. |
| P1 | Case transform on selected span | `uppercase_last_word`, `lowercase_first_word` | Return the last word in uppercase. | single transformed word | Combines extraction plus local transformation. Already implemented and still easy to score. |
| P1 | Synthetic operator | `at_operator_mod_minus_left`, `hash_operator_add_mul` | Define `a@b = a % b - a`; return the result. | integer | Strong instruction-dependence because operator semantics are invented. Good LoRA-vs-instruction test if instruction-only can solve it. |
| P1 | Token filtering | `words_starting_with_letter`, `words_ending_with_letter`, `words_containing_bigram` | Return words starting with `m`. | list of copied words | From Instruction Induction's "Starting With" style. Useful but variable output length complicates target alignment. |
| P1 | List letters / delimiters | `list_letters_space_separated`, `insert_marker_after_condition` | Split the word into letters separated by spaces. | short sequence | From Instruction Induction. Good sequence task if generated words are short. |

## Medium Priority Tasks

These are useful after the P0/P1 tasks pass because they stress different mechanisms but add alignment, ambiguity, or shortcut risks.

| Priority | Task family | Candidate task | Keep only if |
| --- | --- | --- | --- |
| P2 | Word order transformations | `reverse_words`, `rotate_words_left`, `swap_first_last` | instruction-only is high and no-instruction is low. Existing `reverse_words` may be too natural/common. |
| P2 | Arithmetic with ordinary operators | `sum_two_numbers`, `difference_two_numbers`, `min_max` | no-instruction baseline is low enough. Ordinary arithmetic may be solved without instruction from format cues. |
| P2 | Format constraints from IFEval | `include_keyword_n_times`, `exact_word_count`, `forbidden_character` | reformulated as deterministic transform over a fixed input, not open-ended writing. |
| P2 | Membership over fixed synthetic taxonomy | `extract_items_from_set`, e.g. return words from a fixed artificial class | taxonomy is synthetic and included in instruction, avoiding pretrained world knowledge. |
| P2 | Boolean predicate output | `contains_letter`, `length_is_prime`, `has_repeated_word` | class balance is controlled and base/no-instruction cannot infer the predicate. |

## Low Priority / Avoid For Core Mechanism Runs

| Task type | Reason to avoid initially |
| --- | --- |
| Sentiment, NLI, semantic similarity, word-in-context | Labels rely on pretrained semantic knowledge; LoRA may amplify existing capabilities rather than encode the instruction rule. |
| Translation, synonyms, antonyms, rhymes, larger animal | Heavy world/lexical knowledge contamination. Hard to separate instruction computation from memorized knowledge. |
| Style transfer/formality/paraphrase | Open-ended outputs make exact scoring and token alignment noisy. |
| Multi-step chain-of-thought tasks from BBH | Useful later, but too hard for clean instruction-only gating and patch interpretation. |
| Long-context retrieval tasks | Alignment and memory effects dominate the LoRA-vs-instruction question. |

## Recommended Initial Task Set

Use a compact suite with different computation profiles:

1. `first_word` or `last_word`: copied-token positional extraction.
2. `second_letter`: character-level extraction.
3. `word_count`: counting.
4. `uppercase_last_word`: extraction plus local transform.
5. `wrap_odd_char_length`: conditional branch.
6. `at_operator_mod_minus_left`: invented symbolic rule.
7. `words_starting_with_letter`: filtered multi-token output, only after alignment checks.

This gives enough diversity without making the first comparison grid too large.

## Tasks Actually Used In `plan.docx` References And Newer Related Work

This section lists tasks and datasets actually used by the literature, not derived variants.

| Source | Training data / task used | Evaluation task used | Fit for this project |
| --- | --- | --- | --- |
| Hu et al. 2021/2022, LoRA | Downstream fine-tuning with RoBERTa, DeBERTa, GPT-2, GPT-3 settings. The paper is commonly associated with GLUE-style NLU and generation benchmarks. Source: https://arxiv.org/abs/2106.09685 | GLUE/NLU and NLG-style task evaluation. | Low direct fit. These are standard NLP tasks and often too contaminated or label-semantic for clean instruction-importance tests. |
| Dettmers et al. 2023, QLoRA | GLUE with RoBERTa-large; Super-NaturalInstructions/TK-Instruct with T5; LLaMA fine-tuned on FLAN v2 and Alpaca. Source: https://arxiv.org/abs/2305.14314 | MMLU, Vicuna benchmark, zero-shot Winogrande/HellaSwag/PIQA/ARC-Easy/ARC-Challenge. | Medium fit as background only. Super-NaturalInstructions has instruction-defined tasks, but MMLU/Vicuna are not clean mechanism tasks. |
| Ghosh et al. 2024, A Closer Look at the Limitations of Instruction Tuning | Alpaca-52K, MedInstruct-52K, LIMA-1K, Databricks Dolly-15K, Tulu-V2-Mix-326K. Source: https://arxiv.org/abs/2402.05119 | Just-Eval-Instruct-1K and MedInstruct-test-216. | Low direct fit. Useful for showing LoRA may learn style/initiation, but the tasks are open-ended and hard to score mechanistically. |
| Biderman et al. 2024, LoRA Learns Less and Forgets Less | StarCoder-Python and OpenWebMath for continued pretraining; Magicoder-Evol-Instruct-110K and MetaMathQA for instruction fine-tuning. Source: https://arxiv.org/abs/2405.09673 | HumanEval for code; GSM8K for math; HellaSwag, WinoGrande, ARC-Challenge for forgetting/source-domain retention. | Medium fit. HumanEval/GSM8K are real used LoRA tasks, but not ideal for instruction-vs-LoRA equivalence because they require code/math reasoning and pretrained knowledge. |
| Wallace et al. 2024, Instruction Hierarchy | Synthetic hierarchy-training data for conflicting privileged vs lower-priority instructions. Source: https://arxiv.org/abs/2404.13208 | Robustness against prompt injections, jailbreaks, and instruction conflicts. | Medium later fit. It directly tests instruction importance, but tasks involve conflict resolution rather than simple transformations. |
| Chen et al. 2024, SIFo | Sequential instruction-following tasks. Source: https://arxiv.org/abs/2406.19999 | Four task groups: text modification, question answering, mathematics, and security rules. | Medium fit. Text modification is the best candidate group; QA/math/security are less clean for this project. |
| Li et al. 2025, Uni-LoRA | LoRA-variant training across GLUE, math reasoning, and instruction-tuning benchmarks. Source: https://arxiv.org/abs/2506.00799 | GLUE, mathematical reasoning, instruction tuning benchmark performance. | Low to medium fit. Good for rank/parameter-sharing background, but task set is benchmark-oriented rather than instruction-contrast-oriented. |
| Chen et al. 2025, Layer-Aware Task Arithmetic | Task-vector/model-merging experiments. Source: https://arxiv.org/abs/2502.20186 | WikiText-2, GSM8K, HumanEval. | Low direct fit. Useful for disentangling task-specific vs instruction-following components, but used tasks are not clean synthetic transformations. |
| Ren et al. 2025, LexInstructEval | Fine-grained lexical instruction following with rule-based grammar. Source: https://arxiv.org/abs/2511.17561 | Programmatically verified lexical instruction constraints. | High conceptual fit if using their actual lexical-constraint categories; exact task instances should be taken from their released dataset/tools. |
| Pyatkin et al. 2025, IFBench | Verifiable instruction-following constraints and RLVR training prompts. Source: https://arxiv.org/abs/2507.02833 | 58 out-of-domain verifiable output constraints. | High conceptual fit for instruction importance, but many tasks are output constraints rather than input transformations. |
| Petty et al. 2025, RELIC | Synthetic formal grammars provided in context. Source: https://arxiv.org/abs/2506.05205 | Language recognition: decide whether strings are generated by the specified formal grammar. | Medium later fit. Strong instruction dependence, but may be too hard for early LoRA/instruction path comparison. |
| Chopra 2026, Mechanistic Investigation of SFT | Four isolated fine-tuning tasks: MultiNLI, GSM8K, WildJailbreak, and OpenAI Tool Calling. Source: https://arxiv.org/abs/2605.11426 | Same held-out task sets, with raw activation and SAE-latent drift analysis. | Medium fit. Tool Calling and MultiNLI are useful categories; GSM8K/WildJailbreak are less clean for the first controlled comparison. |
| Prasanth K K 2026, Feature Geometry of LoRA Adapters | `tatsu-lab/alpaca`, 10,000 training samples, Gemma-2-9B LoRA ranks 4/8/16/32. Source: https://arxiv.org/abs/2605.28896 | Alpaca held-out/probe samples bucketed into creative, factual, reasoning, coding, and practical categories. | Low direct fit. Very relevant methodologically for LoRA delta activations, but Alpaca is open-ended instruction following. |

## Literature-Used Task Shortlist For This Project

If the requirement is that the task family was used in prior literature, the best candidates are:

1. Instruction Induction spelling/extraction tasks: first letter, second letter, list letters, starting-with-word extraction.
2. Instruction Induction arithmetic tasks: sum and difference.
3. SIFo text modification tasks.
4. LexInstructEval lexical instruction constraints.
5. IFBench verifiable output constraints.
6. RELIC simple formal-language recognition, only if instruction-only performance is strong enough.
7. OpenAI Tool Calling from Chopra 2026, if JSON/function-call output alignment is implemented.

The weaker candidates are Alpaca, Vicuna, MMLU, GSM8K, HumanEval, HellaSwag, WinoGrande, ARC, MultiNLI, WildJailbreak, and general instruction-tuning datasets. They are heavily used in the literature, but they are not clean for verifying instruction importance before LoRA mechanism comparison.

## Acceptance Protocol

For each candidate task:

1. Generate 100-500 test examples with balanced input properties.
2. Run `task_acceptance` under at least 3 instruction phrasings.
3. Require instruction-only token or semantic accuracy >= 0.8.
4. Require no-instruction accuracy <= 0.3.
5. Train LoRA without instruction text.
6. Require LoRA-only task semantic accuracy >= 0.8 on held-out examples.
7. Only then include the task in residual similarity, attention-output similarity, and activation patching.

For tasks with long or variable outputs, use semantic correctness as the behavior gate, but restrict RQ1/RQ2/RQ3 to aligned target positions or start with short-output variants.

## Implementation Notes

- Add `first_letter`, `second_letter`, `last_letter`, `char_count_no_space`, `vowel_count`, and `words_starting_with_letter` to `src/lora_instruction_analysis/data/tasks.py`.
- Parameterized tasks such as `nth_word_k` or `words_starting_with_letter` should be registered as fixed variants first, not runtime-free-form tasks. That keeps one LoRA adapter tied to one deterministic transformation.
- Avoid using task instructions that include many examples during core comparison. Few-shot demonstrations can help acceptance, but they also add extra input tokens and mechanisms that confound instruction-token vs LoRA comparisons.
- Keep a separate `task_screening` result table recording instruction-only, no-instruction, and LoRA-only quality before any mechanism interpretation.
