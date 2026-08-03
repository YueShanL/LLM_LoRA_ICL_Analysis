# Project Guide

Generated: 2026-08-01 05:13:43Z

<!-- guideweaver:start -->

## Repo Shape

- Files indexed: 174
- Files changed in this refresh: 174
- Git remotes: none detected
- Manifests: pyproject.toml
- Top-level source roots: .idea, build, configs, data, docs, scripts, src, tests

## File Types

- `.py`: 58
- `.csv`: 27
- `.jsonl`: 17
- `.json`: 14
- `.log`: 11
- `.pt`: 11
- `.html`: 9
- `.md`: 9
- `.xml`: 6
- `(none)`: 3
- `.bat`: 1
- `.docx`: 1
- `.err`: 1
- `.iml`: 1
- `.out`: 1
- `.ps1`: 1
- `.sh`: 1
- `.toml`: 1
- `.txt`: 1

## Changed Files

- `.gitignore`
- `.idea/.gitignore`
- `.idea/.name`
- `.idea/LoRA_Instruction_analysis.iml`
- `.idea/csv-editor.xml`
- `.idea/inspectionProfiles/Project_Default.xml`
- `.idea/inspectionProfiles/profiles_settings.xml`
- `.idea/misc.xml`
- `.idea/modules.xml`
- `.idea/vcs.xml`
- `README.md`
- `RQ1_result.md`
- `RQ2_result.md`
- `RQ3_result.md`
- `TODO.md`
- `build/cmd_python_test.err`
- `build/cmd_python_test.out`
- `build/debug_collect_mask/config.json`
- `build/debug_collect_mask/dataset_snapshot.jsonl`
- `build/debug_collect_mask/metrics.jsonl`
- `build/debug_collect_mask/tensors/last_word-001654__base.pt`
- `build/first_word_smoke/manifest.json`
- `build/first_word_smoke/test.csv`
- `build/first_word_smoke/test.jsonl`
- `build/first_word_smoke/train.csv`
- `build/first_word_smoke/train.jsonl`
- `build/first_word_smoke/validation.csv`
- `build/first_word_smoke/validation.jsonl`
- `build/format_sources_count_1000/download_status.json`
- `build/format_sources_count_1000/manifest.json`
- `build/format_sources_count_1000/sources.jsonl`
- `build/format_sources_normalized_smoke/download_status.json`
- `build/format_sources_normalized_smoke/manifest.json`
- `build/format_sources_normalized_smoke/sources.jsonl`
- `build/format_sources_smoke/download_status.json`
- `build/format_sources_smoke/manifest.json`
- `build/format_sources_smoke/sources.jsonl`
- `build/gemma4_e4b_icl8_pipeline.err.log`
- `build/gemma4_e4b_icl8_pipeline.out.log`
- `build/gpu_monitor_add_zxq_llama32_3b_r8_20260628.log`
- `build/jlens_fit_llama32_3b_instruct_wikitext.err.log`
- `build/jlens_fit_llama32_3b_instruct_wikitext.out.log`
- `build/jlens_fit_llama32_3b_instruct_wikitext_retry2.err.log`
- `build/jlens_fit_llama32_3b_instruct_wikitext_retry2.out.log`
- `build/make_summary_plot.py`
- `build/rq21_smoke/plots/attention_outputs/aggregate_similarity.csv`
- `build/rq21_smoke/plots/attention_outputs/quality_summary.csv`
- `build/rq21_smoke/plots/attention_outputs/token_similarity.csv`
- `build/rq21_smoke/plots/attention_outputs/token_similarity.html`
- `build/rq21_smoke/plots/attention_probs/aggregate_similarity.csv`
- `build/rq21_smoke/plots/attention_probs/quality_summary.csv`
- `build/rq21_smoke/plots/attention_probs/token_similarity.csv`
- `build/rq21_smoke/plots/attention_probs/token_similarity.html`
- `build/rq21_smoke/states/config.json`
- `build/rq21_smoke/states/dataset_snapshot.jsonl`
- `build/rq21_smoke/states/metrics.jsonl`
- `build/rq21_smoke/states/tensors/add_zxq_after_t_or_l-000931__base.pt`
- `build/rq21_smoke/states/tensors/add_zxq_after_t_or_l-000931__instruction_only.pt`
- `build/rq21_smoke/states/tensors/add_zxq_after_t_or_l-000931__lora_only.pt`
- `build/rq3_lora_to_instruction_text_report.py`
- `build/rq3_report.py`
- `build/run_sae_attention_pipeline.py`
- `build/run_sae_residual_pipeline.py`
- `build/sae_attention_pipeline.err.log`
- `build/sae_attention_pipeline.out.log`
- `build/sae_residual_pipeline.err.log`
- `build/sae_residual_pipeline.out.log`
- `build/visualize_attention_smoke/out/aggregate_similarity.csv`
- `build/visualize_attention_smoke/out/quality_summary.csv`
- `build/visualize_attention_smoke/out/token_similarity.csv`
- `build/visualize_attention_smoke/out/token_similarity.html`
- `build/visualize_attention_smoke/out_check/aggregate_similarity.csv`
- `build/visualize_attention_smoke/out_check/quality_summary.csv`
- `build/visualize_attention_smoke/out_check/token_similarity.csv`
- `build/visualize_attention_smoke/out_check/token_similarity.html`
- `build/visualize_residual_smoke/out/aggregate_similarity.csv`
- `build/visualize_residual_smoke/out/quality_summary.csv`
- `build/visualize_residual_smoke/out/token_similarity.csv`
- `build/visualize_residual_smoke/out/token_similarity.html`
- `build/visualize_residual_smoke/out_check/aggregate_similarity.csv`

## Dependency Guides

- `0.1.0`: `.codex/project-guides/dependencies/0.1.0.md`
- `Synthetic`: `.codex/project-guides/dependencies/Synthetic.md`
- `datasets`: `.codex/project-guides/dependencies/datasets.md`
- `jlens`: `.codex/project-guides/dependencies/jlens.md`
- `lora-instruction-analysis`: `.codex/project-guides/dependencies/lora-instruction-analysis.md`
- `matplotlib`: `.codex/project-guides/dependencies/matplotlib.md`
- `peft`: `.codex/project-guides/dependencies/peft.md`
- `setuptools`: `.codex/project-guides/dependencies/setuptools.md`
- `setuptools.build_meta`: `.codex/project-guides/dependencies/setuptools.build_meta.md`
- `src`: `.codex/project-guides/dependencies/src.md`
- `torch`: `.codex/project-guides/dependencies/torch.md`
- `transformers`: `.codex/project-guides/dependencies/transformers.md`
- `trl`: `.codex/project-guides/dependencies/trl.md`

<!-- guideweaver:end -->
