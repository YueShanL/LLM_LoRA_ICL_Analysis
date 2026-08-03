# transformers Guide

Generated fallback guide from the repository venv. No dependency-owned `GUIDE.md`, `SKILL.md`, or `GUIDE_INDEX.json` was present. This is intentionally limited to the public/model-specific surface needed to collect aligned Gemma 4 teacher attention, restricted student residuals, and KV caches.

## Resolution

- Resolved version: `5.12.1`.
- Python requirement: `>=3.10.0`; the `torch` extra requires `torch>=2.4`.
- Local sources: `venv/Lib/site-packages/transformers/` and `venv/Lib/site-packages/transformers-5.12.1.dist-info/METADATA`.
- The project manifest permits `transformers>=4.40.0`, but the installed major version has materially newer cache/output APIs. Implement against `5.12.1` and test before widening compatibility.

## Local Loading and Token Alignment

- `AutoTokenizer.from_pretrained(path, ...)` selects the tokenizer from the local config. Pass `local_files_only=True` for an offline/isolated pipeline and leave `trust_remote_code=False` unless reviewed local custom code is intentionally required.
- Tokenization returns a `BatchEncoding`; request `return_tensors="pt"`. For chat data, `apply_chat_template(conversation, add_generation_prompt=..., tokenize=True, return_dict=True, ...)` is the public entry point.
- Tokenize an aligned example once and derive teacher/student slices from the same token IDs. Do not independently render/tokenize the two branches, because template or special-token drift invalidates position and attention labels.
- `PreTrainedModel.from_pretrained(path, local_files_only=True, dtype=..., attn_implementation=..., ...)` returns a model in evaluation mode. `torch_dtype` remains backward-compatible but is deprecated in favor of `dtype` in this version. Explicitly call `eval()` and freeze parameters for clarity.

## Model Selection and Forward Outputs

- `AutoModelForCausalLM` maps a top-level `gemma4` config to `Gemma4ForConditionalGeneration`, while `gemma4_text` maps to `Gemma4ForCausalLM`. Do not assume the auto class returns a text-only module; validate `model.config.model_type` and isolate access to the text backbone in a Gemma adapter.
- The relevant Gemma 4 text forward accepts `input_ids`, `attention_mask`, `position_ids`, `past_key_values: Cache`, `inputs_embeds`, `use_cache`, and output-control kwargs. The causal-LM wrapper additionally accepts `labels` and `logits_to_keep`.
- Request `output_hidden_states=True` for residual inputs and `output_attentions=True` for teacher labels. In 5.12.1 these outputs are captured by hooks and returned as tuples on the model output; validate tuple length and tensor shapes before aggregation.
- Teacher attention collection must load/set `attn_implementation="eager"`. The installed SDPA adapter explicitly does not support `output_attentions=True` and returns no attention weights.
- Use `logits_to_keep` only for logits-memory reduction; it does not reduce hidden-state or attention collection cost.

## Cache and Position Surface

- `DynamicCache(config=model.config)` is the public growing cache. Model outputs expose `past_key_values` as a `Cache` object, not a legacy tuple. Common public operations include `update(key_states, value_states, layer_idx)`, `get_seq_length(layer_idx=0)`, `crop(...)`, `reset()`, and batch selection/reordering.
- Dynamic cache tensors use `[batch, num_heads, sequence, head_dim]`. Passing the model config is important because it constructs sliding/hybrid layers according to the model.
- Gemma 4 has sliding/full attention types and KV-sharing behavior; its text output can also carry `shared_kv_states`. A “complete all-layer KV block” cannot be implemented as an unvalidated list of identical per-layer tensors. Centralize slicing/packing in a Gemma cache adapter and validate each physical layer, shared state, logical position, dtype, and device.
- Supply explicit `position_ids` whenever selected historical blocks retain non-contiguous original logical positions. `cache_position` is not an explicit Gemma 4 text-forward parameter in this installed implementation.
- Gemma 4 accepts an ordinary attention mask and internally constructs masks for its layer types. Its source also permits a precomputed mask mapping keyed by layer type; treat that path as model-specific, keep it inside the adapter, and test causality/visibility against a tiny deterministic case.

## Pipeline Constraints

- Run teacher full-context and student restricted-context forwards separately with identical token IDs and logical positions. Only teacher attention aggregates cross the boundary as labels.
- Validate that each attention tensor is shaped as expected before reducing layer, head, horizon, or block axes. Save unreduced aggregation metadata so experiments do not silently hard-code reductions.
- Do not use `generate()` for the first teacher-forced collector; direct forwards make the first future position affected by retrieval and the attention horizon explicit.
- `output_attentions=True` is memory intensive and eager attention is slower. Collect only configured layers/horizon queries where the adapter permits it, write detached CPU artifacts incrementally, and fail fast rather than substituting missing weights.
- Keep all custom cache mutation and per-layer mask logic out of the generic router trainer. Those details are model/version-sensitive and are the main compatibility boundary for Transformers upgrades.
