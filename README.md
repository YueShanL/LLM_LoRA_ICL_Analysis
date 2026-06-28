# LoRA Instruction Analysis

This repository contains the first system module from `plan_summary.md`: a reproducible dataset generator for synthetic text transformation tasks.

## Generate a Dataset

Use the project virtual environment once it is repaired or recreated:

```powershell
venv\Scripts\python.exe -m pip install -e .
venv\Scripts\python.exe -m lora_instruction_analysis.data.cli --task add_zxq_after_t_or_l --source wikitext --output-dir data/generated/add_zxq
```

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
  --task add_zxq_after_t_or_l `
  --source wikitext `
  --max-source-rows 500 `
  --train-size 300 `
  --validation-size 50 `
  --test-size 50 `
  --output-dir data/generated/add_zxq
```

The command writes:

```text
data/generated/add_zxq/
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
