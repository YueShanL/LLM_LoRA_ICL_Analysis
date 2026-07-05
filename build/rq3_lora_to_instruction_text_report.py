from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


root = Path("experiments/last_word_llama32_3b_r8_clean/patches")
layers = [
    ("2", root / "rq3_lora_to_instruction_text_l2"),
    ("21", root / "rq3_lora_to_instruction_text_l21"),
    ("27", root / "rq3_lora_to_instruction_text_l27"),
]
report = [
    "# RQ3 last_word lora -> instruction text-span patch report",
    "",
    "samples: 16; max_new_tokens: 20; source=lora_only; target=instruction_only; patch_span=text",
    "",
]

print("layer,patched,mean_loss,mean_token_acc,mean_seq_acc,mean_gen_token_acc,mean_gen_seq_acc")
for layer_name, layer_dir in layers:
    metrics = [json.loads(line) for line in (layer_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    generations = [
        json.loads(line) for line in (layer_dir / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generations_by_key = {(row["sample_id"], row["patched"]): row for row in generations}

    report.extend(
        [
            f"## Layer {layer_name}",
            "",
            "| sample | patched | tf_loss | tf_token_acc | tf_seq_acc | autoreg_pred | target | gen_token_acc | gen_seq_acc |",
            "|---|---:|---:|---:|---:|---|---|---:|---:|",
        ]
    )

    for patched in (False, True):
        metric_rows = [row for row in metrics if row["patched"] is patched]
        gen_rows = [row for row in generations if row["patched"] is patched]
        print(
            f"{layer_name},{patched},"
            f"{mean(row['loss'] for row in metric_rows):.6f},"
            f"{mean(row['token_accuracy'] for row in metric_rows):.6f},"
            f"{mean(row['sequence_accuracy'] for row in metric_rows):.6f},"
            f"{mean(row['token_accuracy'] for row in gen_rows):.6f},"
            f"{mean(row['sequence_accuracy'] for row in gen_rows):.6f}"
        )

    for row in metrics:
        gen = generations_by_key[(row["sample_id"], row["patched"])]
        pred = gen["pred_text"].replace("|", "\\|").replace("\n", "\\n")
        target = row["target_text"].replace("|", "\\|").replace("\n", "\\n")
        report.append(
            f"| {row['sample_id']} | {row['patched']} | {row['loss']:.6f} | "
            f"{row['token_accuracy']:.3f} | {row['sequence_accuracy']:.3f} | "
            f"{pred} | {target} | {gen['token_accuracy']:.3f} | {gen['sequence_accuracy']:.3f} |"
        )
    report.append("")

out = root / "rq3_lora_to_instruction_text_layers_2_21_27_report.md"
out.write_text("\n".join(report), encoding="utf-8")
print(f"report={out}")
