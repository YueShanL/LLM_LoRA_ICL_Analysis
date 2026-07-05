import csv
import pathlib
import statistics

import matplotlib.pyplot as plt


root = pathlib.Path(r"experiments\add_zxq_llama32_3b_r8_20260628_retry2")
out = root / "plots" / "summary"
out.mkdir(parents=True, exist_ok=True)

metrics = {
    "Residual delta": root / "plots" / "rq1" / "token_similarity.csv",
    "Attention probs": root / "plots" / "rq21" / "attention_probs" / "token_similarity.csv",
    "Attention output": root / "plots" / "rq21" / "attention_outputs" / "token_similarity.csv",
}

rows = []
for name, path in metrics.items():
    vals = sorted(float(r["cosine_similarity"]) for r in csv.DictReader(path.open(encoding="utf-8", newline="")))
    q = lambda p: vals[min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))]
    rows.append(
        {
            "metric": name,
            "count": len(vals),
            "mean": statistics.fmean(vals),
            "median": q(0.5),
            "q25": q(0.25),
            "q75": q(0.75),
            "min": vals[0],
            "max": vals[-1],
        }
    )

quality_path = root / "plots" / "rq21" / "attention_probs" / "quality_summary.csv"
quality = [
    {
        "condition": r["condition"],
        "token_accuracy": float(r["mean_token_accuracy"]),
        "sequence_accuracy": float(r["mean_sequence_accuracy"]),
        "loss": float(r["mean_loss"]),
    }
    for r in csv.DictReader(quality_path.open(encoding="utf-8", newline=""))
]

with (out / "summary_stats.csv").open("w", encoding="utf-8", newline="") as handle:
    fields = ["metric", "count", "mean", "median", "q25", "q75", "min", "max"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

with (out / "quality_stats.csv").open("w", encoding="utf-8", newline="") as handle:
    fields = ["condition", "token_accuracy", "sequence_accuracy", "loss"]
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(quality)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), dpi=180)
colors = ["#4c78a8", "#59a14f", "#e15759"]
xs = range(len(rows))
means = [r["mean"] for r in rows]
low = [r["mean"] - r["q25"] for r in rows]
high = [r["q75"] - r["mean"] for r in rows]

ax1.bar(xs, means, color=colors, width=0.62)
ax1.errorbar(xs, means, yerr=[low, high], fmt="none", ecolor="#222", elinewidth=1, capsize=3)
ax1.set_xticks(list(xs), [r["metric"] for r in rows], rotation=18, ha="right")
ax1.set_ylim(-0.05, 1.02)
ax1.set_ylabel("Cosine similarity")
ax1.set_title("Mechanism similarity")
for i, row in enumerate(rows):
    ax1.text(i, min(0.98, row["mean"] + 0.04), f"{row['mean']:.2f}", ha="center", fontsize=8)

order = ["base", "instruction_only", "lora_only"]
quality_by_condition = {r["condition"]: r for r in quality}
xs2 = range(len(order))
token_acc = [quality_by_condition[c]["token_accuracy"] for c in order]
seq_acc = [quality_by_condition[c]["sequence_accuracy"] for c in order]
ax2.bar([i - 0.16 for i in xs2], token_acc, width=0.32, label="Token acc", color="#76b7b2")
ax2.bar([i + 0.16 for i in xs2], seq_acc, width=0.32, label="Seq acc", color="#f28e2b")
ax2.set_xticks(list(xs2), ["Base", "Instruction", "LoRA"], rotation=18, ha="right")
ax2.set_ylim(0, 1.02)
ax2.set_title("Task quality")
ax2.legend(frameon=False, fontsize=8, loc="upper left")
for i, value in enumerate(token_acc):
    ax2.text(i - 0.16, value + 0.03, f"{value:.2f}", ha="center", fontsize=7)
for i, value in enumerate(seq_acc):
    ax2.text(i + 0.16, value + 0.03, f"{value:.2f}", ha="center", fontsize=7)

fig.suptitle("RQ1/RQ2.1 summary: add_zxq r8, 16 test samples", fontsize=10)
fig.tight_layout()
fig.savefig(out / "rq_summary.png", bbox_inches="tight")
fig.savefig(out / "rq_summary.svg", bbox_inches="tight")
print(out / "rq_summary.png")
print(out / "summary_stats.csv")
