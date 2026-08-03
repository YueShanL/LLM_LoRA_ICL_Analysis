"""Fit a small SAE encoder on saved condition-minus-base vectors."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lora_instruction_analysis.model.visualize import _load_tensor, _metrics_by_key


def _vectors_from_run(torch, run_dir: Path, mode: str):
    metrics = _metrics_by_key(run_dir)
    sample_ids = sorted(
        sample_id
        for sample_id in {sample_id for sample_id, _ in metrics}
        if all((sample_id, condition) in metrics for condition in ("base", "instruction_only", "lora_only"))
    )
    key = {
        "residual": "hidden_states",
        "attention_outputs": "attention_outputs",
        "attention_post_o_proj_outputs": "attention_post_o_proj_outputs",
    }[mode]
    for sample_id in sample_ids:
        tensors = {condition: _load_tensor(torch, metrics[(sample_id, condition)], run_dir) for condition in ("base", "instruction_only", "lora_only")}
        base = tensors["base"][key]
        for condition in ("instruction_only", "lora_only"):
            delta = (tensors[condition][key] - base).float().reshape(-1, base.shape[-1])
            for row in delta:
                yield row.cpu()


def _sample_vectors(torch, run_dirs: list[Path], mode: str, max_vectors: int, seed: int):
    rng = random.Random(seed)
    kept = []
    seen = 0
    for run_dir in run_dirs:
        for vector in _vectors_from_run(torch, run_dir, mode):
            seen += 1
            if len(kept) < max_vectors:
                kept.append(vector)
            else:
                index = rng.randrange(seen)
                if index < max_vectors:
                    kept[index] = vector
    if not kept:
        raise ValueError(f"No vectors found for mode {mode}.")
    return torch.stack(kept), seen


def fit_sae(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    mode: str,
    features: int = 1024,
    max_vectors: int = 20000,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    l1: float = 1e-3,
    seed: int = 13,
    device: str = "cuda",
) -> None:
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("SAE fitting requires torch.") from exc

    torch.manual_seed(seed)
    data, total_seen = _sample_vectors(torch, run_dirs, mode, max_vectors, seed)
    mean = data.mean(dim=0)
    scale = data.std(dim=0).clamp_min(1e-6)
    data = (data - mean) / scale
    dim = data.shape[1]
    target_device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder = nn.Linear(dim, features).to(target_device)
    decoder = nn.Linear(features, dim).to(target_device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=lr)
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for epoch in range(epochs):
        order = torch.randperm(data.shape[0], generator=generator)
        epoch_losses = []
        for start in range(0, data.shape[0], batch_size):
            batch = data[order[start : start + batch_size]].to(target_device)
            acts = torch.relu(encoder(batch))
            recon = decoder(acts)
            loss = torch.nn.functional.mse_loss(recon, batch) + l1 * acts.abs().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append({"epoch": epoch, "loss": sum(epoch_losses) / len(epoch_losses)})

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder_weight": encoder.weight.detach().cpu(),
            "encoder_bias": encoder.bias.detach().cpu(),
            "input_mean": mean,
            "input_scale": scale,
            "mode": mode,
            "features": features,
            "input_dim": dim,
        },
        output_dir / "sae.pt",
    )
    (output_dir / "sae_fit_config.json").write_text(
        json.dumps(
            {
                "run_dirs": [str(path) for path in run_dirs],
                "mode": mode,
                "features": features,
                "max_vectors": max_vectors,
                "total_vectors_seen": total_seen,
                "vectors_used": int(data.shape[0]),
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "l1": l1,
                "seed": seed,
                "device": str(target_device),
                "losses": losses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a small SAE on saved condition-minus-base vectors.")
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("residual", "attention_outputs", "attention_post_o_proj_outputs"), default="residual")
    parser.add_argument("--features", type=int, default=1024)
    parser.add_argument("--max-vectors", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_sae(
        args.run,
        args.output_dir,
        mode=args.mode,
        features=args.features,
        max_vectors=args.max_vectors,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        l1=args.l1,
        seed=args.seed,
        device=args.device,
    )
    print(f"Wrote SAE to {args.output_dir}")


if __name__ == "__main__":
    main()
