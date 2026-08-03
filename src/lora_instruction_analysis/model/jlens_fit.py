"""Optional Jacobian-lens fitting entry point."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from lora_instruction_analysis.model.collect import _torch_dtype

INSTALL_GUIDANCE = (
    "J-lens fitting requires the optional jacobian-lens research package. "
    "Install it via the project's jlens extra before running this command."
)


def _load_dataset_split(dataset_name: str, dataset_config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("J-lens fitting requires datasets. Install the project dependencies first.") from exc
    return load_dataset(dataset_name, dataset_config, split=split) if dataset_config else load_dataset(dataset_name, split=split)


def _sample_rows(dataset, *, text_column: str, count: int, seed: int, offset: int = 0) -> list[dict]:
    if text_column not in dataset.column_names:
        raise ValueError(f"Dataset split is missing text column {text_column!r}; available: {dataset.column_names}")
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    selected = []
    for row_index in indices[offset:]:
        text = str(dataset[int(row_index)][text_column]).strip()
        if text:
            selected.append({"row_index": int(row_index), "text": text})
        if len(selected) == count:
            return selected
    raise ValueError(f"Dataset split has fewer than {count + offset} usable rows in {text_column!r}.")


def _sample_dataset_texts(
    *,
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    validation_split: str | None,
    text_column: str,
    num_sequences: int,
    validation_sequences: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    fit_dataset = _load_dataset_split(dataset_name, dataset_config, dataset_split)
    if validation_split and validation_split != dataset_split:
        validation_dataset = _load_dataset_split(dataset_name, dataset_config, validation_split)
        return (
            _sample_rows(fit_dataset, text_column=text_column, count=num_sequences, seed=seed),
            _sample_rows(validation_dataset, text_column=text_column, count=validation_sequences, seed=seed + 1),
        )
    combined = _sample_rows(
        fit_dataset,
        text_column=text_column,
        count=num_sequences + validation_sequences,
        seed=seed,
    )
    return combined[:num_sequences], combined[num_sequences:]


def fit_jlens(
    *,
    model_name: str,
    output_dir: Path,
    dataset_name: str,
    dataset_config: str | None = None,
    dataset_split: str = "train",
    validation_split: str | None = None,
    text_column: str = "text",
    num_sequences: int = 100,
    validation_sequences: int = 50,
    sequence_length: int = 128,
    dtype: str = "bfloat16",
    device: str = "cuda",
    seed: int = 13,
    dim_batch: int = 8,
    skip_first: int = 16,
    checkpoint_every: int | None = 1,
) -> None:
    try:
        import jlens  # type: ignore
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(INSTALL_GUIDANCE) from exc

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_rows, validation_rows = _sample_dataset_texts(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        dataset_split=dataset_split,
        validation_split=validation_split,
        text_column=text_column,
        num_sequences=num_sequences,
        validation_sequences=validation_sequences,
        seed=seed,
    )

    if not hasattr(jlens, "fit"):
        raise RuntimeError("Installed jlens package does not expose jlens.fit; adapt jlens_fit.py to the installed API.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {"torch_dtype": _torch_dtype(torch, dtype)}
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        hf_model.to(device)
    lens_model = jlens.HFLensModel(hf_model, tokenizer)
    lens = jlens.fit(  # type: ignore[attr-defined]
        lens_model,
        [row["text"] for row in fit_rows],
        max_seq_len=sequence_length,
        dim_batch=dim_batch,
        skip_first=skip_first,
        checkpoint_path=str(output_dir / "fit_checkpoint.pt"),
        checkpoint_every=checkpoint_every,
        resume=True,
    )
    lens_dir = output_dir / "lens"
    lens_dir.mkdir(exist_ok=True)
    if hasattr(lens, "save_pretrained"):
        lens.save_pretrained(lens_dir)
    elif hasattr(lens, "save"):
        lens.save(str(lens_dir / "lens.pt"))
    else:
        torch.save(lens, lens_dir / "lens.pt")

    config = {
        "model_name": model_name,
        "tokenizer_name": model_name,
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "validation_split": validation_split or dataset_split,
        "text_column": text_column,
        "fit_sample_row_indices": [row["row_index"] for row in fit_rows],
        "validation_sample_row_indices": [row["row_index"] for row in validation_rows],
        "sequence_length": sequence_length,
        "num_fit_sequences": len(fit_rows),
        "num_validation_sequences": len(validation_rows),
        "seed": seed,
        "dtype": dtype,
        "device": device,
        "dim_batch": dim_batch,
        "skip_first": skip_first,
        "checkpoint_every": checkpoint_every,
        "jlens_package": getattr(jlens, "__version__", "unknown"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "validation_summary.json").write_text(
        json.dumps({"validation_sequences": len(validation_rows), "status": "fit_complete"}, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit an optional base-model Jacobian lens on generic text.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-config")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--validation-split")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--num-sequences", type=int, default=100)
    parser.add_argument("--validation-sequences", type=int, default=50)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dim-batch", type=int, default=8)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit_jlens(
        model_name=args.model_name,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        validation_split=args.validation_split,
        text_column=args.text_column,
        num_sequences=args.num_sequences,
        validation_sequences=args.validation_sequences,
        sequence_length=args.sequence_length,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        dim_batch=args.dim_batch,
        skip_first=args.skip_first,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Wrote J-lens fit artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
