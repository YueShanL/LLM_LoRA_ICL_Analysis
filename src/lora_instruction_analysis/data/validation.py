"""Validate DatasetModule artifacts without loading a model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "sample_id", "task_id", "input_text", "instruction_text", "target_text", "condition"
}
SPLITS = ("train", "validation", "test")


class DatasetValidationError(ValueError):
    pass


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_splits(
    splits: dict[str, list[dict]], expected_sizes: dict[str, int] | None = None
) -> dict:
    errors: list[str] = []
    rows = [(split, row) for split, split_rows in splits.items() for row in split_rows]
    for split, row in rows:
        missing = sorted(REQUIRED_FIELDS - row.keys())
        if missing:
            errors.append(f"{split}/{row.get('sample_id', '<unknown>')}: missing fields {missing}")
        if not str(row.get("target_text", "")).strip():
            errors.append(f"{split}/{row.get('sample_id', '<unknown>')}: empty target_text")

    input_locations: dict[str, list[str]] = {}
    for split, row in rows:
        input_locations.setdefault(str(row.get("input_text", "")), []).append(
            f"{split}/{row.get('sample_id', '<unknown>')}"
        )
    duplicates = {text: locations for text, locations in input_locations.items() if len(locations) > 1}
    if duplicates:
        errors.append(f"duplicate input_text in {len(duplicates)} group(s)")

    task_ids = sorted({str(row["task_id"]) for _, row in rows if row.get("task_id")})
    if len(task_ids) != 1:
        errors.append(f"expected one task_id, found {task_ids}")

    actual_sizes = {split: len(splits.get(split, [])) for split in SPLITS}
    if expected_sizes is not None:
        mismatches = {
            split: {"expected": int(expected_sizes.get(split, 0)), "actual": actual_sizes[split]}
            for split in SPLITS
            if actual_sizes[split] != int(expected_sizes.get(split, 0))
        }
        if mismatches:
            errors.append(f"split size mismatch: {mismatches}")

    report = {
        "valid": not errors,
        "errors": errors,
        "required_fields": sorted(REQUIRED_FIELDS),
        "splits": actual_sizes,
        "task_id": task_ids[0] if len(task_ids) == 1 else None,
        "duplicate_input_groups": len(duplicates),
    }
    if errors:
        raise DatasetValidationError("; ".join(errors))
    return report


def validate_dataset(dataset_path: Path) -> dict:
    manifest_path = dataset_path / "manifest.json"
    if not manifest_path.exists():
        raise DatasetValidationError(f"Missing dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = {}
    for split in SPLITS:
        path = dataset_path / f"{split}.jsonl"
        if not path.exists():
            raise DatasetValidationError(f"Missing dataset split: {path}")
        splits[split] = _read_jsonl(path)
    report = validate_splits(splits, manifest.get("splits"))
    report.update(
        dataset_path=str(dataset_path),
        data_route=manifest.get("data_route"),
        target_tokenization=manifest.get("target_tokenization"),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset_path)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
