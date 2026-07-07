from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json")
DEFAULT_OUT_DIR = Path("data/benchmarks/pubmedqa/official_pqal_test/quick")


def main() -> None:
    args = _parse_args()
    data = _load_dataset(args.dataset)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    balanced = _balanced_cases(data, per_label=args.per_label)
    first_yes = _first_label_cases(data, label="yes", count=args.yes_count)

    balanced_path = args.out_dir / f"balanced{args.per_label * 3}.json"
    first_yes_path = args.out_dir / f"first{args.yes_count}_yes.json"
    manifest_path = args.out_dir / "manifest.json"

    _write_json(balanced_path, balanced)
    _write_json(first_yes_path, first_yes)
    _write_json(
        manifest_path,
        {
            "source_dataset": str(args.dataset),
            "balanced": {
                "path": str(balanced_path),
                "count": len(balanced),
                "labels": _label_counts(balanced),
            },
            "first_yes": {
                "path": str(first_yes_path),
                "count": len(first_yes),
                "labels": _label_counts(first_yes),
            },
        },
    )

    print(f"Wrote balanced quick set: {balanced_path} cases={len(balanced)} labels={_label_counts(balanced)}")
    print(f"Wrote first-yes quick set: {first_yes_path} cases={len(first_yes)} labels={_label_counts(first_yes)}")
    print(f"Wrote quick-set manifest: {manifest_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build deterministic quick PQA-L eval subsets.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--per-label", type=int, default=30)
    parser.add_argument("--yes-count", type=int, default=100)
    return parser.parse_args()


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected a list of cases in {path}.")
    return data


def _balanced_cases(data: list[dict[str, Any]], *, per_label: int) -> list[dict[str, Any]]:
    selected_by_label = {label: [] for label in ("yes", "no", "maybe")}
    for item in data:
        label = str(item.get("expected_label") or "").lower()
        if label in selected_by_label and len(selected_by_label[label]) < per_label:
            selected_by_label[label].append(item)

    missing = {
        label: per_label - len(items)
        for label, items in selected_by_label.items()
        if len(items) < per_label
    }
    if missing:
        raise RuntimeError(f"Dataset does not have enough cases for balanced split: {missing}")

    interleaved: list[dict[str, Any]] = []
    for index in range(per_label):
        for label in ("yes", "no", "maybe"):
            interleaved.append(selected_by_label[label][index])
    return interleaved


def _first_label_cases(data: list[dict[str, Any]], *, label: str, count: int) -> list[dict[str, Any]]:
    selected = [item for item in data if str(item.get("expected_label") or "").lower() == label][:count]
    if len(selected) < count:
        raise RuntimeError(f"Dataset has only {len(selected)} cases for label={label}; requested {count}.")
    return selected


def _label_counts(data: list[dict[str, Any]]) -> dict[str, int]:
    counts = {label: 0 for label in ("yes", "no", "maybe")}
    for item in data:
        label = str(item.get("expected_label") or "").lower()
        if label in counts:
            counts[label] += 1
    return counts


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
