from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    report = _read_json(args.report)
    baseline = _read_json(args.baseline)

    score = _nested_float(report, args.metric_path)
    previous_best = _nested_float(baseline, "previous_best_score")
    allowed_floor = previous_best - args.allowed_drop
    passed = score >= allowed_floor

    result = {
        "metric_path": args.metric_path,
        "score": score,
        "previous_best_score": previous_best,
        "allowed_drop": args.allowed_drop,
        "allowed_floor": allowed_floor,
        "passed": passed,
        "baseline_path": str(args.baseline),
        "report_path": str(args.report),
    }
    _write_json(args.out, result)

    status = "PASS" if passed else "FAIL"
    print(
        f"{status}: {args.metric_path}={score:.4f}, "
        f"previous_best={previous_best:.4f}, floor={allowed_floor:.4f}"
    )
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail when an eval metric regresses below previous best minus tolerance.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metric-path", default="summary.label_accuracy")
    parser.add_argument("--allowed-drop", type=float, default=0.01)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}.")
    return data


def _nested_float(data: dict[str, Any], path: str) -> float:
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise RuntimeError(f"Missing metric path `{path}` at `{part}`.")
        value = value[part]
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Metric path `{path}` is not numeric: {value!r}") from exc


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
