from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    report = _read_json(args.report)
    checks = []

    for spec in args.min_metric:
        path, expected = _parse_metric_spec(spec)
        actual = _nested_float(report, path)
        checks.append(
            {
                "type": "min",
                "path": path,
                "actual": actual,
                "expected": expected,
                "passed": actual >= expected,
            }
        )

    for spec in args.max_metric:
        path, expected = _parse_metric_spec(spec)
        actual = _nested_float(report, path)
        checks.append(
            {
                "type": "max",
                "path": path,
                "actual": actual,
                "expected": expected,
                "passed": actual <= expected,
            }
        )

    passed = all(check["passed"] for check in checks)
    result = {
        "passed": passed,
        "report_path": str(args.report),
        "checks": checks,
    }
    _write_json(args.out, result)

    for check in checks:
        operator = ">=" if check["type"] == "min" else "<="
        status = "PASS" if check["passed"] else "FAIL"
        print(f"{status}: {check['path']}={check['actual']:.4f} {operator} {check['expected']:.4f}")
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check absolute medical eval gates.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-metric", action="append", default=[], help="Metric gate as path:value, e.g. summary.safety_pass_rate:1.0")
    parser.add_argument("--max-metric", action="append", default=[], help="Metric gate as path:value, e.g. summary.severe_harm_count:0")
    return parser.parse_args()


def _parse_metric_spec(spec: str) -> tuple[str, float]:
    if ":" not in spec:
        raise RuntimeError(f"Expected metric spec in `path:value` form, got `{spec}`.")
    path, value = spec.rsplit(":", 1)
    return path, float(value)


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
