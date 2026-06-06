from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    official_report = _read_optional_json(args.official_report)
    official_gate = _read_optional_json(args.official_gate)
    clinical_report = _read_optional_json(args.clinical_report)
    clinical_gate = _read_optional_json(args.clinical_gate)

    suite = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "suite": "medical_eval_core",
        "reports": {
            "official_pqal500": _report_entry(args.official_report, official_report, official_gate),
            "clinical_safety_golden": _report_entry(args.clinical_report, clinical_report, clinical_gate),
        },
        "overall_passed": _gate_passed(official_gate) and _gate_passed(clinical_gate),
    }
    _write_json(args.json_out, suite)
    _write_markdown(args.md_out, suite)
    print(f"Wrote medical eval suite JSON report: {args.json_out}")
    print(f"Wrote medical eval suite Markdown report: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine medical eval suite reports.")
    parser.add_argument("--official-report", type=Path, required=True)
    parser.add_argument("--official-gate", type=Path, required=True)
    parser.add_argument("--clinical-report", type=Path, required=True)
    parser.add_argument("--clinical-gate", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args()


def _report_entry(path: Path, report: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_optional(path),
        "summary": report.get("summary"),
        "gate": gate,
    }


def _gate_passed(gate: dict[str, Any]) -> bool:
    return bool(gate) and bool(gate.get("passed"))


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _sha256_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_markdown(path: Path, suite: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Medical Eval Core Suite",
        "",
        f"- Overall passed: {suite['overall_passed']}",
        f"- Created at: `{suite['created_at']}`",
        "",
        "## Reports",
        "",
        "| Suite | Passed | Key metrics | Report |",
        "|---|---:|---|---|",
    ]
    for suite_id, entry in suite["reports"].items():
        summary = entry.get("summary") or {}
        gate = entry.get("gate") or {}
        metrics = _summary_metrics(suite_id, summary)
        lines.append(f"| `{suite_id}` | {bool(gate.get('passed'))} | {metrics} | `{entry.get('path')}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary_metrics(suite_id: str, summary: dict[str, Any]) -> str:
    if suite_id == "official_pqal500":
        return (
            f"label_accuracy={_fmt(summary.get('label_accuracy'))}, "
            f"case_pass_rate={_fmt(summary.get('case_pass_rate'))}"
        )
    if suite_id == "clinical_safety_golden":
        return (
            f"safety_pass_rate={_fmt(summary.get('safety_pass_rate'))}, "
            f"severe_harm_count={summary.get('severe_harm_count', '-')}"
        )
    return "-"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
