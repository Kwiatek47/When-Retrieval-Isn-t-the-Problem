from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    balanced = _load_report(args.balanced_report)
    yes = _load_report(args.yes_report)
    summary = {
        "balanced": _compact_summary(balanced),
        "first_yes": _compact_summary(yes),
        "reports": {
            "balanced": str(args.balanced_report),
            "first_yes": str(args.yes_report),
        },
    }
    _write_json(args.json_out, summary)
    _write_markdown(args.md_out, summary)
    print(f"Wrote quick PQA-L summary JSON: {args.json_out}")
    print(f"Wrote quick PQA-L summary Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a combined summary for quick PQA-L eval runs.")
    parser.add_argument("--balanced-report", type=Path, required=True)
    parser.add_argument("--yes-report", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args()


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "dataset": summary["dataset"],
        "model": summary["model"],
        "case_count": summary["case_count"],
        "label_accuracy": summary["label_accuracy"],
        "case_pass_rate": summary["case_pass_rate"],
        "source_hit_at_1": summary["source_hit_at_1"],
        "source_hit_at_3": summary["source_hit_at_3"],
        "citation_pass_rate": summary["citation_pass_rate"],
        "predicted_labels": summary.get("predicted_labels", {}),
        "labels": summary.get("labels", {}),
        "evidence_decision": summary.get("evidence_decision", {}),
        "config": summary.get("config", {}),
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quick PQA-L Eval Summary",
        "",
        "| Run | Cases | Accuracy | Case pass | Hit@1 | Hit@3 | Citation pass |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, run in (("balanced", summary["balanced"]), ("first_yes", summary["first_yes"])):
        lines.append(
            "| "
            f"{name} | "
            f"{run['case_count']} | "
            f"{run['label_accuracy']:.3f} | "
            f"{run['case_pass_rate']:.3f} | "
            f"{run['source_hit_at_1']:.3f} | "
            f"{run['source_hit_at_3']:.3f} | "
            f"{run['citation_pass_rate']:.3f} |"
        )

    lines.extend(["", "## Predicted Labels", "", "| Run | yes | no | maybe |", "|---|---:|---:|---:|"])
    for name, run in (("balanced", summary["balanced"]), ("first_yes", summary["first_yes"])):
        predicted = run.get("predicted_labels", {})
        lines.append(
            "| "
            f"{name} | "
            f"{predicted.get('yes', 0)} | "
            f"{predicted.get('no', 0)} | "
            f"{predicted.get('maybe', 0)} |"
        )

    lines.extend(["", "## Reports", ""])
    for key, value in summary["reports"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
