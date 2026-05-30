from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


LABELS = ("yes", "no", "maybe")
METRIC_FILES = (
    "dev_metrics_threshold_tuned.json",
    "dev_metrics_calibrated.json",
    "dev_metrics.json",
)


def main() -> None:
    args = _parse_args()
    rows = _collect_rows(args.run_root)
    report = _build_report(rows, run_root=args.run_root)
    json_out = args.json_out or args.run_root / "experiment_summary.json"
    md_out = args.md_out or args.run_root / "experiment_summary.md"
    _write_json(json_out, report)
    _write_markdown(md_out, report)
    print(f"Wrote experiment summary JSON: {json_out}")
    print(f"Wrote experiment summary Markdown: {md_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize PubMedQA classifier experiment runs.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--md-out", type=Path, default=None)
    return parser.parse_args()


def _collect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for best_dir in sorted(run_root.rglob("best")):
        if not best_dir.is_dir():
            continue
        metrics_path = _select_metrics_path(best_dir)
        if metrics_path is None:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        training_config = _load_json(best_dir / "training_config.json") or {}
        threshold_config = _load_json(best_dir / "decision_thresholds.json") or {}
        relative = best_dir.relative_to(run_root)
        parts = relative.parts
        dataset = parts[0] if len(parts) > 0 else ""
        model = parts[1] if len(parts) > 1 else str(training_config.get("model_name") or "")
        seed = parts[2].removeprefix("seed_") if len(parts) > 2 else str(training_config.get("seed") or "")
        row = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "best_dir": str(best_dir),
            "metrics_file": metrics_path.name,
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "ece": float(metrics.get("ece", 0.0)),
            "per_label": metrics.get("per_label", {}),
            "confusion_matrix": metrics.get("confusion_matrix", []),
            "predicted_labels": _predicted_labels(metrics.get("confusion_matrix", [])),
            "collapse_flags": _collapse_flags(metrics),
            "thresholds": threshold_config.get("thresholds", {}),
        }
        rows.append(row)
    return sorted(rows, key=_ranking_key, reverse=True)


def _select_metrics_path(best_dir: Path) -> Path | None:
    for filename in METRIC_FILES:
        path = best_dir / filename
        if path.exists():
            return path
    return None


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    per_label = row.get("per_label", {})
    maybe_f1 = float((per_label.get("maybe") or {}).get("f1", 0.0))
    no_f1 = float((per_label.get("no") or {}).get("f1", 0.0))
    return (float(row["macro_f1"]), float(row["accuracy"]), maybe_f1, no_f1)


def _build_report(rows: list[dict[str, Any]], *, run_root: Path) -> dict[str, Any]:
    viable = [row for row in rows if not row["collapse_flags"]]
    selected = viable[0] if viable else (rows[0] if rows else None)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "run_count": len(rows),
        "selection_policy": "Rank by dev macro_f1, then accuracy, then maybe/no F1; prefer rows without collapse flags.",
        "recommended_checkpoint": selected,
        "leaderboard": rows,
        "notes": [
            "Use this summary for model selection before official PQA-L 500.",
            "Do not tune thresholds, prompts, or checkpoint choice on official PQA-L 500.",
            "Inspect `maybe` and `no` recall/F1 before selecting a high-accuracy model.",
        ],
    }


def _predicted_labels(confusion: list[list[int]]) -> dict[str, int]:
    if not confusion:
        return {label: 0 for label in LABELS}
    return {
        label: sum(int(row[index]) for row in confusion if len(row) > index)
        for index, label in enumerate(LABELS)
    }


def _collapse_flags(metrics: dict[str, Any]) -> list[str]:
    flags = []
    per_label = metrics.get("per_label", {})
    predicted = _predicted_labels(metrics.get("confusion_matrix", []))
    total_predictions = sum(predicted.values())
    for label in LABELS:
        recall = float((per_label.get(label) or {}).get("recall", 0.0))
        f1 = float((per_label.get(label) or {}).get("f1", 0.0))
        if recall == 0.0:
            flags.append(f"{label}_zero_recall")
        if f1 < 0.05:
            flags.append(f"{label}_very_low_f1")
    if total_predictions:
        top_label, top_count = max(predicted.items(), key=lambda item: item[1])
        if top_count / total_predictions >= 0.80:
            flags.append(f"prediction_collapse_to_{top_label}")
    return flags


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    selected = report.get("recommended_checkpoint")
    lines = [
        "# PubMedQA Classifier Experiment Summary",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Run root: `{report['run_root']}`",
        f"- Runs found: {report['run_count']}",
        f"- Selection policy: {report['selection_policy']}",
        "",
    ]
    if selected:
        lines.extend(
            [
                "## Recommended Checkpoint",
                "",
                f"- Path: `{selected['best_dir']}`",
                f"- Dataset: `{selected['dataset']}`",
                f"- Model: `{selected['model']}`",
                f"- Seed: `{selected['seed']}`",
                f"- Metrics file: `{selected['metrics_file']}`",
                f"- Accuracy: {selected['accuracy']:.4f}",
                f"- Macro F1: {selected['macro_f1']:.4f}",
                f"- ECE: {selected['ece']:.4f}",
                f"- Collapse flags: `{selected['collapse_flags']}`",
                f"- Thresholds: `{selected['thresholds']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Leaderboard",
            "",
            "| Rank | Dataset | Model | Seed | Metrics | Acc | Macro F1 | ECE | yes F1 | no F1 | maybe F1 | Flags |",
            "|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for rank, row in enumerate(report["leaderboard"], start=1):
        per_label = row.get("per_label", {})
        yes_f1 = float((per_label.get("yes") or {}).get("f1", 0.0))
        no_f1 = float((per_label.get("no") or {}).get("f1", 0.0))
        maybe_f1 = float((per_label.get("maybe") or {}).get("f1", 0.0))
        lines.append(
            f"| {rank} | {row['dataset']} | `{row['model']}` | {row['seed']} | {row['metrics_file']} | "
            f"{row['accuracy']:.4f} | {row['macro_f1']:.4f} | {row['ece']:.4f} | "
            f"{yes_f1:.4f} | {no_f1:.4f} | {maybe_f1:.4f} | `{row['collapse_flags']}` |"
        )

    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
