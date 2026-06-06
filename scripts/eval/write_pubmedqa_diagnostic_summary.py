from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import mean
from typing import Any


LABELS = ("yes", "no", "maybe")


@dataclass(frozen=True)
class Row:
    name: str
    kind: str
    model: str
    cases: int
    accuracy: float
    macro_f1: float | None
    maybe_recall: float | None
    no_recall: float | None
    yes_recall: float | None
    hit1: float | None
    hit3: float | None
    citation: float | None
    latency_ms: float | None
    ece: float | None
    high_conf_error_rate: float | None
    sufficient_false_rate: float | None
    predicted_labels: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]]
    report: str


def main() -> None:
    args = _parse_args()
    rows = _load_rows(args.report_root)
    rows.sort(key=lambda row: (row.kind, -row.cases, -(row.accuracy or 0.0), row.name))
    data = {
        "report_root": str(args.report_root),
        "runs": [row.__dict__ for row in rows],
        "conclusions": _conclusions(rows),
    }
    _write_json(args.json_out, data)
    _write_markdown(args.md_out, rows, data["conclusions"])
    print(f"Wrote diagnostic summary JSON: {args.json_out}")
    print(f"Wrote diagnostic summary Markdown: {args.md_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate PubMedQA diagnostic ablation reports.")
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    return parser.parse_args()


def _load_rows(report_root: Path) -> list[Row]:
    rows = []
    for path in sorted(report_root.rglob("*.json")):
        if path.name.endswith((".summary.json", ".gate.json", ".methods.json", ".lock.json")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        row = _row_from_report(path, data)
        if row is not None:
            rows.append(row)
    return rows


def _row_from_report(path: Path, data: dict[str, Any]) -> Row | None:
    if data.get("kind") == "direct_judge":
        return _row_from_direct(path, data)
    summary = data.get("summary")
    if isinstance(summary, dict) and "label_accuracy" in summary:
        return _row_from_rag(path, data)
    return None


def _row_from_direct(path: Path, data: dict[str, Any]) -> Row:
    metrics = data["metrics"]
    confusion = metrics.get("confusion_matrix", {})
    per_label = metrics.get("per_label", {})
    confidence = data.get("confidence_metrics") or {}
    sufficiency = data.get("sufficiency") or {}
    suff_counts = sufficiency.get("counts") or {}
    false_count = int(suff_counts.get("false", 0))
    total_suff = sum(int(suff_counts.get(key, 0)) for key in ("true", "false", "unknown"))
    kind = f"direct_{data.get('evidence_mode')}_{data.get('prompt_style')}"
    return Row(
        name=path.stem,
        kind=kind,
        model=str(data.get("model") or ""),
        cases=int(data.get("case_count") or 0),
        accuracy=float(metrics.get("accuracy") or 0.0),
        macro_f1=_float_or_none(metrics.get("macro_f1")),
        maybe_recall=_label_metric(per_label, "maybe", "recall"),
        no_recall=_label_metric(per_label, "no", "recall"),
        yes_recall=_label_metric(per_label, "yes", "recall"),
        hit1=None,
        hit3=None,
        citation=None,
        latency_ms=_mean_latency(data.get("cases") or []),
        ece=_float_or_none(confidence.get("ece")),
        high_conf_error_rate=_float_or_none(confidence.get("high_confidence_error_rate")),
        sufficient_false_rate=false_count / total_suff if total_suff else None,
        predicted_labels={str(k): int(v) for k, v in (data.get("predicted_labels") or {}).items()},
        confusion_matrix=confusion,
        report=str(path),
    )


def _row_from_rag(path: Path, data: dict[str, Any]) -> Row:
    summary = data["summary"]
    confusion = summary.get("confusion_matrix", {})
    per_label = _per_label_from_confusion(confusion)
    config = summary.get("config") or {}
    method = str(config.get("RAG_EVIDENCE_JUDGE_METHOD") or "rag")
    classifier = str(config.get("RAG_EVIDENCE_CLASSIFIER_ENABLED") or "")
    top_k = str(config.get("top_k") or "")
    reranker = _reranker_name(path, data)
    kind = f"rag_{method}_classifier-{classifier}_top{top_k}_{reranker}".strip("_")
    return Row(
        name=path.stem,
        kind=kind,
        model=str(summary.get("model") or ""),
        cases=int(summary.get("case_count") or 0),
        accuracy=float(summary.get("label_accuracy") or 0.0),
        macro_f1=_macro_f1(per_label),
        maybe_recall=_label_metric(per_label, "maybe", "recall"),
        no_recall=_label_metric(per_label, "no", "recall"),
        yes_recall=_label_metric(per_label, "yes", "recall"),
        hit1=_float_or_none(summary.get("source_hit_at_1")),
        hit3=_float_or_none(summary.get("source_hit_at_3")),
        citation=_float_or_none(summary.get("citation_pass_rate")),
        latency_ms=_float_or_none(summary.get("mean_latency_ms")),
        ece=None,
        high_conf_error_rate=_high_confidence_error_rate(data.get("cases") or []),
        sufficient_false_rate=_rag_uncertain_or_insufficient_rate(summary),
        predicted_labels={str(k): int(v) for k, v in (summary.get("predicted_labels") or {}).items()},
        confusion_matrix=confusion,
        report=str(path),
    )


def _per_label_from_confusion(confusion: dict[str, dict[str, int]]) -> dict[str, dict[str, float]]:
    labels_with_none = (*LABELS, "none")
    per_label = {}
    for label in LABELS:
        row = confusion.get(label, {})
        tp = int(row.get(label, 0))
        fp = sum(int((confusion.get(gold) or {}).get(label, 0)) for gold in labels_with_none if gold != label)
        fn = sum(int(row.get(pred, 0)) for pred in labels_with_none if pred != label)
        support = sum(int(row.get(pred, 0)) for pred in labels_with_none)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return per_label


def _macro_f1(per_label: dict[str, dict[str, float]]) -> float:
    return mean((per_label.get(label) or {}).get("f1", 0.0) for label in LABELS)


def _label_metric(per_label: dict[str, Any], label: str, metric: str) -> float | None:
    try:
        return float((per_label.get(label) or {}).get(metric))
    except (TypeError, ValueError):
        return None


def _mean_latency(cases: list[dict[str, Any]]) -> float | None:
    values = [_float_or_none(case.get("latency_ms")) for case in cases]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _evidence_confidences(cases: list[dict[str, Any]]) -> list[float]:
    values = []
    for case in cases:
        decision = case.get("evidence_decision") or {}
        confidence = _float_or_none(decision.get("confidence"))
        if confidence is not None:
            values.append(confidence)
    return values


def _high_confidence_error_rate(cases: list[dict[str, Any]]) -> float | None:
    high = []
    for case in cases:
        decision = case.get("evidence_decision") or {}
        confidence = _float_or_none(decision.get("confidence"))
        if confidence is not None and confidence >= 0.80:
            high.append(case)
    if not high:
        return None
    return sum(1 for case in high if not case.get("label_pass")) / len(high)


def _rag_uncertain_or_insufficient_rate(summary: dict[str, Any]) -> float | None:
    statuses = ((summary.get("evidence_decision") or {}).get("statuses") or {})
    total = sum(int(value) for value in statuses.values())
    if total == 0:
        return None
    weak = int(statuses.get("uncertain", 0)) + int(statuses.get("insufficient", 0))
    return weak / total


def _reranker_name(path: Path, data: dict[str, Any]) -> str:
    config = (data.get("summary") or {}).get("config") or {}
    value = str(config.get("CROSS_ENCODER_MODEL") or "")
    if "reranker_off" in path.stem:
        return "reranker-off"
    if "reranker_on" in path.stem:
        return "reranker-on"
    if not value:
        return "reranker-unknown"
    return "reranker-off" if value in {"-", "none", "disabled"} else "reranker-on"


def _conclusions(rows: list[Row]) -> list[str]:
    conclusions = []
    rag_rows = [row for row in rows if row.kind.startswith("rag_")]
    direct_oracle = [row for row in rows if row.kind.startswith("direct_oracle")]
    direct_none = [row for row in rows if row.kind.startswith("direct_none")]

    if rag_rows:
        best_rag = max(rag_rows, key=lambda row: row.accuracy)
        conclusions.append(
            f"Best RAG-style run is `{best_rag.name}` with accuracy {best_rag.accuracy:.3f} "
            f"and maybe recall {_fmt(best_rag.maybe_recall)}."
        )
        high_hit = [row for row in rag_rows if (row.hit1 or 0.0) >= 0.90 and row.accuracy < 0.70]
        if high_hit:
            conclusions.append(
                "At least one run has high source_hit@1 but low label accuracy, supporting the interpretation "
                "that evidence interpretation, not retrieval alone, is the bottleneck."
            )

    if direct_oracle and direct_none:
        best_oracle = max(direct_oracle, key=lambda row: row.accuracy)
        best_none = max(direct_none, key=lambda row: row.accuracy)
        conclusions.append(
            f"Best oracle-evidence direct judge accuracy is {best_oracle.accuracy:.3f}; "
            f"best no-evidence direct judge accuracy is {best_none.accuracy:.3f}. "
            "This estimates the benefit of giving gold evidence to the judge."
        )

    topk_rows = [row for row in rag_rows if "_top" in row.kind]
    if topk_rows:
        grouped: dict[str, list[Row]] = {}
        for row in topk_rows:
            key = row.kind.rsplit("_top", 1)[0]
            grouped.setdefault(key, []).append(row)
        for group_rows in grouped.values():
            if len(group_rows) < 2:
                continue
            best = max(group_rows, key=lambda row: row.accuracy)
            worst = min(group_rows, key=lambda row: row.accuracy)
            if best.accuracy - worst.accuracy >= 0.05:
                conclusions.append(
                    f"Top-k changes matter for `{best.kind}` family: best `{best.name}` "
                    f"beats worst `{worst.name}` by {best.accuracy - worst.accuracy:.3f} accuracy."
                )

    maybe_rows = [row for row in rows if row.maybe_recall is not None]
    if maybe_rows:
        best_maybe = max(maybe_rows, key=lambda row: row.maybe_recall or 0.0)
        conclusions.append(
            f"Best maybe recall is {_fmt(best_maybe.maybe_recall)} in `{best_maybe.name}`; "
            "use this when discussing inconclusive-case handling."
        )

    high_conf_errors = [
        row for row in rows if row.high_conf_error_rate is not None and row.high_conf_error_rate > 0.20
    ]
    if high_conf_errors:
        conclusions.append(
            "Some runs show high-confidence errors, so confidence/calibration should be treated as a safety metric, "
            "not a cosmetic diagnostic."
        )
    return conclusions


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_markdown(path: Path, rows: list[Row], conclusions: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PubMedQA Diagnostic Ablation Summary",
        "",
        "## Main Table",
        "",
        "| Run | Kind | Cases | Acc | Macro F1 | yes R | no R | maybe R | Hit@1 | Hit@3 | Citation | ECE | Hi-conf err | Weak evidence | Latency ms |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row.name}` | "
            f"{row.kind} | "
            f"{row.cases} | "
            f"{row.accuracy:.3f} | "
            f"{_fmt(row.macro_f1)} | "
            f"{_fmt(row.yes_recall)} | "
            f"{_fmt(row.no_recall)} | "
            f"{_fmt(row.maybe_recall)} | "
            f"{_fmt(row.hit1)} | "
            f"{_fmt(row.hit3)} | "
            f"{_fmt(row.citation)} | "
            f"{_fmt(row.ece)} | "
            f"{_fmt(row.high_conf_error_rate)} | "
            f"{_fmt(row.sufficient_false_rate)} | "
            f"{_fmt(row.latency_ms, digits=1)} |"
        )

    lines.extend(["", "## Conclusions", ""])
    if conclusions:
        for item in conclusions:
            lines.append(f"- {item}")
    else:
        lines.append("- No diagnostic conclusions generated; check whether report files were found.")

    lines.extend(["", "## Confusion Matrices", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.name}",
                "",
                "| True \\ Pred | yes | no | maybe | none |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for label in (*LABELS, "none"):
            matrix_row = row.confusion_matrix.get(label, {})
            lines.append(
                f"| {label} | {int(matrix_row.get('yes', 0))} | {int(matrix_row.get('no', 0))} | "
                f"{int(matrix_row.get('maybe', 0))} | {int(matrix_row.get('none', 0))} |"
            )
        lines.append("")

    lines.extend(["", "## Report Files", ""])
    for row in rows:
        lines.append(f"- `{row.name}`: `{row.report}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
