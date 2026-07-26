"""Aggregate all `maybe`-uncertainty experiment artifacts into one paper dataset.

Ingests:
  - debate eval JSON reports (per-case uncertainty_signals, audit_score, labels)
  - standalone NLI-audit signal JSONLs (reports/debate/signals/*.jsonl)

Produces, under reports/debate/analysis/:
  - maybe_analysis.csv          : one row per (case, method) with every signal + gold
  - signal_auroc.json           : AUROC(maybe vs rest) for every scalar signal, per method
  - risk_coverage.json          : selective-prediction risk-coverage per method
  - analysis_summary.md         : human-readable paper tables

Everything is derived deterministically from on-disk artifacts so the analysis
is fully reproducible. Nothing here calls an LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics as st
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.uncertainty import cost_sensitive_analysis, risk_coverage_curve  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "reports/debate"
SIGNAL_DIR = REPORT_DIR / "signals"
OUT_DIR = REPORT_DIR / "analysis"

# Scalar signals we try to extract per case (name -> where to look).
_SIGNAL_KEYS = (
    "uncertainty_score",
    "audit_score",
    "label_entropy",
    "maybe_fraction",
    "inconclusive_fraction",
    "mean_disagreement_with_mode",
    "flip_rate",
    "panel_uncertainty_conf",
    "bert_is_maybe",
    "semantic_entropy",
)


def _auroc(pos: list[float], neg: list[float]) -> float:
    """AUROC (Mann-Whitney) of scores ranking pos above neg. NaN if degenerate."""
    pos = [p for p in pos if p is not None]
    neg = [n for n in neg if n is not None]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


def _rows_from_debate_json(path: Path, method: str) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    if not isinstance(cases, list):
        return []
    rows: list[dict[str, Any]] = []
    for case in cases:
        signals = case.get("uncertainty_signals") or {}
        row: dict[str, Any] = {
            "method": method,
            "id": case.get("id"),
            "gold": case.get("expected_label"),
            "base_label": case.get("base_label") or case.get("predicted_label"),
            "predicted_label": case.get("predicted_label"),
            "biolinkbert_label": case.get("biolinkbert_label"),
            "uncertainty_score": case.get("uncertainty_score"),
            "audit_score": case.get("audit_score"),
        }
        for key in _SIGNAL_KEYS:
            if key not in row and key in signals:
                row[key] = signals.get(key)
        rows.append(row)
    return rows


def _rows_from_signal_jsonl(path: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rows.append(
            {
                "method": method,
                "id": rec.get("id"),
                "gold": rec.get("expected"),
                "audit_score": rec.get("audit_score"),
                "supported": rec.get("supported"),
                "refuted": rec.get("refuted"),
                "silent": rec.get("silent"),
            }
        )
    return rows


def _signal_auroc(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """AUROC(maybe vs rest) for each scalar signal, grouped by method."""
    out: dict[str, dict[str, float]] = {}
    methods = sorted({r["method"] for r in rows})
    for method in methods:
        mrows = [r for r in rows if r["method"] == method]
        per_signal: dict[str, float] = {}
        for key in _SIGNAL_KEYS + ("supported", "refuted", "silent"):
            pos = [r[key] for r in mrows if r.get("gold") == "maybe" and r.get(key) is not None]
            neg = [r[key] for r in mrows if r.get("gold") not in (None, "maybe") and r.get(key) is not None]
            if pos and neg:
                per_signal[key] = round(_auroc(pos, neg), 4)
        if per_signal:
            out[method] = per_signal
    return out


def _risk_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Risk-coverage per method using uncertainty_score as the abstention signal."""
    out: dict[str, Any] = {}
    methods = sorted({r["method"] for r in rows})
    for method in methods:
        mrows = [
            r
            for r in rows
            if r["method"] == method
            and r.get("uncertainty_score") is not None
            and r.get("base_label")
            and r.get("gold")
        ]
        if len(mrows) < 5:
            continue
        scores = [float(r["uncertainty_score"]) for r in mrows]
        base = [str(r["base_label"]) for r in mrows]
        gold = [str(r["gold"]) for r in mrows]
        try:
            out[method] = risk_coverage_curve(scores, base, gold)
        except Exception as exc:  # pragma: no cover
            out[method] = {"error": str(exc)}
    return out


def _cost_sensitive(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Cost-sensitive selective-answer analysis per method (clinical framing)."""
    out: dict[str, Any] = {}
    methods = sorted({r["method"] for r in rows})
    for method in methods:
        mrows = [
            r
            for r in rows
            if r["method"] == method
            and r.get("uncertainty_score") is not None
            and r.get("base_label")
            and r.get("gold")
        ]
        if len(mrows) < 5:
            continue
        scores = [float(r["uncertainty_score"]) for r in mrows]
        base = [str(r["base_label"]) for r in mrows]
        gold = [str(r["gold"]) for r in mrows]
        try:
            out[method] = cost_sensitive_analysis(scores, base, gold)
        except Exception as exc:  # pragma: no cover
            out[method] = {"error": str(exc)}
    return out


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ["method", "id", "gold", "base_label", "predicted_label", "biolinkbert_label"]
    fields += list(_SIGNAL_KEYS)
    fields += ["supported", "refuted", "silent"]
    seen = set(fields)
    for r in rows:
        for k in r:
            if k not in seen:
                fields.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})


def _md_tables(
    auroc: dict[str, dict[str, float]],
    rc: dict[str, Any],
    cost: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    lines = ["# Maybe-uncertainty signal analysis (auto-generated)\n"]
    n_by_method = {}
    for r in rows:
        n_by_method.setdefault(r["method"], {"n": 0, "maybe": 0})
        n_by_method[r["method"]]["n"] += 1
        if r.get("gold") == "maybe":
            n_by_method[r["method"]]["maybe"] += 1
    lines.append("## Datasets per method\n")
    lines.append("| method | n | maybe | maybe frac |")
    lines.append("|---|---|---|---|")
    for m, c in sorted(n_by_method.items()):
        frac = c["maybe"] / c["n"] if c["n"] else 0.0
        lines.append(f"| {m} | {c['n']} | {c['maybe']} | {frac:.2f} |")

    lines.append("\n## Signal AUROC (maybe vs rest) — 0.5 = no separation\n")
    all_signals = sorted({s for d in auroc.values() for s in d})
    header = "| method | " + " | ".join(all_signals) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(all_signals) + 1))
    for method in sorted(auroc):
        cells = [f"{auroc[method].get(s, float('nan')):.3f}" for s in all_signals]
        lines.append(f"| {method} | " + " | ".join(cells) + " |")

    lines.append("\n## Risk-coverage (selective prediction)\n")
    lines.append("| method | AURC (lower better) | full-coverage acc |")
    lines.append("|---|---|---|")
    for method, curve in sorted(rc.items()):
        if not isinstance(curve, dict) or "aurc" not in curve:
            continue
        pts = curve.get("points") or []
        full_acc = pts[-1].get("selective_accuracy") if pts else float("nan")
        lines.append(f"| {method} | {curve['aurc']:.4f} | {full_acc:.3f} |")

    lines.append(
        "\n## Cost-sensitive selective answering "
        "(cost_wrong=1.0, cost_abstain=0.25)\n"
    )
    lines.append(
        "| method | always-answer | best (uncon.) | abstain% | best (>=50% cov) | abstain% | cost reduction |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for method, c in sorted(cost.items()):
        if not isinstance(c, dict) or "best_cost" not in c:
            continue
        lines.append(
            f"| {method} | {c['always_answer_cost']:.3f} | {c['best_cost']:.3f} | "
            f"{c['best_abstain_rate']*100:.0f}% | {c.get('constrained_best_cost', float('nan')):.3f} | "
            f"{c.get('constrained_best_abstain_rate', 0)*100:.0f}% | "
            f"{c.get('constrained_cost_reduction_pct', 0)*100:.1f}% |"
        )
    lines.append(
        "\n> Note: the unconstrained optimum degenerates to abstain-on-everything when the "
        "uncertainty signal is weak (AUROC~0.5); the >=50%-coverage column is the honest "
        "operating point and shows the signal buys little over always answering."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--debate-json",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Debate eval JSON reports as method_label=path",
    )
    parser.add_argument(
        "--signal-jsonl",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="NLI-audit signal JSONLs as method_label=path",
    )
    parser.add_argument("--auto", action="store_true", help="Auto-discover known artifacts")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    specs_debate = list(args.debate_json)
    specs_signal = list(args.signal_jsonl)
    if args.auto:
        for p in sorted(REPORT_DIR.glob("debate_*.json")):
            specs_debate.append(f"{p.stem}={p}")
        for p in sorted(SIGNAL_DIR.glob("*.jsonl")):
            specs_signal.append(f"{p.stem}={p}")

    for spec in specs_debate:
        label, _, path = spec.partition("=")
        p = Path(path)
        if p.exists():
            rows.extend(_rows_from_debate_json(p, label))
            print(f"debate  {label}: {p}")
    for spec in specs_signal:
        label, _, path = spec.partition("=")
        p = Path(path)
        if p.exists():
            rows.extend(_rows_from_signal_jsonl(p, label))
            print(f"signal  {label}: {p}")

    if not rows:
        raise SystemExit("No artifacts found. Pass --auto or explicit LABEL=PATH specs.")

    auroc = _signal_auroc(rows)
    rc = _risk_coverage(rows)
    cost = _cost_sensitive(rows)

    _write_csv(rows, OUT_DIR / "maybe_analysis.csv")
    (OUT_DIR / "signal_auroc.json").write_text(json.dumps(auroc, indent=2), encoding="utf-8")
    (OUT_DIR / "risk_coverage.json").write_text(json.dumps(rc, indent=2), encoding="utf-8")
    (OUT_DIR / "cost_sensitive.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
    (OUT_DIR / "analysis_summary.md").write_text(_md_tables(auroc, rc, cost, rows), encoding="utf-8")

    print(f"\nRows: {len(rows)} across {len({r['method'] for r in rows})} methods")
    print(f"Wrote {OUT_DIR}/maybe_analysis.csv")
    print(f"Wrote {OUT_DIR}/signal_auroc.json")
    print(f"Wrote {OUT_DIR}/risk_coverage.json")
    print(f"Wrote {OUT_DIR}/cost_sensitive.json")
    print(f"Wrote {OUT_DIR}/analysis_summary.md")


if __name__ == "__main__":
    main()
