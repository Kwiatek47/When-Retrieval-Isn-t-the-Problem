"""Build the architecture-comparison tables from per-arm debate reports.

Reads the reports written by `run_paper_b_arms.sh` and emits one markdown file
with the tables the comparison rests on:

  1. Accuracy and cost per arm, with bootstrap CIs and per-1k-token efficiency.
  2. Accuracy split by yes / no / maybe.
  3. Where each arm helps or hurts relative to the round-robin baseline, case by
     case - a net accuracy tie can hide a large number of offsetting flips.
  4. Discussion dynamics: subversion, rescue, adoption, and the identity-bias
     coefficient between the identified and anonymized arms.

Usage:
    python scripts/agents/build_paper_b_tables.py --report-dir reports/paper_b --suffix trial
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.metrics import identity_bias_coefficient

# Order matters: each rung adds exactly one mechanism to the one above it, which
# is what lets a difference be attributed to that mechanism.
ARMS: tuple[tuple[str, str], ...] = (
    ("round_robin", "A0 round-robin, personas, vote"),
    ("neutral", "A1 round-robin, neutral agents, vote"),
    ("supervisor", "B  + supervisor closure"),
    ("anonymized", "C  + anonymized transcript"),
    ("asymmetric", "D  + partitioned evidence (InfoNav)"),
)

BASELINE_ARM = "round_robin"
IDENTIFIED_ARM = "supervisor"
ANONYMIZED_ARM = "anonymized"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _boot_ci_mean(values: list[float], rng: random.Random, n_boot: int) -> dict[str, Any]:
    """Bootstrap 95% CI for a mean; mirrors compute_statistics._boot_ci_mean."""
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    n = len(values)
    point = sum(values) / n
    boots = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        boots.append(total / n)
    return {
        "mean": round(point, 4),
        "ci_low": round(_percentile(boots, 0.025), 4),
        "ci_high": round(_percentile(boots, 0.975), 4),
        "n": n,
    }


def _load_arms(report_dir: Path, suffix: str) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for arm, _ in ARMS:
        path = report_dir / f"paper_b_{arm}_{suffix}.json"
        if path.exists():
            loaded[arm] = json.loads(path.read_text(encoding="utf-8"))
        else:
            print(f"missing (skipped): {path}")
    return loaded


def _cases_by_id(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {case["id"]: case for case in report.get("cases", [])}


def _fmt(value: Any, spec: str = ".3f") -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "n/a"
    return format(value, spec)


def _check_comparability(arms: dict[str, dict[str, Any]]) -> list[str]:
    """Flag anything that would make the arms not comparable.

    A table of numbers produced from different data or different models is worse
    than no table, because it looks authoritative. These warnings go into the
    report itself rather than only to stdout.
    """
    warnings: list[str] = []
    configs = {arm: report["summary"].get("run_config", {}) for arm, report in arms.items()}
    for key in ("dataset_sha256", "corpus_sha256", "model", "num_predict", "num_ctx"):
        values = {arm: config.get(key) for arm, config in configs.items() if key in config}
        distinct = set(values.values())
        if len(distinct) > 1:
            warnings.append(f"`{key}` differs across arms: {values}")

    id_sets = {arm: set(_cases_by_id(report)) for arm, report in arms.items()}
    if len({frozenset(ids) for ids in id_sets.values()}) > 1:
        sizes = {arm: len(ids) for arm, ids in id_sets.items()}
        warnings.append(f"arms were not evaluated on the same case ids: {sizes}")
    return warnings


def _table_accuracy_and_cost(arms: dict[str, dict[str, Any]], rng: random.Random, n_boot: int) -> list[str]:
    lines = [
        "## Table 1 - accuracy and cost",
        "",
        "| Arm | Accuracy [95% CI] | LLM calls/case | Tokens/case | Acc per 1k tokens |",
        "|---|---|---:|---:|---:|",
    ]
    for arm, title in ARMS:
        report = arms.get(arm)
        if not report:
            continue
        summary = report["summary"]
        correct = [1.0 if case.get("label_pass") else 0.0 for case in report["cases"]]
        ci = _boot_ci_mean(correct, rng, n_boot)
        estimated = " (est.)" if summary.get("tokens_estimated") else ""
        lines.append(
            f"| {title} | {_fmt(ci['mean'])} [{_fmt(ci['ci_low'])}, {_fmt(ci['ci_high'])}] "
            f"| {_fmt(summary.get('mean_llm_calls'), '.2f')} "
            f"| {_fmt(summary.get('mean_total_tokens'), '.0f')}{estimated} "
            f"| {_fmt(summary.get('accuracy_per_1k_tokens'), '.4f')} |"
        )
    lines.append("")
    return lines


def _table_per_label(arms: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Table 2 - accuracy by label",
        "",
        "| Arm | yes | no | maybe | predicted maybe |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, title in ARMS:
        report = arms.get(arm)
        if not report:
            continue
        summary = report["summary"]
        per_label = summary.get("per_label_accuracy", {})
        predicted = summary.get("predicted_label_counts", {})
        lines.append(
            f"| {title} | {_fmt(per_label.get('yes'))} | {_fmt(per_label.get('no'))} "
            f"| {_fmt(per_label.get('maybe'))} | {predicted.get('maybe', 0)} |"
        )
    lines.append("")
    return lines


def _table_helps_hurts(arms: dict[str, dict[str, Any]]) -> list[str]:
    """Case-level wins and losses against the baseline arm."""
    baseline = arms.get(BASELINE_ARM)
    lines = [
        f"## Table 3 - helps / hurts vs `{BASELINE_ARM}`",
        "",
    ]
    if not baseline:
        lines.extend([f"Baseline arm `{BASELINE_ARM}` is missing; table skipped.", ""])
        return lines

    base_cases = _cases_by_id(baseline)
    lines.extend(
        [
            "| Arm | fixed | broke | net | both right | both wrong |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, title in ARMS:
        if arm == BASELINE_ARM:
            continue
        report = arms.get(arm)
        if not report:
            continue
        fixed = broke = both_right = both_wrong = 0
        for case in report["cases"]:
            base = base_cases.get(case["id"])
            if base is None:
                continue
            was = bool(base.get("label_pass"))
            now = bool(case.get("label_pass"))
            if now and not was:
                fixed += 1
            elif was and not now:
                broke += 1
            elif was:
                both_right += 1
            else:
                both_wrong += 1
        lines.append(
            f"| {title} | {fixed} | {broke} | {fixed - broke:+d} | {both_right} | {both_wrong} |"
        )
    lines.append("")
    return lines


def _table_dynamics(arms: dict[str, dict[str, Any]]) -> list[str]:
    lines = [
        "## Table 4 - discussion dynamics",
        "",
        "Subversion and rescue are reported together on purpose: subversion alone "
        "cannot tell a discussion that destroys correct answers from one that is "
        "merely churning.",
        "",
        "| Arm | subversion | rescue | net | adoption |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, title in ARMS:
        report = arms.get(arm)
        if not report:
            continue
        summary = report["summary"]
        lines.append(
            f"| {title} | {_fmt(summary.get('subversion_rate'))} "
            f"| {_fmt(summary.get('rescue_rate'))} "
            f"| {_fmt(summary.get('net_flip_rate'))} "
            f"| {_fmt(summary.get('adoption_rate'))} |"
        )
    lines.append("")

    identified = arms.get(IDENTIFIED_ARM)
    anonymized = arms.get(ANONYMIZED_ARM)
    if identified and anonymized:
        left = identified["summary"].get("adoption_rate")
        right = anonymized["summary"].get("adoption_rate")
        if left is not None and right is not None:
            ibc = identity_bias_coefficient(left, right)
            lines.extend(
                [
                    "### Identity bias coefficient",
                    "",
                    f"- adoption with identities visible (`{IDENTIFIED_ARM}`): {_fmt(left)}",
                    f"- adoption with identities hidden (`{ANONYMIZED_ARM}`): {_fmt(right)}",
                    f"- **IBC (identified - anonymized): {_fmt(ibc)}**",
                    "",
                    "Positive means agents conformed more when they could see who was "
                    "speaking. The two arms differ only in anonymization, so the "
                    "difference is attributable to it.",
                    "",
                ]
            )
    return lines


def build_report(arms: dict[str, dict[str, Any]], *, seed: int, n_boot: int) -> str:
    rng = random.Random(seed)
    any_report = next(iter(arms.values()))
    summary = any_report["summary"]

    lines = [
        "# Debate architecture comparison (paper B)",
        "",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Cases per arm: {summary.get('cases')}",
        f"- Backend: `{summary.get('backend')}`",
        f"- Bootstrap: n={n_boot}, seed={seed}",
        "",
    ]

    warnings = _check_comparability(arms)
    if warnings:
        lines.extend(["> **Arms are not directly comparable:**", ""])
        lines.extend(f"> - {warning}" for warning in warnings)
        lines.append("")

    lines.extend(["## Run configuration", "", "```json"])
    lines.append(json.dumps({arm: r["summary"].get("run_config", {}) for arm, r in arms.items()}, indent=1))
    lines.extend(["```", ""])

    lines.extend(_table_accuracy_and_cost(arms, rng, n_boot))
    lines.extend(_table_per_label(arms))
    lines.extend(_table_helps_hurts(arms))
    lines.extend(_table_dynamics(arms))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports" / "paper_b")
    parser.add_argument("--suffix", type=str, default="trial", help="trial | full | mock")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    arms = _load_arms(args.report_dir, args.suffix)
    if not arms:
        raise SystemExit(f"No arm reports found in {args.report_dir} for suffix '{args.suffix}'.")

    out = args.out or (args.report_dir / f"paper_b_tabele_{args.suffix}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(arms, seed=args.seed, n_boot=args.n_boot), encoding="utf-8")
    print(f"Wrote {out} ({len(arms)}/{len(ARMS)} arms)")


if __name__ == "__main__":
    main()
