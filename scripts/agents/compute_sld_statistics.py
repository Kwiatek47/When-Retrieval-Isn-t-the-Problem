#!/usr/bin/env python3
"""Significance testing for SLD v2 arm comparisons (design doc §9/§10).

Reuses the exact McNemar + paired-bootstrap implementation already used for
the legacy debate arms (`scripts/agents/compute_statistics.py`), applied to
report JSONs from either `evaluate_sld_pubmedqa.py` (SLD arms: L1, L3, L4,
L5, L6, L8) or the legacy `evaluate_debate_pubmedqa.py` /
`evaluate_self_consistency_pubmedqa.py` scripts (L2, L7) — both report
formats are accepted so an SLD arm can be compared directly against a legacy
one without a conversion step.

The two comparisons the design doc calls out as the actual hypothesis tests
(not just descriptive numbers):
  - L5 vs L2 (cost-matched self-consistency) — is the ledger architecture
    better than the same compute spent on more samples of a single call?
  - L5 vs L4 (R1+R2 without the ledger) — does the ledger itself help, at an
    identical round/call count?

Usage:
  python3 scripts/agents/compute_sld_statistics.py \\
    --report reports/sld/sld_L5_pqal500.json --label L5 \\
    --report reports/self_consistency/arm_sc.json --label L2 \\
    --report reports/sld/sld_L4_pqal500.json --label L4 \\
    --n-boot 5000 --seed 47 --out reports/sld/analysis/statistics.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.compute_statistics import _mcnemar, _paired_bootstrap_diff_ci  # noqa: E402


def _load_label_pass(path: Path) -> dict[str, float]:
    """1.0/0.0 per case id, for either report schema.

    SLD reports (evaluate_sld_pubmedqa.py): cases carry ``case_id``,
    ``predicted_label``, ``expected_label``. Legacy debate/self-consistency
    reports: cases carry ``id`` and an already-computed ``label_pass``.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    out: dict[str, float] = {}
    for case in cases:
        case_id = case.get("case_id") or case.get("id")
        if case_id is None:
            continue
        if "label_pass" in case:
            out[case_id] = 1.0 if case["label_pass"] else 0.0
        elif "predicted_label" in case and "expected_label" in case:
            out[case_id] = 1.0 if case["predicted_label"] == case["expected_label"] else 0.0
    return out


def _load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("summary", {}) if isinstance(data, dict) else {}


def compare_arms(
    report_paths: list[Path], labels: list[str], rng: random.Random, n_boot: int
) -> dict[str, Any]:
    per_label_correct = {label: _load_label_pass(path) for path, label in zip(report_paths, labels)}
    pairwise: dict[str, Any] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a_label, b_label = labels[i], labels[j]
            ids = sorted(set(per_label_correct[a_label]) & set(per_label_correct[b_label]))
            if not ids:
                pairwise[f"{a_label}_vs_{b_label}"] = {"error": "no shared case ids between reports"}
                continue
            a = [per_label_correct[a_label][cid] for cid in ids]
            b = [per_label_correct[b_label][cid] for cid in ids]
            result = _mcnemar(a, b)
            result.update(_paired_bootstrap_diff_ci(a, b, rng, n_boot))
            result["n_paired"] = len(ids)
            pairwise[f"{a_label}_vs_{b_label}"] = result
    return pairwise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", action="append", required=True, type=Path, dest="reports")
    parser.add_argument("--label", action="append", required=True, dest="labels")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "sld" / "analysis" / "statistics.json")
    args = parser.parse_args()

    if len(args.reports) != len(args.labels):
        raise SystemExit("--report and --label must be given the same number of times, in matching order")

    rng = random.Random(args.seed)
    pairwise = compare_arms(args.reports, args.labels, rng, args.n_boot)

    summaries = {label: _load_summary(path) for path, label in zip(args.reports, args.labels)}
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "arms": {
            label: {
                k: summary.get(k)
                for k in ("cases", "label_accuracy", "macro_f1", "per_label_f1", "arm", "dataset")
                if k in summary
            }
            for label, summary in summaries.items()
        },
        "pairwise_label_pass": pairwise,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Arm summaries ===")
    for label, summary in result["arms"].items():
        acc = summary.get("label_accuracy")
        f1 = summary.get("macro_f1")
        n = summary.get("cases")
        print(f"{label:6s} n={n} accuracy={acc} macro_f1={f1}")

    print("\n=== Pairwise McNemar + bootstrap accuracy-gap CI ===")
    for key, stats in pairwise.items():
        if "error" in stats:
            print(f"{key}: {stats['error']}")
            continue
        sig = "*" if stats["p_value"] < 0.05 else " "
        print(
            f"{key:20s} diff={stats['diff']:+.4f} "
            f"[{stats['ci_low']:+.4f},{stats['ci_high']:+.4f}] "
            f"p={stats['p_value']:.5f}{sig} "
            f"discordant={stats['n_discordant']} n={stats['n_paired']}"
        )

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
