#!/usr/bin/env python3
"""Faza 2 step 8 (design doc §8): tune TriggerConfig on DEV-90, freeze + hash.

compose_label() is pure and deterministic given a DirectorVerdict + question_type
— it needs no new LLM calls to re-evaluate under a different TriggerConfig. This
script re-derives the label for all 16 TriggerConfig combinations directly from
the DirectorVerdict already cached in a completed L5 report
(evaluate_sld_pubmedqa.py --arm L5 on dev-90), picks the macro-F1-maximizing
combination, and writes it to a frozen config file with a content hash — so a
later PQA-L 500 run can assert it's using the exact config that was tuned,
not a config that silently drifted.

Usage:
  python3 scripts/agents/tune_sld_trigger_config.py \\
    --report reports/sld/sld_L5_dev90.json \\
    --out reports/sld/frozen_trigger_config.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.sld.decision import TriggerConfig, compose_label  # noqa: E402
from app.agents.sld.ledger import DirectorVerdict  # noqa: E402

_FIELDS = (
    "coverage_gap_enabled",
    "mixed_findings_enabled",
    "null_result_enabled",
    "hedged_conclusion_enabled",
)


def _macro_f1(pairs: list[tuple[str, str]]) -> tuple[float, dict[str, float]]:
    """``pairs`` = [(predicted, expected), ...]."""
    labels = ("yes", "no", "maybe")
    per_label: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for p, e in pairs if p == label and e == label)
        fp = sum(1 for p, e in pairs if p == label and e != label)
        fn = sum(1 for p, e in pairs if p != label and e == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label[label] = f1
    return sum(per_label.values()) / len(labels), per_label


def _load_cases(report_path: Path) -> list[dict[str, Any]]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    usable = [c for c in cases if c.get("director_verdict") and c.get("expected_label")]
    skipped = len(cases) - len(usable)
    if skipped:
        print(
            f"Skipping {skipped}/{len(cases)} cases with no director_verdict "
            "(arms without a Director call, e.g. L4, can't be tuned this way)."
        )
    return usable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--report", type=Path, required=True, help="Completed L5 report JSON (dev-90)")
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "reports" / "sld" / "frozen_trigger_config.json"
    )
    args = parser.parse_args()

    cases = _load_cases(args.report)
    if not cases:
        raise SystemExit(f"No usable cases (with director_verdict + expected_label) in {args.report}")

    results = []
    for combo in itertools.product((True, False), repeat=len(_FIELDS)):
        config = TriggerConfig(**dict(zip(_FIELDS, combo)))
        pairs = []
        for case in cases:
            verdict = DirectorVerdict.model_validate(case["director_verdict"])
            label, _rule = compose_label(verdict, case.get("question_type"), config)
            pairs.append((label, case["expected_label"]))
        macro_f1, per_label_f1 = _macro_f1(pairs)
        results.append(
            {
                "config": asdict(config),
                "macro_f1": macro_f1,
                "per_label_f1": per_label_f1,
                "label_accuracy": sum(1 for p, e in pairs if p == e) / len(pairs),
            }
        )

    results.sort(key=lambda r: r["macro_f1"], reverse=True)
    best = results[0]

    baseline_maybe_f1 = _macro_f1([("maybe", c["expected_label"]) for c in cases])
    print(f"n={len(cases)} cases from {args.report}")
    print(f"Naive always-maybe baseline: macro_f1={baseline_maybe_f1[0]:.4f}")
    print("\nAll 16 TriggerConfig combinations, ranked by macro_f1:")
    for r in results:
        flags = "".join("1" if r["config"][f] else "0" for f in _FIELDS)
        print(f"  {flags} macro_f1={r['macro_f1']:.4f} acc={r['label_accuracy']:.4f} per_label={r['per_label_f1']}")

    config_hash = hashlib.sha256(
        json.dumps(best["config"], sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    frozen = {
        "trigger_config": best["config"],
        "config_hash": config_hash,
        "tuned_on": str(args.report),
        "n_cases": len(cases),
        "macro_f1": best["macro_f1"],
        "per_label_f1": best["per_label_f1"],
        "label_accuracy": best["label_accuracy"],
        "naive_always_maybe_macro_f1": baseline_maybe_f1[0],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nBest config (hash {config_hash}): {best['config']}")
    print(f"macro_f1={best['macro_f1']:.4f} (naive always-maybe: {baseline_maybe_f1[0]:.4f})")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
