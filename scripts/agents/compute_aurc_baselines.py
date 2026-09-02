"""AURC baselines from a debate JSON (no GPU).

Ranks cases by (1) BioLinkBERT confidence as certainty, (2) debate
uncertainty_score, (3) shuffled random ranks. Writes a small JSON next
to the report. Do not invent numbers — refuses if confidence is missing.

Usage:
  .venv/bin/python scripts/agents/compute_aurc_baselines.py \\
      --debate-json reports/debate/debate_balanced90_ml4h_v1.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.uncertainty import risk_coverage_curve


def _cases(report: dict) -> list[dict]:
    raw = report.get("cases") or report.get("results") or []
    if not isinstance(raw, list):
        raise SystemExit("debate JSON has no cases/results list")
    return [c for c in raw if isinstance(c, dict)]


def _labels(cases: list[dict]) -> tuple[list[str], list[str]]:
    gold = [str(c.get("expected_label") or "") for c in cases]
    base = [
        str(c.get("biolinkbert_label") or c.get("base_label") or c.get("predicted_label") or "")
        for c in cases
    ]
    if any(not g or not b for g, b in zip(gold, base)):
        raise SystemExit("missing expected_label or BioLinkBERT/base label")
    return gold, base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debate-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = json.loads(args.debate_json.read_text(encoding="utf-8"))
    cases = _cases(report)
    gold, base = _labels(cases)

    bert_conf = [c.get("biolinkbert_confidence") for c in cases]
    if any(x is None for x in bert_conf):
        raise SystemExit("biolinkbert_confidence missing — re-run debate with --hint biolinkbert")
    # High confidence = more certain → invert so the curve abstains on LOW confidence
    bert_uncertainty = [1.0 - float(x) for x in bert_conf]

    u_scores = [c.get("uncertainty_score") for c in cases]
    have_u = all(x is not None for x in u_scores)

    rng = random.Random(args.seed)
    random_u = [rng.random() for _ in cases]

    out = {
        "n": len(cases),
        "source": str(args.debate_json),
        "aurc_bert_confidence": risk_coverage_curve(bert_uncertainty, base, gold)["aurc"],
        "aurc_random": risk_coverage_curve(random_u, base, gold)["aurc"],
    }
    if have_u:
        out["aurc_debate_uncertainty"] = risk_coverage_curve(
            [float(x) for x in u_scores], base, gold
        )["aurc"]
    else:
        out["aurc_debate_uncertainty"] = None
        out["note"] = "uncertainty_score missing on some cases"

    dest = args.out or args.debate_json.with_name(args.debate_json.stem + ".aurc_baselines.json")
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
