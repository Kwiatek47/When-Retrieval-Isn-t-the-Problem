"""Hypothesis H4: abstention helps by avoiding likely errors, not by detecting ``maybe``.

Each system has an uncertainty signal per question: BioLinkBERT 1 - confidence,
self-consistency 1 - sample agreement, the debate's panel split and its u-score.
Abstaining on the most uncertain questions lowers risk. H4 says this works because the
signal flags ordinary mistakes, while among mistakes it does not single out the
questions whose gold label is ``maybe``.

Comparing AUROC(error) with AUROC(maybe) on all questions would be circular: BioLinkBERT
is wrong on 51 of the 55 gold-maybe questions, so the two targets nearly coincide.
The test therefore separates them:

  A = AUROC of the signal for *error* among questions whose gold label is yes or no
      (does it flag ordinary mistakes?);
  B = AUROC of the signal for *gold maybe* among the system's errors
      (among mistakes, does it single out maybe?).

Primary test, fixed before A or B were computed:
  - system and signal: BioLinkBERT, 1 - confidence (the paper's abstention signal);
  - statistic: D = A - B, bootstrap over the 500 test questions (95% CI).
  H4 is supported if the CI of A lies above 0.5 AND the CI of D lies above 0;
  refuted if the CI of D lies at or below 0 (the signal singles out maybe among errors at
  least as well as it flags ordinary mistakes); inconclusive otherwise.

Secondary: A, B and D for self-consistency and the hinted debate (panel split, u-score);
AUROC for error and for gold maybe on all 500 questions; AUROC for questions on which the
annotator without the conclusion departed from gold (H2's disputed set); and, at abstention
rates of 10-50%, how many abstained questions are gold maybe and how the avoided errors
split between maybe and yes/no questions.

Known before registration: from the paper draft, BioLinkBERT 1 - confidence has AUROC
0.637 for gold maybe on all 500, 39 of its 51 maybe errors have confidence >= 0.90, and
21 of 55 maybe questions fall in the abstained set at the draft's threshold; from
``ANALYSIS_research_findings.md``, error AUROC about 0.62 (BioLinkBERT), 0.70 (panel
split), 0.62 (self-consistency) on all 500. Not known: A and B.

Output: ``reports/debate/analysis/h4_abstention.json``. No LLM calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.analyze_h1_hedging import auroc  # noqa: E402
from scripts.agents.analyze_h2_human_ceiling import DEBATE, PQAL, bootstrap_ci  # noqa: E402

OUT = DEBATE / "analysis/h4_abstention.json"
_ID_PREFIX = "pubmedqa-official-"
RATES = (0.1, 0.2, 0.3, 0.4, 0.5)
PRIMARY = "biolinkbert_1_minus_confidence"


def _panel_split(case: dict) -> float:
    return 1.0 - max(case["vote_share"].values())


# name -> (report file, label field, uncertainty of one case)
SIGNALS = {
    "biolinkbert_1_minus_confidence": (
        "debate7b_dissent_pqal500_v1.json",
        "biolinkbert_label",
        lambda case: 1.0 - case["biolinkbert_confidence"],
    ),
    "self_consistency_1_minus_agreement": (
        "selfconsistency_qwen3_8b_k4_pqal500.json",
        "predicted_label",
        lambda case: 1.0 - case["agreement"],
    ),
    "debate_dissent_panel_split": ("debate7b_dissent_pqal500_v1.json", "predicted_label", _panel_split),
    "debate_dissent_u_score": (
        "debate7b_dissent_pqal500_v1.json",
        "predicted_label",
        lambda case: case["uncertainty_score"],
    ),
}


def load_signal(name: str, data: dict, debate_dir: Path = DEBATE) -> dict[str, np.ndarray]:
    """Aligned arrays (sorted by PMID) for one system: prediction, uncertainty, gold, RR."""
    file, label_field, uncertainty = SIGNALS[name]
    cases = json.loads((debate_dir / file).read_text(encoding="utf-8"))["cases"]
    cases = sorted(cases, key=lambda c: c["id"])
    pmids = [c["id"][len(_ID_PREFIX):] for c in cases]
    return {
        "pred": np.array([c[label_field] for c in cases]),
        "unc": np.array([float(uncertainty(c)) for c in cases]),
        "gold": np.array([data[p]["final_decision"] for p in pmids]),
        "rr": np.array([data[p]["reasoning_required_pred"] for p in pmids]),
    }


def auroc_error_on_yes_no(unc: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> float:
    """A: does the signal flag errors among questions whose gold label is yes or no?"""
    mask = gold != "maybe"
    return auroc(unc[mask], (pred != gold)[mask])


def auroc_maybe_among_errors(unc: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> float:
    """B: among the system's errors, does the signal single out gold-maybe questions?"""
    mask = pred != gold
    return auroc(unc[mask], (gold == "maybe")[mask])


def separation(unc: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> float:
    """D = A - B."""
    return auroc_error_on_yes_no(unc, pred, gold) - auroc_maybe_among_errors(unc, pred, gold)


def auroc_error(unc: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> float:
    return auroc(unc, pred != gold)


def auroc_gold_maybe(unc: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> float:
    return auroc(unc, gold == "maybe")


def verdict(a: dict, d: dict) -> str:
    if a["ci_low"] > 0.5 and d["ci_low"] > 0:
        return "supported"
    if d["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def abstention_table(unc: np.ndarray, pred: np.ndarray, gold: np.ndarray) -> list[dict]:
    """At each abstention rate: who is abstained on and which errors are avoided.

    The most uncertain questions are abstained first; ties keep PMID order, so signals with
    few distinct values (self-consistency has four) cut a tie group at an arbitrary point.
    """
    order = np.argsort(-unc, kind="mergesort")
    wrong = pred != gold
    maybe = gold == "maybe"
    rows = []
    for rate in RATES:
        k = int(round(rate * len(unc)))
        out = np.zeros(len(unc), dtype=bool)
        out[order[:k]] = True
        kept = ~out
        rows.append(
            {
                "abstention_rate": rate,
                "abstained": k,
                "abstained_gold_maybe": int((out & maybe).sum()),
                "expected_gold_maybe_if_random": round(k * float(maybe.mean()), 1),
                "errors_avoided_on_maybe": int((out & wrong & maybe).sum()),
                "errors_avoided_on_yes_no": int((out & wrong & ~maybe).sum()),
                "correct_answers_lost": int((out & ~wrong).sum()),
                "selective_accuracy": round(float((~wrong)[kept].mean()), 4),
            }
        )
    return rows


def analyze_signal(frame: dict, rng: np.random.Generator, n_boot: int) -> dict:
    unc, pred, gold, rr = frame["unc"], frame["pred"], frame["gold"], frame["rr"]
    arrays = (unc, pred, gold)
    a = bootstrap_ci(auroc_error_on_yes_no, arrays, rng, n_boot)
    b = bootstrap_ci(auroc_maybe_among_errors, arrays, rng, n_boot)
    d = bootstrap_ci(separation, arrays, rng, n_boot)
    wrong = pred != gold
    return {
        "n_errors": int(wrong.sum()),
        "n_errors_on_gold_maybe": int((wrong & (gold == "maybe")).sum()),
        "A_error_among_gold_yes_no": a,
        "B_maybe_among_errors": b,
        "D_separation": {**d, "verdict": verdict(a, d)},
        "auroc_error_all": bootstrap_ci(auroc_error, arrays, rng, n_boot),
        "auroc_gold_maybe_all": bootstrap_ci(auroc_gold_maybe, arrays, rng, n_boot),
        "auroc_disputed_by_rr": bootstrap_ci(lambda u, r, g: auroc(u, r != g), (unc, rr, gold), rng, n_boot),
        "abstention": abstention_table(unc, pred, gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "primary_signal": PRIMARY,
        "signals": {name: analyze_signal(load_signal(name, data), rng, args.n_boot) for name in SIGNALS},
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== H4 ===")
    for name, s in result["signals"].items():
        a, b, d = s["A_error_among_gold_yes_no"], s["B_maybe_among_errors"], s["D_separation"]
        tag = "PRIMARY " if name == PRIMARY else ""
        print(
            f"{tag}{name}: A {a['value']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]  "
            f"B {b['value']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]  "
            f"D {d['value']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] -> {d['verdict']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
