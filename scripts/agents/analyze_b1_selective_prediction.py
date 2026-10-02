"""BRAKI B1: selective prediction on the 500 PQA-L test questions, recomputed from the runs.

The draft's selective-prediction numbers (always-answer cost 0.246, random AURC 0.231,
BioLinkBERT AURC 0.150, cost 0.1975) have no source file and do not describe BioLinkBERT,
whose error rate is 137/500 = 0.274. This script recomputes the table and the curve from
the per-question predictions, for the same signals as H4.

Definitions (standard selective classification):
  - a question is answered when its uncertainty is at or below a threshold; risk is the
    error rate among answered questions, and an answered gold-maybe question is an error;
  - AURC is the mean risk over all coverages, questions sorted from least to most
    uncertain; tied uncertainties share their errors equally, so ties cannot flatter a signal;
  - the random-order reference is the system's own error rate at every coverage;
  - cost = wrong answers + ``abstain_cost`` per abstention, divided by the number of questions.

Operating point: the threshold is chosen out-of-fold (5 folds, seed 47) as the one that
minimises cost on the other four folds while answering at least half of their questions.
Bootstrap CIs resample questions with their out-of-fold decisions held fixed.

Descriptive numbers for the paper, not a hypothesis test.
Output: ``reports/debate/analysis/b1_selective_prediction.json``. No LLM calls.
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

from scripts.agents.analyze_h2_human_ceiling import PQAL, bootstrap_ci  # noqa: E402
from scripts.agents.analyze_h4_abstention import SIGNALS, load_signal  # noqa: E402

OUT = PROJECT_ROOT / "reports/debate/analysis/b1_selective_prediction.json"
ABSTAIN_COST = 0.25
MIN_COVERAGE = 0.5
COST_SENSITIVITY = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)
CURVE_COVERAGES = tuple(round(0.05 * i, 2) for i in range(1, 21))


def tie_shared_errors(unc: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """Errors in order of increasing uncertainty; tied questions each carry their group's mean."""
    order = np.argsort(unc, kind="mergesort")
    sorted_unc, sorted_wrong = unc[order], wrong[order].astype(float)
    _, inverse = np.unique(sorted_unc, return_inverse=True)
    group_mean = np.bincount(inverse, weights=sorted_wrong) / np.bincount(inverse)
    return group_mean[inverse]


def risk_by_coverage(unc: np.ndarray, wrong: np.ndarray) -> np.ndarray:
    """Risk after answering the k least uncertain questions, for k = 1..n."""
    errors = tie_shared_errors(unc, wrong)
    return np.cumsum(errors) / np.arange(1, len(errors) + 1)


def aurc(unc: np.ndarray, wrong: np.ndarray) -> float:
    return float(risk_by_coverage(unc, wrong).mean())


def oracle_aurc(wrong: np.ndarray) -> float:
    """AURC of a signal that ranks every error after every correct answer."""
    return aurc(wrong.astype(float), wrong)


def cost(wrong: np.ndarray, abstain: np.ndarray, abstain_cost: float) -> float:
    return float(((wrong & ~abstain).sum() + abstain_cost * abstain.sum()) / len(wrong))


def best_threshold(unc: np.ndarray, wrong: np.ndarray, abstain_cost: float, min_coverage: float) -> float:
    """Threshold (answer when unc <= t) with the lowest cost at coverage >= ``min_coverage``."""
    best_t, best_cost = float("inf"), cost(wrong, np.zeros(len(wrong), dtype=bool), abstain_cost)
    for t in np.unique(unc):
        abstain = unc > t
        if 1.0 - abstain.mean() < min_coverage:
            continue
        c = cost(wrong, abstain, abstain_cost)
        if c < best_cost:
            best_t, best_cost = float(t), c
    return best_t


def out_of_fold_abstentions(
    unc: np.ndarray, wrong: np.ndarray, abstain_cost: float, rng: np.random.Generator, folds: int = 5
) -> np.ndarray:
    """Abstention decision for every question, from a threshold fitted on the other folds."""
    fold_of = rng.permutation(len(unc)) % folds
    abstain = np.zeros(len(unc), dtype=bool)
    for fold in range(folds):
        held_out = fold_of == fold
        t = best_threshold(unc[~held_out], wrong[~held_out], abstain_cost, MIN_COVERAGE)
        abstain[held_out] = unc[held_out] > t
    return abstain


def cost_gain(wrong: np.ndarray, abstain: np.ndarray, abstain_cost: float = ABSTAIN_COST) -> float:
    """Always-answer cost minus the cost with abstentions (positive = abstaining helps)."""
    return float(wrong.mean()) - cost(wrong, abstain, abstain_cost)


def analyze_signal(frame: dict, seed: int, n_boot: int) -> dict:
    unc, wrong = frame["unc"], frame["pred"] != frame["gold"]
    rng = np.random.default_rng(seed)
    risks = risk_by_coverage(unc, wrong)
    n = len(unc)
    abstain = out_of_fold_abstentions(unc, wrong, ABSTAIN_COST, np.random.default_rng(seed))
    sensitivity = {}
    for c in COST_SENSITIVITY:
        a = out_of_fold_abstentions(unc, wrong, c, np.random.default_rng(seed))
        sensitivity[f"{c:.2f}"] = {
            "cost": round(cost(wrong, a, c), 4),
            "abstention_rate": round(float(a.mean()), 4),
            "gain_vs_always_answer": round(float(wrong.mean()) - cost(wrong, a, c), 4),
        }
    return {
        "n": n,
        "errors": int(wrong.sum()),
        "always_answer_cost": round(float(wrong.mean()), 4),
        "aurc": bootstrap_ci(aurc, (unc, wrong), rng, n_boot),
        "aurc_random_order": round(float(wrong.mean()), 4),
        "aurc_oracle": round(oracle_aurc(wrong), 4),
        "operating_point": {
            "abstain_cost": ABSTAIN_COST,
            "min_coverage": MIN_COVERAGE,
            "cost": round(cost(wrong, abstain, ABSTAIN_COST), 4),
            "abstention_rate": round(float(abstain.mean()), 4),
            "selective_accuracy": round(float((~wrong)[~abstain].mean()), 4),
            "gold_maybe_abstained": int((abstain & (frame["gold"] == "maybe")).sum()),
            "gain_vs_always_answer": bootstrap_ci(cost_gain, (wrong, abstain), rng, n_boot),
        },
        "abstain_cost_sensitivity": sensitivity,
        "risk_coverage_curve": [
            {"coverage": c, "risk": round(float(risks[max(int(round(c * n)), 1) - 1]), 4)} for c in CURVE_COVERAGES
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "signals": {name: analyze_signal(load_signal(name, data), args.seed, args.n_boot) for name in SIGNALS},
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== B1 selective prediction, PQA-L test 500 ===")
    for name, s in result["signals"].items():
        a, op = s["aurc"], s["operating_point"]
        g = op["gain_vs_always_answer"]
        print(
            f"{name}: always-answer {s['always_answer_cost']:.3f} | AURC {a['value']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] "
            f"(random {s['aurc_random_order']:.3f}, oracle {s['aurc_oracle']:.3f}) | cost {op['cost']:.3f} at "
            f"{op['abstention_rate']:.1%} abstained, gain {g['value']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}]"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
