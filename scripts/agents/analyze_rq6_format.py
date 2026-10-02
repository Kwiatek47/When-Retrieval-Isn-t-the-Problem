"""RQ6: does gold `maybe` (or annotator disagreement) depend on abstract/question format?

Logistic regression on features a model also sees (question + context; never the conclusion), on all 1000
PQA-L questions of ``pqal_label_table.jsonl``. Odds ratios are per 1 SD with bootstrap CIs; predictive value is
5-fold cross-validated AUROC (repeated), reported with the best single feature as reference.
Per-system prediction analysis (RQ6 second bullet) needs the system reports, which are not in this repo.

Output: ``reports/debate/analysis/rq6_format.json``. No LLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.analyze_h1_hedging import TABLE, auroc, load_rows  # noqa: E402

OUT = TABLE.parent / "rq6_format.json"

FEATURES = {
    "log_context_tokens": lambda r: np.log(r["context_tokens"]),
    "n_sections": lambda r: r["n_sections"],
    "log1p_numbers": lambda r: np.log1p(r["context_number_count"]),
    "has_pvalue": lambda r: float(r["context_pvalue_count"] > 0),
    "hedge_density_context": lambda r: r["context_hedge_density"],
    "log_question_tokens": lambda r: np.log(r["question_tokens"]),
}
TARGETS = {
    "gold_maybe": lambda r: r["gold_is_maybe"],
    "annotators_disagree": lambda r: not r["annotators_agree"],
}


def design(rows: list[dict]) -> np.ndarray:
    x = np.array([[f(r) for f in FEATURES.values()] for r in rows], dtype=float)
    return (x - x.mean(0)) / x.std(0)


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1.0, iters: int = 50) -> np.ndarray:
    """Newton/IRLS with L2 on slopes; returns [intercept, *slopes]."""
    a = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(a.shape[1])
    pen = ridge * np.eye(a.shape[1])
    pen[0, 0] = 0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a @ w))
        step = np.linalg.solve(a.T @ (a * (p * (1 - p))[:, None]) + pen, a.T @ (y - p) - pen @ w)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(x), 1)), x]) @ w


def cv_auroc(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, repeats: int = 10, folds: int = 5) -> float:
    scores = []
    for _ in range(repeats):
        fold = rng.permutation(len(y)) % folds
        oof = np.empty(len(y))
        for k in range(folds):
            oof[fold == k] = predict(fit_logistic(x[fold != k], y[fold != k]), x[fold == k])
        scores.append(auroc(oof, y.astype(bool)))
    return float(np.mean(scores))


def analyse(rows: list[dict], target: str, rng: np.random.Generator, n_boot: int) -> dict:
    x = design(rows)
    y = np.array([TARGETS[target](r) for r in rows], dtype=float)
    w = fit_logistic(x, y)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(fit_logistic(x[i], y[i])[1:])
    lo, hi = np.percentile(np.exp(boots), [2.5, 97.5], axis=0)
    names = list(FEATURES)
    single = {n: auroc(x[:, j], y.astype(bool)) for j, n in enumerate(names)}
    return {
        "n": len(y),
        "n_positive": int(y.sum()),
        "odds_ratio_per_sd": {
            n: {"or": round(float(np.exp(w[j + 1])), 3), "ci_low": round(float(lo[j]), 3), "ci_high": round(float(hi[j]), 3)}
            for j, n in enumerate(names)
        },
        "cv_auroc_all_features": round(cv_auroc(x, y, rng), 4),
        "single_feature_auroc": {n: round(v, 4) for n, v in single.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    rows = load_rows(args.table)
    rng = np.random.default_rng(args.seed)
    result = {"seed": args.seed, "n_boot": args.n_boot, **{t: analyse(rows, t, rng, args.n_boot) for t in TARGETS}}
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for t in TARGETS:
        print(t, "CV AUROC", result[t]["cv_auroc_all_features"])


if __name__ == "__main__":
    main()
