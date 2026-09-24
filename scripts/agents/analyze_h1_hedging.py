"""Hypothesis H1: gold ``maybe`` in PubMedQA tracks hedging in the conclusion the model never sees.

Reads ``reports/debate/analysis/pqal_label_table.jsonl`` (``build_pqal_label_table.py``).
No LLM calls; deterministic bootstrap.

Primary test, fixed before looking at the results:
  - questions: every PQA-L question whose context has a RESULTS section;
  - target: ``final_decision == maybe``;
  - A = core hedge density in the conclusion, B = core hedge density in the RESULTS section;
  - statistic: dAUROC = AUROC(A) - AUROC(B), paired bootstrap over questions (95% CI).
  H1 is supported if the CI of AUROC(A) lies above 0.5 AND the CI of dAUROC lies above 0.
  H1 fails if the CI of dAUROC lies at or below 0 (the results predict ``maybe`` as well
  as the conclusion does).

Secondary analyses: the same scores against each annotator's label (annotator 1 saw the
conclusion, annotator 2 did not), the whole context instead of RESULTS, the extended
lexicon, null-result cues, both data splits, and hedge prevalence per annotator pattern.

Output: ``reports/debate/analysis/h1_hedging.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
TABLE = ANALYSIS / "pqal_label_table.jsonl"
OUT = ANALYSIS / "h1_hedging.json"

PRIMARY_A = "hedge_density_conclusion"
PRIMARY_B = "hedge_density_results"

_SCORES = (
    "hedge_density_conclusion",
    "hedge_density_results",
    "hedge_density_context",
    "hedge_ext_density_conclusion",
    "hedge_ext_density_results",
    "null_density_conclusion",
    "null_density_results",
)
_TARGETS = {"final": "final", "annotator_with_conclusion": "rf", "annotator_without_conclusion": "rr"}


def rankdata(values: np.ndarray) -> np.ndarray:
    """1-based ranks; tied values share their average rank."""
    order = np.argsort(values, kind="mergesort")
    _, first, counts = np.unique(values[order], return_index=True, return_counts=True)
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return ranks


def auroc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney AUROC of ``scores`` for the boolean ``positive``; ties count 0.5."""
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _ci(boots: list[float]) -> tuple[float, float]:
    arr = np.asarray([b for b in boots if b == b])
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def bootstrap(
    pairs: list[tuple[np.ndarray, np.ndarray]], rng: np.random.Generator, n_boot: int
) -> list[list[float]]:
    """Resample questions once per iteration and score every (scores, positive) pair on it."""
    n = len(pairs[0][0])
    out: list[list[float]] = [[] for _ in pairs]
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for k, (scores, positive) in enumerate(pairs):
            out[k].append(auroc(scores[idx], positive[idx]))
    return out


def _summary(point: float, boots: list[float]) -> dict:
    lo, hi = _ci(boots)
    return {"auroc": round(point, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def _diff_summary(a_point: float, b_point: float, a_boots: list[float], b_boots: list[float]) -> dict:
    diffs = [a - b for a, b in zip(a_boots, b_boots)]
    lo, hi = _ci(diffs)
    return {"d_auroc": round(a_point - b_point, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}


def _column(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in rows], dtype=float)


def _is_maybe(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] == "maybe" for r in rows], dtype=bool)


def primary_test(
    rows: list[dict],
    rng: np.random.Generator,
    n_boot: int,
    a_key: str = PRIMARY_A,
    b_key: str = PRIMARY_B,
) -> dict:
    frame = [r for r in rows if r["has_results_section"]]
    y = _is_maybe(frame, "final")
    a, b = _column(frame, a_key), _column(frame, b_key)
    a_boots, b_boots = bootstrap([(a, y), (b, y)], rng, n_boot)
    a_point, b_point = auroc(a, y), auroc(b, y)
    result = {
        "n": len(frame),
        "n_maybe": int(y.sum()),
        "conclusion": _summary(a_point, a_boots),
        "results_section": _summary(b_point, b_boots),
        "difference": _diff_summary(a_point, b_point, a_boots, b_boots),
    }
    supported = result["conclusion"]["ci_low"] > 0.5 and result["difference"]["ci_low"] > 0
    refuted = result["difference"]["ci_high"] <= 0
    result["verdict"] = "supported" if supported else "refuted" if refuted else "inconclusive"
    return result


def score_matrix(rows: list[dict], rng: np.random.Generator, n_boot: int) -> dict:
    frame = [r for r in rows if r["has_results_section"]]
    keys = [(s, t) for s in _SCORES for t in _TARGETS]
    pairs = [(_column(frame, s), _is_maybe(frame, _TARGETS[t])) for s, t in keys]
    boots = bootstrap(pairs, rng, n_boot)
    matrix: dict = {s: {} for s in _SCORES}
    for (s, t), pair, bs in zip(keys, pairs, boots):
        matrix[s][t] = _summary(auroc(*pair), bs)
    return {"n": len(frame), "auroc": matrix}


def annotator_contrast(
    rows: list[dict],
    rng: np.random.Generator,
    n_boot: int,
    scores: tuple[str, ...] = (PRIMARY_A, PRIMARY_B),
) -> dict:
    """Does a score track annotator 1's maybe (saw the conclusion) more than annotator 2's?"""
    frame = [r for r in rows if r["has_results_section"]]
    rf, rr = _is_maybe(frame, "rf"), _is_maybe(frame, "rr")
    out = {}
    for score in scores:
        x = _column(frame, score)
        rf_boots, rr_boots = bootstrap([(x, rf), (x, rr)], rng, n_boot)
        out[score] = _diff_summary(auroc(x, rf), auroc(x, rr), rf_boots, rr_boots)
    return out


def by_split(
    rows: list[dict],
    rng: np.random.Generator,
    n_boot: int,
    a_key: str = PRIMARY_A,
    b_key: str = PRIMARY_B,
) -> dict:
    return {
        split: primary_test([r for r in rows if r["split"] == split], rng, n_boot, a_key, b_key)
        for split in ("test", "cv")
    }


def prevalence(rows: list[dict]) -> dict:
    """Share of questions with at least one core hedge, by gold label and by annotator pattern."""

    def share(group: list[dict], key: str) -> float:
        return round(sum(r[key] > 0 for r in group) / len(group), 4) if group else float("nan")

    by_label = {}
    for label in ("yes", "no", "maybe"):
        group = [r for r in rows if r["final"] == label]
        by_label[label] = {
            "n": len(group),
            "conclusion": share(group, "hedge_count_conclusion"),
            "results_section": share([r for r in group if r["has_results_section"]], "hedge_count_results"),
        }
    by_pattern = {}
    for pattern in sorted({r["pattern"] for r in rows if r["final"] == "maybe"}):
        group = [r for r in rows if r["final"] == "maybe" and r["pattern"] == pattern]
        by_pattern[pattern] = {
            "n": len(group),
            "conclusion": share(group, "hedge_count_conclusion"),
            "results_section": share([r for r in group if r["has_results_section"]], "hedge_count_results"),
        }
    return {"by_gold_label": by_label, "gold_maybe_by_annotator_pattern": by_pattern}


def load_rows(path: Path = TABLE) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    rows = load_rows(args.table)
    rng = np.random.default_rng(args.seed)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_questions": len(rows),
        "primary": primary_test(rows, rng, args.n_boot),
        "primary_by_split": by_split(rows, rng, args.n_boot),
        "annotator_contrast": annotator_contrast(rows, rng, args.n_boot),
        "score_matrix": score_matrix(rows, rng, args.n_boot),
        "prevalence": prevalence(rows),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    p = result["primary"]
    print(f"=== H1 primary (n={p['n']}, gold maybe={p['n_maybe']}) ===")
    for name in ("conclusion", "results_section"):
        s = p[name]
        print(f"AUROC {name:16s} {s['auroc']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
    d = p["difference"]
    print(f"dAUROC conclusion - results {d['d_auroc']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  -> {p['verdict']}")
    for split, s in result["primary_by_split"].items():
        sd = s["difference"]
        print(f"  {split:4s} n={s['n']} maybe={s['n_maybe']} dAUROC {sd['d_auroc']:+.3f} [{sd['ci_low']:+.3f}, {sd['ci_high']:+.3f}] -> {s['verdict']}")
    print("=== annotator contrast: AUROC(annotator with conclusion) - AUROC(without) ===")
    for score, s in result["annotator_contrast"].items():
        print(f"  {score:28s} {s['d_auroc']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
