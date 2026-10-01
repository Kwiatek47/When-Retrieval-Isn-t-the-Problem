"""RQ8 / RQ9a: does a dedicated head or balanced sampling let BioLinkBERT detect ``maybe``?

Reads the test-set ``maybe`` probabilities written by
``scripts/classifier/train_maybe_detector.py`` (2 x 2 design, five seeds per cell, trained
on the 500 non-test PQA-L questions) and scores them on the 500 official test questions.

Each cell is scored as the mean probability over its five seeds. The metric is average
precision (AP) for gold ``maybe``: it needs no threshold, so nothing is tuned on the test set.
With 55 of 500 questions ``maybe``, a detector with no signal has AP of about 0.11.

Tests, fixed before any model was trained (paired bootstrap over test questions, 95% CI):
  RQ8  head effect      = mean over sampling of [AP(binary) - AP(three_class)];
  RQ9a balance effect   = mean over head     of [AP(balanced) - AP(natural)].
  Each is supported if its CI lies above 0, refuted if its CI lies at or below 0,
  inconclusive otherwise. The two tests are reported side by side, without correction.

Secondary: AP and AUROC of each cell with CIs; spread of AP across seeds; the deployed
checkpoint (three classes, trained with PQA-A) as a reference; and the comparison with the
annotator who read the same text: precision of each cell at the threshold where it recovers
as many ``maybe`` as that annotator did (30 of 55), next to the annotator's precision (30/47 = 0.638).

Known before registration: the deployed checkpoint recovers 4/55 at argmax, and its
1 - confidence has AUROC 0.637 for gold maybe. Nothing is known about the new cells.
Limits fixed by the data: 55 ``maybe`` training examples; one test set; five seeds.

Output: ``reports/debate/analysis/rq8_maybe_detector.json``. No training here.
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
from scripts.agents.analyze_h2_human_ceiling import PQAL, bootstrap_ci  # noqa: E402

SCORES_DIR = PROJECT_ROOT / "reports/debate/analysis/maybe_detector"
OUT = PROJECT_ROOT / "reports/debate/analysis/rq8_maybe_detector.json"
HEADS = ("binary", "three_class")
SAMPLINGS = ("natural", "balanced")
SEEDS = (11, 23, 42, 47, 101)
REFERENCE = "reference_deployed_three_class"
HUMAN_HITS = 30  # gold maybe recovered by the annotator who did not see the conclusion


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    """Area under the precision-recall curve; tied scores share one threshold."""
    n_pos = int(positive.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores, hits = scores[order], positive[order].astype(float)
    last_of_group = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    true_positives = np.cumsum(hits)[last_of_group]
    retrieved = (np.arange(len(scores)) + 1)[last_of_group]
    recall = true_positives / n_pos
    precision = true_positives / retrieved
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def precision_at_hits(scores: np.ndarray, positive: np.ndarray, hits: int = HUMAN_HITS) -> float:
    """Precision at the loosest threshold needed to recover ``hits`` positives."""
    order = np.argsort(-scores, kind="mergesort")
    cumulative = np.cumsum(positive[order])
    reached = np.nonzero(cumulative >= hits)[0]
    if reached.size == 0:
        return float("nan")
    k = reached[0] + 1
    return float(hits / k)


def load_scores(path: Path) -> dict[str, float]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["pmid"]: row["p_maybe"] for row in rows if "pmid" in row}


def load_cells(scores_dir: Path = SCORES_DIR, seeds: tuple[int, ...] = SEEDS) -> tuple[list[str], dict, dict]:
    """PMIDs, per-cell seed-averaged scores, and per-cell per-seed scores."""
    per_seed: dict[tuple[str, str], list[dict[str, float]]] = {}
    for head in HEADS:
        for sampling in SAMPLINGS:
            per_seed[(head, sampling)] = [
                load_scores(scores_dir / f"{head}.{sampling}.seed{seed}.jsonl") for seed in seeds
            ]
    pmids = sorted(per_seed[(HEADS[0], SAMPLINGS[0])][0])
    seed_arrays = {cell: [np.array([run[p] for p in pmids]) for run in runs] for cell, runs in per_seed.items()}
    ensemble = {cell: np.mean(arrays, axis=0) for cell, arrays in seed_arrays.items()}
    return pmids, ensemble, seed_arrays


def head_effect(bn, bb, tn, tb, positive) -> float:
    ap = average_precision
    return ((ap(bn, positive) - ap(tn, positive)) + (ap(bb, positive) - ap(tb, positive))) / 2


def balance_effect(bn, bb, tn, tb, positive) -> float:
    ap = average_precision
    return ((ap(bb, positive) - ap(bn, positive)) + (ap(tb, positive) - ap(tn, positive))) / 2


def verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "supported"
    if ci["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def describe(scores: np.ndarray, positive: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    return {
        "average_precision": bootstrap_ci(average_precision, (scores, positive), rng, n_boot),
        "auroc": bootstrap_ci(auroc, (scores, positive), rng, n_boot),
        "precision_at_30_hits": round(precision_at_hits(scores, positive), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=SCORES_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    pmids, ensemble, seed_arrays = load_cells(args.scores_dir)
    positive = np.array([data[p]["final_decision"] == "maybe" for p in pmids])
    rng = np.random.default_rng(args.seed)

    cells = {}
    for cell, scores in ensemble.items():
        seed_aps = [average_precision(s, positive) for s in seed_arrays[cell]]
        cells[".".join(cell)] = {
            **describe(scores, positive, rng, args.n_boot),
            "ap_per_seed_mean": round(float(np.mean(seed_aps)), 4),
            "ap_per_seed_sd": round(float(np.std(seed_aps, ddof=1)), 4),
        }

    design = tuple(ensemble[(h, s)] for h in HEADS for s in SAMPLINGS)  # bn, bb, tn, tb
    head = bootstrap_ci(head_effect, (*design, positive), rng, args.n_boot)
    balance = bootstrap_ci(balance_effect, (*design, positive), rng, args.n_boot)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_questions": len(pmids),
        "n_gold_maybe": int(positive.sum()),
        "chance_average_precision": round(float(positive.mean()), 4),
        "human_without_conclusion": {"hits": HUMAN_HITS, "precision": round(30 / 47, 4), "recall": round(30 / 55, 4)},
        "rq8_head_effect": {**head, "verdict": verdict(head)},
        "rq9a_balance_effect": {**balance, "verdict": verdict(balance)},
        "cells": cells,
    }
    reference_path = args.scores_dir / f"{REFERENCE}.jsonl"
    if reference_path.exists():
        reference = load_scores(reference_path)
        result["reference_deployed"] = describe(np.array([reference[p] for p in pmids]), positive, rng, args.n_boot)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== RQ8 / RQ9a (n={len(pmids)}, gold maybe={result['n_gold_maybe']}, chance AP {result['chance_average_precision']}) ===")
    for name, key in (("RQ8 head (binary - three_class)", "rq8_head_effect"), ("RQ9a balance (balanced - natural)", "rq9a_balance_effect")):
        e = result[key]
        print(f"{name}: {e['value']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] -> {e['verdict']}")
    for name, c in {**cells, **({"deployed (reference)": result["reference_deployed"]} if "reference_deployed" in result else {})}.items():
        ap = c["average_precision"]
        print(f"  {name:28s} AP {ap['value']:.3f} [{ap['ci_low']:.3f}, {ap['ci_high']:.3f}]  precision at 30 hits {c['precision_at_30_hits']}")
    print(f"  annotator without conclusion: precision {30 / 47:.3f} at 30 hits")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
