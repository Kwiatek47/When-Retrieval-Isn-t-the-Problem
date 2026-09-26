"""AUROC and bootstrap CIs, shared by the `maybe` analyses. Pure stdlib, no numpy.

Extracted from `compute_statistics.py` when `analyze_h1_hedging.py` needed the same
estimators — a third private copy of "AUROC with ties counted as 0.5" was one copy
too many. The arithmetic is unchanged, so previously published numbers reproduce.

Two bootstrap designs live here, and the difference matters:

  * `bootstrap_auroc_ci`   — resamples the positive and negative score pools
    independently. Right for a CI on *one* predictor.
  * `paired_auroc_delta_ci` — resamples *items*, then scores every predictor on the
    same resampled items. Right for a CI on the *difference* between two predictors
    measured on the same data, where the shared item noise cancels and an unpaired
    interval would be too wide.
"""

from __future__ import annotations

import random


def auroc(pos: list[float], neg: list[float]) -> float:
    """Probability a random positive outranks a random negative; ties count 0.5."""
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def bootstrap_auroc_ci(
    pos: list[float], neg: list[float], rng: random.Random, n_boot: int
) -> dict:
    point = auroc(pos, neg)
    boots = []
    for _ in range(n_boot):
        rp = [pos[rng.randrange(len(pos))] for _ in pos]
        rn = [neg[rng.randrange(len(neg))] for _ in neg]
        boots.append(auroc(rp, rn))
    boots = [b for b in boots if b == b]  # drop NaN
    return {
        "auroc": round(point, 4),
        "ci_low": round(percentile(boots, 0.025), 4),
        "ci_high": round(percentile(boots, 0.975), 4),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "brackets_chance": percentile(boots, 0.025) <= 0.5 <= percentile(boots, 0.975),
    }


def bootstrap_mean_ci(vals: list[float], rng: random.Random, n_boot: int) -> dict:
    """Bootstrap 95% CI for the mean of a 0/1 (or real) per-case vector."""
    if not vals:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    n = len(vals)
    point = sum(vals) / n
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    return {
        "mean": round(point, 4),
        "ci_low": round(percentile(boots, 0.025), 4),
        "ci_high": round(percentile(boots, 0.975), 4),
        "n": n,
    }


def paired_auroc_delta_ci(
    labels: list[bool],
    predictors: dict[str, list[float]],
    reference: str,
    comparison: str,
    rng: random.Random,
    n_boot: int,
) -> dict:
    """AUROC per predictor plus a paired CI for `reference - comparison`.

    `labels[i]` is whether item i is a positive; each predictor supplies one score per
    item, in the same order. The bootstrap resamples positives and negatives separately
    (so class sizes stay fixed) but reuses the drawn item indices for both predictors.
    """
    for name, scores in predictors.items():
        if len(scores) != len(labels):
            raise ValueError(f"predictor {name!r} has {len(scores)} scores for {len(labels)} labels")
    for name in (reference, comparison):
        if name not in predictors:
            raise KeyError(f"{name!r} is not among the predictors {sorted(predictors)}")

    pos_idx = [i for i, y in enumerate(labels) if y]
    neg_idx = [i for i, y in enumerate(labels) if not y]
    if not pos_idx or not neg_idx:
        raise ValueError("need at least one positive and one negative item")

    def auroc_on(name: str, pi: list[int], ni: list[int]) -> float:
        scores = predictors[name]
        return auroc([scores[i] for i in pi], [scores[i] for i in ni])

    point = {name: auroc_on(name, pos_idx, neg_idx) for name in predictors}
    delta_point = point[reference] - point[comparison]

    boots: dict[str, list[float]] = {name: [] for name in predictors}
    deltas: list[float] = []
    for _ in range(n_boot):
        pi = [pos_idx[rng.randrange(len(pos_idx))] for _ in pos_idx]
        ni = [neg_idx[rng.randrange(len(neg_idx))] for _ in neg_idx]
        drawn = {name: auroc_on(name, pi, ni) for name in predictors}
        for name, value in drawn.items():
            boots[name].append(value)
        deltas.append(drawn[reference] - drawn[comparison])

    per_predictor = {
        name: {
            "auroc": round(point[name], 4),
            "ci_low": round(percentile(boots[name], 0.025), 4),
            "ci_high": round(percentile(boots[name], 0.975), 4),
            "brackets_chance": percentile(boots[name], 0.025) <= 0.5 <= percentile(boots[name], 0.975),
        }
        for name in predictors
    }
    delta_low = percentile(deltas, 0.025)
    delta_high = percentile(deltas, 0.975)
    return {
        "n_pos": len(pos_idx),
        "n_neg": len(neg_idx),
        "n_boot": n_boot,
        "per_predictor": per_predictor,
        "delta": {
            "reference": reference,
            "comparison": comparison,
            "value": round(delta_point, 4),
            "ci_low": round(delta_low, 4),
            "ci_high": round(delta_high, 4),
            "brackets_zero": delta_low <= 0.0 <= delta_high,
        },
    }


def paired_auroc_delta_across_targets(
    scores: list[float],
    targets: dict[str, list[bool]],
    reference: str,
    comparison: str,
    rng: random.Random,
    n_boot: int,
) -> dict:
    """AUROC of one predictor against several *different* label sets, plus a paired delta.

    The mirror image of `paired_auroc_delta_ci`: there, one label set and competing
    predictors; here, one predictor and competing label sets (e.g. "is this item `maybe`
    according to the annotator who saw the conclusion" vs "according to the context-only
    annotator"). Because the positive sets differ, class sizes cannot be held fixed —
    the bootstrap resamples *items* and lets each target find its own positives, which
    is also what makes the two AUROCs paired on the same drawn items.
    """
    n = len(scores)
    for name, ys in targets.items():
        if len(ys) != n:
            raise ValueError(f"target {name!r} has {len(ys)} labels for {n} scores")
    for name in (reference, comparison):
        if name not in targets:
            raise KeyError(f"{name!r} is not among the targets {sorted(targets)}")

    def auroc_on(name: str, idx: list[int]) -> float:
        ys = targets[name]
        pos = [scores[i] for i in idx if ys[i]]
        neg = [scores[i] for i in idx if not ys[i]]
        return auroc(pos, neg)

    all_idx = list(range(n))
    point = {name: auroc_on(name, all_idx) for name in targets}

    boots: dict[str, list[float]] = {name: [] for name in targets}
    deltas: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        drawn = {name: auroc_on(name, idx) for name in targets}
        if any(v != v for v in drawn.values()):  # a draw with no positives for some target
            continue
        for name, value in drawn.items():
            boots[name].append(value)
        deltas.append(drawn[reference] - drawn[comparison])

    per_target = {
        name: {
            "auroc": round(point[name], 4),
            "ci_low": round(percentile(boots[name], 0.025), 4),
            "ci_high": round(percentile(boots[name], 0.975), 4),
            "n_pos": sum(1 for y in targets[name] if y),
            "n_neg": sum(1 for y in targets[name] if not y),
            "brackets_chance": percentile(boots[name], 0.025) <= 0.5 <= percentile(boots[name], 0.975),
        }
        for name in targets
    }
    delta_low = percentile(deltas, 0.025)
    delta_high = percentile(deltas, 0.975)
    return {
        "n_items": n,
        "n_boot": len(deltas),
        "per_target": per_target,
        "delta": {
            "reference": reference,
            "comparison": comparison,
            "value": round(point[reference] - point[comparison], 4),
            "ci_low": round(delta_low, 4),
            "ci_high": round(delta_high, 4),
            "brackets_zero": delta_low <= 0.0 <= delta_high,
        },
    }
