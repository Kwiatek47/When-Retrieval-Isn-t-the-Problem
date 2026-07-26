"""Structured uncertainty scoring for the multi-agent debate (maybe routing).

Motivation
----------
On PubMedQA the ``maybe`` class is an *uncertainty* class: the correct answer is
"the abstract does not conclusively settle the question". Empirically neither the
BioLinkBERT classifier nor a naive debate panel expose this: both tend to commit
confidently to yes/no on true-maybe cases (silent agreement). See
``docs/research`` and the debate error analysis.

Instead of trusting a single confidence number, we derive an interpretable
uncertainty score ``u`` in [0, 1] from *multiple* debate signals, following the
uncertainty-aware multi-agent literature (DebUnc, MARC, BELIEF) and the
"Silence is not consensus" catfish-agent finding. A high ``u`` routes the final
answer to ``maybe``; otherwise the normal panel+classifier decision stands. The
routing threshold is *calibrated* on held-out data rather than hand-picked.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

from app.agents.aggregation import opinion_label
from app.agents.models import AgentRoundOpinion, ClinicalOpinion

_LABELS = ("yes", "no", "maybe")


def _label_entropy(labels: list[str]) -> float:
    """Normalized Shannon entropy of a label distribution in [0, 1]."""
    labels = [label for label in labels if label]
    if not labels:
        return 0.0
    counts = Counter(labels)
    total = sum(counts.values())
    entropy = -sum((c / total) * math.log2(c / total) for c in counts.values())
    max_entropy = math.log2(len(_LABELS))
    return entropy / max_entropy if max_entropy else 0.0


@dataclass
class UncertaintySignals:
    """Interpretable per-case uncertainty features (all in [0, 1] unless noted)."""

    label_entropy: float
    maybe_fraction: float
    inconclusive_fraction: float
    mean_disagreement_with_mode: float
    flip_rate: float
    panel_uncertainty_conf: float
    bert_is_maybe: float
    semantic_entropy: float
    audit_score: float
    score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "label_entropy": self.label_entropy,
            "maybe_fraction": self.maybe_fraction,
            "inconclusive_fraction": self.inconclusive_fraction,
            "mean_disagreement_with_mode": self.mean_disagreement_with_mode,
            "flip_rate": self.flip_rate,
            "panel_uncertainty_conf": self.panel_uncertainty_conf,
            "bert_is_maybe": self.bert_is_maybe,
            "semantic_entropy": self.semantic_entropy,
            "audit_score": self.audit_score,
            "score": self.score,
        }


# Feature weights for the linear uncertainty score. Chosen to reflect the
# analysis: explicit "inconclusive" reasoning and panel `maybe` votes are the
# strongest honest signals; semantic disagreement among the 8 agent opinions
# (semantic entropy) captures deeper divergence than raw label counts; label
# disagreement / round flips indicate an unresolved debate; a low-confidence or
# maybe classifier reinforces it.
# The evidence condition-audit (audit_score) is the only *evidence-anchored*
# signal, so when present it carries the most weight; the self-reported debate
# signals (inconclusive/maybe fractions) are secondary, and the semantic /
# label entropies are weak tie-breakers per the balanced90 analysis.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "audit_score": 0.40,
    "inconclusive_fraction": 0.22,
    "maybe_fraction": 0.15,
    "label_entropy": 0.08,
    "semantic_entropy": 0.05,
    "flip_rate": 0.05,
    "mean_disagreement_with_mode": 0.03,
    "bert_is_maybe": 0.02,
}


# When no evidence audit is supplied, re-normalize the remaining weights so the
# score still spans [0, 1] and behaves like the pre-audit configuration.
_NO_AUDIT_WEIGHTS: dict[str, float] = {
    "inconclusive_fraction": 0.38,
    "maybe_fraction": 0.24,
    "label_entropy": 0.12,
    "semantic_entropy": 0.08,
    "flip_rate": 0.08,
    "mean_disagreement_with_mode": 0.05,
    "bert_is_maybe": 0.05,
}


_EMBEDDER: Any = None


def _default_embed(texts: list[str]) -> list[list[float]]:
    """Lazily load a local sentence embedder and encode ``texts``.

    Uses the already-cached multilingual MiniLM model so no download is needed.
    Returns an empty list if sentence-transformers is unavailable.
    """
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer

            _EMBEDDER = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        except Exception:  # pragma: no cover - environment without the model
            _EMBEDDER = False
    if not _EMBEDDER:
        return []
    return [list(map(float, vec)) for vec in _EMBEDDER.encode(texts, normalize_embeddings=True)]


def _opinion_text(opinion: ClinicalOpinion) -> str:
    """Render an opinion's semantic content (label + reasoning) for clustering."""
    parts = [opinion.top_1_diagnosis or ""]
    if opinion.evidence_conclusiveness:
        parts.append(opinion.evidence_conclusiveness)
    parts.extend(opinion.pros[:2])
    parts.extend(opinion.cons[:2])
    if opinion.missing_information:
        parts.append(opinion.missing_information)
    return " | ".join(p for p in parts if p).strip()


def semantic_entropy(
    opinions: list[ClinicalOpinion],
    *,
    embed_fn: Any = None,
    threshold: float = 0.80,
) -> float:
    """Normalized semantic entropy over all agent opinions (Kuhn et al. 2023 style).

    We embed each opinion's meaning (label + reasoning), greedily cluster by
    cosine similarity (>= ``threshold`` joins a cluster), and take the normalized
    Shannon entropy of the cluster-size distribution. High entropy => the panel
    is semantically divided, a strong honest uncertainty signal that plain label
    counts miss (four agents can all say "yes" for contradictory reasons, or say
    different labels for essentially the same hedge).

    Falls back to label entropy if no embedder is available.
    """
    opinions = [o for o in opinions if o is not None]
    if len(opinions) <= 1:
        return 0.0
    embed_fn = embed_fn or _default_embed
    texts = [_opinion_text(o) for o in opinions]
    vectors = embed_fn(texts)
    if not vectors or len(vectors) != len(texts):
        return _label_entropy([opinion_label(o) or "" for o in opinions])

    clusters: list[list[int]] = []
    centroids: list[list[float]] = []

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a)) or 1.0
        nb = math.sqrt(sum(y * y for y in b)) or 1.0
        return dot / (na * nb)

    for idx, vec in enumerate(vectors):
        best_c = -1
        best_sim = -1.0
        for c, centroid in enumerate(centroids):
            sim = cosine(vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_c = c
        if best_c >= 0 and best_sim >= threshold:
            clusters[best_c].append(idx)
            members = clusters[best_c]
            dim = len(vec)
            centroids[best_c] = [
                sum(vectors[m][d] for m in members) / len(members) for d in range(dim)
            ]
        else:
            clusters.append([idx])
            centroids.append(list(vec))

    sizes = [len(c) for c in clusters]
    total = sum(sizes)
    entropy = -sum((s / total) * math.log2(s / total) for s in sizes if s)
    max_entropy = math.log2(len(opinions))
    return entropy / max_entropy if max_entropy else 0.0


def compute_uncertainty(
    rounds: list[list[AgentRoundOpinion]],
    final_opinions: list[AgentRoundOpinion],
    *,
    bert_label: str | None = None,
    bert_confidence: float | None = None,
    weights: dict[str, float] | None = None,
    embed_fn: Any = None,
    audit_score: float | None = None,
) -> UncertaintySignals:
    """Compute interpretable uncertainty signals + a combined score in [0, 1].

    ``rounds`` is the full debate history (per round, per agent); ``final_opinions``
    is each agent's last opinion. ``bert_label`` / ``bert_confidence`` are the
    optional BioLinkBERT signal for the same case.
    """
    weights = weights or (_DEFAULT_WEIGHTS if audit_score is not None else _NO_AUDIT_WEIGHTS)

    final_labels = [opinion_label(entry.opinion) for entry in final_opinions]
    final_labels = [label for label in final_labels if label]
    n = len(final_labels) or 1

    label_entropy = _label_entropy(final_labels)
    maybe_fraction = sum(1 for label in final_labels if label == "maybe") / n

    # Explicit self-reported inconclusiveness from the schema field (strongest
    # honest signal: an agent that argues yes/no but flags evidence as
    # inconclusive is really uncertain).
    inconclusive = 0
    conclusiveness_seen = 0
    for entry in final_opinions:
        value = (entry.opinion.evidence_conclusiveness or "").strip().lower()
        if value in {"conclusive", "inconclusive"}:
            conclusiveness_seen += 1
            if value == "inconclusive":
                inconclusive += 1
    inconclusive_fraction = (inconclusive / conclusiveness_seen) if conclusiveness_seen else 0.0

    # Disagreement with the modal label, confidence-weighted: agents that
    # confidently dissent from the majority signal an unresolved question.
    if final_labels:
        mode_label = Counter(final_labels).most_common(1)[0][0]
    else:
        mode_label = None
    disagreements: list[float] = []
    for entry in final_opinions:
        label = opinion_label(entry.opinion)
        if label is None:
            continue
        conf = float(entry.opinion.confidence_level)
        disagreements.append(conf if label != mode_label else 0.0)
    mean_disagreement = sum(disagreements) / len(disagreements) if disagreements else 0.0

    # Round-to-round instability: how often an agent changed its label between
    # its first and last opinion (a debate that keeps flipping is unresolved).
    flip_rate = _flip_rate(rounds)

    # Confidence attached specifically to uncertainty: mean confidence among
    # agents that voted `maybe` (a confident `maybe` is a strong signal).
    maybe_confs = [
        float(entry.opinion.confidence_level)
        for entry in final_opinions
        if opinion_label(entry.opinion) == "maybe"
    ]
    panel_uncertainty_conf = (sum(maybe_confs) / len(maybe_confs)) if maybe_confs else 0.0

    bert_is_maybe = 1.0 if (bert_label == "maybe") else 0.0

    # Semantic entropy over all opinions across all rounds (the 4 agents x
    # rounds samples): captures divergence in *meaning*, not just labels.
    all_opinions = [entry.opinion for round_entries in rounds for entry in round_entries]
    if not all_opinions:
        all_opinions = [entry.opinion for entry in final_opinions]
    sem_entropy = semantic_entropy(all_opinions, embed_fn=embed_fn)

    features = {
        "audit_score": float(audit_score) if audit_score is not None else 0.0,
        "inconclusive_fraction": inconclusive_fraction,
        "maybe_fraction": maybe_fraction,
        "semantic_entropy": sem_entropy,
        "label_entropy": label_entropy,
        "flip_rate": flip_rate,
        "mean_disagreement_with_mode": mean_disagreement,
        "bert_is_maybe": bert_is_maybe,
    }
    score = sum(weights.get(key, 0.0) * value for key, value in features.items())
    score = max(0.0, min(1.0, score))

    return UncertaintySignals(
        label_entropy=label_entropy,
        maybe_fraction=maybe_fraction,
        inconclusive_fraction=inconclusive_fraction,
        mean_disagreement_with_mode=mean_disagreement,
        flip_rate=flip_rate,
        panel_uncertainty_conf=panel_uncertainty_conf,
        bert_is_maybe=bert_is_maybe,
        semantic_entropy=sem_entropy,
        audit_score=float(audit_score) if audit_score is not None else 0.0,
        score=score,
    )


def _flip_rate(rounds: list[list[AgentRoundOpinion]]) -> float:
    """Fraction of agents whose label changed between first and last round."""
    if len(rounds) < 2:
        return 0.0
    first: dict[str, str | None] = {}
    last: dict[str, str | None] = {}
    for entry in rounds[0]:
        first[entry.agent_id] = opinion_label(entry.opinion)
    for entry in rounds[-1]:
        last[entry.agent_id] = opinion_label(entry.opinion)
    agents = [a for a in first if a in last and first[a] and last[a]]
    if not agents:
        return 0.0
    flips = sum(1 for a in agents if first[a] != last[a])
    return flips / len(agents)


def calibrate_threshold(
    scores: list[float],
    labels: list[str],
    *,
    objective: str = "macro_f1",
    grid: int = 101,
    base_labels: list[str] | None = None,
) -> tuple[float, dict[str, float]]:
    """Pick the uncertainty threshold that maximizes an objective on held-out data.

    For each candidate threshold ``t`` a case is routed to ``maybe`` iff its
    uncertainty ``score >= t`` (otherwise it keeps ``base_labels`` — the
    panel+classifier decision). The objective is then computed on the resulting
    *3-class* predictions, so raising the maybe recall is traded honestly
    against the yes/no accuracy it costs.

    ``objective`` is one of:
      - ``accuracy``     : overall 3-class accuracy of the routed predictions
      - ``macro_f1``     : macro-averaged 3-class F1 of the routed predictions
      - ``maybe_f1``     : F1 of the maybe class only (recall-heavy, legacy)
      - ``balanced_acc`` : mean per-class recall of the routed predictions

    If ``base_labels`` is not given we fall back to the legacy binary
    (maybe vs not-maybe) detection objective for backward compatibility.

    Returns ``(threshold, metrics_at_threshold)``.
    """
    if not scores:
        return 0.5, {}

    labels_t = ("yes", "no", "maybe")

    def _three_class_metrics(threshold: float) -> dict[str, float]:
        preds = [
            "maybe" if s >= threshold else (base_labels[i] if base_labels else "maybe")
            for i, s in enumerate(scores)
        ]
        tp = {L: 0 for L in labels_t}
        fp = {L: 0 for L in labels_t}
        fn = {L: 0 for L in labels_t}
        support = {L: 0 for L in labels_t}
        correct = 0
        for pred, gold in zip(preds, labels):
            support[gold] = support.get(gold, 0) + 1
            if pred == gold:
                correct += 1
            for L in labels_t:
                if pred == L and gold == L:
                    tp[L] += 1
                elif pred == L and gold != L:
                    fp[L] += 1
                elif pred != L and gold == L:
                    fn[L] += 1
        f1 = {}
        recall_by = {}
        for L in labels_t:
            prec = tp[L] / (tp[L] + fp[L]) if (tp[L] + fp[L]) else 0.0
            rec = tp[L] / (tp[L] + fn[L]) if (tp[L] + fn[L]) else 0.0
            f1[L] = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            recall_by[L] = rec
        n = len(labels) or 1
        seen = [L for L in labels_t if support.get(L, 0) > 0]
        return {
            "accuracy": correct / n,
            "macro_f1": sum(f1.values()) / len(labels_t),
            "maybe_f1": f1["maybe"],
            "balanced_acc": (sum(recall_by[L] for L in seen) / len(seen)) if seen else 0.0,
            "maybe_precision": (tp["maybe"] / (tp["maybe"] + fp["maybe"]))
            if (tp["maybe"] + fp["maybe"])
            else 0.0,
            "maybe_recall": recall_by["maybe"],
        }

    def _binary_metrics(threshold: float) -> dict[str, float]:
        is_maybe = [1 if label == "maybe" else 0 for label in labels]
        tp = sum(1 for s, m in zip(scores, is_maybe) if s >= threshold and m == 1)
        fp = sum(1 for s, m in zip(scores, is_maybe) if s >= threshold and m == 0)
        fn = sum(1 for s, m in zip(scores, is_maybe) if s < threshold and m == 1)
        tn = sum(1 for s, m in zip(scores, is_maybe) if s < threshold and m == 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        maybe_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        n_prec = tn / (tn + fn) if (tn + fn) else 0.0
        n_rec = tn / (tn + fp) if (tn + fp) else 0.0
        not_f1 = (2 * n_prec * n_rec / (n_prec + n_rec)) if (n_prec + n_rec) else 0.0
        return {
            "precision": precision,
            "recall": recall,
            "maybe_f1": maybe_f1,
            "macro_f1": (maybe_f1 + not_f1) / 2.0,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    use_three_class = base_labels is not None
    best_t = 0.5
    best_obj = -1.0
    best_metrics: dict[str, float] = {}

    for i in range(grid):
        t = i / (grid - 1)
        metrics = _three_class_metrics(t) if use_three_class else _binary_metrics(t)
        obj = metrics.get(objective)
        if obj is None:
            obj = metrics.get("macro_f1", 0.0)
        if obj > best_obj:
            best_obj = obj
            best_t = t
            best_metrics = {"threshold": t, **metrics}

    return best_t, best_metrics


def risk_coverage_curve(
    scores: list[float],
    base_labels: list[str],
    gold_labels: list[str],
    *,
    steps: int = 21,
) -> dict[str, Any]:
    """Selective-prediction risk-coverage analysis for maybe-routing.

    We treat routing a case to ``maybe`` as *abstaining* from a definitive
    yes/no answer. For a sweep of uncertainty thresholds we report, over the
    non-abstained (answered) cases only:
      - coverage: fraction of cases answered with a definitive yes/no
      - selective_accuracy: accuracy on those answered cases
    plus the AURC (area under risk-coverage, lower is better), following the
    abstention-aware scientific-reasoning literature (arXiv 2602.14189).

    Cases whose gold label is ``maybe`` are, by construction, "correctly
    abstained" and excluded from the answered-accuracy denominator.
    """
    order = sorted(scores)
    points: list[dict[str, float]] = []
    thresholds = sorted({order[min(int(round(i / (steps - 1) * (len(order) - 1))), len(order) - 1)] for i in range(steps)}) if order else [0.5]
    thresholds = sorted(set([0.0, *thresholds, 1.0]))

    for t in thresholds:
        answered = 0
        answered_correct = 0
        for s, base, gold in zip(scores, base_labels, gold_labels):
            routed = "maybe" if s >= t else base
            if routed == "maybe":
                continue  # abstained
            answered += 1
            if routed == gold:
                answered_correct += 1
        coverage = answered / len(scores) if scores else 0.0
        sel_acc = answered_correct / answered if answered else 1.0
        points.append(
            {
                "threshold": t,
                "coverage": coverage,
                "selective_accuracy": sel_acc,
                "risk": 1.0 - sel_acc,
            }
        )

    # AURC via trapezoid over coverage-sorted points.
    pts = sorted(points, key=lambda p: p["coverage"])
    aurc = 0.0
    for a, b in zip(pts, pts[1:]):
        dc = b["coverage"] - a["coverage"]
        aurc += dc * (a["risk"] + b["risk"]) / 2.0
    return {"points": points, "aurc": aurc}


def cost_sensitive_analysis(
    scores: list[float],
    base_labels: list[str],
    gold_labels: list[str],
    *,
    cost_wrong: float = 1.0,
    cost_abstain: float = 0.25,
    steps: int = 51,
) -> dict[str, Any]:
    """Expected clinical cost of a *selective* yes/no policy vs always answering.

    Clinical motivation (the "knows when not to answer" framing): in a diagnostic
    setting a confidently wrong definitive answer is far more costly than deferring
    to a human ("maybe"/abstain). We assign:
      - correct definitive yes/no  -> cost 0
      - wrong definitive yes/no    -> cost ``cost_wrong``
      - abstain (routed to maybe)  -> cost ``cost_abstain`` (0 < cost_abstain < cost_wrong)

    A true-``maybe`` gold case answered definitively counts as *wrong* (the abstract
    did not settle it), so abstaining on it is the correct, cheaper action. We sweep
    the uncertainty threshold and report the minimum-cost operating point, comparing
    it against the always-answer baseline (threshold = 1.0, never abstain).
    """
    n = len(scores)
    if n == 0:
        return {"error": "empty"}

    def policy_cost(t: float) -> tuple[float, float]:
        total = 0.0
        abstained = 0
        for s, base, gold in zip(scores, base_labels, gold_labels):
            routed = "maybe" if s >= t else base
            if routed == "maybe":
                total += cost_abstain
                abstained += 1
            elif routed == gold and gold != "maybe":
                total += 0.0
            else:
                total += cost_wrong
        return total / n, abstained / n

    always_cost, _ = policy_cost(1.0)  # never abstain
    grid = [i / (steps - 1) for i in range(steps)]
    best_t = 1.0
    best_cost = always_cost
    best_abstain = 0.0
    # Constrained optimum: require answering at least ``min_coverage`` of cases so we
    # do not reward the degenerate "abstain on everything" policy that a weak signal
    # produces when cost_abstain < the base error rate.
    min_coverage = 0.5
    best_t_cov = 1.0
    best_cost_cov = always_cost
    best_abstain_cov = 0.0
    curve: list[dict[str, float]] = []
    for t in grid:
        c, ab = policy_cost(t)
        curve.append({"threshold": round(t, 4), "mean_cost": c, "abstain_rate": ab})
        if c < best_cost:
            best_cost = c
            best_t = t
            best_abstain = ab
        if (1.0 - ab) >= min_coverage and c < best_cost_cov:
            best_cost_cov = c
            best_t_cov = t
            best_abstain_cov = ab

    return {
        "cost_wrong": cost_wrong,
        "cost_abstain": cost_abstain,
        "always_answer_cost": always_cost,
        "best_threshold": best_t,
        "best_cost": best_cost,
        "best_abstain_rate": best_abstain,
        "cost_reduction": always_cost - best_cost,
        "cost_reduction_pct": (always_cost - best_cost) / always_cost if always_cost else 0.0,
        "min_coverage": min_coverage,
        "constrained_best_threshold": best_t_cov,
        "constrained_best_cost": best_cost_cov,
        "constrained_best_abstain_rate": best_abstain_cov,
        "constrained_cost_reduction_pct": (
            (always_cost - best_cost_cov) / always_cost if always_cost else 0.0
        ),
        "curve": curve,
    }

