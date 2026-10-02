"""Hypothesis H3: scoring against soft labels changes how systems compare on PubMedQA.

PQA-L keeps both raw annotations of every question: annotator 1 (saw the conclusion, RF)
and annotator 2 (did not, RR). The usual evaluation scores against the single label they
agreed on after discussion. Following Lionetti et al. (2025), a soft label keeps the
annotations as a distribution, and metrics sum probabilities instead of counting hits.

Soft label (primary): q = 1/2 one-hot(RF) + 1/2 one-hot(RR). Where the annotators agreed
it equals the hard label.
Soft accuracy of a system = mean probability the soft label gives to the system's answer.
Soft precision / recall / F1 for maybe use soft counts: TP = sum of q[maybe] over the
questions the system answered maybe; recall divides by the total q[maybe] mass
(Lionetti et al., eq. 3, at the system's single operating point).

Primary test, fixed before any soft score was computed:
  - pair: self-consistency (qwen3:8b, k=4) vs BioLinkBERT, the two systems that did not
    see each other's answers;
  - statistic: I = [soft_acc(SC) - soft_acc(BERT)] - [hard_acc(SC) - hard_acc(BERT)],
    paired bootstrap over the 500 test questions (95% CI);
  H3 is supported if the CI of I excludes 0; refuted if the CI lies inside [-0.02, +0.02]
  (the comparison moves by less than 2 points whichever labels are used); inconclusive otherwise.
  The four systems' hard accuracies sit within 3 points of each other, so a change of rank
  order alone would not be evidence; the test is on the size of the change.

Secondary: I for every pair of systems (uncorrected); how often each pair's order differs
between hard and soft accuracy within one bootstrap resample; hard vs soft maybe
precision / recall / F1; an always-yes reference; and the same analysis with the agreed
label as a third vote (q = 1/3 each).

Known before registration: every system's hard accuracy and its accuracy against RR
(H2 output), so half of each soft accuracy. Not known: accuracy against RF.
The two annotators read different inputs, so q mixes reader variation with the effect of
seeing the conclusion; it is the distribution PQA-L's raw annotations give, not a
population of independent readers.

Output: ``reports/debate/analysis/h3_soft_labels.json``. No LLM calls.
"""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.analyze_h2_human_ceiling import SYSTEMS, bootstrap_ci, load_frame  # noqa: E402

OUT = PROJECT_ROOT / "reports/debate/analysis/h3_soft_labels.json"
LABELS = ("yes", "no", "maybe")
PRIMARY_PAIR = ("self_consistency_qwen3_8b_k4", "biolinkbert")
EQUIVALENCE = 0.02


def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.stack([(labels == lab).astype(float) for lab in LABELS], axis=1)


def soft_labels(*votes: np.ndarray) -> np.ndarray:
    """Label distribution per question: equal weight to each annotation in ``votes``."""
    return np.mean([one_hot(v) for v in votes], axis=0)


def soft_accuracy(pred: np.ndarray, q: np.ndarray) -> float:
    """Mean probability that the soft label assigns to the predicted label."""
    return float((one_hot(pred) * q).sum(axis=1).mean())


def interaction(pred_a: np.ndarray, pred_b: np.ndarray, gold: np.ndarray, q: np.ndarray) -> float:
    """(A - B) in soft accuracy minus (A - B) in hard accuracy."""
    soft = soft_accuracy(pred_a, q) - soft_accuracy(pred_b, q)
    hard = float((pred_a == gold).mean() - (pred_b == gold).mean())
    return soft - hard


def order_differs(pred_a: np.ndarray, pred_b: np.ndarray, gold: np.ndarray, q: np.ndarray) -> float:
    """1.0 if the sign of A - B differs between hard and soft accuracy, else 0.0."""
    soft = soft_accuracy(pred_a, q) - soft_accuracy(pred_b, q)
    hard = float((pred_a == gold).mean() - (pred_b == gold).mean())
    return float(np.sign(soft) != np.sign(hard))


def soft_maybe_prf(pred: np.ndarray, q: np.ndarray) -> dict:
    """Precision / recall / F1 for maybe with soft counts (probability mass, not hits)."""
    p_maybe = q[:, LABELS.index("maybe")]
    predicted = pred == "maybe"
    tp, n_pred, mass = float(p_maybe[predicted].sum()), int(predicted.sum()), float(p_maybe.sum())
    precision = tp / n_pred if n_pred else float("nan")
    recall = tp / mass if mass else float("nan")
    f1 = 2 * tp / (n_pred + mass) if n_pred + mass else float("nan")
    return {
        "soft_true_positives": round(tp, 2),
        "predicted": n_pred,
        "gold_mass": round(mass, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def bootstrap_mean(stat, arrays: tuple[np.ndarray, ...], rng: np.random.Generator, n_boot: int) -> float:
    """Mean of ``stat`` over bootstrap resamples of the questions."""
    n = len(arrays[0])
    return float(np.mean([stat(*(a[idx] for a in arrays)) for idx in (rng.integers(0, n, size=n) for _ in range(n_boot))]))


def verdict(ci: dict) -> str:
    if ci["ci_low"] > 0 or ci["ci_high"] < 0:
        return "supported"
    if -EQUIVALENCE <= ci["ci_low"] and ci["ci_high"] <= EQUIVALENCE:
        return "refuted"
    return "inconclusive"


def analyze(frame: dict, q: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    gold = frame["gold"]
    preds = {name: frame[name] for name in SYSTEMS}
    preds["always_yes"] = np.full(len(gold), "yes")

    scores = {}
    for name, pred in preds.items():
        hard_hits = pred == gold
        scores[name] = {
            "hard_accuracy": round(float(hard_hits.mean()), 4),
            "soft_accuracy": bootstrap_ci(soft_accuracy, (pred, q), rng, n_boot),
            "maybe_hard": soft_maybe_prf(pred, one_hot(gold)),
            "maybe_soft": soft_maybe_prf(pred, q),
        }

    pairs = {}
    for a, b in combinations(SYSTEMS, 2):
        if (b, a) == PRIMARY_PAIR:
            a, b = b, a
        arrays = (preds[a], preds[b], gold, q)
        ci = bootstrap_ci(interaction, arrays, rng, n_boot)
        pairs[f"{a} - {b}"] = {
            "interaction": {**ci, "verdict": verdict(ci)},
            "hard_difference": round(float((preds[a] == gold).mean() - (preds[b] == gold).mean()), 4),
            "soft_difference": round(soft_accuracy(preds[a], q) - soft_accuracy(preds[b], q), 4),
            "order_differs_rate": round(bootstrap_mean(order_differs, arrays, rng, n_boot), 4),
        }
    return {"systems": scores, "pairs": pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    frame = load_frame()
    rng = np.random.default_rng(args.seed)
    q_primary = soft_labels(frame["rf"], frame["rr"])
    q_three = soft_labels(frame["rf"], frame["rr"], frame["gold"])
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_questions": int(len(frame["gold"])),
        "n_soft_questions": int((frame["rf"] != frame["rr"]).sum()),
        "primary_pair": " - ".join(PRIMARY_PAIR),
        "two_annotators": analyze(frame, q_primary, rng, args.n_boot),
        "two_annotators_plus_agreed_label": analyze(frame, q_three, rng, args.n_boot),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    main_pair = result["two_annotators"]["pairs"][result["primary_pair"]]
    ci = main_pair["interaction"]
    print(f"=== H3 (n={result['n_questions']}, annotators differ on {result['n_soft_questions']}) ===")
    print(
        f"PRIMARY {result['primary_pair']}: hard {main_pair['hard_difference']:+.3f}, soft {main_pair['soft_difference']:+.3f}; "
        f"I {ci['value']:+.3f} [{ci['ci_low']:+.3f}, {ci['ci_high']:+.3f}] -> {ci['verdict']}"
    )
    for name, s in result["two_annotators"]["systems"].items():
        soft = s["soft_accuracy"]
        print(f"  {name:32s} hard {s['hard_accuracy']:.3f}  soft {soft['value']:.3f} [{soft['ci_low']:.3f}, {soft['ci_high']:.3f}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
