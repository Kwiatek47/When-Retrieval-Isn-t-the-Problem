"""Hypothesis H2: models fail where a person with the same information also departs from gold.

PQA-L's second annotator (``reasoning_required_pred``, RR) read exactly what the models
read: the question and the abstract without its conclusion. The gold label
(``final_decision``) was settled with a first annotator who also read the conclusion.
H2 says a model's errors against gold are concentrated on the questions where RR, too,
departed from gold, i.e. they reflect what the given text supports rather than a failure
to read it.

Data: the 500 official test questions; per-question predictions already in the repo.
A question is *disputed* when RR != gold and *agreed* when RR == gold.

Primary test, fixed before any of these overlaps were computed:
  - system: BioLinkBERT (the paper's decision baseline);
  - statistic: D = error rate on disputed questions - error rate on agreed questions,
    errors counted against gold; bootstrap over questions (95% CI).
  H2 is supported if the CI of D lies above 0, refuted if it lies at or below 0,
  inconclusive otherwise.

Secondary, same data:
  - D for self-consistency (qwen3:8b, k=4) and the two hinted debate runs (the debate runs
    saw BioLinkBERT's answer, so they are not independent of the primary system);
  - siding: among a system's errors on disputed questions, how often its label equals RR's
    (two wrong labels exist, so 0.5 is the reference);
  - accuracy against RR minus accuracy against gold (paired bootstrap);
  - share of a system's errors that fall on disputed questions, next to their base rate;
  - on gold-maybe questions: the system's maybe recall where RR said maybe vs where RR did not;
  - maybe precision / recall / F1 against gold for every system and for both annotators.

Known before registration (so not evidence for or against H2): each system's accuracy and
maybe recall against gold, and RR's maybe recall (30/55) and precision (30/47). RR helped
set the gold label, so its agreement with gold overstates what an independent reader would reach.

Output: ``reports/debate/analysis/h2_human_ceiling.json``. No LLM calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
DEBATE = PROJECT_ROOT / "reports/debate"
OUT = DEBATE / "analysis/h2_human_ceiling.json"

PRIMARY = "biolinkbert"
# name -> (report file, field holding that system's label)
SYSTEMS = {
    "biolinkbert": ("debate7b_dissent_pqal500_v1.json", "biolinkbert_label"),
    "self_consistency_qwen3_8b_k4": ("selfconsistency_qwen3_8b_k4_pqal500.json", "predicted_label"),
    "debate_dissent_hinted": ("debate7b_dissent_pqal500_v1.json", "predicted_label"),
    "debate_majority_hinted": ("debate7b_sup14b_majority_pqal500_v1.json", "predicted_label"),
}
_ID_PREFIX = "pubmedqa-official-"


def load_predictions(path: Path, field: str) -> dict[str, str]:
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    return {case["id"][len(_ID_PREFIX):]: case[field] for case in cases}


def load_frame(pqal_path: Path = PQAL, debate_dir: Path = DEBATE) -> dict[str, np.ndarray]:
    """Aligned label arrays for the questions every system answered."""
    data = json.loads(pqal_path.read_text(encoding="utf-8"))
    predictions = {
        name: load_predictions(debate_dir / file, field) for name, (file, field) in SYSTEMS.items()
    }
    pmids = sorted(set.intersection(*(set(p) for p in predictions.values())))
    frame = {
        "gold": np.array([data[p]["final_decision"] for p in pmids]),
        "rr": np.array([data[p]["reasoning_required_pred"] for p in pmids]),
        "rf": np.array([data[p]["reasoning_free_pred"] for p in pmids]),
    }
    for name, pred in predictions.items():
        frame[name] = np.array([pred[p] for p in pmids])
    return frame


def _rate(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def error_gap(pred: np.ndarray, gold: np.ndarray, rr: np.ndarray) -> float:
    """Error rate on disputed questions (RR != gold) minus error rate on agreed ones."""
    wrong = pred != gold
    disputed = rr != gold
    return _rate(wrong[disputed]) - _rate(wrong[~disputed])


def siding_with_rr(pred: np.ndarray, gold: np.ndarray, rr: np.ndarray) -> float:
    """Among errors on disputed questions, the share where the system gave RR's label."""
    mask = (rr != gold) & (pred != gold)
    return _rate(pred[mask] == rr[mask])


def accuracy_shift(pred: np.ndarray, gold: np.ndarray, rr: np.ndarray) -> float:
    """Accuracy against RR minus accuracy against gold."""
    return _rate(pred == rr) - _rate(pred == gold)


def error_share_on_disputed(pred: np.ndarray, gold: np.ndarray, rr: np.ndarray) -> float:
    wrong = pred != gold
    return _rate((rr != gold)[wrong])


def maybe_prf(pred: np.ndarray, gold: np.ndarray) -> dict:
    hit = int(((pred == "maybe") & (gold == "maybe")).sum())
    n_pred, n_gold = int((pred == "maybe").sum()), int((gold == "maybe").sum())
    precision = hit / n_pred if n_pred else float("nan")
    recall = hit / n_gold if n_gold else float("nan")
    f1 = 2 * hit / (n_pred + n_gold) if n_pred + n_gold else float("nan")
    return {
        "hits": hit,
        "predicted": n_pred,
        "gold": n_gold,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def maybe_recall_by_rr(pred: np.ndarray, gold: np.ndarray, rr: np.ndarray) -> dict:
    """On gold-maybe questions: maybe recall where RR also said maybe vs where RR did not."""
    out = {}
    for name, mask in (("rr_maybe", (gold == "maybe") & (rr == "maybe")), ("rr_not_maybe", (gold == "maybe") & (rr != "maybe"))):
        out[name] = {"n": int(mask.sum()), "hits": int((pred[mask] == "maybe").sum())}
    return out


def bootstrap_ci(stat, arrays: tuple[np.ndarray, ...], rng: np.random.Generator, n_boot: int) -> dict:
    """Point estimate and 95% CI of ``stat(*arrays)``, resampling questions."""
    n = len(arrays[0])
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(stat(*(a[idx] for a in arrays)))
    boots = np.asarray([b for b in boots if b == b])
    return {
        "value": round(stat(*arrays), 4),
        "ci_low": round(float(np.percentile(boots, 2.5)), 4),
        "ci_high": round(float(np.percentile(boots, 97.5)), 4),
    }


def verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "supported"
    if ci["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def analyze_system(pred: np.ndarray, frame: dict, rng: np.random.Generator, n_boot: int) -> dict:
    gold, rr = frame["gold"], frame["rr"]
    arrays = (pred, gold, rr)
    disputed = rr != gold
    wrong = pred != gold
    gap = bootstrap_ci(error_gap, arrays, rng, n_boot)
    return {
        "accuracy_vs_gold": round(_rate(pred == gold), 4),
        "accuracy_vs_rr": round(_rate(pred == rr), 4),
        "errors": int(wrong.sum()),
        "error_rate_disputed": round(_rate(wrong[disputed]), 4),
        "error_rate_agreed": round(_rate(wrong[~disputed]), 4),
        "error_gap": {**gap, "verdict": verdict(gap)},
        "siding_with_rr": {
            **bootstrap_ci(siding_with_rr, arrays, rng, n_boot),
            "n_errors_on_disputed": int((wrong & disputed).sum()),
        },
        "accuracy_shift_rr_minus_gold": bootstrap_ci(accuracy_shift, arrays, rng, n_boot),
        "error_share_on_disputed": bootstrap_ci(error_share_on_disputed, arrays, rng, n_boot),
        "maybe_recall_by_rr": maybe_recall_by_rr(*arrays),
        "maybe_vs_gold": maybe_prf(pred, gold),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    frame = load_frame()
    gold, rr = frame["gold"], frame["rr"]
    rng = np.random.default_rng(args.seed)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_questions": int(len(gold)),
        "n_disputed": int((rr != gold).sum()),
        "primary_system": PRIMARY,
        "systems": {name: analyze_system(frame[name], frame, rng, args.n_boot) for name in SYSTEMS},
        "annotators_maybe_vs_gold": {
            "without_conclusion_rr": maybe_prf(rr, gold),
            "with_conclusion_rf": maybe_prf(frame["rf"], gold),
        },
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== H2 (n={result['n_questions']}, disputed={result['n_disputed']}) ===")
    for name, s in result["systems"].items():
        g = s["error_gap"]
        tag = "PRIMARY " if name == PRIMARY else ""
        print(
            f"{tag}{name}: errors disputed {s['error_rate_disputed']:.3f} vs agreed {s['error_rate_agreed']:.3f}; "
            f"gap {g['value']:+.3f} [{g['ci_low']:+.3f}, {g['ci_high']:+.3f}] -> {g['verdict']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
