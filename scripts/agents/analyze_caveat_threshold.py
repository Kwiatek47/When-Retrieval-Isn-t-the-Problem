"""Can a stated caveat rule or a maybe threshold bring LLM labels closer to how humans said maybe?

Two fixes for what the protocol replays showed: qwen3:30b and mistral-small3.2:24b say ``maybe``
far less often than the annotators, and less often once they see the authors' conclusion, while
the annotator who saw it said ``maybe`` more often (110 vs 84 on the 995 questions).

Runs (``run_label_probe.py``, round 0 only, all 1000 PQA-L questions, temperature 0, thinking off
for qwen3): 2 models x 2 prompts x 2 inputs.
  label-probs@1         label-defined@1's definition plus a probability for each answer;
  label-caveat-probs@1  the same plus the rule that an answer the text itself limits is maybe;
  input: ``context`` (as annotator 2 saw it) or ``context+conclusion`` (as annotator 1 saw it).
"With-conclusion seat" is the ``context+conclusion`` input; its human counterpart is the
annotator who saw the conclusion (``reasoning_free_pred``), and the context seat's is
``reasoning_required_pred``. Pilot questions (``PILOT_PMIDS``) are excluded everywhere.

Tests, fixed before the runs were made. Three co-primary tests, Bonferroni-adjusted: 98.33%
bootstrap intervals over questions (5000 resamples, seed 47). Each statistic is the mean over the
two models. Verdict: supported if the interval lies above 0, refuted if at or below 0,
inconclusive otherwise.
  C1 conclusion effect (point 1): the conclusion effect is the maybe rate with the conclusion
     minus without it, from the stated labels. C1 = effect under label-caveat-probs@1 minus
     effect under label-probs@1. All questions.
  C2 maybe like the annotator with the conclusion (point 1): maybe F1 of the with-conclusion
     seat's stated label against that annotator, label-caveat-probs@1 minus label-probs@1.
     All questions.
  T1 threshold (point 2), label-probs@1: maybe F1 against the human counterpart, threshold
     labels minus stated labels, averaged over the 2 models x 2 seats. Test questions only.
     The threshold is set on the 500 non-test questions, per model and seat, so that the share
     of questions whose normalised p_maybe reaches it is as close as possible to the human
     counterpart's maybe share there (ties broken toward the higher threshold). Threshold labels:
     maybe if p_maybe reaches the threshold; otherwise the stated label if it is yes or no, else
     the larger of p_yes and p_no (yes on a tie). The threshold is held fixed in the bootstrap.

Secondary, uncorrected: every statistic per model; the conclusion effect under each prompt with
a 95% interval, next to the humans' (+0.026); maybe counts per model, prompt and input; maybe F1
against gold and against each annotator for stated and threshold labels, both prompts; the
threshold applied to label-caveat-probs@1; AP and AUROC of p_maybe for the counterpart's maybe
and for gold maybe (test); accuracy against gold, stated vs threshold labels (test); chosen
thresholds and the maybe shares they give on the non-test and test halves.

Known before registration: label-defined@1 round-0 labels of both models with and without the
conclusion (protocol replays; qwen3 also from the label probes); a 5-question pilot of both new
prompts on the pilot questions to check that the probabilities parse.

Output: ``reports/debate/analysis/caveat_threshold.json``. No LLM calls.
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

from scripts.agents.analyze_annotation_protocol import PILOT_PMIDS, PQAL  # noqa: E402
from scripts.agents.analyze_h1_hedging import auroc  # noqa: E402
from scripts.agents.analyze_label_probe import maybe_f1  # noqa: E402
from scripts.agents.analyze_rq8_maybe_detector import average_precision  # noqa: E402
from scripts.agents.run_label_probe import OUT_DIR, select_pmids  # noqa: E402

MODELS = {"qwen3": ("qwen3-30b", ".think-off"), "mistral": ("mistral-small3.2-24b", "")}
PROMPTS = {"probs": "label-probs-v1", "caveat": "label-caveat-probs-v1"}
INPUTS = {"with": "context-conclusion", "without": "context"}
OUT = PROJECT_ROOT / "reports/debate/analysis/caveat_threshold.json"
CI_LEVEL = 1 - 0.05 / 3


def run_path(model: str, prompt: str, seat: str) -> Path:
    slug, think = MODELS[model]
    return OUT_DIR / f"{PROMPTS[prompt]}.{INPUTS[seat]}.{slug}{think}.all.jsonl"


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("label") is not None and row.get("repeat", 0) == 0:
            rows[row["pmid"]] = row
    return rows


def choose_threshold(p_maybe: np.ndarray, target_share: float) -> float:
    """Observed value whose share of p_maybe >= it is closest to the target; ties to the higher one."""
    best, best_gap = float("inf"), float("inf")
    for value in sorted(set(p_maybe.tolist()), reverse=True):
        gap = round(abs(float((p_maybe >= value).mean()) - target_share), 9)  # equal gaps tie exactly
        if gap < best_gap:
            best, best_gap = value, gap
    return best


def threshold_labels(stated: np.ndarray, probs: np.ndarray, threshold: float) -> np.ndarray:
    """``probs`` columns are yes, no, maybe."""
    out = np.where(stated == "maybe", np.where(probs[:, 0] >= probs[:, 1], "yes", "no"), stated)
    return np.where(probs[:, 2] >= threshold, "maybe", out)


def conclusion_effect(with_labels: np.ndarray, without_labels: np.ndarray) -> float:
    return float((with_labels == "maybe").mean() - (without_labels == "maybe").mean())


def bootstrap(stat, arrays: tuple[np.ndarray, ...], rng: np.random.Generator, n_boot: int, level: float) -> dict:
    n = len(arrays[0])
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(stat(*(a[idx] for a in arrays)))
    boots = np.asarray([b for b in boots if b == b])
    tail = (1 - level) / 2 * 100
    return {
        "value": round(stat(*arrays), 4),
        "ci_low": round(float(np.percentile(boots, tail)), 4),
        "ci_high": round(float(np.percentile(boots, 100 - tail)), 4),
        "ci_level": round(level, 4),
    }


def verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "supported"
    if ci["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def mean_of(stat, k: int):
    """Mean of ``stat`` over consecutive groups of ``k`` arrays."""
    def f(*arrays: np.ndarray) -> float:
        groups = [arrays[i:i + k] for i in range(0, len(arrays), k)]
        return float(np.mean([stat(*g) for g in groups]))
    return f


def c1_one(caveat_with, caveat_without, probs_with, probs_without) -> float:
    return conclusion_effect(caveat_with, caveat_without) - conclusion_effect(probs_with, probs_without)


def c2_one(caveat_with, probs_with, human) -> float:
    return maybe_f1(caveat_with, human) - maybe_f1(probs_with, human)


def t1_one(threshold, stated, human) -> float:
    return maybe_f1(threshold, human) - maybe_f1(stated, human)


def analyze(runs: dict, data: dict, test: set[str], rng: np.random.Generator, n_boot: int) -> dict:
    """``runs[(model, prompt, seat)]`` maps pmid -> row."""
    pmids = sorted(set.intersection(*(set(r) for r in runs.values())) - set(PILOT_PMIDS))
    test_ids = [p for p in pmids if p in test]
    cv_ids = [p for p in pmids if p not in test]
    human = {"with": "reasoning_free_pred", "without": "reasoning_required_pred"}

    def labels(key, ids):
        return np.array([runs[key][p]["label"] for p in ids])

    def probs(key, ids):
        return np.array([[runs[key][p]["probs"][lab] for lab in ("yes", "no", "maybe")] for p in ids])

    def human_labels(seat, ids):
        return np.array([data[p][human[seat]] for p in ids])

    gold_all = np.array([data[p]["final_decision"] for p in pmids])
    rf_all = human_labels("with", pmids)

    c1_arrays = tuple(a for m in MODELS for a in (
        labels((m, "caveat", "with"), pmids), labels((m, "caveat", "without"), pmids),
        labels((m, "probs", "with"), pmids), labels((m, "probs", "without"), pmids)))
    c2_arrays = tuple(a for m in MODELS for a in (
        labels((m, "caveat", "with"), pmids), labels((m, "probs", "with"), pmids), rf_all))
    c1 = bootstrap(mean_of(c1_one, 4), c1_arrays, rng, n_boot, CI_LEVEL)
    c2 = bootstrap(mean_of(c2_one, 3), c2_arrays, rng, n_boot, CI_LEVEL)

    thresholds, t1_arrays, per_cell = {}, [], {}
    gold_test = np.array([data[p]["final_decision"] for p in test_ids])
    for prompt in PROMPTS:
        for m in MODELS:
            for seat in INPUTS:
                key = (m, prompt, seat)
                cv_p = probs(key, cv_ids)[:, 2]
                target = float((human_labels(seat, cv_ids) == "maybe").mean())
                t = choose_threshold(cv_p, target)
                stated = labels(key, test_ids)
                pr = probs(key, test_ids)
                thr = threshold_labels(stated, pr, t)
                hum = human_labels(seat, test_ids)
                name = f"{m}|{prompt}|{seat}"
                thresholds[name] = {
                    "threshold": t, "target_share_nontest": round(target, 4),
                    "share_nontest": round(float((cv_p >= t).mean()), 4),
                    "share_test": round(float((pr[:, 2] >= t).mean()), 4),
                }
                per_cell[name] = {
                    "maybe_f1_vs_counterpart": {"stated": round(maybe_f1(stated, hum), 4), "threshold": round(maybe_f1(thr, hum), 4)},
                    "maybe_f1_vs_gold": {"stated": round(maybe_f1(stated, gold_test), 4), "threshold": round(maybe_f1(thr, gold_test), 4)},
                    "accuracy_vs_gold": {"stated": round(float((stated == gold_test).mean()), 4),
                                         "threshold": round(float((thr == gold_test).mean()), 4)},
                    "maybe_count": {"stated": int((stated == "maybe").sum()), "threshold": int((thr == "maybe").sum()),
                                    "counterpart": int((hum == "maybe").sum()), "gold": int((gold_test == "maybe").sum())},
                    "p_maybe_ap": {"counterpart": round(average_precision(pr[:, 2], hum == "maybe"), 4),
                                   "gold": round(average_precision(pr[:, 2], gold_test == "maybe"), 4)},
                    "p_maybe_auroc": {"counterpart": round(auroc(pr[:, 2], hum == "maybe"), 4),
                                      "gold": round(auroc(pr[:, 2], gold_test == "maybe"), 4)},
                }
                if prompt == "probs":
                    t1_arrays += [thr, stated, hum]
    t1 = bootstrap(mean_of(t1_one, 3), tuple(t1_arrays), rng, n_boot, CI_LEVEL)

    per_model = {}
    for m in MODELS:
        per_model[m] = {
            "C1": bootstrap(c1_one, c1_arrays[:4] if m == "qwen3" else c1_arrays[4:], rng, n_boot, 0.95),
            "C2": bootstrap(c2_one, c2_arrays[:3] if m == "qwen3" else c2_arrays[3:], rng, n_boot, 0.95),
            "conclusion_effect": {
                prompt: bootstrap(conclusion_effect, (labels((m, prompt, "with"), pmids), labels((m, prompt, "without"), pmids)),
                                  rng, n_boot, 0.95)
                for prompt in PROMPTS
            },
            "maybe_count_all": {f"{prompt}|{seat}": int((labels((m, prompt, seat), pmids) == "maybe").sum())
                                for prompt in PROMPTS for seat in INPUTS},
            "maybe_f1_all_stated": {
                f"{prompt}|{seat}": {"vs_gold": round(maybe_f1(labels((m, prompt, seat), pmids), gold_all), 4),
                                     "vs_counterpart": round(maybe_f1(labels((m, prompt, seat), pmids), human_labels(seat, pmids)), 4)}
                for prompt in PROMPTS for seat in INPUTS},
        }
    rr_all = human_labels("without", pmids)
    return {
        "n_questions": len(pmids), "n_test": len(test_ids), "n_nontest": len(cv_ids),
        "human": {"maybe_with_conclusion": int((rf_all == "maybe").sum()), "maybe_without": int((rr_all == "maybe").sum()),
                  "conclusion_effect": round(conclusion_effect(rf_all, rr_all), 4), "gold_maybe": int((gold_all == "maybe").sum())},
        "C1_conclusion_effect_caveat_minus_probs": {**c1, "verdict": verdict(c1)},
        "C2_maybe_f1_vs_annotator_with_conclusion_caveat_minus_probs": {**c2, "verdict": verdict(c2)},
        "T1_maybe_f1_vs_counterpart_threshold_minus_stated": {**t1, "verdict": verdict(t1)},
        "per_model": per_model,
        "thresholds": thresholds,
        "test_cells": per_cell,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    runs = {(m, p, s): load(run_path(m, p, s)) for m in MODELS for p in PROMPTS for s in INPUTS}
    result = analyze(runs, data, set(select_pmids(data, "test")), np.random.default_rng(args.seed), args.n_boot)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("C1_conclusion_effect_caveat_minus_probs", "C2_maybe_f1_vs_annotator_with_conclusion_caveat_minus_probs",
                "T1_maybe_f1_vs_counterpart_threshold_minus_stated"):
        r = result[key]
        print(f"{key}: {r['value']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] -> {r['verdict']}")
    print(f"{result['n_questions']} questions ({result['n_test']} test); wrote {args.out}")


if __name__ == "__main__":
    main()
