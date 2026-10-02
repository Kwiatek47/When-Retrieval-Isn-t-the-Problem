"""Can a large LLM that is told what ``maybe`` means read it from the abstract?

After RQ8 / RQ9a an encoder trained on 55 ``maybe`` examples stayed near chance, which
leaves two readings: too little training data, or a task that is hard for models in
general. This probe needs no training: qwen3:30b answers the 500 official test questions
with prompt ``label-defined@1`` (PubMedQA's definition of maybe), once from the abstract
without its conclusion (what annotator 2 and every model saw) and once with the conclusion
added (what annotator 1 saw). Runs come from ``scripts/agents/run_label_probe.py``.

Tests, fixed before any test question was sent to the model (bootstrap over questions, 95% CI):
  T1 gap to the human reader = F1_maybe(annotator 2) - F1_maybe(model, no conclusion), both
     against gold. CI above 0: the model is below the same-information human; CI below 0:
     above; otherwise not distinguishable.
  T2 conclusion effect = F1_maybe(model, with conclusion) - F1_maybe(model, no conclusion).
     Supported if the CI lies above 0 (the direct H1 test with a model), refuted if at or
     below 0, inconclusive otherwise. The annotators' own difference is reported next to it.
  Primary: thinking on (the model's default mode). Thinking off is secondary.

Secondary: accuracy against gold and against each annotator; maybe precision / recall and
how often the model says maybe; how many answers change to or from maybe when the
conclusion is added. A reply that could not be parsed after retries counts as a wrong
answer that is not maybe.

Known before registration: annotator 2 has maybe F1 0.588 and annotator 1 0.660 against
gold; qwen3:8b with four samples and another prompt reached accuracy 0.742 and maybe F1
0.282; the timing pilot used 16 non-test questions. Annotator 2 helped set the gold label,
so the human F1 overstates what an independent reader would reach.

Output: ``reports/debate/analysis/label_probe_qwen3_30b.json``. No LLM calls.
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

from scripts.agents.analyze_h2_human_ceiling import PQAL, bootstrap_ci, maybe_prf  # noqa: E402
from scripts.agents.probe_prompts import get_prompt  # noqa: E402
from scripts.agents.run_label_probe import default_out  # noqa: E402

OUT = PROJECT_ROOT / "reports/debate/analysis/label_probe_qwen3_30b.json"
MODEL = "qwen3:30b"
PROMPT = "label-defined@1"
CONDITIONS = ("context", "context+conclusion")
THINK_MODES = ("on", "off")
PRIMARY_THINK = "on"
NO_ANSWER = "none"


def maybe_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    hit = ((pred == "maybe") & (gold == "maybe")).sum()
    total = (pred == "maybe").sum() + (gold == "maybe").sum()
    return float(2 * hit / total) if total else float("nan")


def gap_to_human(model: np.ndarray, human: np.ndarray, gold: np.ndarray) -> float:
    return maybe_f1(human, gold) - maybe_f1(model, gold)


def conclusion_effect(with_conclusion: np.ndarray, without: np.ndarray, gold: np.ndarray) -> float:
    return maybe_f1(with_conclusion, gold) - maybe_f1(without, gold)


def load_labels(path: Path, pmids: list[str]) -> np.ndarray:
    """First-repeat label per question; unanswered questions become ``NO_ANSWER``."""
    labels = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("repeat") == 0 and row.get("label") is not None:
            labels[row["pmid"]] = row["label"]
    return np.array([labels.get(p, NO_ANSWER) for p in pmids])


def gap_verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "model below human"
    if ci["ci_high"] < 0:
        return "model above human"
    return "not distinguishable"


def effect_verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "supported"
    if ci["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def describe(pred: np.ndarray, frame: dict) -> dict:
    return {
        "unanswered": int((pred == NO_ANSWER).sum()),
        "accuracy_vs_gold": round(float((pred == frame["gold"]).mean()), 4),
        "accuracy_vs_annotator_without_conclusion": round(float((pred == frame["rr"]).mean()), 4),
        "accuracy_vs_annotator_with_conclusion": round(float((pred == frame["rf"]).mean()), 4),
        "maybe_vs_gold": maybe_prf(pred, frame["gold"]),
    }


def analyze_mode(without: np.ndarray, with_conclusion: np.ndarray, frame: dict, rng: np.random.Generator, n_boot: int) -> dict:
    gold, rr = frame["gold"], frame["rr"]
    gap = bootstrap_ci(gap_to_human, (without, rr, gold), rng, n_boot)
    effect = bootstrap_ci(conclusion_effect, (with_conclusion, without, gold), rng, n_boot)
    return {
        "T1_gap_to_human": {**gap, "verdict": gap_verdict(gap)},
        "T2_conclusion_effect": {**effect, "verdict": effect_verdict(effect)},
        "context": describe(without, frame),
        "context+conclusion": describe(with_conclusion, frame),
        "answers_changed_to_maybe": int(((without != "maybe") & (with_conclusion == "maybe")).sum()),
        "answers_changed_from_maybe": int(((without == "maybe") & (with_conclusion != "maybe")).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    spec = get_prompt(PROMPT)
    data = json.loads(PQAL.read_text(encoding="utf-8"))
    paths = {
        (think, condition): default_out(spec, condition, MODEL, "test", think)
        for think in THINK_MODES
        for condition in CONDITIONS
    }
    first = paths[(PRIMARY_THINK, "context")]
    pmids = sorted({json.loads(line)["pmid"] for line in first.read_text(encoding="utf-8").splitlines() if line.strip()})
    frame = {
        "gold": np.array([data[p]["final_decision"] for p in pmids]),
        "rr": np.array([data[p]["reasoning_required_pred"] for p in pmids]),
        "rf": np.array([data[p]["reasoning_free_pred"] for p in pmids]),
    }
    rng = np.random.default_rng(args.seed)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "model": MODEL,
        "prompt": {"id": spec.id, "sha": spec.sha},
        "n_questions": len(pmids),
        "primary_think_mode": PRIMARY_THINK,
        "annotators": {
            "without_conclusion": maybe_prf(frame["rr"], frame["gold"]),
            "with_conclusion": maybe_prf(frame["rf"], frame["gold"]),
            "conclusion_effect": bootstrap_ci(conclusion_effect, (frame["rf"], frame["rr"], frame["gold"]), rng, args.n_boot),
        },
        "think": {},
    }
    for think in THINK_MODES:
        if not all(paths[(think, c)].exists() for c in CONDITIONS):
            continue
        without = load_labels(paths[(think, "context")], pmids)
        with_conclusion = load_labels(paths[(think, "context+conclusion")], pmids)
        result["think"][think] = analyze_mode(without, with_conclusion, frame, rng, args.n_boot)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== label probe {MODEL}, {spec.id} (n={len(pmids)}) ===")
    for think, r in result["think"].items():
        tag = "PRIMARY " if think == PRIMARY_THINK else ""
        t1, t2 = r["T1_gap_to_human"], r["T2_conclusion_effect"]
        print(f"{tag}think {think}:")
        print(f"  T1 human - model maybe F1: {t1['value']:+.3f} [{t1['ci_low']:+.3f}, {t1['ci_high']:+.3f}] -> {t1['verdict']}")
        print(f"  T2 conclusion effect:      {t2['value']:+.3f} [{t2['ci_low']:+.3f}, {t2['ci_high']:+.3f}] -> {t2['verdict']}")
        for condition in CONDITIONS:
            d = r[condition]
            m = d["maybe_vs_gold"]
            print(f"  {condition:20s} acc {d['accuracy_vs_gold']:.3f}  maybe P/R/F1 {m['precision']}/{m['recall']}/{m['f1']} ({m['predicted']} predicted)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
