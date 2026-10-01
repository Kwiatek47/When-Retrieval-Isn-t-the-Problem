"""Does the model's gap to the human reader on ``maybe`` survive an independent reference?

The first label probe (``analyze_label_probe.py``) found qwen3:30b far below annotator 2 on
``maybe`` when both are scored against the final label. Annotator 2 helped set that label,
so the comparison favours the human. Here each reader is scored against the *other*
annotator, whom neither the model nor annotator 2 influenced, on the 500 PQA-L questions
outside the official test set (the model was never trained on PQA-L). The questions assumed
to have been used in the timing pilot (the first ``PILOT_N`` in runner order) are left out.
Runs come from ``scripts/agents/run_label_probe.py --split cv``.

Tests, fixed before any of these questions was scored (bootstrap over questions, 95% CI):
  S1 same-seat gap = F1_maybe(annotator 2 vs annotator 1) - F1_maybe(model without the
     conclusion vs annotator 1). CI above 0: the model is below the human; CI below 0:
     above; otherwise not distinguishable.
  S2 share of the gap owed to the co-created label = (gap against the final label) -
     (gap against annotator 1). Supported if the CI lies above 0, refuted if at or below 0,
     inconclusive otherwise.
  R1 replication of T1 = F1_maybe(annotator 2 vs final) - F1_maybe(model vs final).
  T2 conclusion effect = F1_maybe(model with conclusion) - F1_maybe(model without), against
     the final label, on these questions (confirmatory) and pooled with the test questions
     (more power, but the test half was already seen).
  Primary: thinking on. Thinking off is secondary.

Secondary: the mirrored seat (model with the conclusion and annotator 1, both against
annotator 2); the same-seat difference in overall agreement; maybe precision / recall
against each reference. A reply that could not be parsed counts as a wrong answer that is
not maybe.

Known before registration (exploratory, test questions, thinking on): annotator 2 vs
annotator 1 maybe F1 0.232, model 0.209 (0.216 thinking off); S1 +0.023 [-0.132, +0.178];
S2 +0.403 [+0.258, +0.565]; T1 +0.426 [+0.262, +0.569]; T2 +0.022 [-0.104, +0.149]; mirrored
seat -0.063 [-0.233, +0.114]; agreement gap -0.086 [-0.128, -0.046]. On the questions scored
here the annotators' labels alone give: annotator 2 vs annotator 1 maybe F1 0.247 (12
shared, 37 and 60 answers), annotator 2 vs final 0.489. Model answers on these questions:
only the pilot's.

Output: ``reports/debate/analysis/label_probe_qwen3_30b_independent.json``. No LLM calls.
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
from scripts.agents.analyze_label_probe import (  # noqa: E402
    CONDITIONS,
    MODEL,
    NO_ANSWER,
    PRIMARY_THINK,
    PROMPT,
    THINK_MODES,
    conclusion_effect,
    effect_verdict,
    gap_to_human,
    gap_verdict,
    load_labels,
)
from scripts.agents.probe_prompts import get_prompt  # noqa: E402
from scripts.agents.run_label_probe import default_out, select_pmids  # noqa: E402

OUT = PROJECT_ROOT / "reports/debate/analysis/label_probe_qwen3_30b_independent.json"
SPLIT = "cv"
PILOT_N = 16


def label_gap_share(model: np.ndarray, human: np.ndarray, gold: np.ndarray, independent: np.ndarray) -> float:
    """How much of the human-model gap disappears when the reference is independent."""
    return gap_to_human(model, human, gold) - gap_to_human(model, human, independent)


def agreement_gap(model: np.ndarray, human: np.ndarray, reference: np.ndarray) -> float:
    return float((human == reference).mean() - (model == reference).mean())


def confirmatory_pmids(data: dict, pilot_n: int = PILOT_N) -> list[str]:
    return select_pmids(data, SPLIT)[pilot_n:]


def require_complete(path: Path, pmids: list[str]) -> None:
    """Stop if a run is unfinished: a missing row would otherwise be scored as a wrong answer."""
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            seen.add(json.loads(line)["pmid"])
        except (ValueError, KeyError):
            continue
    missing = [p for p in pmids if p not in seen]
    if missing:
        raise SystemExit(f"{path.name} has no row for {len(missing)} questions; finish the run first.")


def make_frame(data: dict, pmids: list[str]) -> dict:
    return {
        "gold": np.array([data[p]["final_decision"] for p in pmids]),
        "rr": np.array([data[p]["reasoning_required_pred"] for p in pmids]),
        "rf": np.array([data[p]["reasoning_free_pred"] for p in pmids]),
    }


def analyze_mode(without: np.ndarray, with_conclusion: np.ndarray, frame: dict, rng: np.random.Generator, n_boot: int) -> dict:
    gold, rr, rf = frame["gold"], frame["rr"], frame["rf"]
    same_seat = bootstrap_ci(gap_to_human, (without, rr, rf), rng, n_boot)
    share = bootstrap_ci(label_gap_share, (without, rr, gold, rf), rng, n_boot)
    replication = bootstrap_ci(gap_to_human, (without, rr, gold), rng, n_boot)
    effect = bootstrap_ci(conclusion_effect, (with_conclusion, without, gold), rng, n_boot)
    mirrored = bootstrap_ci(gap_to_human, (with_conclusion, rf, rr), rng, n_boot)
    return {
        "S1_same_seat_gap": {**same_seat, "verdict": gap_verdict(same_seat)},
        "S2_gap_owed_to_co_created_label": {**share, "verdict": effect_verdict(share)},
        "R1_gap_against_final_label": {**replication, "verdict": gap_verdict(replication)},
        "T2_conclusion_effect": {**effect, "verdict": effect_verdict(effect)},
        "mirrored_seat_gap": {**mirrored, "verdict": gap_verdict(mirrored)},
        "agreement_gap_same_seat": bootstrap_ci(agreement_gap, (without, rr, rf), rng, n_boot),
        "model_without_conclusion": {
            "unanswered": int((without == NO_ANSWER).sum()),
            "maybe_vs_annotator_with_conclusion": maybe_prf(without, rf),
            "maybe_vs_final": maybe_prf(without, gold),
            "agreement_with_annotator_with_conclusion": round(float((without == rf).mean()), 4),
        },
        "model_with_conclusion": {
            "unanswered": int((with_conclusion == NO_ANSWER).sum()),
            "maybe_vs_annotator_without_conclusion": maybe_prf(with_conclusion, rr),
            "maybe_vs_final": maybe_prf(with_conclusion, gold),
            "agreement_with_annotator_without_conclusion": round(float((with_conclusion == rr).mean()), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    spec = get_prompt(PROMPT)
    data = json.loads(PQAL.read_text(encoding="utf-8"))
    pmids = confirmatory_pmids(data)
    test_pmids = select_pmids(data, "test")
    frame = make_frame(data, pmids)
    pooled_frame = make_frame(data, pmids + test_pmids)
    rng = np.random.default_rng(args.seed)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "model": MODEL,
        "prompt": {"id": spec.id, "sha": spec.sha},
        "split": SPLIT,
        "pilot_questions_left_out": PILOT_N,
        "n_questions": len(pmids),
        "primary_think_mode": PRIMARY_THINK,
        "annotators": {
            "without_vs_with_conclusion": maybe_prf(frame["rr"], frame["rf"]),
            "without_conclusion_vs_final": maybe_prf(frame["rr"], frame["gold"]),
            "with_conclusion_vs_final": maybe_prf(frame["rf"], frame["gold"]),
            "agreement": round(float((frame["rr"] == frame["rf"]).mean()), 4),
        },
        "think": {},
    }
    for think in THINK_MODES:
        paths = {c: default_out(spec, c, MODEL, SPLIT, think) for c in CONDITIONS}
        if not all(p.exists() for p in paths.values()):
            continue
        for path in paths.values():
            require_complete(path, pmids)
        without = load_labels(paths["context"], pmids)
        with_conclusion = load_labels(paths["context+conclusion"], pmids)
        r = analyze_mode(without, with_conclusion, frame, rng, args.n_boot)
        test_paths = {c: default_out(spec, c, MODEL, "test", think) for c in CONDITIONS}
        if all(p.exists() for p in test_paths.values()):
            pooled_without = np.concatenate([without, load_labels(test_paths["context"], test_pmids)])
            pooled_with = np.concatenate([with_conclusion, load_labels(test_paths["context+conclusion"], test_pmids)])
            pooled = bootstrap_ci(conclusion_effect, (pooled_with, pooled_without, pooled_frame["gold"]), rng, args.n_boot)
            r["T2_conclusion_effect_pooled_with_test"] = {**pooled, "verdict": effect_verdict(pooled), "n_questions": len(pooled_without)}
        result["think"][think] = r
    if not result["think"]:
        raise SystemExit(f"no finished {SPLIT} runs for {MODEL}; run scripts/agents/run_label_probe.py --split {SPLIT} first.")
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"=== label probe {MODEL}, {spec.id}, independent reference (split {SPLIT}, n={len(pmids)}) ===")
    rows = (
        ("S1 same-seat gap", "S1_same_seat_gap"),
        ("S2 gap owed to co-created label", "S2_gap_owed_to_co_created_label"),
        ("R1 gap against final label", "R1_gap_against_final_label"),
        ("T2 conclusion effect", "T2_conclusion_effect"),
        ("T2 pooled with test", "T2_conclusion_effect_pooled_with_test"),
    )
    for think, r in result["think"].items():
        print(f"{'PRIMARY ' if think == PRIMARY_THINK else ''}think {think}:")
        for title, key in rows:
            if key in r:
                t = r[key]
                print(f"  {title:32s} {t['value']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] -> {t['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
