"""Exploratory: is the "same seat" result specific to one LLM, or does it hold for our systems?

The registered independent-reference probe (``analyze_label_probe_independent.py``) found
that qwen3:30b and the annotator who read the same text agree with the *other* annotator
on ``maybe`` equally poorly (S1), and that most of the human's lead against the final label
comes from having co-created that label (S2). That rests on one model and one prompt.
Here the same two statistics are computed for the systems whose predictions already exist:
BioLinkBERT, self-consistency (qwen3:8b, k=4) and the two hinted debate runs.

  S1 same-seat gap = F1_maybe(annotator 2 vs annotator 1) - F1_maybe(system vs annotator 1)
  S2 share owed to the co-created label = (gap against the final label) - (gap against annotator 1)

NOT pre-registered and not confirmatory. These predictions exist only for the 500 official
test questions, which earlier analyses already inspected: every system's accuracy against
the final label and against annotator 2 (H2), and annotator 2's maybe F1 against annotator 1
on these questions (0.232). BioLinkBERT was trained on final labels of other PQA-L questions
and the debate runs saw its answer, so these systems lean towards the final label by
construction. Read the output as a robustness check on S1 / S2, with the same caveat as
there: "not distinguishable" is not "equal".

Output: ``reports/debate/analysis/same_seat_systems.json``. No LLM calls.
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

from scripts.agents.analyze_h2_human_ceiling import SYSTEMS, bootstrap_ci, load_frame, maybe_prf  # noqa: E402
from scripts.agents.analyze_label_probe import effect_verdict, gap_to_human, gap_verdict  # noqa: E402
from scripts.agents.analyze_label_probe_independent import agreement_gap, label_gap_share  # noqa: E402

OUT = PROJECT_ROOT / "reports/debate/analysis/same_seat_systems.json"


def analyze_system(pred: np.ndarray, frame: dict, rng: np.random.Generator, n_boot: int) -> dict:
    gold, rr, rf = frame["gold"], frame["rr"], frame["rf"]
    same_seat = bootstrap_ci(gap_to_human, (pred, rr, rf), rng, n_boot)
    share = bootstrap_ci(label_gap_share, (pred, rr, gold, rf), rng, n_boot)
    against_final = bootstrap_ci(gap_to_human, (pred, rr, gold), rng, n_boot)
    return {
        "S1_same_seat_gap": {**same_seat, "verdict": gap_verdict(same_seat)},
        "S2_gap_owed_to_co_created_label": {**share, "verdict": effect_verdict(share)},
        "gap_against_final_label": {**against_final, "verdict": gap_verdict(against_final)},
        "agreement_gap_same_seat": bootstrap_ci(agreement_gap, (pred, rr, rf), rng, n_boot),
        "maybe_vs_annotator_with_conclusion": maybe_prf(pred, rf),
        "maybe_vs_final": maybe_prf(pred, gold),
        "agreement_with_annotator_with_conclusion": round(float((pred == rf).mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    frame = load_frame()
    rng = np.random.default_rng(args.seed)
    result = {
        "status": "exploratory, not pre-registered; test questions were inspected before",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_questions": int(len(frame["gold"])),
        "annotator_without_conclusion": {
            "maybe_vs_annotator_with_conclusion": maybe_prf(frame["rr"], frame["rf"]),
            "maybe_vs_final": maybe_prf(frame["rr"], frame["gold"]),
            "agreement_with_annotator_with_conclusion": round(float((frame["rr"] == frame["rf"]).mean()), 4),
        },
        "systems": {name: analyze_system(frame[name], frame, rng, args.n_boot) for name in SYSTEMS},
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    human = result["annotator_without_conclusion"]
    print(f"=== same seat, exploratory (n={result['n_questions']}) ===")
    print(
        f"annotator 2: maybe F1 {human['maybe_vs_annotator_with_conclusion']['f1']} vs annotator 1, "
        f"{human['maybe_vs_final']['f1']} vs final"
    )
    for name, s in result["systems"].items():
        s1, s2 = s["S1_same_seat_gap"], s["S2_gap_owed_to_co_created_label"]
        print(
            f"{name}: maybe F1 {s['maybe_vs_annotator_with_conclusion']['f1']} vs annotator 1, {s['maybe_vs_final']['f1']} vs final | "
            f"S1 {s1['value']:+.3f} [{s1['ci_low']:+.3f}, {s1['ci_high']:+.3f}] {s1['verdict']} | "
            f"S2 {s2['value']:+.3f} [{s2['ci_low']:+.3f}, {s2['ci_high']:+.3f}] {s2['verdict']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
