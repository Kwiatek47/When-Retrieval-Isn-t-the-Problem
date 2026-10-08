"""Can two LLM annotators reproduce how PubMedQA's ``maybe`` was made?

Reads the protocol replay (``run_annotation_protocol.py``): two qwen3:30b agents, one with the
authors' conclusion and one without, label each of the 1000 PQA-L questions and negotiate when
they disagree. The human run of the same protocol is in ``ori_pqal.json``.

Human reference values (computed here from ``ori_pqal.json``, not typed in):
  - share of gold ``maybe`` that came out of a negotiation rather than agreement: 87/110 = 0.791;
  - in disputes settled on one side, share settled on the annotator who saw the conclusion:
    215/295 = 0.729.

Tests, fixed before the replay was run. Three co-primary tests, Bonferroni-adjusted: each uses a
98.33% bootstrap interval (5000 resamples of questions, seed 47).
  P1 maybe through negotiation: share of the replay's final ``maybe`` that came out of a
     negotiation. Reproduced if the interval lies inside the human value +/- 0.15; not
     reproduced if it lies entirely outside; inconclusive otherwise.
  P2 the conclusion wins: in replay disputes settled on one side, share settled on the agent
     that saw the conclusion. Same rule against 0.729.
  P3 does the protocol add anything: maybe F1 against the gold label of the replay's final label
     minus that of the agent with the conclusion alone (its round-0 label). Removed questions
     count as answered "not maybe". Supported if the interval lies above 0, refuted if at or
     below 0, inconclusive otherwise.

Secondary, uncorrected: status counts and removal rate; final-label distribution; each agent's
round-0 agreement with its human counterpart (accuracy and maybe F1); maybe F1 of the final label
against gold and against each human annotator; the replay's 3 x 3 initial-label matrix next to the
human one; how often a yes-versus-no dispute ends in ``maybe`` (human: 2 of 151); maybe recall on
gold ``maybe`` that was unanimous versus negotiated among the humans.

Known before registration: the human values above; the single-agent behaviour of qwen3:30b with
``label-defined@1`` and thinking off on all 1000 questions (round 0 of this replay uses the same
prompt), e.g. maybe F1 against gold 0.167 with the conclusion and 0.222 without on the test half.
Pilot questions (``PILOT_PMIDS``) are excluded.

Output: ``reports/debate/analysis/annotation_protocol.json``. No LLM calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.analyze_label_probe import maybe_f1  # noqa: E402

PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
RUN = PROJECT_ROOT / "reports/debate/analysis/annotation_protocol/protocol-qwen3-30b.think-off.all.jsonl"
OUT = PROJECT_ROOT / "reports/debate/analysis/annotation_protocol.json"
WITH, WITHOUT = "with_conclusion", "context_only"
MARGIN = 0.15
CI_LEVEL = 1 - 0.05 / 3
# The first five questions in runner order were used for the 2026-10-08 pilot (5/5 completed, 0 failures).
PILOT_PMIDS: tuple[str, ...] = ("10135926", "10158597", "10173769", "10201555", "10223070")
LABELS = ("yes", "no", "maybe")


def human_reference(data: dict) -> dict:
    gold_maybe = [v for v in data.values() if v["final_decision"] == "maybe"]
    negotiated_maybe = sum(v["reasoning_free_pred"] != v["reasoning_required_pred"] for v in gold_maybe)
    disputes = [v for v in data.values() if v["reasoning_free_pred"] != v["reasoning_required_pred"]]
    to_with = sum(v["final_decision"] == v["reasoning_free_pred"] for v in disputes)
    to_without = sum(v["final_decision"] == v["reasoning_required_pred"] for v in disputes)
    yes_no = [v for v in disputes if {v["reasoning_free_pred"], v["reasoning_required_pred"]} == {"yes", "no"}]
    return {
        "maybe_through_negotiation": negotiated_maybe / len(gold_maybe),
        "maybe_through_negotiation_counts": [negotiated_maybe, len(gold_maybe)],
        "conclusion_wins": to_with / (to_with + to_without),
        "conclusion_wins_counts": [to_with, to_with + to_without],
        "yes_no_dispute_to_maybe": [sum(v["final_decision"] == "maybe" for v in yes_no), len(yes_no)],
    }


def load_run(path: Path, pilot: tuple[str, ...] = PILOT_PMIDS) -> list[dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row["pmid"] in pilot:
            continue
        if row.get("status") != "failed" or row["pmid"] not in rows:
            rows[row["pmid"]] = row  # a later successful retry replaces a failure
    return [rows[p] for p in sorted(rows)]


def maybe_through_negotiation(status: np.ndarray, final: np.ndarray) -> float:
    m = final == "maybe"
    return float((status[m] == "negotiated").mean()) if m.any() else float("nan")


def conclusion_wins(follows: np.ndarray) -> float:
    settled = (follows == WITH) | (follows == WITHOUT)
    return float((follows[settled] == WITH).mean()) if settled.any() else float("nan")


def protocol_gain(final: np.ndarray, with_initial: np.ndarray, gold: np.ndarray) -> float:
    return maybe_f1(final, gold) - maybe_f1(with_initial, gold)


def bootstrap(stat, arrays: tuple[np.ndarray, ...], rng: np.random.Generator, n_boot: int, level: float = CI_LEVEL) -> dict:
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


def reproduction_verdict(ci: dict, human: float, margin: float = MARGIN) -> str:
    lo, hi = human - margin, human + margin
    if lo <= ci["ci_low"] and ci["ci_high"] <= hi:
        return "reproduced"
    if ci["ci_high"] < lo or ci["ci_low"] > hi:
        return "not reproduced"
    return "inconclusive"


def gain_verdict(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "supported"
    if ci["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def matrix(rows_a: list[str], rows_b: list[str]) -> dict:
    counts = Counter(zip(rows_a, rows_b))
    return {f"with={a}|without={b}": counts.get((a, b), 0) for a in LABELS for b in LABELS}


def analyze(rows: list[dict], data: dict, rng: np.random.Generator, n_boot: int) -> dict:
    ok = [r for r in rows if r["status"] != "failed"]
    status = np.array([r["status"] for r in ok])
    final = np.array([r["final"] if r["final"] else "removed" for r in ok])
    follows = np.array([r["final_follows"] for r in ok])
    with_initial = np.array([r["initial"][WITH] for r in ok])
    without_initial = np.array([r["initial"][WITHOUT] for r in ok])
    gold = np.array([data[r["pmid"]]["final_decision"] for r in ok])
    rf = np.array([data[r["pmid"]]["reasoning_free_pred"] for r in ok])
    rr = np.array([data[r["pmid"]]["reasoning_required_pred"] for r in ok])
    human = human_reference(data)

    p1 = bootstrap(maybe_through_negotiation, (status, final), rng, n_boot)
    p2 = bootstrap(conclusion_wins, (follows,), rng, n_boot)
    p3 = bootstrap(protocol_gain, (final, with_initial, gold), rng, n_boot)

    yes_no = {(a, b) for a in ("yes", "no") for b in ("yes", "no") if a != b}
    yn = np.array([(a, b) in yes_no for a, b in zip(with_initial, without_initial)])
    human_unanimous = (rf == "maybe") & (rr == "maybe") & (gold == "maybe")
    human_negotiated = (gold == "maybe") & ~human_unanimous
    return {
        "n_questions": len(rows),
        "n_failed": len(rows) - len(ok),
        "human_reference": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in human.items()},
        "P1_maybe_through_negotiation": {**p1, "human": round(human["maybe_through_negotiation"], 4),
                                         "verdict": reproduction_verdict(p1, human["maybe_through_negotiation"])},
        "P2_conclusion_wins": {**p2, "human": round(human["conclusion_wins"], 4),
                               "verdict": reproduction_verdict(p2, human["conclusion_wins"])},
        "P3_protocol_gain_maybe_f1": {**p3, "verdict": gain_verdict(p3)},
        "status_counts": dict(Counter(status.tolist())),
        "final_label_counts": dict(Counter(final.tolist())),
        "round0_agreement_with_human_counterpart": {
            "with_conclusion_vs_rf": {"accuracy": round(float((with_initial == rf).mean()), 4), "maybe_f1": round(maybe_f1(with_initial, rf), 4)},
            "context_only_vs_rr": {"accuracy": round(float((without_initial == rr).mean()), 4), "maybe_f1": round(maybe_f1(without_initial, rr), 4)},
        },
        "maybe_f1_of_final": {
            "vs_gold": round(maybe_f1(final, gold), 4),
            "vs_annotator_with_conclusion": round(maybe_f1(final, rf), 4),
            "vs_annotator_without_conclusion": round(maybe_f1(final, rr), 4),
        },
        "maybe_f1_round0_vs_gold": {"with_conclusion": round(maybe_f1(with_initial, gold), 4),
                                    "context_only": round(maybe_f1(without_initial, gold), 4)},
        "initial_label_matrix": {"replay": matrix(with_initial.tolist(), without_initial.tolist()),
                                 "human": matrix(rf.tolist(), rr.tolist())},
        "yes_no_dispute_to_maybe": {"replay": [int(((final == "maybe") & yn).sum()), int(yn.sum())],
                                    "human": human["yes_no_dispute_to_maybe"]},
        "final_maybe_on_gold_maybe": {
            "human_unanimous": [int((final[human_unanimous] == "maybe").sum()), int(human_unanimous.sum())],
            "human_negotiated": [int((final[human_negotiated] == "maybe").sum()), int(human_negotiated.sum())],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    result = analyze(load_run(args.run), data, np.random.default_rng(args.seed), args.n_boot)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("P1_maybe_through_negotiation", "P2_conclusion_wins", "P3_protocol_gain_maybe_f1"):
        r = result[key]
        human = f" (human {r['human']})" if "human" in r else ""
        print(f"{key}: {r['value']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]{human} -> {r['verdict']}")
    print(f"status {result['status_counts']}; wrote {args.out}")


if __name__ == "__main__":
    main()
