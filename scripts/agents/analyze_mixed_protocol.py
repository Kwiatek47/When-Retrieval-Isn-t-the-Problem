"""Do two different models reproduce PubMedQA's ``maybe`` better than one model twice?

The first protocol replay (``analyze_annotation_protocol.py``) used qwen3:30b in both seats. Its
agents agreed far more often than the two humans (848 vs 698 of 995 questions) and ended with 24
``maybe`` against 109 gold. One reason may be that one model in both seats shares its own blind
spots. This analysis compares that run with two mixed runs of the same protocol, same prompts
(``label-defined@1``, ``negotiate@1``), thinking off for qwen3, temperature 0:

  QQ  qwen3:30b with the conclusion,            qwen3:30b without       (existing run)
  QM  qwen3:30b with the conclusion,            mistral-small3.2:24b without
  MQ  mistral-small3.2:24b with the conclusion, qwen3:30b without

Running both mixed orders keeps model and seat from being confounded: "mixed" below is the mean of
QM and MQ. All statistics are paired by question over the questions all three runs completed.

Tests, fixed before the mixed runs were made. Two co-primary tests, Bonferroni-adjusted: each uses
a 97.5% bootstrap interval (5000 resamples of questions, seed 47).
  M1 more disputes: share of questions on which the two agents disagree at round 0, mixed minus
     QQ. Supported if the interval lies above 0, refuted if at or below 0, inconclusive otherwise.
  M2 better maybe: maybe F1 of the final label against gold, mixed minus QQ. Removed questions
     count as answered "not maybe". Same rule.

Secondary, uncorrected: M1 and M2 per order (QM, MQ); the dispute rate of each run next to the
humans' (0.298 on these questions); for each mixed run the replay tests P1-P3 of
``analyze_annotation_protocol.py`` with their rules and human values; status and final-label
counts; round-0 ``maybe`` counts per model and seat (qwen3 and mistral, with and without the
conclusion) next to the humans' (110 with, 84 without); round-0 agreement of a virtual
mistral-mistral pair (mistral's round-0 labels from MQ and QM); maybe F1 of each run's final label
against each human annotator; final ``maybe`` on gold ``maybe`` that was unanimous versus
negotiated among the humans; accuracy of the final label against gold.

Known before registration: everything about QQ and the human labels; a 5-question pilot of the
mixed pair on the pilot questions (``PILOT_PMIDS``, excluded here) to check that mistral's replies
parse. Mistral's labels on the analysed questions were not seen.

Output: ``reports/debate/analysis/mixed_protocol.json``. No LLM calls.
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

from scripts.agents.analyze_annotation_protocol import (  # noqa: E402
    PQAL,
    WITH,
    WITHOUT,
    conclusion_wins,
    gain_verdict,
    human_reference,
    load_run,
    maybe_through_negotiation,
    protocol_gain,
    reproduction_verdict,
)
from scripts.agents.analyze_label_probe import maybe_f1  # noqa: E402

RUN_DIR = PROJECT_ROOT / "reports/debate/analysis/annotation_protocol"
RUNS = {
    "QQ": RUN_DIR / "protocol-qwen3-30b.think-off.all.jsonl",
    "QM": RUN_DIR / "protocol-qwen3-30b+mistral-small3.2-24b.think-off.all.jsonl",
    "MQ": RUN_DIR / "protocol-mistral-small3.2-24b+qwen3-30b.think-off.all.jsonl",
}
OUT = PROJECT_ROOT / "reports/debate/analysis/mixed_protocol.json"
CI_LEVEL = 1 - 0.05 / 2
SECONDARY_LEVEL = 0.95


def run_arrays(rows: list[dict], pmids: list[str]) -> dict[str, np.ndarray]:
    """Per-question arrays in ``pmids`` order."""
    by = {r["pmid"]: r for r in rows}
    pick = [by[p] for p in pmids]
    return {
        "status": np.array([r["status"] for r in pick]),
        "final": np.array([r["final"] if r["final"] else "removed" for r in pick]),
        "follows": np.array([r["final_follows"] for r in pick]),
        "with": np.array([r["initial"][WITH] for r in pick]),
        "without": np.array([r["initial"][WITHOUT] for r in pick]),
    }


def dispute_rate(with_initial: np.ndarray, without_initial: np.ndarray) -> float:
    return float((with_initial != without_initial).mean())


def mixed_minus_qq(stat):
    """Mean of the statistic over QM and MQ minus its value on QQ; arrays are QQ, QM, MQ in that order."""
    def diff(*arrays: np.ndarray) -> float:
        k = len(arrays) // 3
        qq, qm, mq = arrays[:k], arrays[k:2 * k], arrays[2 * k:]
        return (stat(*qm) + stat(*mq)) / 2 - stat(*qq)
    return diff


def one_minus_other(stat):
    def diff(*arrays: np.ndarray) -> float:
        k = len(arrays) // 2
        return stat(*arrays[:k]) - stat(*arrays[k:])
    return diff


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


def describe(run: dict[str, np.ndarray], gold: np.ndarray, rf: np.ndarray, rr: np.ndarray) -> dict:
    final = run["final"]
    unanimous = (rf == "maybe") & (rr == "maybe") & (gold == "maybe")
    negotiated = (gold == "maybe") & ~unanimous
    return {
        "dispute_rate": round(dispute_rate(run["with"], run["without"]), 4),
        "status_counts": dict(Counter(run["status"].tolist())),
        "final_label_counts": dict(Counter(final.tolist())),
        "round0_maybe": {"with_conclusion": int((run["with"] == "maybe").sum()),
                         "context_only": int((run["without"] == "maybe").sum())},
        "accuracy_vs_gold": round(float((final == gold).mean()), 4),
        "maybe_f1_of_final": {"vs_gold": round(maybe_f1(final, gold), 4),
                              "vs_annotator_with_conclusion": round(maybe_f1(final, rf), 4),
                              "vs_annotator_without_conclusion": round(maybe_f1(final, rr), 4)},
        "final_maybe_on_gold_maybe": {
            "human_unanimous": [int((final[unanimous] == "maybe").sum()), int(unanimous.sum())],
            "human_negotiated": [int((final[negotiated] == "maybe").sum()), int(negotiated.sum())],
        },
    }


def replay_tests(run: dict[str, np.ndarray], gold: np.ndarray, human: dict, rng, n_boot: int) -> dict:
    p1 = bootstrap(maybe_through_negotiation, (run["status"], run["final"]), rng, n_boot, SECONDARY_LEVEL)
    p2 = bootstrap(conclusion_wins, (run["follows"],), rng, n_boot, SECONDARY_LEVEL)
    p3 = bootstrap(protocol_gain, (run["final"], run["with"], gold), rng, n_boot, SECONDARY_LEVEL)
    return {
        "P1_maybe_through_negotiation": {**p1, "human": round(human["maybe_through_negotiation"], 4),
                                         "verdict": reproduction_verdict(p1, human["maybe_through_negotiation"])},
        "P2_conclusion_wins": {**p2, "human": round(human["conclusion_wins"], 4),
                               "verdict": reproduction_verdict(p2, human["conclusion_wins"])},
        "P3_protocol_gain_maybe_f1": {**p3, "verdict": gain_verdict(p3)},
    }


def analyze(rows: dict[str, list[dict]], data: dict, rng: np.random.Generator, n_boot: int) -> dict:
    ok = {k: {r["pmid"] for r in v if r["status"] != "failed"} for k, v in rows.items()}
    pmids = sorted(set.intersection(*ok.values()))
    runs = {k: run_arrays(v, pmids) for k, v in rows.items()}
    gold = np.array([data[p]["final_decision"] for p in pmids])
    rf = np.array([data[p]["reasoning_free_pred"] for p in pmids])
    rr = np.array([data[p]["reasoning_required_pred"] for p in pmids])
    human = human_reference({p: data[p] for p in pmids})

    def f1_final(final: np.ndarray, g: np.ndarray) -> float:
        return maybe_f1(final, g)

    order = ("QQ", "QM", "MQ")
    m1_arrays = tuple(a for k in order for a in (runs[k]["with"], runs[k]["without"]))
    m2_arrays = tuple(a for k in order for a in (runs[k]["final"], gold))
    m1 = bootstrap(mixed_minus_qq(dispute_rate), m1_arrays, rng, n_boot, CI_LEVEL)
    m2 = bootstrap(mixed_minus_qq(f1_final), m2_arrays, rng, n_boot, CI_LEVEL)

    per_order = {}
    for k in ("QM", "MQ"):
        per_order[k] = {
            "M1_dispute_rate_minus_QQ": bootstrap(
                one_minus_other(dispute_rate),
                (runs[k]["with"], runs[k]["without"], runs["QQ"]["with"], runs["QQ"]["without"]),
                rng, n_boot, SECONDARY_LEVEL),
            "M2_maybe_f1_minus_QQ": bootstrap(
                one_minus_other(f1_final), (runs[k]["final"], gold, runs["QQ"]["final"], gold),
                rng, n_boot, SECONDARY_LEVEL),
        }

    mistral_with, mistral_without = runs["MQ"]["with"], runs["QM"]["without"]
    qwen_with, qwen_without = runs["QM"]["with"], runs["MQ"]["without"]
    return {
        "n_questions": len(pmids),
        "n_completed": {k: len(v) for k, v in ok.items()},
        "human": {
            "dispute_rate": round(dispute_rate(rf, rr), 4),
            "round0_maybe": {"with_conclusion": int((rf == "maybe").sum()), "context_only": int((rr == "maybe").sum())},
            "gold_maybe": int((gold == "maybe").sum()),
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in human.items()},
        },
        "M1_dispute_rate_mixed_minus_QQ": {**m1, "verdict": gain_verdict(m1)},
        "M2_maybe_f1_mixed_minus_QQ": {**m2, "verdict": gain_verdict(m2)},
        "per_order": per_order,
        "runs": {k: describe(runs[k], gold, rf, rr) for k in order},
        "replay_tests": {k: replay_tests(runs[k], gold, human, rng, n_boot) for k in ("QM", "MQ")},
        "round0_maybe_by_model_and_seat": {
            "qwen3_with_conclusion": int((qwen_with == "maybe").sum()),
            "qwen3_context_only": int((qwen_without == "maybe").sum()),
            "mistral_with_conclusion": int((mistral_with == "maybe").sum()),
            "mistral_context_only": int((mistral_without == "maybe").sum()),
        },
        "virtual_mistral_pair_round0": {
            "dispute_rate": round(dispute_rate(mistral_with, mistral_without), 4),
            "both_maybe": int(((mistral_with == "maybe") & (mistral_without == "maybe")).sum()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    rows = {k: load_run(path) for k, path in RUNS.items()}
    result = analyze(rows, data, np.random.default_rng(args.seed), args.n_boot)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("M1_dispute_rate_mixed_minus_QQ", "M2_maybe_f1_mixed_minus_QQ"):
        r = result[key]
        print(f"{key}: {r['value']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] -> {r['verdict']}")
    print(f"{result['n_questions']} questions; wrote {args.out}")


if __name__ == "__main__":
    main()
