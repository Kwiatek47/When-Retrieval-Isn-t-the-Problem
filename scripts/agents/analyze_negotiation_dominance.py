"""Does arguing from quoted evidence move dispute wins from the stronger model to the better-informed side?

In the mixed-model protocol replays the dispute went to qwen3:30b whichever side it played: the side
with the conclusion won 0.874 of settled disputes when qwen3 held it (QM) and 0.344 when mistral held
it (MQ). Among the annotators the side with the conclusion won 0.728. ``run_negotiation_replay.py``
discusses every QM and MQ dispute again from the same round-0 labels, in two arms: ``old``
(negotiate@1, a rerun of the source prompt) and ``evidence`` (justify@1 + negotiate-evidence@1).

With w_QM and w_MQ the share of settled disputes won by the side with the conclusion in each run:
  model advantage        D = (w_QM - w_MQ) / 2   (qwen3's share of wins over both runs, minus 0.5)
  information advantage  I = (w_QM + w_MQ) / 2 - 0.5
"Settled" = the final label is one side's round-0 label; ties of both sides do not occur in disputes.

Tests, fixed before the replay was run. Two co-primary tests, Bonferroni-adjusted: 97.5% bootstrap
intervals (5000 resamples of disputes, drawn within QM and within MQ, paired across arms, seed 47).
  N1 less model advantage: D(evidence) - D(old). Supported if the interval lies below 0, refuted
     if at or above 0, inconclusive otherwise.
  N2 more information advantage: I(evidence) - I(old). Supported if the interval lies above 0,
     refuted if at or below 0, inconclusive otherwise.

Secondary, uncorrected: D, I and w per run and arm with 95% intervals; the same from the source runs,
and old-arm vs source agreement on the final label (how much a rerun moves); the humans' I (+0.228);
status counts, rounds used and final ``maybe`` among the disputes; quotes: share that appear word for
word in the side's own text (whitespace and case normalised) and share of the conclusion side's
quotes taken from the conclusion; final-label accuracy and maybe F1 against gold over all 995
questions (agreed questions from the source run plus the replayed disputes).

Known before registration: the source runs (QQ, QM, MQ) and all their statistics; a pilot of both
arms on the pilot questions' disputes (3 disputes, QM only: MQ's pilot had none) to check parsing.

Output: ``reports/debate/analysis/negotiation_dominance.json``. No LLM calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.analyze_annotation_protocol import PILOT_PMIDS, PQAL, human_reference  # noqa: E402
from scripts.agents.analyze_label_probe import maybe_f1  # noqa: E402

RUN_DIR = PROJECT_ROOT / "reports/debate/analysis"
SOURCES = {
    "QM": RUN_DIR / "annotation_protocol/protocol-qwen3-30b+mistral-small3.2-24b.think-off.all.jsonl",
    "MQ": RUN_DIR / "annotation_protocol/protocol-mistral-small3.2-24b+qwen3-30b.think-off.all.jsonl",
}
ARMS = ("old", "evidence")
OUT = RUN_DIR / "negotiation_dominance.json"
CI_LEVEL = 1 - 0.05 / 2
WITH, WITHOUT = "with_conclusion", "context_only"


def load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row["pmid"] in PILOT_PMIDS:
            continue
        if row.get("status") != "failed" or row["pmid"] not in rows:
            rows[row["pmid"]] = row
    return rows


def win_with(follows: np.ndarray) -> float:
    settled = (follows == WITH) | (follows == WITHOUT)
    return float((follows[settled] == WITH).mean()) if settled.any() else float("nan")


def model_advantage(f_qm: np.ndarray, f_mq: np.ndarray) -> float:
    return (win_with(f_qm) - win_with(f_mq)) / 2


def information_advantage(f_qm: np.ndarray, f_mq: np.ndarray) -> float:
    return (win_with(f_qm) + win_with(f_mq)) / 2 - 0.5


def stratified_bootstrap(stat, qm: tuple[np.ndarray, ...], mq: tuple[np.ndarray, ...], rng, n_boot: int, level: float) -> dict:
    """``stat(*qm, *mq)``; resamples QM disputes and MQ disputes separately, same index for all arrays of a run."""
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(qm[0]), size=len(qm[0]))
        j = rng.integers(0, len(mq[0]), size=len(mq[0]))
        boots.append(stat(*(a[i] for a in qm), *(a[j] for a in mq)))
    boots = np.asarray([b for b in boots if b == b])
    tail = (1 - level) / 2 * 100
    return {"value": round(stat(*qm, *mq), 4), "ci_low": round(float(np.percentile(boots, tail)), 4),
            "ci_high": round(float(np.percentile(boots, 100 - tail)), 4), "ci_level": round(level, 4)}


def verdict_below(ci: dict) -> str:
    if ci["ci_high"] < 0:
        return "supported"
    if ci["ci_low"] >= 0:
        return "refuted"
    return "inconclusive"


def verdict_above(ci: dict) -> str:
    if ci["ci_low"] > 0:
        return "supported"
    if ci["ci_high"] <= 0:
        return "refuted"
    return "inconclusive"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_stats(rows: list[dict], data: dict) -> dict:
    from scripts.agents.run_label_probe import evidence_text

    verbatim, total, from_conclusion, with_total = 0, 0, 0, 0
    for row in rows:
        item = data[row["pmid"]]
        texts = {WITH: _norm(evidence_text(item, "context+conclusion")), WITHOUT: _norm(evidence_text(item, "context"))}
        conclusion = _norm(item["LONG_ANSWER"])
        for t in row["turns"]:
            quote = _norm(t.get("quote", "").strip('"'))
            if not quote:
                continue
            total += 1
            verbatim += quote in texts[t["annotator"]]
            if t["annotator"] == WITH:
                with_total += 1
                from_conclusion += quote in conclusion
    return {"quotes": total, "verbatim_share": round(verbatim / total, 4) if total else None,
            "conclusion_side_quotes": with_total,
            "conclusion_side_from_conclusion_share": round(from_conclusion / with_total, 4) if with_total else None}


def analyze(source: dict[str, dict], arms: dict[tuple[str, str], dict], data: dict, rng, n_boot: int) -> dict:
    ids = {}
    for run in SOURCES:
        disputed = {p for p, r in source[run].items() if r["status"] in ("negotiated", "removed")}
        ok = [{p for p, r in arms[(run, arm)].items() if r["status"] != "failed"} for arm in ARMS]
        ids[run] = sorted(disputed.intersection(*ok))

    def follows(rows: dict, run: str) -> np.ndarray:
        return np.array([rows[p]["final_follows"] for p in ids[run]])

    f = {(run, arm): follows(arms[(run, arm)], run) for run in SOURCES for arm in ARMS}
    f_src = {run: follows(source[run], run) for run in SOURCES}

    def diff(stat):
        return lambda qm_old, qm_new, mq_old, mq_new: stat(qm_new, mq_new) - stat(qm_old, mq_old)

    qm = (f[("QM", "old")], f[("QM", "evidence")])
    mq = (f[("MQ", "old")], f[("MQ", "evidence")])
    n1 = stratified_bootstrap(diff(model_advantage), qm, mq, rng, n_boot, CI_LEVEL)
    n2 = stratified_bootstrap(diff(information_advantage), qm, mq, rng, n_boot, CI_LEVEL)

    def per(label: str, fq: np.ndarray, fm: np.ndarray) -> dict:
        return {
            "D_model_advantage": stratified_bootstrap(model_advantage, (fq,), (fm,), rng, n_boot, 0.95),
            "I_information_advantage": stratified_bootstrap(information_advantage, (fq,), (fm,), rng, n_boot, 0.95),
            "w_with_QM": round(win_with(fq), 4), "w_with_MQ": round(win_with(fm), 4),
        }

    gold = {p: data[p]["final_decision"] for p in data if p not in PILOT_PMIDS}
    whole = {}
    for arm in ARMS:
        whole[arm] = {}
        for run in SOURCES:
            final = {p: (r["final"] or "removed") for p, r in source[run].items()}
            final.update({p: (arms[(run, arm)][p]["final"] or "removed") for p in ids[run]})
            pm = sorted(final)
            pred, g = np.array([final[p] for p in pm]), np.array([gold[p] for p in pm])
            whole[arm][run] = {"accuracy_vs_gold": round(float((pred == g).mean()), 4),
                               "maybe_f1_vs_gold": round(maybe_f1(pred, g), 4), "final_maybe": int((pred == "maybe").sum())}

    rerun_agreement = {
        run: round(float(np.mean([(arms[(run, "old")][p]["final"] == source[run][p]["final"]) for p in ids[run]])), 4)
        for run in SOURCES
    }
    return {
        "n_disputes": {run: len(ids[run]) for run in SOURCES},
        "human_I": round(human_reference({p: v for p, v in data.items() if p not in PILOT_PMIDS})["conclusion_wins"] - 0.5, 4),
        "N1_model_advantage_evidence_minus_old": {**n1, "verdict": verdict_below(n1)},
        "N2_information_advantage_evidence_minus_old": {**n2, "verdict": verdict_above(n2)},
        "by_arm": {**{arm: per(arm, f[("QM", arm)], f[("MQ", arm)]) for arm in ARMS},
                   "source": per("source", f_src["QM"], f_src["MQ"])},
        "old_rerun_same_final_as_source": rerun_agreement,
        "disputes": {
            f"{run}|{arm}": {
                "status_counts": dict(Counter(arms[(run, arm)][p]["status"] for p in ids[run])),
                "mean_rounds": round(float(np.mean([arms[(run, arm)][p]["rounds_used"] for p in ids[run]])), 3),
                "final_maybe": sum(arms[(run, arm)][p]["final"] == "maybe" for p in ids[run]),
            }
            for run in SOURCES for arm in ARMS
        },
        "quotes": {f"{run}|evidence": quote_stats([arms[(run, "evidence")][p] for p in ids[run]], data) for run in SOURCES},
        "all_questions": whole,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    data = json.loads(PQAL.read_text(encoding="utf-8"))
    source = {run: load(path) for run, path in SOURCES.items()}
    arms = {(run, arm): load(RUN_DIR / f"negotiation_replay/{run}.{arm}.jsonl") for run in SOURCES for arm in ARMS}
    result = analyze(source, arms, data, np.random.default_rng(args.seed), args.n_boot)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("N1_model_advantage_evidence_minus_old", "N2_information_advantage_evidence_minus_old"):
        r = result[key]
        print(f"{key}: {r['value']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] -> {r['verdict']}")
    print(f"disputes {result['n_disputes']}; wrote {args.out}")


if __name__ == "__main__":
    main()
