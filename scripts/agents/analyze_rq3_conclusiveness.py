"""RQ3 — the panel reports inconclusive evidence, the aggregation throws it away.

Every agent returns `evidence_conclusiveness` ("conclusive" / "inconclusive") next to its
label, so each case carries a continuous panel-level uncertainty signal that never reaches
the output: the aggregation rule emits exactly one of yes/no/maybe. This measures, per arm,
on the cached n=90 runs — no LLM calls, no GPU:

1. **Contradiction** — how often the panel flags inconclusive evidence and the system still
   commits to a confident yes/no.
2. **Discarded error detector** — whether that flag predicts the system being wrong. If
   accuracy is materially lower on flagged cases, the panel knows something the output does
   not carry.
3. **Discarded `maybe` detector** — AUROC of the continuous signal against gold `maybe`,
   compared against the AUROC of the emitted label itself (a 0/1 `maybe` indicator). The
   gap between the two is the information the aggregation destroys.

Usage:
  python scripts/agents/analyze_rq3_conclusiveness.py
  python scripts/agents/analyze_rq3_conclusiveness.py --n-boot 10000 --sample-size 40
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.bootstrap_stats import (  # noqa: E402
    bootstrap_mean_ci,
    paired_auroc_delta_ci,
    percentile,
)

DEBATE = PROJECT_ROOT / "reports/debate"
ANALYSIS = DEBATE / "analysis"
TABLE = ANALYSIS / "pqal_label_table.jsonl"
OUT = ANALYSIS / "rq3_conclusiveness.json"
SAMPLE_OUT = ANALYSIS / "rq3_qualitative_sample.csv"

INCONCLUSIVE = "inconclusive"
# Anything outside this set means the backend never filled the field in. The mock backend
# writes "" rather than omitting the key, so "not inconclusive" must not be read as
# "conclusive" — an unfilled arm has to drop out, not look unanimously confident.
VERDICTS = ("conclusive", INCONCLUSIVE)
CONFIDENT_LABELS = ("yes", "no")


def discover_runs(debate_dir: Path) -> list[Path]:
    """Report JSONs that could carry per-agent opinions, newest-agnostic and sorted."""
    return sorted(
        p
        for p in debate_dir.glob("*.json")
        if not p.name.endswith(".prompts.json") and not p.name.endswith(".checkpoint.json")
    )


def case_signal(case: dict) -> dict | None:
    """Reduce one case to the panel signal plus the committed decision.

    Returns None when the run carries no conclusiveness field at all (mock backends), so a
    silently empty arm cannot be mistaken for an arm where every agent was confident.
    """
    opinions = [o.get("opinion") or {} for o in case.get("final_opinions") or []]
    verdicts = [o.get("evidence_conclusiveness") for o in opinions]
    if not opinions or not any(v in VERDICTS for v in verdicts):
        return None

    # Only agents that actually returned a verdict form the denominator, so a partially
    # filled panel cannot look more confident than it was.
    n_rated = sum(1 for v in verdicts if v in VERDICTS)
    n_agents = len(opinions)
    n_inconclusive = sum(1 for v in verdicts if v == INCONCLUSIVE)
    confidences = [o["confidence_level"] for o in opinions if isinstance(o.get("confidence_level"), (int, float))]
    agent_labels = [o.get("top_1_diagnosis") for o in opinions]
    predicted = case.get("predicted_label") or ""

    return {
        "pmid": str(case.get("id", "")).rsplit("-", 1)[-1],
        "gold": case.get("expected_label") or "",
        "predicted": predicted,
        "correct": predicted == case.get("expected_label"),
        "committed_confidently": predicted in CONFIDENT_LABELS,
        "n_agents": n_agents,
        "n_rated": n_rated,
        "n_inconclusive": n_inconclusive,
        "inconclusive_fraction": n_inconclusive / n_rated,
        "any_inconclusive": n_inconclusive > 0,
        "all_inconclusive": n_inconclusive == n_rated,
        "agent_maybe_fraction": sum(1 for lab in agent_labels if lab == "maybe") / n_agents,
        "agent_labels": agent_labels,
        "agent_flags": {
            str(o.get("agent_id")): 1.0 if (op.get("evidence_conclusiveness") == INCONCLUSIVE) else 0.0
            for o, op in zip(case.get("final_opinions") or [], opinions)
            if op.get("evidence_conclusiveness") in VERDICTS
        },
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "min_confidence": min(confidences) if confidences else None,
        "rounds_run": case.get("rounds_run"),
        "aggregation_rule": case.get("aggregation_rule"),
    }


def load_run(path: Path) -> list[dict] | None:
    """Per-case signals for one report, or None if the report has no conclusiveness data."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    cases = report.get("cases")
    if not isinstance(cases, list):
        return None
    signals = [s for s in (case_signal(c) for c in cases) if s is not None]
    return signals or None


def _two_sample_delta(a: list[float], b: list[float], rng: random.Random, n_boot: int) -> dict:
    """CI for mean(a) - mean(b) with the two groups resampled independently.

    The groups here are different cases (flagged vs not), so there is nothing to pair.
    """
    if not a or not b:
        return {"value": None, "reason": "one group is empty"}
    point = sum(a) / len(a) - sum(b) / len(b)
    boots = []
    for _ in range(n_boot):
        ra = sum(a[rng.randrange(len(a))] for _ in a) / len(a)
        rb = sum(b[rng.randrange(len(b))] for _ in b) / len(b)
        boots.append(ra - rb)
    low, high = percentile(boots, 0.025), percentile(boots, 0.975)
    return {
        "value": round(point, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "brackets_zero": low <= 0.0 <= high,
    }


def contradiction_section(signals: list[dict]) -> dict:
    """How often a flagged panel still produces a confident label."""
    flagged = [s for s in signals if s["any_inconclusive"]]
    unanimous_flag = [s for s in signals if s["all_inconclusive"]]
    committed_despite_flag = [s for s in flagged if s["committed_confidently"]]
    gold_maybe = [s for s in signals if s["gold"] == "maybe"]

    return {
        "n_cases": len(signals),
        "agents_per_case": sorted({s["n_agents"] for s in signals}),
        "agents_returning_a_verdict_per_case": sorted({s["n_rated"] for s in signals}),
        "n_any_inconclusive": len(flagged),
        "n_all_inconclusive": len(unanimous_flag),
        "n_committed_confidently_despite_flag": len(committed_despite_flag),
        "share_of_flagged_that_still_commit": (
            round(len(committed_despite_flag) / len(flagged), 4) if flagged else None
        ),
        "n_unanimous_flag_but_confident": sum(1 for s in unanimous_flag if s["committed_confidently"]),
        # The sharpest count: the label is `maybe`, the panel said the evidence was
        # inconclusive, and the system answered yes/no anyway.
        "n_gold_maybe_flagged_but_committed": sum(
            1 for s in gold_maybe if s["any_inconclusive"] and s["committed_confidently"]
        ),
        "n_gold_maybe": len(gold_maybe),
        "emitted_label_distribution": dict(Counter(s["predicted"] for s in signals)),
        "emitted_by_flag": {
            "all_conclusive": dict(Counter(s["predicted"] for s in signals if not s["any_inconclusive"])),
            "some_inconclusive": dict(
                Counter(s["predicted"] for s in signals if s["any_inconclusive"] and not s["all_inconclusive"])
            ),
            "all_inconclusive": dict(Counter(s["predicted"] for s in unanimous_flag)),
        },
    }


def error_detector_section(signals: list[dict], rng: random.Random, n_boot: int) -> dict:
    """Does the discarded flag mark the cases the system gets wrong?"""
    flagged = [1.0 if s["correct"] else 0.0 for s in signals if s["any_inconclusive"]]
    clean = [1.0 if s["correct"] else 0.0 for s in signals if not s["any_inconclusive"]]
    return {
        "accuracy_when_all_agents_conclusive": bootstrap_mean_ci(clean, rng, n_boot) if clean else None,
        "accuracy_when_any_agent_inconclusive": bootstrap_mean_ci(flagged, rng, n_boot) if flagged else None,
        "accuracy_delta_flagged_minus_clean": _two_sample_delta(flagged, clean, rng, n_boot),
    }


def maybe_detector_section(signals: list[dict], rng: random.Random, n_boot: int) -> dict:
    """Continuous panel signal vs the emitted label, both scored against gold `maybe`.

    `emitted_maybe` is the system's own output read as a score (1 for `maybe`, else 0). Its
    AUROC is (sensitivity + specificity) / 2, so comparing the two numbers is a direct
    measure of what collapsing the panel into one label costs.
    """
    labels = [s["gold"] == "maybe" for s in signals]
    if not any(labels) or all(labels):
        return {"skipped": "arm has no usable gold `maybe` / non-`maybe` split"}

    predictors = {
        "inconclusive_fraction": [s["inconclusive_fraction"] for s in signals],
        "agent_maybe_fraction": [s["agent_maybe_fraction"] for s in signals],
        "emitted_maybe": [1.0 if s["predicted"] == "maybe" else 0.0 for s in signals],
    }
    if all(s["mean_confidence"] is not None for s in signals):
        predictors["one_minus_mean_confidence"] = [1.0 - s["mean_confidence"] for s in signals]

    out = paired_auroc_delta_ci(labels, predictors, "inconclusive_fraction", "emitted_maybe", rng, n_boot)
    # Every candidate signal against the system's own output, paired per case. A predictor
    # that beats `emitted_maybe` is information the panel had and the aggregation dropped.
    out["deltas_vs_emitted"] = {
        name: paired_auroc_delta_ci(labels, predictors, name, "emitted_maybe", rng, n_boot)["delta"]
        for name in predictors
        if name != "emitted_maybe"
    }
    return out


def per_agent_section(signals: list[dict], rng: random.Random, n_boot: int) -> dict:
    """Per-agent flag rate, plus how much the rate varies between agents.

    This is what explains the arm-level numbers: a flag that one persona raises on almost
    every item is a property of the persona, not of the item, and "at least one agent
    flagged it" then reduces to "that persona is on the panel".
    """
    agents = sorted({name for s in signals for name in s["agent_flags"]})
    per_agent: dict[str, dict] = {}
    labels = [s["gold"] == "maybe" for s in signals]
    usable_target = any(labels) and not all(labels)

    for name in agents:
        flags = [s["agent_flags"][name] for s in signals if name in s["agent_flags"]]
        entry = {"n_rated": len(flags), "inconclusive_rate": bootstrap_mean_ci(flags, rng, n_boot)}
        # An agent whose flag is constant carries no information about the item, so the
        # AUROC is only meaningful when the flag actually varies.
        if usable_target and len(flags) == len(signals) and 0.0 < sum(flags) < len(flags):
            paired = paired_auroc_delta_ci(
                labels,
                {"flag": flags, "emitted_maybe": [1.0 if s["predicted"] == "maybe" else 0.0 for s in signals]},
                "flag",
                "emitted_maybe",
                rng,
                n_boot,
            )
            entry["auroc_vs_gold_maybe"] = paired["per_predictor"]["flag"]
        else:
            entry["auroc_vs_gold_maybe"] = None
        per_agent[name] = entry

    rates = [e["inconclusive_rate"]["mean"] for e in per_agent.values()]
    return {
        "per_agent": per_agent,
        "rate_spread": round(max(rates) - min(rates), 4) if rates else None,
        "max_rate_agent": max(per_agent, key=lambda k: per_agent[k]["inconclusive_rate"]["mean"]) if rates else None,
    }


def arm_section(signals: list[dict], rng: random.Random, n_boot: int) -> dict:
    return {
        "aggregation_rules": sorted({str(s["aggregation_rule"]) for s in signals}),
        "rounds_run": sorted({s["rounds_run"] for s in signals if s["rounds_run"] is not None}),
        "overall_accuracy": bootstrap_mean_ci([1.0 if s["correct"] else 0.0 for s in signals], rng, n_boot),
        "contradiction": contradiction_section(signals),
        "per_agent_flag": per_agent_section(signals, rng, n_boot),
        "discarded_error_detector": error_detector_section(signals, rng, n_boot),
        "discarded_maybe_detector": maybe_detector_section(signals, rng, n_boot),
    }


def qualitative_sample(
    per_arm: dict[str, list[dict]], table: Path, size: int
) -> tuple[list[dict], str | None]:
    """Cases where gold is `maybe`, the panel flagged it, and the system committed anyway.

    Ranked by how loudly the panel objected, so the manual write-up starts with the clearest
    instances. Question and conclusion text come from `ori_pqal.json` (cached, hash-pinned).
    """
    candidates = [
        {**s, "arm": arm}
        for arm, signals in per_arm.items()
        for s in signals
        if s["gold"] == "maybe" and s["any_inconclusive"] and s["committed_confidently"]
    ]
    candidates.sort(key=lambda s: (-s["inconclusive_fraction"], s["mean_confidence"] is None, -(s["mean_confidence"] or 0.0)))

    by_pmid: dict[str, dict] = {}
    if table.exists():
        for line in table.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                by_pmid[row["pmid"]] = row

    texts: dict[str, dict] = {}
    note = None
    try:
        from scripts.agents.pqal_official import load_ori_pqal

        texts = load_ori_pqal(download=False)
    except (ImportError, SystemExit, FileNotFoundError, OSError) as exc:
        note = f"question/conclusion text omitted: {exc}"

    rows = []
    for c in candidates[:size]:
        item = texts.get(c["pmid"], {})
        label_row = by_pmid.get(c["pmid"], {})
        rows.append(
            {
                "arm": c["arm"],
                "pmid": c["pmid"],
                "gold": c["gold"],
                "predicted": c["predicted"],
                "inconclusive_agents": f"{c['n_inconclusive']}/{c['n_agents']}",
                "mean_confidence": round(c["mean_confidence"], 3) if c["mean_confidence"] is not None else "",
                "agent_labels": "|".join(str(lab) for lab in c["agent_labels"]),
                "context_only_annotator": label_row.get("context_only_pred", ""),
                "sees_conclusion_annotator": label_row.get("sees_conclusion_pred", ""),
                "question": item.get("QUESTION", ""),
                "conclusion": item.get("LONG_ANSWER", ""),
            }
        )
    return rows, note


def write_sample(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debate-dir", type=Path, default=DEBATE)
    parser.add_argument("--runs", type=Path, nargs="*", default=None, help="explicit report JSONs")
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--sample-out", type=Path, default=SAMPLE_OUT)
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    paths = args.runs if args.runs else discover_runs(args.debate_dir)

    per_arm: dict[str, list[dict]] = {}
    skipped: dict[str, str] = {}
    for path in paths:
        signals = load_run(path)
        if signals is None:
            skipped[path.stem] = "no per-agent evidence_conclusiveness"
            continue
        per_arm[path.stem] = signals
    if not per_arm:
        raise SystemExit(f"No run in {args.debate_dir} carries per-agent conclusiveness verdicts.")

    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "runs_without_conclusiveness": skipped,
        "arms": {arm: arm_section(signals, rng, args.n_boot) for arm, signals in sorted(per_arm.items())},
    }

    sample, note = qualitative_sample(per_arm, args.table, args.sample_size)
    result["qualitative_sample"] = {
        "n_rows": len(sample),
        "path": str(args.sample_out.relative_to(PROJECT_ROOT)) if sample else None,
        "selection": "gold `maybe`, at least one agent flagged inconclusive, system answered yes/no",
        "note": note,
    }
    if sample:
        write_sample(sample, args.sample_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== RQ3: panel flags inconclusive evidence, system answers anyway ===")
    for arm, sec in result["arms"].items():
        c = sec["contradiction"]
        share = c["share_of_flagged_that_still_commit"]
        print(
            f"{arm:44s} n={c['n_cases']:3d}  flagged {c['n_any_inconclusive']:3d}  "
            f"still confident {c['n_committed_confidently_despite_flag']:3d}"
            f"{f' ({share:.0%})' if share is not None else ''}  "
            f"gold-maybe flagged-but-committed {c['n_gold_maybe_flagged_but_committed']:2d}/{c['n_gold_maybe']:2d}"
        )

    print("\n=== Who raises the flag? (per-agent inconclusive rate) ===")
    for arm, sec in result["arms"].items():
        pa = sec["per_agent_flag"]
        rates = " ".join(
            f"{name}={e['inconclusive_rate']['mean']:.2f}" for name, e in pa["per_agent"].items()
        )
        print(f"{arm:44s} spread {pa['rate_spread']:.2f}  ({rates})")

    print("\n=== Is the discarded flag an error detector? (accuracy) ===")
    for arm, sec in result["arms"].items():
        d = sec["discarded_error_detector"]
        clean, flagged, delta = d["accuracy_when_all_agents_conclusive"], d["accuracy_when_any_agent_inconclusive"], d["accuracy_delta_flagged_minus_clean"]
        if not clean or not flagged or delta.get("value") is None:
            print(f"{arm:44s} (one group empty)")
            continue
        flag = "" if delta["brackets_zero"] else "  *"
        print(
            f"{arm:44s} conclusive {clean['mean']:.3f} (n={clean['n']:3d})  "
            f"flagged {flagged['mean']:.3f} (n={flagged['n']:3d})  "
            f"delta {delta['value']:+.3f} [{delta['ci_low']:+.3f},{delta['ci_high']:+.3f}]{flag}"
        )

    print("\n=== Continuous panel signal vs the emitted label, on gold `maybe` (AUROC) ===")
    for arm, sec in result["arms"].items():
        m = sec["discarded_maybe_detector"]
        if "skipped" in m:
            print(f"{arm:44s} {m['skipped']}")
            continue
        emitted = m["per_predictor"]["emitted_maybe"]
        print(f"{arm}  (emitted label as a score: {emitted['auroc']:.3f})")
        for name, delta in m["deltas_vs_emitted"].items():
            p = m["per_predictor"][name]
            beats = "" if delta["brackets_zero"] else "  * beats the emitted label"
            sep = "" if p["brackets_chance"] else "  sep!"
            print(
                f"    {name:28s} {p['auroc']:.3f} [{p['ci_low']:.3f},{p['ci_high']:.3f}]{sep}  "
                f"delta {delta['value']:+.3f} [{delta['ci_low']:+.3f},{delta['ci_high']:+.3f}]{beats}"
            )

    if sample:
        print(f"\nWrote {len(sample)} qualitative cases to {args.sample_out}")
    if note:
        print(f"  {note}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
