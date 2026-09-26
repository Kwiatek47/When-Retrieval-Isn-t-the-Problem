"""Audit the PQA-L labelling protocol, and re-score every system against the label a
context-only reader could actually produce (RQ5 + RQ8b).

Three sections, all from `pqal_label_table.jsonl` plus the cached per-case predictions
in `maybe_analysis.csv` — no LLM calls, no GPU.

1. **Protocol** — how the gold label was produced. The two single annotators disagree on
   299/1000 items, and in 215 of those the final label is the one from the annotator who
   read the author conclusion. `maybe` is where this bites hardest: of 110 gold `maybe`,
   only 23 are unanimous; 56 come from the conclusion-reading annotator alone.

2. **Ceilings** — each annotator's agreement with the gold label, overall and per class,
   with bootstrap CIs. The context-only annotator is the honest ceiling for any system
   that sees question + abstract: 78.1% overall. The 90.4% figure PubMedQA reports as
   "single human performance" belongs to the annotator who read the conclusion *and*
   whose vote the gold label usually adopted, so it is partly definitional.

3. **RQ8b** — every cached system re-scored against the context-only annotator's label
   instead of `final_decision`, paired per case. This is the difference between calling a
   prediction a model error and calling it a difference in information.

Usage:
  python scripts/agents/audit_pqal_labels.py
  python scripts/agents/audit_pqal_labels.py --n-boot 10000 --split test
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

from scripts.agents.bootstrap_stats import bootstrap_mean_ci, percentile  # noqa: E402
from scripts.agents.pqal_official import LABELS  # noqa: E402

ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
TABLE = ANALYSIS / "pqal_label_table.jsonl"
PREDICTIONS = ANALYSIS / "maybe_analysis.csv"
OUT = ANALYSIS / "pqal_protocol_audit.json"


def load_table(path: Path, split: str | None) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"{path} is missing — run scripts/agents/build_pqal_label_table.py first.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split:
        rows = [r for r in rows if r["split"] == split]
    if not rows:
        raise SystemExit(f"No rows left after filtering to split={split!r}.")
    return rows


def protocol_section(rows: list[dict], rng: random.Random, n_boot: int) -> dict:
    disputes = [r for r in rows if not r["annotators_agree"]]
    follows_conclusion = [
        1.0 if r["agreement_pattern"] == "dispute_final_follows_conclusion_annotator" else 0.0
        for r in disputes
    ]
    gold_maybe = [r for r in rows if r["gold_is_maybe"]]

    def maybe_source(r: dict) -> str:
        sees = r["sees_conclusion_pred"] == "maybe"
        ctx = r["context_only_pred"] == "maybe"
        if sees and ctx:
            return "both_annotators"
        if sees:
            return "conclusion_annotator_only"
        if ctx:
            return "context_only_annotator_only"
        return "neither_annotator"

    return {
        "n_items": len(rows),
        "n_disputes": len(disputes),
        "dispute_rate": round(len(disputes) / len(rows), 4),
        "dispute_resolution": dict(Counter(r["agreement_pattern"] for r in disputes)),
        "share_of_disputes_following_conclusion_annotator": bootstrap_mean_ci(
            follows_conclusion, rng, n_boot
        ),
        "n_gold_maybe": len(gold_maybe),
        "gold_maybe_source": dict(Counter(maybe_source(r) for r in gold_maybe)),
        "gold_maybe_unanimous": sum(1 for r in gold_maybe if maybe_source(r) == "both_annotators"),
        "gold_maybe_disputed": sum(1 for r in gold_maybe if maybe_source(r) != "both_annotators"),
        # A `maybe` gold label is itself, in most cases, a record of disagreement:
        # what share of all disputes ended as `maybe`?
        "share_of_disputes_ending_as_maybe": bootstrap_mean_ci(
            [1.0 if r["gold_is_maybe"] else 0.0 for r in disputes], rng, n_boot
        ),
    }


def _label_metrics(gold: list[str], pred: list[str], rng: random.Random, n_boot: int) -> dict:
    per_class = {}
    for lab in LABELS:
        support_idx = [i for i, g in enumerate(gold) if g == lab]
        hits = [1.0 if pred[i] == lab else 0.0 for i in support_idx]
        predicted = sum(1 for p in pred if p == lab)
        tp = sum(hits)
        per_class[lab] = {
            "support": len(support_idx),
            "n_predicted": predicted,
            "recall": bootstrap_mean_ci(hits, rng, n_boot) if hits else None,
            "precision": round(tp / predicted, 4) if predicted else None,
        }
    return {
        "overall_accuracy": bootstrap_mean_ci(
            [1.0 if p == g else 0.0 for p, g in zip(pred, gold)], rng, n_boot
        ),
        "per_class": per_class,
    }


def ceilings_section(rows: list[dict], rng: random.Random, n_boot: int) -> dict:
    gold = [r["gold"] for r in rows]
    return {
        "context_only_annotator": {
            "saw": "question + CONTEXTS — the model's information set",
            "is_ceiling_for_models": True,
            **_label_metrics(gold, [r["context_only_pred"] for r in rows], rng, n_boot),
        },
        "sees_conclusion_annotator": {
            "saw": "question + CONTEXTS + author conclusion",
            "is_ceiling_for_models": False,
            "caveat": (
                "Reported by PubMedQA as single-human performance. Inflated twice over: this "
                "annotator read the conclusion, and in disputes the gold label usually adopted "
                "this annotator's vote."
            ),
            **_label_metrics(gold, [r["sees_conclusion_pred"] for r in rows], rng, n_boot),
        },
    }


def _paired_accuracy_delta(
    pred: list[str], gold_a: list[str], gold_b: list[str], rng: random.Random, n_boot: int
) -> dict:
    """CI for accuracy(pred vs gold_a) - accuracy(pred vs gold_b), paired per case."""
    diffs = [
        (1.0 if p == a else 0.0) - (1.0 if p == b else 0.0)
        for p, a, b in zip(pred, gold_a, gold_b)
    ]
    n = len(diffs)
    point = sum(diffs) / n
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    low, high = percentile(boots, 0.025), percentile(boots, 0.975)
    return {
        "value": round(point, 4),
        "ci_low": round(low, 4),
        "ci_high": round(high, 4),
        "brackets_zero": low <= 0.0 <= high,
    }


def rq8b_section(rows: list[dict], predictions: Path, rng: random.Random, n_boot: int) -> dict:
    """Re-score cached system predictions against the context-only annotator's label.

    `maybe_analysis.csv` carries one row per (method, case) with the prediction and the
    gold label; the pmid in the case id joins it to the label table.
    """
    if not predictions.exists():
        return {"skipped": f"{predictions.name} not on this machine"}

    by_pmid = {r["pmid"]: r for r in rows}
    per_method: dict[str, dict] = {}
    with predictions.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pmid = row["id"].rsplit("-", 1)[-1]
            entry = by_pmid.get(pmid)
            pred = row.get("predicted_label") or ""
            if entry is None or pred not in LABELS:
                continue
            bucket = per_method.setdefault(row["method"], {"pred": [], "final": [], "context_only": []})
            bucket["pred"].append(pred)
            bucket["final"].append(entry["gold"])
            bucket["context_only"].append(entry["context_only_pred"])

    out: dict = {}
    for method, b in sorted(per_method.items()):
        # The human ceiling has to be recomputed on exactly the items this method was run
        # on, otherwise a 90-case arm is being compared against a 1000-item ceiling.
        ceiling = bootstrap_mean_ci(
            [1.0 if c == f else 0.0 for c, f in zip(b["context_only"], b["final"])], rng, n_boot
        )
        model = bootstrap_mean_ci(
            [1.0 if p == f else 0.0 for p, f in zip(b["pred"], b["final"])], rng, n_boot
        )
        out[method] = {
            "n_cases": len(b["pred"]),
            "vs_final_decision": _label_metrics(b["final"], b["pred"], rng, n_boot),
            "vs_context_only_annotator": _label_metrics(b["context_only"], b["pred"], rng, n_boot),
            "accuracy_delta_context_only_minus_final": _paired_accuracy_delta(
                b["pred"], b["context_only"], b["final"], rng, n_boot
            ),
            # The quantity that survives: how far the system is from the best a human with
            # the same information (no conclusion) manages on the same items.
            "context_only_human_ceiling_on_these_items": ceiling,
            "gap_to_ceiling": round(model["mean"] - ceiling["mean"], 4),
            # How many of the method's apparent errors are cases where a context-only
            # reader would not have produced the gold label either?
            "cases_where_the_two_golds_differ": sum(
                1 for f, c in zip(b["final"], b["context_only"]) if f != c
            ),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--split", choices=["test", "train_dev"], default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = load_table(args.table, args.split)

    result = {
        "split": args.split or "all",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "protocol": protocol_section(rows, rng, args.n_boot),
        "ceilings": ceilings_section(rows, rng, args.n_boot),
        "rq8b_systems_vs_context_only_label": rq8b_section(rows, args.predictions, rng, args.n_boot),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    p = result["protocol"]
    print(f"=== Protocol ({p['n_items']} items) ===")
    print(f"annotators disagree           {p['n_disputes']}/{p['n_items']} ({p['dispute_rate']:.1%})")
    s = p["share_of_disputes_following_conclusion_annotator"]
    print(
        f"final label follows the conclusion-reading annotator  "
        f"{s['mean']:.1%} [{s['ci_low']:.1%},{s['ci_high']:.1%}] of disputes"
    )
    m = p["share_of_disputes_ending_as_maybe"]
    print(f"disputes that end as gold `maybe`   {m['mean']:.1%} [{m['ci_low']:.1%},{m['ci_high']:.1%}]")
    print(f"gold `maybe` = {p['n_gold_maybe']}: {p['gold_maybe_unanimous']} unanimous, {p['gold_maybe_disputed']} disputed")
    for source, count in sorted(p["gold_maybe_source"].items(), key=lambda kv: -kv[1]):
        print(f"    {source:30s} {count}")

    print("\n=== Ceilings: annotator vs gold label ===")
    for name, c in result["ceilings"].items():
        acc = c["overall_accuracy"]
        rec = c["per_class"]["maybe"]["recall"]
        print(
            f"{name:28s} accuracy {acc['mean']:.3f} [{acc['ci_low']:.3f},{acc['ci_high']:.3f}]  "
            f"maybe recall {rec['mean']:.3f} [{rec['ci_low']:.3f},{rec['ci_high']:.3f}]"
        )

    rq8b = result["rq8b_systems_vs_context_only_label"]
    if "skipped" in rq8b:
        print(f"\n=== RQ8b skipped — {rq8b['skipped']} ===")
    else:
        print("\n=== RQ8b: system accuracy vs final_decision -> vs context-only label ===")
        for method, c in rq8b.items():
            f = c["vs_final_decision"]["overall_accuracy"]
            r = c["vs_context_only_annotator"]["overall_accuracy"]
            d = c["accuracy_delta_context_only_minus_final"]
            flag = "" if d["brackets_zero"] else "  *"
            print(
                f"{method:42s} n={c['n_cases']:3d}  {f['mean']:.3f} -> {r['mean']:.3f}  "
                f"delta {d['value']:+.3f} [{d['ci_low']:+.3f},{d['ci_high']:+.3f}]{flag}"
            )
        print("\n=== System vs the same-information human ceiling (accuracy on final_decision) ===")
        for method, c in rq8b.items():
            ceil = c["context_only_human_ceiling_on_these_items"]
            model = c["vs_final_decision"]["overall_accuracy"]
            print(
                f"{method:42s} model {model['mean']:.3f} vs human(no conclusion) "
                f"{ceil['mean']:.3f} [{ceil['ci_low']:.3f},{ceil['ci_high']:.3f}]  "
                f"gap {c['gap_to_ceiling']:+.3f}"
            )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
