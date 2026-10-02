"""Audit the PQA-L labelling protocol, and re-score every system against the label a
context-only reader could actually produce (RQ5 + RQ8b).

Three sections, all from `pqal_label_table.jsonl` plus the cached per-case predictions
in `maybe_analysis.csv` — no LLM calls, no GPU.

1. **Protocol** — how the gold label was produced. The two single annotators disagree on
   299/1000 items, and in 215 of those the final label is the one from the annotator who
   read the author conclusion. `maybe` is where this bites hardest: of 110 gold `maybe`,
   only 23 are unanimous; 56 come from the conclusion-reading annotator alone.

2. **Annotator agreement with the gold label** — each annotator, overall and per class,
   with bootstrap CIs: 78.1% for the context-only annotator, 91.6% for the one who read
   the conclusion.

   These are NOT ceilings for models, and this section used to claim they were.
   Both annotators co-authored `final_decision`: under Alg. 1 a disagreement is settled by
   those same two people talking, and items they could not settle were dropped from the
   dataset. So each number is partly agreement-with-oneself. The registered test on an
   independent reference (commit `4a120c2`) scored annotator 2 and `qwen3:30b` against
   annotator 1, whom neither influenced, and the human advantage on `maybe` vanished:
   F1 0.247 vs 0.237, difference +0.011 [-0.156, +0.178]. Of the apparent gap of +0.286
   against `final_decision`, +0.275 [+0.142, +0.427] is co-authorship of the label.

   Read this section as a description of the protocol, never as a bound on models.
   The 90.4% PubMedQA reports as "single human performance" is the worst case: that
   annotator read the conclusion *and* the gold label usually adopted their vote (215/299).

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
BALANCED90 = (
    PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json"
)
CHECKPOINT = PROJECT_ROOT / "biolinkbert_pubmedqa_seed47_best/best"


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


def label_matrix_section(rows: list[dict], rng: random.Random, n_boot: int) -> dict:
    """Full context-only x sees-conclusion x final cross-tabulation (BRAKI A1, Figure 1).

    The 3x3 annotator-by-annotator table, and inside each cell how `final_decision` came
    out. This is the whole protocol in one object: the diagonal is agreement (where final
    is forced), the off-diagonal is the 299 negotiated items.
    """
    cells = Counter(
        (r["context_only_pred"], r["sees_conclusion_pred"], r["gold"]) for r in rows
    )
    pairs = {}
    for ctx in LABELS:
        for concl in LABELS:
            n = sum(cells[(ctx, concl, f)] for f in LABELS)
            pairs[f"context_only={ctx}|sees_conclusion={concl}"] = {
                "n": n,
                "annotators_agree": ctx == concl,
                "final_decision": {f: cells[(ctx, concl, f)] for f in LABELS},
            }

    # Items where the negotiated label is one neither annotator put forward. Under Alg. 1
    # these can only come out of the discussion step, so they are pure protocol artefacts.
    invented = [r for r in rows if r["gold"] not in (r["context_only_pred"], r["sees_conclusion_pred"])]

    return {
        "n_items": len(rows),
        "note": (
            "Cross-tab of the two raw annotations against final_decision. On the diagonal "
            "the annotators agreed and final_decision follows by construction; off-diagonal "
            "cells are the negotiated ones."
        ),
        "cells": pairs,
        "marginals": {
            "context_only": dict(Counter(r["context_only_pred"] for r in rows)),
            "sees_conclusion": dict(Counter(r["sees_conclusion_pred"] for r in rows)),
            "final_decision": dict(Counter(r["gold"] for r in rows)),
        },
        "final_label_neither_annotator_proposed": {
            "n": len(invented),
            "by_final_label": dict(Counter(r["gold"] for r in invented)),
            "share_of_all_items": bootstrap_mean_ci(
                [
                    1.0 if r["gold"] not in (r["context_only_pred"], r["sees_conclusion_pred"]) else 0.0
                    for r in rows
                ],
                rng,
                n_boot,
            ),
        },
    }


def balanced90_selection_section(
    rows: list[dict], balanced90: Path, rng: random.Random, n_boot: int
) -> dict:
    """Does the 30/30/30 `balanced90` sample distort annotator agreement? (BRAKI A1)

    `balanced90` oversamples gold `maybe` from 11% to 33%. Gold `maybe` is far more often
    a negotiated label than yes/no, so the sample cannot be assumed neutral: every
    human-vs-model number measured on these 90 cases sits on a subset enriched for
    annotator disagreement. Reported as two independent rates with CIs (the item sets are
    nested, not paired, so no paired delta is computed).
    """
    if not balanced90.exists():
        return {"skipped": f"{balanced90.name} not on this machine"}

    wanted = set()
    for case in json.loads(balanced90.read_text(encoding="utf-8")):
        case_id = case.get("id") or ""
        wanted.add(case_id.rsplit("-", 1)[-1])

    by_pmid = {r["pmid"]: r for r in rows}
    sampled = [by_pmid[p] for p in wanted if p in by_pmid]
    test_rows = [r for r in rows if r["split"] == "test"]

    def rates(subset: list[dict]) -> dict:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "gold_distribution": dict(Counter(r["gold"] for r in subset)),
            "annotator_agreement": bootstrap_mean_ci(
                [1.0 if r["annotators_agree"] else 0.0 for r in subset], rng, n_boot
            ),
            "context_only_accuracy_vs_final": bootstrap_mean_ci(
                [1.0 if r["context_only_pred"] == r["gold"] else 0.0 for r in subset], rng, n_boot
            ),
            "sees_conclusion_accuracy_vs_final": bootstrap_mean_ci(
                [1.0 if r["sees_conclusion_pred"] == r["gold"] else 0.0 for r in subset], rng, n_boot
            ),
        }

    return {
        "n_matched_to_label_table": len(sampled),
        "n_unmatched": len(wanted) - len(sampled),
        "balanced90": rates(sampled),
        "official_test_500": rates(test_rows),
        "all_1000": rates(rows),
    }


def training_labels_section(checkpoint: Path) -> dict:
    """Where the deployed classifier's 44 training `maybe` come from (BRAKI A1, B7).

    The training jsonl itself is not in the repo (it was built on Colab; `training_config`
    points at /content/...), so the count is reconstructed from two facts that *are*
    checkable here: PQA-A contributes no `maybe` at all, and the checkpoint's own
    dev_metrics.json records the dev support per class.
    """
    out: dict = {
        "pqa_a_maybe_count": 0,
        "pqa_a_source": "ori_pqaa.json, 211269 items: 196144 yes / 15125 no / 0 maybe (BRAKI A2)",
        "pqal_non_test_maybe_count": 55,
        "derivation": (
            "All training `maybe` must come from the 500 PQA-L items outside the official "
            "test split, which hold 55. The dev split takes some of them; training keeps "
            "the rest."
        ),
    }
    dev_metrics = checkpoint / "dev_metrics.json"
    if not dev_metrics.exists():
        out["dev_metrics"] = f"not on this machine ({dev_metrics})"
        return out

    dev = json.loads(dev_metrics.read_text(encoding="utf-8"))
    per_label = dev.get("per_label", {})
    dev_support = {lab: per_label.get(lab, {}).get("support") for lab in LABELS}
    dev_maybe = dev_support.get("maybe")
    out["dev_support"] = dev_support
    if isinstance(dev_maybe, int):
        out["train_maybe_count"] = 55 - dev_maybe
        out["train_maybe_share_of_34838"] = round((55 - dev_maybe) / 34838, 6)
    # The selection metric was macro-F1 on this dev set, so it is worth recording what the
    # selected checkpoint actually scored on the 11 dev `maybe`.
    out["dev_maybe_f1_of_selected_checkpoint"] = per_label.get("maybe", {}).get("f1")
    out["dev_macro_f1"] = dev.get("macro_f1")
    out["dev_ece"] = dev.get("ece")
    calibrated = checkpoint / "dev_metrics_calibrated.json"
    if calibrated.exists():
        out["dev_ece_after_temperature_scaling"] = json.loads(
            calibrated.read_text(encoding="utf-8")
        ).get("ece")
    temperature = checkpoint / "calibration.json"
    if temperature.exists():
        out["temperature"] = json.loads(temperature.read_text(encoding="utf-8")).get("temperature")
    return out


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


_CO_AUTHORSHIP_CAVEAT = (
    "NOT a ceiling for models. This annotator co-authored final_decision: under Alg. 1 "
    "disagreements are settled by the same two annotators talking, and unresolved items were "
    "dropped from the dataset, so this is partly agreement-with-oneself. Scored against an "
    "independent reference (commit 4a120c2) the human advantage on `maybe` disappears: "
    "annotator 2 F1 0.247 vs qwen3:30b 0.237, difference +0.011 [-0.156, +0.178]."
)


def annotator_agreement_section(rows: list[dict], rng: random.Random, n_boot: int) -> dict:
    """Each annotator's agreement with `final_decision` — a description of the protocol.

    Deliberately NOT called a ceiling: see the module docstring. Both annotators helped
    produce the label they are being scored against.
    """
    gold = [r["gold"] for r in rows]
    return {
        "context_only_annotator": {
            "saw": "question + CONTEXTS — the model's information set",
            "caveat": _CO_AUTHORSHIP_CAVEAT,
            **_label_metrics(gold, [r["context_only_pred"] for r in rows], rng, n_boot),
        },
        "sees_conclusion_annotator": {
            "saw": "question + CONTEXTS + author conclusion",
            "caveat": (
                _CO_AUTHORSHIP_CAVEAT + " Reported by PubMedQA as single-human performance, and "
                "inflated twice over: this annotator read the conclusion, and in disputes the "
                "gold label usually adopted this annotator's vote (215/299)."
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
        # Recomputed on exactly the items this method was run on, otherwise a 90-case arm
        # gets compared against a 1000-item number. This is the context-only annotator's
        # agreement with final_decision — not a bound on the model (see module docstring).
        human = bootstrap_mean_ci(
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
            # Descriptive only: how the context-only annotator scores against a label they
            # helped write, on the same items. Do not read the delta as distance-to-ceiling.
            "context_only_human_vs_final_on_these_items": human,
            "accuracy_delta_model_minus_context_only_human": round(
                model["mean"] - human["mean"], 4
            ),
            "interpretation_warning": _CO_AUTHORSHIP_CAVEAT,
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
    parser.add_argument("--balanced90", type=Path, default=BALANCED90)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--split", choices=["test", "train_dev"], default=None)
    args = parser.parse_args()

    rows = load_table(args.table, args.split)

    # The original three sections share one stream, in this order, because their CIs are
    # already quoted in docs/research and in the team plan — reordering them or inserting
    # a bootstrap in front would silently move the last digits.
    legacy_rng = random.Random(args.seed)
    protocol = protocol_section(rows, legacy_rng, args.n_boot)
    annotator_agreement = annotator_agreement_section(rows, legacy_rng, args.n_boot)
    rq8b = rq8b_section(rows, args.predictions, legacy_rng, args.n_boot)

    # Sections added later get their own derived stream, so adding the next one does not
    # perturb anything already published.
    def section_rng(name: str) -> random.Random:
        return random.Random(f"{args.seed}:{name}")

    result = {
        "split": args.split or "all",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "protocol": protocol,
        "label_matrix": label_matrix_section(rows, section_rng("label_matrix"), args.n_boot),
        "annotator_agreement_with_final": annotator_agreement,
        "balanced90_selection": balanced90_selection_section(
            rows, args.balanced90, section_rng("balanced90_selection"), args.n_boot
        ),
        "training_labels": training_labels_section(args.checkpoint),
        "rq8b_systems_vs_context_only_label": rq8b,
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

    lm = result["label_matrix"]
    print("\n=== Figure 1: context-only (rows) x sees-conclusion (cols), cell = n ===")
    print(f"{'':>8}" + "".join(f"{c:>22}" for c in LABELS))
    for ctx in LABELS:
        line = f"{ctx:>8}"
        for concl in LABELS:
            cell = lm["cells"][f"context_only={ctx}|sees_conclusion={concl}"]
            split = "/".join(str(cell["final_decision"][f]) for f in LABELS)
            line += f"{cell['n']:>6}  (f {split:>10})"
        print(line)
    print(f"    cell shows n, then final_decision split as {'/'.join(LABELS)}")
    inv = lm["final_label_neither_annotator_proposed"]
    print(f"    final label neither annotator proposed: {inv['n']}  {inv['by_final_label']}")

    b90 = result["balanced90_selection"]
    if "skipped" in b90:
        print(f"\n=== balanced90 check skipped — {b90['skipped']} ===")
    else:
        print("\n=== Does the 30/30/30 balanced90 sample distort annotator agreement? ===")
        for name in ("balanced90", "official_test_500", "all_1000"):
            s = b90[name]
            a = s["annotator_agreement"]
            print(
                f"{name:20s} n={s['n']:4d}  annotators agree "
                f"{a['mean']:.3f} [{a['ci_low']:.3f},{a['ci_high']:.3f}]  gold {s['gold_distribution']}"
            )

    tl = result["training_labels"]
    print("\n=== Deployed classifier: where the training `maybe` come from ===")
    print(f"    PQA-A `maybe`: {tl['pqa_a_maybe_count']}; PQA-L non-test `maybe`: {tl['pqal_non_test_maybe_count']}")
    if "train_maybe_count" in tl:
        print(
            f"    dev support {tl['dev_support']} -> train keeps {tl['train_maybe_count']} "
            f"`maybe` ({tl['train_maybe_share_of_34838']:.4%} of 34838)"
        )
        print(
            f"    selected on dev macro-F1 {tl['dev_macro_f1']:.4f}, but dev `maybe` F1 = "
            f"{tl['dev_maybe_f1_of_selected_checkpoint']}"
        )
        print(
            f"    dev ECE {tl['dev_ece']:.4f} -> {tl.get('dev_ece_after_temperature_scaling')} "
            f"at T={tl.get('temperature')}"
        )

    print("\n=== Annotator vs final_decision (NOT a ceiling — label is co-authored) ===")
    for name, c in result["annotator_agreement_with_final"].items():
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
        print("\n=== System vs context-only annotator, both on final_decision (descriptive) ===")
        print("    The annotator co-authored this label; the gap is NOT distance to a ceiling.")
        for method, c in rq8b.items():
            hum = c["context_only_human_vs_final_on_these_items"]
            model = c["vs_final_decision"]["overall_accuracy"]
            print(
                f"{method:42s} model {model['mean']:.3f} vs human(no conclusion) "
                f"{hum['mean']:.3f} [{hum['ci_low']:.3f},{hum['ci_high']:.3f}]  "
                f"delta {c['accuracy_delta_model_minus_context_only_human']:+.3f}"
            )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
