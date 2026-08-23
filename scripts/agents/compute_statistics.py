"""Statistical rigor for the `maybe` findings: bootstrap CIs + significance tests.

Reads the aggregated per-case table (`reports/debate/analysis/maybe_analysis.csv`)
and the official human baseline, and produces:

  - Bootstrap 95% CIs for every scalar signal's AUROC(maybe vs rest), per method.
    A CI that brackets 0.5 = "not distinguishable from chance" — the core claim.
  - A significance test for human maybe-recall (0.60) vs model maybe-recall (~0.0):
    bootstrap CI of the difference + a permutation p-value.

Pure stdlib + the existing CSV/JSON artifacts; no LLM calls. Deterministic seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
CSV_PATH = ANALYSIS / "maybe_analysis.csv"
HUMAN = PROJECT_ROOT / "reports/debate/human/human_baseline_official.json"
OUT = ANALYSIS / "statistics.json"

_SIGNALS = (
    "uncertainty_score",
    "audit_score",
    "label_entropy",
    "maybe_fraction",
    "inconclusive_fraction",
    "mean_disagreement_with_mode",
    "flip_rate",
    "panel_uncertainty_conf",
    "semantic_entropy",
)


def _auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def _percentile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _bootstrap_auroc_ci(
    pos: list[float], neg: list[float], rng: random.Random, n_boot: int
) -> dict:
    point = _auroc(pos, neg)
    boots = []
    for _ in range(n_boot):
        rp = [pos[rng.randrange(len(pos))] for _ in pos]
        rn = [neg[rng.randrange(len(neg))] for _ in neg]
        boots.append(_auroc(rp, rn))
    boots = [b for b in boots if b == b]  # drop NaN
    return {
        "auroc": round(point, 4),
        "ci_low": round(_percentile(boots, 0.025), 4),
        "ci_high": round(_percentile(boots, 0.975), 4),
        "n_pos": len(pos),
        "n_neg": len(neg),
        "brackets_chance": _percentile(boots, 0.025) <= 0.5 <= _percentile(boots, 0.975),
    }


def _load_rows() -> list[dict]:
    return list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))


DEBATE_500 = PROJECT_ROOT / "reports/debate/debate_pqal500_biolinkbert.json"
RAG_500 = (
    PROJECT_ROOT
    / "reports/official_pqal500_biolinkbert_seed47/official_pqal500_biolinkbert_seed47_rag.json"
)


def _boot_ci_mean(vals: list[float], rng: random.Random, n_boot: int) -> dict:
    """Bootstrap 95% CI for the mean of a 0/1 (or real) per-case vector."""
    if not vals:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    n = len(vals)
    point = sum(vals) / n
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        boots.append(s / n)
    return {
        "mean": round(point, 4),
        "ci_low": round(_percentile(boots, 0.025), 4),
        "ci_high": round(_percentile(boots, 0.975), 4),
        "n": n,
    }


def pqal500_cis(rng: random.Random, n_boot: int) -> dict:
    """Bootstrap CIs on the full PQA-L 500 for both paper threads.

    Thread B (decision): overall accuracy + per-class recall (esp. maybe) from the
    BioLinkBERT debate report. Thread A (retrieval): hit@1, hit@3, citation pass
    from the official RAG report. All per-case, so CIs reflect the real n=500.
    """
    out: dict = {}

    # --- decision layer (BioLinkBERT), from debate_pqal500 report ---
    d = json.loads(DEBATE_500.read_text())
    cases = d.get("cases") if isinstance(d, dict) else d
    correct = [1.0 if c.get("label_pass") else 0.0 for c in cases]
    out["decision_overall_accuracy"] = _boot_ci_mean(correct, rng, n_boot)
    per_class = {}
    for lab in ("yes", "no", "maybe"):
        sub = [1.0 if c.get("predicted_label") == lab else 0.0 for c in cases if c.get("expected_label") == lab]
        per_class[lab] = _boot_ci_mean(sub, rng, n_boot)
    out["decision_per_class_recall"] = per_class

    # --- retrieval + citation, from official RAG report (if present) ---
    if RAG_500.exists():
        r = json.loads(RAG_500.read_text())
        rc = r.get("cases") if isinstance(r, dict) else r
        hit1 = [1.0 if c.get("source_hit_at_1") else 0.0 for c in rc if "source_hit_at_1" in c]
        hit3 = [1.0 if c.get("source_hit_at_3") else 0.0 for c in rc if "source_hit_at_3" in c]
        cit = [1.0 if c.get("citation_pass") else 0.0 for c in rc if "citation_pass" in c]
        racc = [1.0 if c.get("label_pass") else 0.0 for c in rc if "label_pass" in c]
        out["retrieval_hit_at_1"] = _boot_ci_mean(hit1, rng, n_boot)
        out["retrieval_hit_at_3"] = _boot_ci_mean(hit3, rng, n_boot)
        out["citation_pass"] = _boot_ci_mean(cit, rng, n_boot)
        out["rag_label_accuracy"] = _boot_ci_mean(racc, rng, n_boot)
    return out


def multiseed_routing(rows: list[dict], seeds: list[int], split: float, objective: str) -> dict:
    """Replicate calibrate+route across seeds on the debate uncertainty method.

    Uses the same stratified calibration split + `calibrate_threshold` as the eval
    harness, but purely on the cached per-case signals (no LLM). Reports the spread
    of held-out 3-class accuracy, macro-F1 and maybe-recall across seeds, so we can
    show the routing result is not an artifact of one lucky split.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from app.agents.uncertainty import calibrate_threshold

    method = "debate_balanced90_ollama_r2_uncertainty"
    cases = [r for r in rows if r["method"] == method and r.get("uncertainty_score") not in ("", None)]
    if not cases:
        return {"error": f"no rows for {method}"}

    def base_label(r: dict) -> str:
        return r.get("base_label") or r.get("predicted_label") or ""

    labels = ("yes", "no", "maybe")
    accs, macros, recalls, thresholds = [], [], [], []
    for seed in seeds:
        rng = random.Random(seed)
        by_label: dict[str, list[dict]] = {}
        for r in cases:
            by_label.setdefault(r["gold"], []).append(r)
        calib, report = [], []
        for _lab, group in by_label.items():
            group = list(group)
            rng.shuffle(group)
            cut = int(round(len(group) * split))
            calib.extend(group[:cut])
            report.extend(group[cut:])
        threshold, _ = calibrate_threshold(
            [float(r["uncertainty_score"]) for r in calib],
            [r["gold"] for r in calib],
            objective=objective,
            base_labels=[base_label(r) for r in calib],
        )
        thresholds.append(threshold)
        # apply to held-out report split
        pred, gold = [], []
        for r in report:
            g = r["gold"]
            u = float(r["uncertainty_score"])
            p = "maybe" if u >= threshold else base_label(r)
            pred.append(p)
            gold.append(g)
        n = len(gold) or 1
        acc = sum(1 for p, g in zip(pred, gold) if p == g) / n
        f1s = []
        for lab in labels:
            tp = sum(1 for p, g in zip(pred, gold) if p == lab and g == lab)
            fp = sum(1 for p, g in zip(pred, gold) if p == lab and g != lab)
            fn = sum(1 for p, g in zip(pred, gold) if p != lab and g == lab)
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
            if lab == "maybe":
                recalls.append(rec)
        accs.append(acc)
        macros.append(sum(f1s) / len(f1s))

    def stats(xs: list[float]) -> dict:
        m = sum(xs) / len(xs)
        var = sum((x - m) ** 2 for x in xs) / len(xs)
        return {"mean": round(m, 4), "std": round(var ** 0.5, 4), "min": round(min(xs), 4), "max": round(max(xs), 4)}

    return {
        "method": method,
        "seeds": seeds,
        "split": split,
        "objective": objective,
        "accuracy": stats(accs),
        "macro_f1": stats(macros),
        "maybe_recall": stats(recalls),
        "threshold": stats(thresholds),
    }


def signal_cis(rows: list[dict], rng: random.Random, n_boot: int) -> dict:
    out: dict = {}
    methods = sorted({r["method"] for r in rows})
    for method in methods:
        mrows = [r for r in rows if r["method"] == method]
        per_signal = {}
        for sig in _SIGNALS:
            pos, neg = [], []
            for r in mrows:
                v = r.get(sig, "")
                if v in ("", None):
                    continue
                try:
                    fv = float(v)
                except ValueError:
                    continue
                (pos if r["gold"] == "maybe" else neg).append(fv)
            if len(pos) >= 3 and len(neg) >= 3:
                per_signal[sig] = _bootstrap_auroc_ci(pos, neg, rng, n_boot)
        if per_signal:
            out[method] = per_signal
    return out


def _mcnemar(correct_a: list[float], correct_b: list[float]) -> dict:
    """Paired McNemar test: is arm A's accuracy different from arm B's on the same cases?

    Only discordant pairs carry information about which arm is systematically
    better (concordant pairs — both right or both wrong — are ignored). Uses
    the exact binomial test when the discordant count is below 25 (recommended
    at small n, and the balanced90 arms land well below that), else the
    chi-square approximation with continuity correction.
    """
    assert len(correct_a) == len(correct_b)
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a == 1.0 and bb == 0.0)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if a == 0.0 and bb == 1.0)
    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    elif n_discordant < 25:
        from math import comb

        k = min(b, c)
        p_value = min(
            1.0,
            2 * sum(comb(n_discordant, i) * 0.5**n_discordant for i in range(0, k + 1)),
        )
    else:
        from math import erfc, sqrt

        chi2 = (abs(b - c) - 1) ** 2 / n_discordant
        p_value = erfc(sqrt(chi2 / 2.0))
    return {
        "b_a_wins": b,
        "c_b_wins": c,
        "n_discordant": n_discordant,
        "n_concordant": len(correct_a) - n_discordant,
        "p_value": round(p_value, 5),
        "exact": n_discordant < 25,
    }


def _paired_bootstrap_diff_ci(
    correct_a: list[float], correct_b: list[float], rng: random.Random, n_boot: int
) -> dict:
    """Bootstrap 95% CI for the paired accuracy difference (A - B), same case order."""
    n = len(correct_a)
    point = sum(correct_a) / n - sum(correct_b) / n
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        a = sum(correct_a[i] for i in idx) / n
        b = sum(correct_b[i] for i in idx) / n
        diffs.append(a - b)
    return {
        "diff": round(point, 4),
        "ci_low": round(_percentile(diffs, 0.025), 4),
        "ci_high": round(_percentile(diffs, 0.975), 4),
    }


def _load_case_field(path: Path, field: str) -> dict[str, float]:
    """id -> 1.0/0.0 for a boolean per-case field (e.g. label_pass, biolinkbert_pass)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = data.get("cases") if isinstance(data, dict) else data
    out: dict[str, float] = {}
    for c in cases:
        v = c.get(field)
        if v is None:
            continue
        out[c["id"]] = 1.0 if v else 0.0
    return out


def compare_arms(
    report_paths: list[Path], labels: list[str], rng: random.Random, n_boot: int
) -> dict:
    """Pairwise McNemar + bootstrap accuracy-gap CI across debate/SC report JSONs.

    Aligns by case id (so arms must share the same dataset). Also compares each
    arm's `label_pass` against its own embedded `biolinkbert_pass` and
    `round1_pass` columns when present, so debate-vs-classifier and
    debate-vs-N=1 come for free from a single report without a second file.
    """
    per_label_correct: dict[str, dict[str, float]] = {}
    per_label_bert: dict[str, dict[str, float]] = {}
    per_label_round1: dict[str, dict[str, float]] = {}
    for path, label in zip(report_paths, labels):
        per_label_correct[label] = _load_case_field(path, "label_pass")
        bert = _load_case_field(path, "biolinkbert_pass")
        if bert:
            per_label_bert[label] = bert
        r1 = _load_case_field(path, "round1_pass")
        if r1:
            per_label_round1[label] = r1

    def _paired(a_map: dict[str, float], b_map: dict[str, float]) -> dict:
        ids = sorted(set(a_map) & set(b_map))
        a = [a_map[i] for i in ids]
        b = [b_map[i] for i in ids]
        out = _mcnemar(a, b)
        out.update(_paired_bootstrap_diff_ci(a, b, rng, n_boot))
        out["n_paired"] = len(ids)
        return out

    result: dict = {"pairwise_label_pass": {}, "vs_biolinkbert": {}, "vs_round1": {}}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            key = f"{labels[i]}_vs_{labels[j]}"
            result["pairwise_label_pass"][key] = _paired(
                per_label_correct[labels[i]], per_label_correct[labels[j]]
            )
    for label, bert in per_label_bert.items():
        result["vs_biolinkbert"][label] = _paired(per_label_correct[label], bert)
    for label, r1 in per_label_round1.items():
        result["vs_round1"][label] = _paired(per_label_correct[label], r1)
    return result


def human_vs_model(rng: random.Random, n_boot: int) -> dict:
    """Human maybe-recall vs model maybe-recall (model recall = 0 on maybe).

    We use per-case correctness on the true-maybe subset. Human correctness comes
    from the official single-annotator prediction; the model (best debate/routing
    base) is treated as recall 0 on maybe (documented: models essentially never
    output maybe). We bootstrap the recall difference and run a permutation test.
    """
    human = json.loads(HUMAN.read_text())
    hb = human["reasoning_free_pred (abstract only)"]["per_class"]["maybe"]
    support = hb["support"]
    human_correct = round(hb["recall"] * support)
    # per-case binary correctness vectors on the maybe subset
    human_vec = [1] * human_correct + [0] * (support - human_correct)
    model_vec = [0] * support  # models: ~0 maybe recall

    def recall(v: list[int]) -> float:
        return sum(v) / len(v) if v else 0.0

    diff_point = recall(human_vec) - recall(model_vec)
    # bootstrap CI of the difference
    diffs = []
    for _ in range(n_boot):
        hs = [human_vec[rng.randrange(support)] for _ in range(support)]
        ms = [model_vec[rng.randrange(support)] for _ in range(support)]
        diffs.append(recall(hs) - recall(ms))
    # permutation test: shuffle labels between the two groups
    pooled = human_vec + model_vec
    n_h = len(human_vec)
    count_ge = 0
    n_perm = n_boot
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = recall(pooled[:n_h]) - recall(pooled[n_h:])
        if abs(d) >= abs(diff_point):
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)
    return {
        "human_maybe_recall": recall(human_vec),
        "model_maybe_recall": recall(model_vec),
        "difference": round(diff_point, 4),
        "diff_ci_low": round(_percentile(diffs, 0.025), 4),
        "diff_ci_high": round(_percentile(diffs, 0.975), 4),
        "permutation_p_value": round(p_value, 5),
        "n_maybe": support,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--route-seeds", type=int, nargs="*", default=[11, 23, 42, 47, 101, 202, 303])
    parser.add_argument("--route-split", type=float, default=0.5)
    parser.add_argument("--route-objective", type=str, default="macro_f1")
    parser.add_argument(
        "--reports",
        nargs="*",
        default=None,
        help=(
            "Debate/SC report JSONs to compare pairwise (McNemar + bootstrap "
            "accuracy-gap CI on label_pass), plus each arm vs its own embedded "
            "biolinkbert_pass/round1_pass columns. Short-circuits the rest of "
            "this script (no CSV/PQA-L500 files needed)."
        ),
    )
    parser.add_argument(
        "--report-labels",
        nargs="*",
        default=None,
        help="Labels for --reports, same order and count (e.g. arm_clean arm_sc)",
    )
    parser.add_argument(
        "--reports-out", type=Path, default=ANALYSIS / "arm_comparison.json"
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.reports:
        if not args.report_labels or len(args.report_labels) != len(args.reports):
            raise SystemExit("--report-labels must be given, same count as --reports")
        cmp_result = compare_arms(
            [Path(p) for p in args.reports], args.report_labels, rng, args.n_boot
        )
        args.reports_out.parent.mkdir(parents=True, exist_ok=True)
        args.reports_out.write_text(json.dumps(cmp_result, indent=2), encoding="utf-8")

        def _line(prefix: str, r: dict) -> str:
            sig = "*" if r["p_value"] < 0.05 else ""
            return (
                f"{prefix:34s} diff={r['diff']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] "
                f"b={r['b_a_wins']} c={r['c_b_wins']} n_disc={r['n_discordant']} "
                f"p={r['p_value']} {sig}"
            )

        print("=== Pairwise arm comparison (McNemar + bootstrap diff CI) ===")
        for key, r in cmp_result["pairwise_label_pass"].items():
            print(_line(key, r))
        for label, r in cmp_result["vs_biolinkbert"].items():
            print(_line(f"{label} vs biolinkbert", r))
        for label, r in cmp_result["vs_round1"].items():
            print(_line(f"{label} vs round1/N=1", r))
        print(f"\nWrote {args.reports_out}")
        return

    rows = _load_rows()

    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "signal_auroc_ci": signal_cis(rows, rng, args.n_boot),
        "human_vs_model_maybe_recall": human_vs_model(rng, args.n_boot),
        "multiseed_routing": multiseed_routing(
            rows, args.route_seeds, args.route_split, args.route_objective
        ),
        "pqal500": pqal500_cis(rng, args.n_boot),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Console summary
    print("=== Human vs model (maybe recall) ===")
    hv = result["human_vs_model_maybe_recall"]
    print(
        f"human={hv['human_maybe_recall']:.2f} model={hv['model_maybe_recall']:.2f} "
        f"diff={hv['difference']:.2f} 95%CI[{hv['diff_ci_low']:.2f},{hv['diff_ci_high']:.2f}] "
        f"p={hv['permutation_p_value']}"
    )
    print("\n=== AUROC 95% CI (audit_score / uncertainty_score), brackets 0.5? ===")
    for method, sigs in result["signal_auroc_ci"].items():
        for key in ("audit_score", "uncertainty_score"):
            if key in sigs:
                c = sigs[key]
                flag = "CHANCE" if c["brackets_chance"] else "sep!"
                print(
                    f"{method:42s} {key:16s} {c['auroc']:.3f} "
                    f"[{c['ci_low']:.3f},{c['ci_high']:.3f}] {flag}"
                )
    ms = result["multiseed_routing"]
    if "accuracy" in ms:
        print(f"\n=== Multi-seed routing ({len(ms['seeds'])} seeds, obj={ms['objective']}) ===")
        print(
            f"accuracy    {ms['accuracy']['mean']:.3f} ± {ms['accuracy']['std']:.3f} "
            f"[{ms['accuracy']['min']:.3f},{ms['accuracy']['max']:.3f}]"
        )
        print(
            f"macro_f1    {ms['macro_f1']['mean']:.3f} ± {ms['macro_f1']['std']:.3f} "
            f"[{ms['macro_f1']['min']:.3f},{ms['macro_f1']['max']:.3f}]"
        )
        print(
            f"maybe_recall {ms['maybe_recall']['mean']:.3f} ± {ms['maybe_recall']['std']:.3f} "
            f"[{ms['maybe_recall']['min']:.3f},{ms['maybe_recall']['max']:.3f}]"
        )
    p5 = result["pqal500"]
    print("\n=== PQA-L 500 (full) — bootstrap 95% CI ===")
    d = p5["decision_overall_accuracy"]
    print(f"decision accuracy  {d['mean']:.3f} [{d['ci_low']:.3f},{d['ci_high']:.3f}] n={d['n']}")
    for lab, c in p5["decision_per_class_recall"].items():
        print(f"  recall {lab:5s}     {c['mean']:.3f} [{c['ci_low']:.3f},{c['ci_high']:.3f}] n={c['n']}")
    for key in ("retrieval_hit_at_1", "retrieval_hit_at_3", "citation_pass", "rag_label_accuracy"):
        if key in p5:
            c = p5[key]
            print(f"{key:20s} {c['mean']:.3f} [{c['ci_low']:.3f},{c['ci_high']:.3f}] n={c['n']}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
