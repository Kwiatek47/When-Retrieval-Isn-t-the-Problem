"""Hypothesis H1b: gold ``maybe`` tracks a conditional answer stated in the hidden conclusion.

Reads the per-question table (``build_pqal_label_table.py``) and the LLM conditionality
ratings (``rate_conditionality.py``). No LLM calls; deterministic bootstrap.

Primary test, fixed before the ratings were collected (same form as H1):
  - questions: PQA-L questions with a RESULTS section and at least 2 valid ratings of each passage;
  - target: ``final_decision == maybe``;
  - A = mean conditionality (0-2) of the conclusion, B = mean conditionality of the RESULTS section;
  - statistic: dAUROC = AUROC(A) - AUROC(B), paired bootstrap over questions (95% CI).
  H1b is supported if the CI of AUROC(A) lies above 0.5 AND the CI of dAUROC lies above 0;
  refuted if the CI of dAUROC lies at or below 0.

Secondary: each annotator's label as target (annotator 1 saw the conclusion, annotator 2
did not), both data splits, and rating stability across repeats.

Calibration against people (run before trusting the ratings):
  ``--export-calibration`` writes a blind sheet of 50 passages for two raters;
  ``--score-calibration`` computes agreement between the raters and with the LLM.

Output: ``reports/debate/analysis/h1b_conditionality.json``.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import random
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.analyze_h1_hedging import (  # noqa: E402
    TABLE,
    annotator_contrast,
    by_split,
    load_rows,
    primary_test,
)
from scripts.agents.rate_conditionality import OUT as RATINGS  # noqa: E402
from scripts.agents.rate_conditionality import PQAL, passages  # noqa: E402

ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
OUT = ANALYSIS / "h1b_conditionality.json"
CALIBRATION_SHEET = ANALYSIS / "h1b_calibration_sheet.csv"
CALIBRATION_KEY = ANALYSIS / "h1b_calibration_key.json"

A_KEY = "cond_conclusion"
B_KEY = "cond_results"
MIN_VALID = 2


def load_ratings(path: Path = RATINGS) -> dict[tuple[str, str], list[int]]:
    """Valid ratings per (pmid, part); a partial last line is ignored."""
    ratings: dict[tuple[str, str], list[int]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("conditionality") is not None:
            ratings[(row["pmid"], row["part"])].append(row["conditionality"])
    return ratings


def mean_rating(values: list[int]) -> float | None:
    return float(np.mean(values)) if len(values) >= MIN_VALID else None


def attach_ratings(rows: list[dict], ratings: dict[tuple[str, str], list[int]]) -> list[dict]:
    """Rows that have a mean rating for both passages, with the ratings added as columns."""
    out = []
    for row in rows:
        a = mean_rating(ratings.get((row["pmid"], "conclusion"), []))
        b = mean_rating(ratings.get((row["pmid"], "results"), []))
        if a is not None and b is not None:
            out.append({**row, A_KEY: a, B_KEY: b})
    return out


def stability(ratings: dict[tuple[str, str], list[int]]) -> dict:
    """How often the repeated ratings of one passage agree exactly."""
    out = {}
    for part in ("conclusion", "results"):
        groups = [v for (_, p), v in ratings.items() if p == part and len(v) >= 2]
        out[part] = {
            "n_passages": len(groups),
            "all_repeats_equal": round(float(np.mean([len(set(v)) == 1 for v in groups])), 4) if groups else None,
        }
    return out


def distribution(rows: list[dict]) -> dict:
    """Mean conditionality of each passage by gold label."""
    out = {}
    for label in ("yes", "no", "maybe"):
        group = [r for r in rows if r["final"] == label]
        out[label] = {
            "n": len(group),
            "conclusion_mean": round(float(np.mean([r[A_KEY] for r in group])), 3) if group else None,
            "results_mean": round(float(np.mean([r[B_KEY] for r in group])), 3) if group else None,
        }
    return out


def cohen_kappa(a: list[int], b: list[int], weighted: bool = False) -> float:
    """Cohen's kappa for ratings 0-2; ``weighted`` uses quadratic weights."""
    cats = (0, 1, 2)
    n = len(a)
    observed = np.zeros((3, 3))
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= n
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    if weighted:
        w = np.array([[(i - j) ** 2 for j in cats] for i in cats], dtype=float) / 4.0
    else:
        w = np.array([[0.0 if i == j else 1.0 for j in cats] for i in cats])
    denominator = (w * expected).sum()
    return float(1.0 - (w * observed).sum() / denominator) if denominator else float("nan")


def export_calibration(n_questions: int = 25, n_maybe: int = 10, seed: int = 47) -> None:
    """Blind sheet: both passages of 25 questions (10 gold maybe), shuffled, no labels."""
    data = json.loads(PQAL.read_text(encoding="utf-8"))
    eligible = [pmid for pmid, item in sorted(data.items()) if passages(item)["results"]]
    rng = random.Random(seed)
    maybe = [p for p in eligible if data[p]["final_decision"] == "maybe"]
    other = [p for p in eligible if data[p]["final_decision"] != "maybe"]
    chosen = rng.sample(maybe, n_maybe) + rng.sample(other, n_questions - n_maybe)
    items = [(pmid, part) for pmid in chosen for part in ("conclusion", "results")]
    rng.shuffle(items)

    key = []
    with CALIBRATION_SHEET.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "question", "passage", "rater_1", "rater_2"])
        for i, (pmid, part) in enumerate(items, start=1):
            writer.writerow([i, data[pmid]["QUESTION"], passages(data[pmid])[part], "", ""])
            key.append({"item": i, "pmid": pmid, "part": part})
    CALIBRATION_KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {len(items)} passages -> {CALIBRATION_SHEET} (key: {CALIBRATION_KEY})")


def score_calibration() -> dict:
    key = {k["item"]: k for k in json.loads(CALIBRATION_KEY.read_text(encoding="utf-8"))}
    ratings = load_ratings()
    human_1, human_2, llm = [], [], []
    for row in csv.DictReader(CALIBRATION_SHEET.open(encoding="utf-8")):
        if row["rater_1"].strip() == "" or row["rater_2"].strip() == "":
            continue
        k = key[int(row["item"])]
        values = ratings.get((k["pmid"], k["part"]), [])
        if len(values) < MIN_VALID:
            continue
        human_1.append(int(row["rater_1"]))
        human_2.append(int(row["rater_2"]))
        llm.append(int(round(float(np.median(values)))))
    if not human_1:
        raise SystemExit("no rated rows in the calibration sheet yet")
    return {
        "n": len(human_1),
        "kappa_rater1_rater2": round(cohen_kappa(human_1, human_2), 3),
        "weighted_kappa_rater1_rater2": round(cohen_kappa(human_1, human_2, weighted=True), 3),
        "weighted_kappa_rater1_llm": round(cohen_kappa(human_1, llm, weighted=True), 3),
        "weighted_kappa_rater2_llm": round(cohen_kappa(human_2, llm, weighted=True), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, default=TABLE)
    parser.add_argument("--ratings", type=Path, default=RATINGS)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--export-calibration", action="store_true")
    parser.add_argument("--score-calibration", action="store_true")
    args = parser.parse_args()

    if args.export_calibration:
        export_calibration(seed=args.seed)
        return
    if args.score_calibration:
        print(json.dumps(score_calibration(), indent=2))
        return

    ratings = load_ratings(args.ratings)
    rows = attach_ratings(load_rows(args.table), ratings)
    rng = np.random.default_rng(args.seed)
    result = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_questions_rated": len(rows),
        "primary": primary_test(rows, rng, args.n_boot, A_KEY, B_KEY),
        "primary_by_split": by_split(rows, rng, args.n_boot, A_KEY, B_KEY),
        "annotator_contrast": annotator_contrast(rows, rng, args.n_boot, (A_KEY, B_KEY)),
        "mean_conditionality_by_gold_label": distribution(rows),
        "rating_stability": stability(ratings),
    }
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    p = result["primary"]
    print(f"=== H1b primary (n={p['n']}, gold maybe={p['n_maybe']}) ===")
    for name in ("conclusion", "results_section"):
        s = p[name]
        print(f"AUROC {name:16s} {s['auroc']:.3f} [{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
    d = p["difference"]
    print(f"dAUROC conclusion - results {d['d_auroc']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]  -> {p['verdict']}")
    for split, s in result["primary_by_split"].items():
        sd = s["difference"]
        print(f"  {split:4s} n={s['n']} maybe={s['n_maybe']} dAUROC {sd['d_auroc']:+.3f} [{sd['ci_low']:+.3f}, {sd['ci_high']:+.3f}] -> {s['verdict']}")
    print("=== annotator contrast: AUROC(annotator with conclusion) - AUROC(without) ===")
    for score, s in result["annotator_contrast"].items():
        print(f"  {score:16s} {s['d_auroc']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
    print(f"rating stability: {result['rating_stability']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
