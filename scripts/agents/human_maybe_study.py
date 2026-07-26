"""Human-agreement mini-study for the PubMedQA `maybe` class (experiment #3).

Rationale
---------
Experiments #1 (generative + external-NLI auditors) and #2 (gold-conclusion
oracle) show no model can separate `maybe` from yes/no on PubMedQA (AUROC ~0.5).
The remaining question a reviewer will ask: *is the label even recoverable by a
human from the same text?* If trained humans, given only the question + abstract,
also fail to reproduce the gold `maybe` label, then `maybe` is (partly)
irreducible ambiguity — the strongest possible framing for the negative result.

This script has two subcommands:

  make-sheet : builds a BLIND annotation sheet (question + abstract only, gold
               label removed, cases shuffled) as CSV/Markdown for one or more
               annotators, plus a hidden answer key.
  score      : reads a filled sheet (annotator column added) and reports
               human-vs-gold agreement: overall accuracy, per-class recall,
               maybe recall/precision/F1, and Cohen's kappa.

Usage:
  python scripts/agents/human_maybe_study.py make-sheet --n-maybe 30 --n-yesno 30
  python scripts/agents/human_maybe_study.py score --sheet reports/debate/human/annotations_filled.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json"
CORPUS = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/corpus.json"
ORI_PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
OUT_DIR = PROJECT_ROOT / "reports/debate/human"


def _corpus() -> dict:
    raw = json.loads(CORPUS.read_text())
    if isinstance(raw, list):
        return {str(d["id"]): d for d in raw if isinstance(d, dict) and "id" in d}
    return {str(k): v for k, v in raw.items()}


def _abstract(case: dict, corpus: dict) -> str:
    blocks = []
    for doc_id in case.get("relevant_document_ids", []):
        doc = corpus.get(str(doc_id))
        if doc:
            title = str(doc.get("title") or "").strip()
            content = str(doc.get("content") or "").strip()
            blocks.append((f"{title}\n{content}").strip() if title else content)
    return "\n\n".join(blocks)


def make_sheet(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    cases = json.loads(args.dataset.read_text())
    corpus = _corpus()

    by_label: dict[str, list[dict]] = {"maybe": [], "yes": [], "no": []}
    for c in cases:
        by_label.setdefault(c["expected_label"], []).append(c)

    picked: list[dict] = []
    picked += rng.sample(by_label["maybe"], min(args.n_maybe, len(by_label["maybe"])))
    yesno = by_label["yes"] + by_label["no"]
    picked += rng.sample(yesno, min(args.n_yesno, len(yesno)))
    rng.shuffle(picked)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet_csv = OUT_DIR / "annotation_sheet.csv"
    sheet_md = OUT_DIR / "annotation_sheet.md"
    key_path = OUT_DIR / "answer_key.json"

    key = {}
    with sheet_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "question", "abstract", "your_label(yes/no/maybe)", "your_confidence(1-5)"])
        md_lines = [
            "# Blind PubMedQA annotation sheet",
            "",
            "For each item, read ONLY the question and abstract, then decide: **yes**, "
            "**no**, or **maybe** (maybe = the abstract does not conclusively settle the "
            "question). Do not look anything up. Fill `your_label` and `your_confidence` (1-5).",
            "",
        ]
        for i, c in enumerate(picked, 1):
            item = f"H{i:03d}"
            q = c["question"].strip()
            ab = _abstract(c, corpus)
            writer.writerow([item, q, ab, "", ""])
            key[item] = {"id": c["id"], "gold": c["expected_label"]}
            md_lines += [
                f"## {item}",
                "",
                f"**Question:** {q}",
                "",
                f"**Abstract:** {ab}",
                "",
                "**Your label (yes/no/maybe):** ____   **Confidence (1-5):** __",
                "",
                "---",
                "",
            ]
        sheet_md.write_text("\n".join(md_lines), encoding="utf-8")

    key_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    n_maybe = sum(1 for v in key.values() if v["gold"] == "maybe")
    print(f"Wrote {sheet_csv} ({len(picked)} items, {n_maybe} maybe)")
    print(f"Wrote {sheet_md}")
    print(f"Wrote {key_path} (hidden gold key — do not show annotators)")


def _kappa(a: list[str], b: list[str], labels: list[str]) -> float:
    """Cohen's kappa between two labelings."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = 0.0
    for lab in labels:
        pa = sum(1 for x in a if x == lab) / n
        pb = sum(1 for y in b if y == lab) / n
        pe += pa * pb
    return (po - pe) / (1 - pe) if pe != 1.0 else 1.0


def score(args: argparse.Namespace) -> None:
    key = json.loads((OUT_DIR / "answer_key.json").read_text())
    rows = list(csv.DictReader(args.sheet.open(encoding="utf-8")))
    human: list[str] = []
    gold: list[str] = []
    for r in rows:
        item = r.get("item", "").strip()
        lab = (r.get("your_label(yes/no/maybe)") or r.get("your_label") or "").strip().lower()
        if item not in key or lab not in {"yes", "no", "maybe"}:
            continue
        human.append(lab)
        gold.append(key[item]["gold"])

    if not human:
        raise SystemExit("No usable annotations found — fill `your_label` with yes/no/maybe.")

    n = len(human)
    overall = sum(1 for h, g in zip(human, gold) if h == g) / n
    labels = ["yes", "no", "maybe"]
    per_class = {}
    for lab in labels:
        support = sum(1 for g in gold if g == lab)
        tp = sum(1 for h, g in zip(human, gold) if g == lab and h == lab)
        pred = sum(1 for h in human if h == lab)
        recall = tp / support if support else float("nan")
        precision = tp / pred if pred else float("nan")
        f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and precision + recall) else 0.0
        per_class[lab] = {"support": support, "recall": recall, "precision": precision, "f1": f1}

    result = {
        "n_annotated": n,
        "overall_accuracy": overall,
        "cohen_kappa": _kappa(human, gold, labels),
        "per_class": per_class,
    }
    out = OUT_DIR / "human_agreement.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


def human_baseline(args: argparse.Namespace) -> None:
    """Official-annotator agreement proxy: PubMedQA single-annotator predictions.

    PubMedQA ships two single-annotator predictions per item — reasoning_free_pred
    (from the abstract, no author conclusion) and reasoning_required_pred (with the
    author LONG_ANSWER) — while final_decision is the multi-annotator gold. Agreement
    of a single annotator with the gold is a real, citable human-reproducibility
    signal for the `maybe` label, computed here on the study set.
    """
    ori = json.loads(ORI_PQAL.read_text())
    cases = json.loads(args.dataset.read_text())

    def pmid(c: dict) -> str | None:
        t = str(c["id"]).rsplit("-", 1)[-1]
        return t if t.isdigit() else None

    labels = ["yes", "no", "maybe"]
    rows = [(pmid(c), c) for c in cases]
    rows = [(p, c) for p, c in rows if p in ori]

    def metrics(pred_field: str) -> dict:
        gold = [ori[p]["final_decision"] for p, _ in rows]
        pred = [ori[p][pred_field] for p, _ in rows]
        n = len(gold)
        overall = sum(1 for g, x in zip(gold, pred) if g == x) / n
        per_class = {}
        for lab in labels:
            support = sum(1 for g in gold if g == lab)
            tp = sum(1 for g, x in zip(gold, pred) if g == lab and x == lab)
            predc = sum(1 for x in pred if x == lab)
            recall = tp / support if support else float("nan")
            precision = tp / predc if predc else float("nan")
            f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else 0.0
            per_class[lab] = {"support": support, "recall": recall, "precision": precision, "f1": f1}
        return {
            "n": n,
            "overall_accuracy": overall,
            "cohen_kappa": _kappa(gold, pred, labels),
            "per_class": per_class,
        }

    result = {
        "study_set": str(args.dataset),
        "reasoning_free_pred (abstract only)": metrics("reasoning_free_pred"),
        "reasoning_required_pred (with author conclusion)": metrics("reasoning_required_pred"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "human_baseline_official.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_make = sub.add_parser("make-sheet", help="Build a blind annotation sheet")
    p_make.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p_make.add_argument("--n-maybe", type=int, default=30)
    p_make.add_argument("--n-yesno", type=int, default=30)
    p_make.add_argument("--seed", type=int, default=47)
    p_make.set_defaults(func=make_sheet)

    p_score = sub.add_parser("score", help="Score a filled annotation sheet vs gold")
    p_score.add_argument("--sheet", type=Path, required=True)
    p_score.set_defaults(func=score)

    p_base = sub.add_parser(
        "human-baseline",
        help="Official single-annotator agreement proxy (no manual annotation needed)",
    )
    p_base.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    p_base.set_defaults(func=human_baseline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
