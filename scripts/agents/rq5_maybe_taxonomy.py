"""RQ5 — why is a question labelled `maybe`? Blind coding sheet and its scoring.

Two people independently assign one cause to each sampled gold-`maybe` question, using the
codebook in `docs/research/2026-10-04-rq5-codebook-taksonomia-maybe.md`. The sheet carries no
annotator labels and no stratum, so a coder cannot tell a unanimous `maybe` (both PQA-L
annotators said `maybe`) from a negotiated one. No LLM, no GPU.

Exploratory: these questions have been looked at in earlier analyses.

Usage:
  python scripts/agents/rq5_maybe_taxonomy.py export     # writes the sheet and the key
  python scripts/agents/rq5_maybe_taxonomy.py score      # after coder_1 / coder_2 are filled
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.agents.pqal_official import (  # noqa: E402
    CONTEXT_ONLY_FIELD,
    GOLD_FIELD,
    SEES_CONCLUSION_FIELD,
    load_ori_pqal,
    test_pmids,
)

ANALYSIS = PROJECT_ROOT / "reports/debate/analysis"
SHEET = ANALYSIS / "rq5_coding_sheet.csv"
KEY = ANALYSIS / "rq5_coding_key.json"
OUT = ANALYSIS / "rq5_maybe_taxonomy.json"

# Codes of the codebook, in its order. A cell holds exactly one of these.
CODES = ("A", "B", "C", "D", "E", "F")
STRATA = ("unanimous", "negotiated")


def stratum(item: dict) -> str:
    both = item[CONTEXT_ONLY_FIELD] == "maybe" and item[SEES_CONCLUSION_FIELD] == "maybe"
    return "unanimous" if both else "negotiated"


def sample(data: dict[str, dict], per_stratum: int, seed: int) -> list[str]:
    """`per_stratum` gold-`maybe` pmids from each stratum, shuffled together."""
    rng = random.Random(seed)
    chosen: list[str] = []
    for name in STRATA:
        pool = [p for p, item in sorted(data.items()) if item[GOLD_FIELD] == "maybe" and stratum(item) == name]
        chosen += rng.sample(pool, per_stratum)
    rng.shuffle(chosen)
    return chosen


def context_text(item: dict) -> str:
    contexts, labels = item["CONTEXTS"], item["LABELS"]
    if isinstance(contexts, str):  # the release stores both lists as Python reprs
        contexts, labels = ast.literal_eval(contexts), ast.literal_eval(labels)
    return "\n".join(f"{label}: {text}" for label, text in zip(labels, contexts))


def export(per_stratum: int, seed: int) -> None:
    data = load_ori_pqal()
    test = test_pmids()
    key = []
    with SHEET.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["item", "question", "context", "conclusion", "coder_1", "coder_2"])
        for i, pmid in enumerate(sample(data, per_stratum, seed), start=1):
            item = data[pmid]
            writer.writerow([i, item["QUESTION"], context_text(item), item["LONG_ANSWER"], "", ""])
            key.append(
                {
                    "item": i,
                    "pmid": pmid,
                    "stratum": stratum(item),
                    "split": "test" if pmid in test else "train_dev",
                    "context_only_annotator": item[CONTEXT_ONLY_FIELD],
                    "sees_conclusion_annotator": item[SEES_CONCLUSION_FIELD],
                }
            )
    KEY.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(f"wrote {len(key)} questions -> {SHEET} (key: {KEY})")


def score() -> dict:
    from sklearn.metrics import cohen_kappa_score

    key = {k["item"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8")))
    codes = [(int(r["item"]), r["coder_1"].strip().upper(), r["coder_2"].strip().upper()) for r in rows]
    bad = [i for i, a, b in codes if a not in CODES or b not in CODES]
    if bad:
        raise SystemExit(f"items without a valid code from both coders ({'/'.join(CODES)}): {bad}")

    def section(items: list[tuple[int, str, str]]) -> dict:
        a, b = [x[1] for x in items], [x[2] for x in items]
        return {
            "n": len(items),
            "agreement": round(sum(x == y for x, y in zip(a, b)) / len(items), 3),
            # Undefined when both coders use a single code throughout; sklearn returns nan.
            "cohen_kappa": round(float(cohen_kappa_score(a, b, labels=list(CODES))), 3),
            "coder_1": dict(sorted(Counter(a).items())),
            "coder_2": dict(sorted(Counter(b).items())),
        }

    result = {"exploratory": True, "all": section(codes)}
    for name in STRATA:
        result[name] = section([c for c in codes if key[c[0]]["stratum"] == name])
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("export", "score"))
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    args = parser.parse_args()
    if args.command == "export":
        if SHEET.exists():
            raise SystemExit(f"{SHEET} exists — refusing to overwrite a sheet that may hold codes.")
        export(args.per_stratum, args.seed)
    else:
        score()


if __name__ == "__main__":
    main()
