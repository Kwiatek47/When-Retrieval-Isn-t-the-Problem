"""RQ5 — why is a question labelled `maybe`? Blind coding sheet and its scoring.

Two people independently assign one cause to each sampled gold-`maybe` question, using the
codebook in `docs/research/2026-10-04-rq5-codebook-taksonomia-maybe.md`. The sheet carries no
annotator labels and no stratum, so a coder cannot tell a unanimous `maybe` (both PQA-L
annotators said `maybe`) from a negotiated one. No LLM, no GPU.

Exploratory: these questions have been looked at in earlier analyses.

Usage:
  python scripts/agents/rq5_maybe_taxonomy.py export     # writes the sheet and the key
  python scripts/agents/rq5_maybe_taxonomy.py score      # after coder_1 / coder_2 are filled
  python scripts/agents/rq5_maybe_taxonomy.py score --codes-1 a.csv --codes-2 b.csv
                                                         # codes from rq5_code.py files instead
  python scripts/agents/rq5_maybe_taxonomy.py analyze --codes-1 a.csv --codes-2 b.csv --llm-codes llm.csv
                                                         # the pre-registered analysis (below)

Pre-registered analysis (``analyze``), fixed on 2026-10-08 before the second coder started:

  Gate. Cohen's kappa between the two human coders on the six codes, all 40 items. If it is
  below 0.40 the codes are not reliable enough to compare strata: the primary test is reported
  as "not interpretable" and only the agreement figures are read.

  Item code. Where the two coders agree the item has that code; where they disagree each of the
  two codes gets weight 1/2. Codes are grouped into the three classes of Jiang & de Marneffe
  (2022): content uncertainty (A, B, C), task underspecification (D, E), annotator (F).

  Primary test. D = share of content uncertainty among unanimous items minus the share among
  negotiated items. The thesis (maybe is mostly a recorded dispute) predicts D > 0: where both
  annotators said maybe, the abstract itself is more often unsettled. Two-sided permutation
  test over the stratum labels (10,000 permutations, seed 47) and a bootstrap CI that resamples
  items within each stratum. Supported if p < 0.05 and D > 0; contradicted if p < 0.05 and D < 0;
  otherwise inconclusive. With 20 items per stratum only a large difference is detectable.

  Secondary (no correction): the same comparison for F alone (predicted higher among negotiated
  items) and for D/E; kappa per stratum and on the three classes; agreement of the LLM coder
  with each human (it coded the English original, the humans mostly a Polish translation, so this
  mixes coder and language); the class distribution over all 110 gold-maybe questions, reweighting
  the strata to their population sizes (23 unanimous, 87 negotiated).
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


def read_code_file(path: Path) -> dict[int, str]:
    """Codes written by ``rq5_code.py`` (or any CSV with ``item`` and ``code`` columns)."""
    with path.open(encoding="utf-8", newline="") as fh:
        return {int(r["item"]): r["code"].strip().upper() for r in csv.DictReader(fh)}


def score(codes_1: Path | None = None, codes_2: Path | None = None, out: Path = OUT) -> dict:
    from sklearn.metrics import cohen_kappa_score

    key = {k["item"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))}
    rows = list(csv.DictReader(SHEET.open(encoding="utf-8")))
    first = read_code_file(codes_1) if codes_1 else {int(r["item"]): r["coder_1"] for r in rows}
    second = read_code_file(codes_2) if codes_2 else {int(r["item"]): r["coder_2"] for r in rows}
    codes = [
        (int(r["item"]), first.get(int(r["item"]), "").strip().upper(), second.get(int(r["item"]), "").strip().upper())
        for r in rows
    ]
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

    result = {
        "exploratory": True,
        "coder_1_source": str(codes_1) if codes_1 else "sheet column coder_1",
        "coder_2_source": str(codes_2) if codes_2 else "sheet column coder_2",
        "all": section(codes),
    }
    for name in STRATA:
        result[name] = section([c for c in codes if key[c[0]]["stratum"] == name])
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


CLASSES = {"content": ("A", "B", "C"), "task": ("D", "E"), "annotator": ("F",)}
POPULATION = {"unanimous": 23, "negotiated": 87}
KAPPA_GATE = 0.40


def cohen_kappa(a: list[str], b: list[str], labels: tuple[str, ...] = CODES) -> float:
    """Cohen's kappa for two nominal codings of the same items; nan when chance agreement is 1."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[l] * cb[l] for l in labels) / (n * n)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def class_of(code: str) -> str:
    return next(name for name, codes in CLASSES.items() if code in codes)


def soft_share(pairs: list[tuple[str, str]], members: tuple[str, ...]) -> float:
    """Share of items in ``members``: an item coded differently by the two coders counts half for each code."""
    return sum(((a in members) + (b in members)) / 2 for a, b in pairs) / len(pairs)


def strata_difference(unanimous: list[tuple[str, str]], negotiated: list[tuple[str, str]], members: tuple[str, ...]) -> float:
    return soft_share(unanimous, members) - soft_share(negotiated, members)


def permutation_p(
    unanimous: list[tuple[str, str]], negotiated: list[tuple[str, str]], members: tuple[str, ...], n_perm: int, seed: int
) -> float:
    """Two-sided p for the strata difference, permuting which items belong to which stratum."""
    observed = abs(strata_difference(unanimous, negotiated, members))
    pooled = unanimous + negotiated
    rng = random.Random(seed)
    k = len(unanimous)
    hits = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        if abs(strata_difference(pooled[:k], pooled[k:], members)) >= observed - 1e-12:
            hits += 1
    return (hits + 1) / (n_perm + 1)


def bootstrap_difference(
    unanimous: list[tuple[str, str]], negotiated: list[tuple[str, str]], members: tuple[str, ...], n_boot: int, seed: int
) -> tuple[float, float]:
    """95% CI of the strata difference, resampling items within each stratum."""
    rng = random.Random(seed)
    boots = sorted(
        strata_difference(
            [rng.choice(unanimous) for _ in unanimous], [rng.choice(negotiated) for _ in negotiated], members
        )
        for _ in range(n_boot)
    )
    return boots[int(0.025 * (n_boot - 1))], boots[int(0.975 * (n_boot - 1))]


def compare(unanimous, negotiated, members, n_perm, seed) -> dict:
    low, high = bootstrap_difference(unanimous, negotiated, members, n_perm, seed)
    return {
        "unanimous_share": round(soft_share(unanimous, members), 3),
        "negotiated_share": round(soft_share(negotiated, members), 3),
        "difference": round(strata_difference(unanimous, negotiated, members), 3),
        "ci_low": round(low, 3),
        "ci_high": round(high, 3),
        "permutation_p": round(permutation_p(unanimous, negotiated, members, n_perm, seed), 4),
    }


def primary_verdict(kappa: float, test: dict) -> str:
    if not kappa >= KAPPA_GATE:
        return "not interpretable (kappa below gate)"
    if test["permutation_p"] < 0.05:
        return "supported" if test["difference"] > 0 else "contradicted"
    return "inconclusive"


def population_distribution(by_stratum: dict[str, list[tuple[str, str]]]) -> dict:
    """Class shares over all gold-maybe questions, each stratum weighted by its population size."""
    total = sum(POPULATION.values())
    return {
        name: round(sum(POPULATION[s] / total * soft_share(by_stratum[s], members) for s in STRATA), 3)
        for name, members in CLASSES.items()
    }


def analyze(codes_1: Path, codes_2: Path, llm_codes: Path | None = None, n_perm: int = 10_000, seed: int = 47) -> dict:
    """The pre-registered RQ5 analysis; see the module docstring."""
    key = {k["item"]: k for k in json.loads(KEY.read_text(encoding="utf-8"))}
    first, second = read_code_file(codes_1), read_code_file(codes_2)
    items = sorted(key)
    bad = [i for i in items if first.get(i) not in CODES or second.get(i) not in CODES]
    if bad:
        raise SystemExit(f"items without a valid code from both coders: {bad}")
    pairs = {i: (first[i], second[i]) for i in items}
    by_stratum = {s: [pairs[i] for i in items if key[i]["stratum"] == s] for s in STRATA}
    a, b = [pairs[i][0] for i in items], [pairs[i][1] for i in items]
    kappa = cohen_kappa(a, b)

    primary = compare(by_stratum["unanimous"], by_stratum["negotiated"], CLASSES["content"], n_perm, seed)
    result = {
        "exploratory_material": "questions were inspected in earlier analyses; the analysis itself is pre-registered",
        "coders": {"coder_1": str(codes_1), "coder_2": str(codes_2)},
        "gate": {"kappa_six_codes": round(kappa, 3), "threshold": KAPPA_GATE, "passed": kappa >= KAPPA_GATE},
        "agreement_six_codes": round(sum(x == y for x, y in zip(a, b)) / len(a), 3),
        "kappa_three_classes": round(
            cohen_kappa([class_of(x) for x in a], [class_of(y) for y in b], tuple(CLASSES)), 3
        ),
        "kappa_by_stratum": {
            s: round(cohen_kappa([p[0] for p in by_stratum[s]], [p[1] for p in by_stratum[s]]), 3) for s in STRATA
        },
        "primary_content_uncertainty": {**primary, "verdict": primary_verdict(kappa, primary)},
        "secondary_annotator_F": compare(by_stratum["unanimous"], by_stratum["negotiated"], CLASSES["annotator"], n_perm, seed),
        "secondary_task_DE": compare(by_stratum["unanimous"], by_stratum["negotiated"], CLASSES["task"], n_perm, seed),
        "codes_by_stratum": {
            s: {"coder_1": dict(sorted(Counter(p[0] for p in by_stratum[s]).items())),
                "coder_2": dict(sorted(Counter(p[1] for p in by_stratum[s]).items()))}
            for s in STRATA
        },
        "population_class_distribution": population_distribution(by_stratum),
    }
    if llm_codes is not None:
        with llm_codes.open(encoding="utf-8", newline="") as fh:
            llm = {int(r["item"]): r["code"].strip().upper() for r in csv.DictReader(fh)}
        m = [llm.get(i, "") for i in items]
        result["llm_coder"] = {
            "source": str(llm_codes),
            "kappa_with_coder_1": round(cohen_kappa(m, a), 3),
            "kappa_with_coder_2": round(cohen_kappa(m, b), 3),
            "caveat": "LLM coded the English original; humans mostly a Polish translation",
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("export", "score", "analyze"))
    parser.add_argument("--per-stratum", type=int, default=20)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--codes-1", type=Path, default=None, help="score: coder 1's file from rq5_code.py")
    parser.add_argument("--codes-2", type=Path, default=None, help="score: coder 2's file from rq5_code.py")
    parser.add_argument("--out", type=Path, default=OUT, help="score / analyze: where to write the result")
    parser.add_argument("--llm-codes", type=Path, default=None, help="analyze: the LLM coder's file")
    parser.add_argument("--n-perm", type=int, default=10_000)
    args = parser.parse_args()
    if args.command == "export":
        if SHEET.exists():
            raise SystemExit(f"{SHEET} exists — refusing to overwrite a sheet that may hold codes.")
        export(args.per_stratum, args.seed)
    elif args.command == "analyze":
        if not (args.codes_1 and args.codes_2):
            raise SystemExit("analyze needs --codes-1 and --codes-2 (one file per human coder)")
        out = args.out if args.out != OUT else ANALYSIS / "rq5_analysis.json"
        result = analyze(args.codes_1, args.codes_2, args.llm_codes, args.n_perm, args.seed)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    else:
        score(args.codes_1, args.codes_2, args.out)


if __name__ == "__main__":
    main()
