"""Per-question table for the 1000 PubMedQA PQA-L questions: labels, annotators, text features.

PQA-L was labelled by two annotators (Jin et al. 2019, Algorithm 1):
  - annotator 1 saw the question, the context AND the authors' conclusion
    (``LONG_ANSWER``) -> ``reasoning_free_pred`` (RF);
  - annotator 2 saw only the question and the context -> ``reasoning_required_pred`` (RR);
  - on disagreement they discussed to a shared ``final_decision``.
Models see what annotator 2 saw. This table keeps both annotators' labels next to
cheap text features, so that label analyses (hypothesis H1: gold ``maybe`` tracks
hedging in the hidden conclusion) and format analyses need no LLM calls.

Hedging is counted with a fixed lexicon in three places: the whole context, the
RESULTS section of the context, and the conclusion. Densities are cues per 100 tokens.

Output: ``reports/debate/analysis/pqal_label_table.jsonl`` (one row per question).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
TEST_SET = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/eval.json"
OUT = PROJECT_ROOT / "reports/debate/analysis/pqal_label_table.jsonl"

_WORD = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:[.,]\d+)?")

# Epistemic hedges and calls for more evidence. Conservative on purpose: words that are
# mostly assertive in abstracts ("indicate", "can", "potential") stay out of the core list.
_CORE_HEDGE = (
    r"\bmay\b",  # lower case only: capitalised "May" is usually the month
    r"(?:^|[.!?]\s+)May\b(?!\s+\d)",
    r"\bmight\b",
    r"\bcould\b",
    r"\bperhaps\b",
    r"\bpossibly\b",
    r"\bprobabl[ey]\b",
    r"\b(?:un)?likely\b",
    r"\bpresumabl[ey]\b",
    r"\bsuggest\w*",
    r"\bappear(?:s|ed)?\s+to\b",
    r"\bseem\w*",
    r"\buncertain\w*",
    r"\bunclear\b",
    r"\bnot\s+clear\b",
    r"\binconclusive\b",
    r"\bcontroversial\b",
    r"\bconflicting\b",
    r"\bequivocal\b",
    r"\bfurther\s+(?:\w+\s+){0,3}(?:stud(?:y|ies)|research|investigations?|trials?|evaluations?|work|data)\b",
    r"\bremains?\s+to\s+be\b",
    r"\bneeds?\s+to\s+be\s+(?:confirmed|clarified|determined|elucidated|established)\b",
)

# Weaker cues, reported only as a sensitivity check.
_EXTENDED_HEDGE = _CORE_HEDGE + (
    r"\bpossible\b",
    r"\bpotential(?:ly)?\b",
    r"\blimited\b",
    r"\bindicat\w*",
    r"\bquestionable\b",
    r"\btend(?:s|ed|ency)?\b",
    r"\btrend\w*",
)

# Null or non-significant results. Not hedging: a clear null result supports "no".
_NULL_RESULT = (
    r"\bnot\s+(?:statistically\s+)?significant\w*",
    r"\bno\s+(?:statistically\s+)?significant\b",
    r"\bnon-?significant\w*",
    r"\bdid\s+not\s+(?:differ|reach|show|improve|reduce|increase|change|affect)\b",
    r"\bfailed\s+to\b",
    r"\bno\s+(?:\w+\s+){0,2}differences?\b",
    r"\b(?:was|were)\s+not\s+associated\b",
)

# The two "may" patterns are case-sensitive so that the month ("in May 2010") is not a hedge;
# every other cue ignores case.
_CASE_SENSITIVE = {_CORE_HEDGE[0], _CORE_HEDGE[1]}


def _compile(patterns: tuple[str, ...]) -> list[re.Pattern]:
    return [re.compile(p) if p in _CASE_SENSITIVE else re.compile(p, re.IGNORECASE) for p in patterns]


_CUE_SETS = {
    "hedge": _compile(_CORE_HEDGE),
    "hedge_ext": _compile(_EXTENDED_HEDGE),
    "null": _compile(_NULL_RESULT),
}

_PVALUE = re.compile(r"\b[pP]\s*[<>=≤≥]\s*0?\.\d+")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")


def tokens(text: str) -> list[str]:
    return _WORD.findall(text)


def count_cues(text: str, cue_set: str) -> int:
    """Number of cue matches of one family (``hedge``, ``hedge_ext`` or ``null``)."""
    return sum(len(pattern.findall(text)) for pattern in _CUE_SETS[cue_set])


def results_text(contexts: list[str], labels: list[str]) -> str:
    """Concatenated context sections whose label mentions RESULT; empty if none."""
    return " ".join(c for c, lab in zip(contexts, labels) if "RESULT" in lab.upper())


def _density(count: int, n_tokens: int) -> float | None:
    return round(100.0 * count / n_tokens, 4) if n_tokens else None


def _pattern(rf: str, rr: str) -> str:
    return f"rf={rf}|rr={rr}"


def build_row(pmid: str, item: dict, split: str) -> dict:
    contexts = item["CONTEXTS"]
    labels = item["LABELS"]
    parts = {
        "context": " ".join(contexts),
        "results": results_text(contexts, labels),
        "conclusion": item["LONG_ANSWER"],
    }
    rf = item["reasoning_free_pred"]
    rr = item["reasoning_required_pred"]
    final = item["final_decision"]
    row: dict = {
        "pmid": pmid,
        "split": split,
        "year": item.get("YEAR"),
        "final": final,
        "rf": rf,
        "rr": rr,
        "pattern": _pattern(rf, rr),
        "annotators_agree": rf == rr,
        "all_agree": rf == rr == final,
        "final_follows": "both" if rf == rr else ("rf" if final == rf else "rr" if final == rr else "neither"),
        "n_sections": len(contexts),
        "has_results_section": bool(parts["results"]),
        "n_tokens_question": len(tokens(item["QUESTION"])),
        "n_numbers_context": len(_NUMBER.findall(parts["context"])),
        "n_pvalues_context": len(_PVALUE.findall(parts["context"])),
    }
    for name, text in parts.items():
        n_tok = len(tokens(text))
        row[f"n_tokens_{name}"] = n_tok
        for cue_set in _CUE_SETS:
            count = count_cues(text, cue_set)
            row[f"{cue_set}_count_{name}"] = count
            row[f"{cue_set}_density_{name}"] = _density(count, n_tok)
    return row


def test_pmids(path: Path = TEST_SET) -> set[str]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    return {str(p) for case in cases for p in case["relevant_pmids"]}


def build_table(pqal_path: Path = PQAL, test_path: Path = TEST_SET) -> list[dict]:
    data = json.loads(pqal_path.read_text(encoding="utf-8"))
    in_test = test_pmids(test_path)
    return [
        build_row(pmid, item, "test" if pmid in in_test else "cv")
        for pmid, item in sorted(data.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pqal", type=Path, default=PQAL)
    parser.add_argument("--test-set", type=Path, default=TEST_SET)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = build_table(args.pqal, args.test_set)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    n_test = sum(r["split"] == "test" for r in rows)
    n_maybe = sum(r["final"] == "maybe" for r in rows)
    print(f"wrote {len(rows)} rows ({n_test} test, {len(rows) - n_test} cv; {n_maybe} gold maybe) -> {args.out}")


if __name__ == "__main__":
    main()
