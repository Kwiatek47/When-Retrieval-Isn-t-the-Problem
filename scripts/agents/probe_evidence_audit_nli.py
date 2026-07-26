"""External-NLI evidence-audit probe (+ optional oracle mode).

Two paper experiments in one script:

  #1 External auditor: score `maybe`-inconclusiveness with a fixed DeBERTa NLI
     cross-encoder instead of a generative chat model. Directly extends the
     abstention-aware verification setup of arXiv 2602.14189 (which used a fixed
     NLI auditor). If this also lands at AUROC ~0.5, the bottleneck is the *data*
     (the abstract lacks the signal), not the choice of generative auditor.

  #2 Oracle upper-bound (--evidence-source long_answer): replace the abstract
     with the authors' own gold LONG_ANSWER (conclusions) from ori_pqal.json. If
     even the gold conclusions do not make `maybe` separable, the label is not an
     evidence-conclusiveness property an auditor can read off text at all.

Usage:
  python scripts/agents/probe_evidence_audit_nli.py --label nli_abstract
  python scripts/agents/probe_evidence_audit_nli.py --evidence-source long_answer --label nli_oracle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics as st
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.evidence_audit_nli import audit_evidence_nli  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json"
CORPUS = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/corpus.json"
ORI_PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"
REPORT_DIR = PROJECT_ROOT / "reports/debate/signals"


def _corpus() -> dict:
    raw = json.loads(CORPUS.read_text())
    if isinstance(raw, list):
        return {str(d["id"]): d for d in raw if isinstance(d, dict) and "id" in d}
    return {str(k): v for k, v in raw.items()}


def _abstract_evidence(case: dict, corpus: dict) -> str:
    blocks = []
    for doc_id in case.get("relevant_document_ids", []):
        doc = corpus.get(str(doc_id))
        if doc:
            blocks.append(str(doc.get("content") or ""))
    return "\n\n".join(blocks)


def _pmid_from_case(case: dict) -> str | None:
    # ids look like "pubmedqa-official-12377809"; the trailing number is the PMID.
    cid = str(case.get("id") or "")
    tail = cid.rsplit("-", 1)[-1]
    if tail.isdigit():
        return tail
    pmids = case.get("relevant_pmids") or []
    return str(pmids[0]) if pmids else None


def _long_answer_index() -> dict[str, str]:
    data = json.loads(ORI_PQAL.read_text())
    return {str(k): str(v.get("LONG_ANSWER") or "").strip() for k, v in data.items()}


def _auroc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--evidence-source",
        choices=("abstract", "long_answer"),
        default="abstract",
        help="abstract = normal PubMedQA input; long_answer = gold oracle conclusions",
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    label = args.label or f"nli_{args.evidence_source}_balanced90"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{label}.jsonl"
    summary_path = REPORT_DIR / f"{label}.summary.json"

    cases = json.loads(args.dataset.read_text())
    if args.limit:
        cases = cases[: args.limit]
    corpus = _corpus()
    long_answers = _long_answer_index() if args.evidence_source == "long_answer" else {}

    results = []
    missing_oracle = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for i, case in enumerate(cases, 1):
            q = case["question"]
            if args.evidence_source == "long_answer":
                pmid = _pmid_from_case(case)
                ev = long_answers.get(pmid or "", "")
                if not ev:
                    missing_oracle += 1
            else:
                ev = _abstract_evidence(case, corpus)
            t0 = time.perf_counter()
            audit = audit_evidence_nli(question=q, evidence=ev)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            row = {
                "id": case["id"],
                "expected": case["expected_label"],
                "auditor": "deberta-nli",
                "evidence_source": args.evidence_source,
                "audit_score": audit.audit_score,
                "p_entail_yes": audit.p_entail_yes,
                "p_neutral_yes": audit.p_neutral_yes,
                "p_entail_no": audit.p_entail_no,
                "p_neutral_no": audit.p_neutral_no,
                "error": audit.error,
                "latency_ms": latency_ms,
            }
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{i}/{len(cases)}] exp={case['expected_label']:5s} "
                f"audit={audit.audit_score:.3f} "
                f"e_yes={audit.p_entail_yes:.2f} n_yes={audit.p_neutral_yes:.2f} "
                f"({latency_ms:.0f} ms) {audit.error}"
            )

    mb = [r["audit_score"] for r in results if r["expected"] == "maybe"]
    nm = [r["audit_score"] for r in results if r["expected"] != "maybe"]
    summary = {
        "auditor": "deberta-nli",
        "evidence_source": args.evidence_source,
        "dataset": str(args.dataset),
        "n": len(results),
        "errors": sum(1 for r in results if r["error"]),
        "missing_oracle_evidence": missing_oracle,
        "maybe_mean": st.mean(mb) if mb else None,
        "non_maybe_mean": st.mean(nm) if nm else None,
        "auroc_maybe_vs_rest": _auroc(mb, nm),
        "mean_latency_ms": st.mean([r["latency_ms"] for r in results]) if results else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n[deberta-nli / {args.evidence_source}] maybe={summary['maybe_mean']} "
        f"non={summary['non_maybe_mean']} AUROC={summary['auroc_maybe_vs_rest']:.3f} "
        f"missing_oracle={missing_oracle}"
    )
    print(f"Wrote {out_path}\nWrote {summary_path}")


if __name__ == "__main__":
    main()
