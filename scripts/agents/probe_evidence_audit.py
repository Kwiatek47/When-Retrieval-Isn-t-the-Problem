"""Signal-collection probe: does the evidence condition-audit separate PubMedQA `maybe`?

Runs the LLM evidence audit on every case and logs ALL per-case signals to a
structured JSONL for paper analysis. Parametrized by model so we can ablate the
NLI-audit signal across backbones (qwen 7b, qwen 14b, deepseek-r1 reasoning).

Usage:
  OLLAMA_MODEL=qwen2.5:14b python scripts/agents/probe_evidence_audit.py \
      --dataset .../balanced90.json --label audit_14b
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import statistics as st
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.evidence_audit import audit_evidence

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "agents"))
from evaluate_debate_pubmedqa import _build_backend  # noqa: E402


DEFAULT_DATA = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json"
CORPUS = PROJECT_ROOT / "data/benchmarks/pubmedqa/official_pqal_test/corpus.json"
REPORT_DIR = PROJECT_ROOT / "reports/debate/signals"


def _corpus() -> dict:
    raw = json.loads(CORPUS.read_text())
    if isinstance(raw, list):
        return {str(d["id"]): d for d in raw if isinstance(d, dict) and "id" in d}
    return {str(k): v for k, v in raw.items()}


def _evidence(case: dict, corpus: dict) -> str:
    blocks = []
    for doc_id in case.get("relevant_document_ids", []):
        doc = corpus.get(str(doc_id))
        if not doc:
            continue
        content = str(doc.get("content") or "")
        blocks.append(content)
    return "\n\n".join(blocks)


ORI_PQAL = PROJECT_ROOT / "data/raw/pubmedqa_official/data/ori_pqal.json"


def _pmid_from_case(case: dict) -> str | None:
    tail = str(case.get("id") or "").rsplit("-", 1)[-1]
    if tail.isdigit():
        return tail
    pmids = case.get("relevant_pmids") or []
    return str(pmids[0]) if pmids else None


def _long_answer_index() -> dict[str, str]:
    data = json.loads(ORI_PQAL.read_text())
    return {str(k): str(v.get("LONG_ANSWER") or "").strip() for k, v in data.items()}


def _auroc(pos: list[float], neg: list[float]) -> float:
    """AUROC of scores ranking pos above neg (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else (0.5 if a == b else 0.0)
    return wins / (len(pos) * len(neg))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", type=str, default=None, help="Override OLLAMA_MODEL for this probe")
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--num-predict", type=int, default=400)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--evidence-source",
        choices=("abstract", "long_answer"),
        default="abstract",
        help="abstract = normal input; long_answer = gold oracle conclusions (ori_pqal)",
    )
    args = parser.parse_args()

    from app.core.config import get_settings

    model_name = args.model or get_settings().default_model
    backend = _build_backend("ollama", num_predict=args.num_predict, model=model_name)

    label = args.label or f"audit_{model_name.replace(':', '_').replace('/', '_')}"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / f"{label}.jsonl"
    summary_path = REPORT_DIR / f"{label}.summary.json"

    cases = json.loads(args.dataset.read_text())
    if args.limit:
        cases = cases[: args.limit]
    corpus = _corpus()
    long_answers = _long_answer_index() if args.evidence_source == "long_answer" else {}
    missing_oracle = 0

    results = []
    with out_path.open("w", encoding="utf-8") as handle:
        for i, case in enumerate(cases, 1):
            q = case["question"]
            if args.evidence_source == "long_answer":
                pmid = _pmid_from_case(case)
                ev = long_answers.get(pmid or "", "")
                if not ev:
                    missing_oracle += 1
            else:
                ev = _evidence(case, corpus)
            t0 = time.perf_counter()
            audit = await audit_evidence(backend, question=q, evidence=ev)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            row = {
                "id": case["id"],
                "expected": case["expected_label"],
                "model": model_name,
                "evidence_source": args.evidence_source,
                "audit_score": audit.audit_score,
                "supported": audit.supported,
                "refuted": audit.refuted,
                "silent": audit.silent,
                "n_conditions": len(audit.conditions),
                "conditions": audit.conditions,
                "error": audit.error,
                "latency_ms": latency_ms,
            }
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{i}/{len(cases)}] exp={case['expected_label']:5s} "
                f"audit={audit.audit_score:.2f} sup={audit.supported} ref={audit.refuted} "
                f"sil={audit.silent} ({latency_ms:.0f} ms) {audit.error}"
            )

    mb = [r["audit_score"] for r in results if r["expected"] == "maybe"]
    nm = [r["audit_score"] for r in results if r["expected"] != "maybe"]
    errors = sum(1 for r in results if r["error"])
    summary = {
        "model": model_name,
        "evidence_source": args.evidence_source,
        "missing_oracle_evidence": missing_oracle,
        "dataset": str(args.dataset),
        "n": len(results),
        "errors": errors,
        "maybe_mean": st.mean(mb) if mb else None,
        "non_maybe_mean": st.mean(nm) if nm else None,
        "auroc_maybe_vs_rest": _auroc(mb, nm),
        "mean_latency_ms": st.mean([r["latency_ms"] for r in results]) if results else 0.0,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\n[{model_name}] audit_score maybe={summary['maybe_mean']} "
        f"non={summary['non_maybe_mean']} AUROC={summary['auroc_maybe_vs_rest']:.3f} "
        f"errors={errors}/{len(results)}"
    )
    print(f"Wrote {out_path}\nWrote {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
