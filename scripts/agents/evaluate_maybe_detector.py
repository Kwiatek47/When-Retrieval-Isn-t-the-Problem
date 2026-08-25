#!/usr/bin/env python
"""Score the standalone `maybe` detector on PubMedQA — one LLM call per case.

The detector targets category 1 from docs/agents/maybe-detector-spec.md (answer
splits). It is scored as a binary detector (maybe vs not-maybe), and separately
composed with BioLinkBERT's binary answer to show whether it clears the
break-even bar (precision ~0.55; below that it loses ground regardless of recall).

Example:
    python scripts/agents/evaluate_maybe_detector.py \
        --dataset data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json \
        --corpus data/benchmarks/pubmedqa/official_pqal_test/corpus.json \
        --model qwen2.5:14b --base-url http://127.0.0.1:11434 \
        --label maybe_detector_14b_v1
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.backends import OllamaInferenceBackend  # noqa: E402
from app.agents.maybe_detector import (  # noqa: E402
    AnswerSplitVerdict,
    apply_split_detector,
    detect_answer_split,
)

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "debate"


@dataclass
class DetectorCaseResult:
    id: str
    expected_label: str
    is_maybe: bool
    split_detected: bool
    split_kind: str
    confidence: float
    conflicting_findings: list[str] = field(default_factory=list)
    rationale: str = ""
    latency_ms: float = 0.0


def _load_cases(dataset: Path, corpus: Path, limit: int | None) -> list[dict[str, Any]]:
    raw = json.loads(dataset.read_text())
    docs = json.loads(corpus.read_text())
    by_id = {d["id"]: d for d in docs} if isinstance(docs, list) else docs
    cases = []
    for case in raw if isinstance(raw, list) else raw.values():
        text = (by_id.get(case["id"]) or {}).get("content", "")
        # Strip the synthetic preamble; keep the abstract body.
        body = text.split("Abstract context:", 1)[-1].strip()
        if not body:
            continue
        cases.append(
            {
                "id": case["id"],
                "question": case.get("benchmark_question") or case["question"],
                "abstract": body,
                "expected_label": case["expected_label"],
            }
        )
    return cases[:limit] if limit else cases


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _summarize(
    results: list[DetectorCaseResult],
    *,
    bert_labels: dict[str, str] | None,
    expected: dict[str, str],
    min_confidence: float,
) -> dict[str, Any]:
    n = len(results) or 1
    fired = [r for r in results if r.split_detected and r.confidence >= min_confidence]
    tp = sum(1 for r in fired if r.is_maybe)
    fp = len(fired) - tp
    fn = sum(1 for r in results if r.is_maybe and r not in fired)
    base_rate = sum(1 for r in results if r.is_maybe) / n
    stats = _prf(tp, fp, fn)

    summary: dict[str, Any] = {
        "cases": len(results),
        "maybe_base_rate": base_rate,
        "fire_rate": len(fired) / n,
        **stats,
        "lift_over_base_rate": stats["precision"] - base_rate,
        "break_even_precision": 0.55,
        "clears_break_even": stats["precision"] >= 0.55,
        "by_split_kind": {},
        "min_confidence": min_confidence,
    }
    for kind in ("subgroup", "compound_question", "outcome_conflict"):
        sub = [r for r in fired if r.split_kind == kind]
        if sub:
            hits = sum(1 for r in sub if r.is_maybe)
            summary["by_split_kind"][kind] = {
                "fired": len(sub),
                "precision": hits / len(sub),
            }

    # Composition with the binary classifier: does it actually gain accuracy?
    if bert_labels:
        ids = [r.id for r in results if r.id in bert_labels]
        base_acc = sum(1 for i in ids if bert_labels[i] == expected[i]) / (len(ids) or 1)
        by_id = {r.id: r for r in results}
        composed = 0
        for i in ids:
            verdict = AnswerSplitVerdict(
                split_detected=by_id[i].split_detected,
                conflicting_findings=by_id[i].conflicting_findings,
                confidence=by_id[i].confidence,
            )
            label, _ = apply_split_detector(
                bert_labels[i], verdict, min_confidence=min_confidence
            )
            composed += label == expected[i]
        summary["composed_with_biolinkbert"] = {
            "baseline_accuracy": base_acc,
            "composed_accuracy": composed / (len(ids) or 1),
            "delta": composed / (len(ids) or 1) - base_acc,
        }
    return summary


async def _run(args: argparse.Namespace) -> None:
    from app.core.config import get_settings
    from app.providers.ollama import OllamaProvider

    settings = get_settings()
    cases = _load_cases(Path(args.dataset), Path(args.corpus), args.limit)
    provider = OllamaProvider(
        base_url=args.base_url.rstrip("/"),
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        num_predict=600,
        num_ctx=max(settings.ollama_num_ctx, 8192),
    )
    backend = OllamaInferenceBackend(provider, model=args.model, temperature=0.0)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(case: dict[str, Any]) -> DetectorCaseResult:
        async with semaphore:
            started = perf_counter()
            verdict = await detect_answer_split(
                backend, question=case["question"], abstract=case["abstract"]
            )
            result = DetectorCaseResult(
                id=case["id"],
                expected_label=case["expected_label"],
                is_maybe=case["expected_label"] == "maybe",
                split_detected=verdict.is_usable,
                split_kind=verdict.split_kind,
                confidence=verdict.confidence,
                conflicting_findings=verdict.conflicting_findings,
                rationale=verdict.rationale,
                latency_ms=(perf_counter() - started) * 1000.0,
            )
            mark = "SPLIT" if result.split_detected else "  -  "
            hit = "ok " if result.split_detected == result.is_maybe else "MISS"
            print(
                f"[{hit}] {mark} {result.id} exp={result.expected_label:5s} "
                f"kind={result.split_kind:18s} ({result.latency_ms:.0f} ms)",
                flush=True,
            )
            return result

    results = list(await asyncio.gather(*[one(c) for c in cases]))

    bert_labels = None
    if args.biolinkbert_from:
        prior = json.loads(Path(args.biolinkbert_from).read_text())
        bert_labels = {
            c["id"]: c["biolinkbert_label"]
            for c in prior.get("cases", [])
            if c.get("biolinkbert_label")
        }

    summary = _summarize(
        results,
        bert_labels=bert_labels,
        expected={c["id"]: c["expected_label"] for c in cases},
        min_confidence=args.min_confidence,
    )
    summary["model"] = args.model
    summary["dataset"] = args.dataset

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{args.label}.json"
    out.write_text(
        json.dumps(
            {"summary": summary, "cases": [asdict(r) for r in results]},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nWrote {out}")
    print(
        f"precision={summary['precision']:.3f} recall={summary['recall']:.3f} "
        f"f1={summary['f1']:.3f} fire_rate={summary['fire_rate']:.3f} "
        f"lift={summary['lift_over_base_rate']:+.3f} "
        f"clears_break_even={summary['clears_break_even']}"
    )
    if "composed_with_biolinkbert" in summary:
        c = summary["composed_with_biolinkbert"]
        print(
            f"composed with BioLinkBERT: {c['baseline_accuracy']:.3f} -> "
            f"{c['composed_accuracy']:.3f} ({c['delta']:+.3f})"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default="qwen2.5:14b")
    p.add_argument("--base-url", default="http://127.0.0.1:11434")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-confidence", type=float, default=0.0)
    p.add_argument(
        "--biolinkbert-from",
        default=None,
        help="Prior debate report JSON to read biolinkbert_label per case from.",
    )
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--label", required=True)
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
