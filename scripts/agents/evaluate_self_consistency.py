#!/usr/bin/env python
"""Self-consistency baseline: k samples of ONE model, majority vote.

The debate is expensive — four 7B agents over three rounds plus supervisor calls,
~180s per case. Standard practice is to check that against the cheaper baseline of
sampling a single (possibly larger) model k times at temperature > 0 and taking
the majority. Without that number in the table there is no evidence the debate
structure earns its cost rather than the extra compute doing the work.

Compute matching, by parameters x calls:
    debate  = 12 x 7B agent calls + ~2 x 14B supervisor calls  ~= 112 B-calls
    k=4 of qwen3:30b                                           ~= 120 B-calls

Example:
    python scripts/agents/evaluate_self_consistency.py \
        --dataset data/benchmarks/pubmedqa/official_pqal_test/eval.json \
        --corpus data/benchmarks/pubmedqa/official_pqal_test/corpus.json \
        --model qwen3:8b --samples 4 --base-url http://127.0.0.1:11436 \
        --label selfconsistency_8b_k4
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.aggregation import extract_label  # noqa: E402
from app.schemas import ChatMessage  # noqa: E402

DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "debate"

SYSTEM = "You answer PubMedQA questions. Return only valid JSON."

PROMPT = """
Answer the research question from the abstract below, as the authors concluded.

- "yes"   : the authors conclude a positive association, effect, or affirmative answer
- "no"    : the authors conclude no association, no effect, or a negative answer
- "maybe" : the findings are genuinely mixed, or the study only partially answers
            the question as posed (surrogate endpoint, subgroup only)

RESEARCH QUESTION:
{question}

ABSTRACT:
{abstract}

Output ONLY: {{"label": "yes" | "no" | "maybe", "reason": "one sentence"}}
""".strip()


@dataclass
class SelfConsistencyResult:
    id: str
    expected_label: str
    predicted_label: str | None
    samples: list[str] = field(default_factory=list)
    vote_share: dict[str, float] = field(default_factory=dict)
    agreement: float = 0.0
    label_pass: bool = False
    latency_ms: float = 0.0


def _load(dataset: Path, corpus: Path, limit: int | None) -> list[dict[str, Any]]:
    raw = json.loads(dataset.read_text())
    docs = json.loads(corpus.read_text())
    by_id = {d["id"]: d for d in docs} if isinstance(docs, list) else docs
    out = []
    for case in raw if isinstance(raw, list) else raw.values():
        text = (by_id.get(case["id"]) or {}).get("content", "")
        body = text.split("Abstract context:", 1)[-1].strip()
        if not body:
            continue
        out.append(
            {
                "id": case["id"],
                "question": case.get("benchmark_question") or case["question"],
                "abstract": body,
                "expected_label": case["expected_label"],
            }
        )
    return out[:limit] if limit else out


async def _run(args: argparse.Namespace) -> None:
    from app.core.config import get_settings
    from app.providers.ollama import OllamaProvider
    from app.agents.backends import OllamaInferenceBackend

    settings = get_settings()
    cases = _load(Path(args.dataset), Path(args.corpus), args.limit)
    urls = [u.strip().rstrip("/") for u in args.base_urls.split(",") if u.strip()]
    backends = [
        OllamaInferenceBackend(
            OllamaProvider(
                base_url=u,
                timeout=settings.ollama_timeout,
                keep_alive=settings.ollama_keep_alive,
                num_predict=args.num_predict,
                num_ctx=max(settings.ollama_num_ctx, 8192),
            ),
            model=args.model,
            temperature=args.temperature,
        )
        for u in urls
    ]
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(index: int, case: dict[str, Any]) -> SelfConsistencyResult:
        async with semaphore:
            started = perf_counter()
            backend = backends[index % len(backends)]
            messages = [
                ChatMessage(role="system", content=SYSTEM),
                ChatMessage(
                    role="user",
                    content=PROMPT.format(question=case["question"], abstract=case["abstract"]),
                ),
            ]

            async def sample() -> str | None:
                try:
                    raw = await backend.complete(messages, temperature=args.temperature)
                except Exception:
                    return None
                try:
                    import re

                    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
                    obj = json.loads(match.group(0) if match else raw)
                    return extract_label(str(obj.get("label") or ""))
                except Exception:
                    return extract_label(raw)

            labels = [lab for lab in await asyncio.gather(*[sample() for _ in range(args.samples)]) if lab]
            counts = Counter(labels)
            predicted = counts.most_common(1)[0][0] if counts else None
            total = sum(counts.values()) or 1
            result = SelfConsistencyResult(
                id=case["id"],
                expected_label=case["expected_label"],
                predicted_label=predicted,
                samples=labels,
                vote_share={lab: counts.get(lab, 0) / total for lab in ("yes", "no", "maybe")},
                agreement=(counts.most_common(1)[0][1] / total) if counts else 0.0,
                label_pass=predicted == case["expected_label"],
                latency_ms=(perf_counter() - started) * 1000.0,
            )
            print(
                f"[{index + 1}/{len(cases)}] {'PASS' if result.label_pass else 'FAIL'} "
                f"{result.id} exp={result.expected_label} pred={result.predicted_label} "
                f"votes={labels} ({result.latency_ms:.0f} ms)",
                flush=True,
            )
            return result

    results = list(await asyncio.gather(*[one(i, c) for i, c in enumerate(cases)]))

    n = len(results) or 1
    by_label: dict[str, list[bool]] = {}
    for r in results:
        by_label.setdefault(r.expected_label, []).append(r.label_pass)
    # Agreement across samples is the natural confidence signal here — the
    # self-consistency analogue of panel unanimity, and free to compute.
    selective = []
    for floor in (0.0, 0.51, 0.75, 1.0):
        answered = [r for r in results if r.agreement >= floor]
        if answered:
            selective.append(
                {
                    "min_agreement": floor,
                    "coverage": len(answered) / n,
                    "selective_accuracy": sum(1 for r in answered if r.label_pass) / len(answered),
                    "unsettled_answered": sum(1 for r in answered if r.expected_label == "maybe"),
                }
            )
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "samples_per_case": args.samples,
        "temperature": args.temperature,
        "cases": len(results),
        "label_accuracy": sum(1 for r in results if r.label_pass) / n,
        "per_label_accuracy": {
            lab: sum(flags) / len(flags) for lab, flags in sorted(by_label.items())
        },
        "predicted_label_counts": dict(Counter(r.predicted_label or "none" for r in results)),
        "mean_agreement": sum(r.agreement for r in results) / n,
        "mean_latency_ms": sum(r.latency_ms for r in results) / n,
        "selective_prediction": selective,
    }
    out_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}.json"
    path.write_text(
        json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2, ensure_ascii=False)
    )
    print(f"\nWrote {path}")
    print(
        f"label_accuracy={summary['label_accuracy']:.3f} "
        f"per_label={ {k: round(v, 3) for k, v in summary['per_label_accuracy'].items()} } "
        f"mean_agreement={summary['mean_agreement']:.3f} "
        f"mean_latency_ms={summary['mean_latency_ms']:.0f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", default="qwen3:8b")
    p.add_argument("--base-urls", default="http://127.0.0.1:11436")
    p.add_argument("--samples", type=int, default=4, help="k samples per case")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--num-predict", type=int, default=1200)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    p.add_argument("--label", required=True)
    asyncio.run(_run(p.parse_args()))


if __name__ == "__main__":
    main()
