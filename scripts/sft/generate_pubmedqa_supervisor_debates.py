#!/usr/bin/env python3
"""Generate resumable two-round peer debates for supervisor SFT."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import build_default_agents
from app.agents.aggregation import build_debate_brief
from app.agents.backends import (
    BioLinkBERTHintProvider,
    EvidenceHint,
    MockInferenceBackend,
    OllamaInferenceBackend,
    build_biolinkbert_hint_from_settings,
)
from app.agents.orchestrator import DebateOrchestrator
from app.rag.models import RetrievedDocument


class _StaticHintProvider:
    def __init__(self, hint: EvidenceHint | None) -> None:
        self.hint = hint

    def get_hint(self, patient_case: str) -> EvidenceHint | None:  # noqa: ARG002
        return self.hint


def build_patient_case(source: dict[str, Any]) -> str:
    """Render the same evidence-only shape used by the PubMedQA benchmark."""
    question = str(source.get("question") or "").strip()
    evidence = str(source.get("evidence") or "").strip()
    if not question or not evidence:
        raise ValueError("Source record requires non-empty question and evidence.")
    return (
        "Task: Answer yes/no/maybe from the evidence only.\n\n"
        f"RESEARCH QUESTION:\n{question}\n\n"
        f"EVIDENCE:\nSOURCE pubmedqa-{source.get('pmid', source.get('id', ''))}\n"
        f"Title: {question}\n{evidence[:2800]}"
    )


async def generate_debate_record(
    source: dict[str, Any],
    *,
    backend: Any,
    hint: EvidenceHint | None,
    agent_concurrency: int = 2,
) -> dict[str, Any]:
    """Run a fixed two-round peer debate without any supervisor calls."""
    patient_case = build_patient_case(source)
    agents = build_default_agents(
        backend,
        hint_provider=_StaticHintProvider(hint),
        task_mode="pubmedqa",
        compact=True,
    )
    orchestrator = DebateOrchestrator(
        agents,
        rounds=2,
        min_rounds=2,
        max_rounds=2,
        adaptive_rounds=False,
        debate_mode="peer",
        agent_concurrency=max(1, int(agent_concurrency)),
        blind_critic="all-rounds",
    )
    debate = await orchestrator.run(patient_case)
    brief = build_debate_brief(debate.rounds, shared_report=None)
    return {
        "id": str(source.get("id") or f"pubmedqa-{source.get('pmid', '')}"),
        "pmid": str(source.get("pmid") or ""),
        "source_dataset": str(source.get("source_dataset") or ""),
        "gold_label": str(source.get("label") or "").strip().lower(),
        "question": str(source.get("question") or ""),
        "evidence": str(source.get("evidence") or ""),
        "long_answer": str(source.get("long_answer") or ""),
        "patient_case": patient_case,
        "biolinkbert_hint": asdict(hint) if hint is not None else None,
        "history": [
            [entry.model_dump() for entry in round_entries]
            for round_entries in debate.rounds
        ],
        "final_opinions": [entry.model_dump() for entry in debate.final_opinions],
        "debate_brief": brief,
    }


def load_completed_ids(path: Path) -> set[str]:
    """Read valid checkpoint lines; ignore a truncated trailing write."""
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("id"):
                completed.add(str(value["id"]))
    return completed


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            records.append(value)
    return records


def _build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "mock":
        return MockInferenceBackend()
    from app.core.config import get_settings
    from app.providers.ollama import OllamaProvider

    settings = get_settings()
    provider = OllamaProvider(
        base_url=args.ollama_base_url.rstrip("/"),
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        num_predict=args.num_predict,
        num_ctx=8192,
    )
    return OllamaInferenceBackend(
        provider,
        model=args.model,
        temperature=0.1,
    )


def _predict_hint(
    provider: BioLinkBERTHintProvider | None,
    source: dict[str, Any],
) -> EvidenceHint | None:
    if provider is None:
        return None
    document = RetrievedDocument(
        id=str(source.get("id") or source.get("pmid") or ""),
        title=str(source.get("question") or ""),
        content=str(source.get("evidence") or ""),
        source="PubMedQA",
    )
    provider.set_case(
        question=str(source.get("question") or ""),
        documents=[document],
    )
    try:
        return provider.predict_current()
    finally:
        provider.clear_case()


async def generate_file(
    *,
    source_file: Path,
    output_file: Path,
    backend: Any,
    hint_provider: BioLinkBERTHintProvider | None,
    agent_concurrency: int,
    case_concurrency: int,
    limit: int | None,
) -> dict[str, int]:
    sources = _load_jsonl(source_file)
    if limit is not None:
        sources = sources[: max(0, limit)]
    completed = load_completed_ids(output_file)
    pending = [
        source
        for source in sources
        if str(source.get("id") or f"pubmedqa-{source.get('pmid', '')}") not in completed
    ]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, int(case_concurrency)))
    write_lock = asyncio.Lock()
    generated = 0

    async def run_one(source: dict[str, Any]) -> None:
        nonlocal generated
        async with semaphore:
            hint = _predict_hint(hint_provider, source)
            record = await generate_debate_record(
                source,
                backend=backend,
                hint=hint,
                agent_concurrency=agent_concurrency,
            )
            async with write_lock:
                with output_file.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                generated += 1
                print(
                    f"[{generated}/{len(pending)}] id={record['id']} "
                    f"gold={record['gold_label']}"
                )

    await asyncio.gather(*(run_one(source) for source in pending))
    return {
        "source_count": len(sources),
        "already_completed": len(completed & {
            str(source.get("id") or f"pubmedqa-{source.get('pmid', '')}")
            for source in sources
        }),
        "generated": generated,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--backend", choices=("ollama", "mock"), default="ollama")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--hint", choices=("biolinkbert", "none"), default="biolinkbert")
    parser.add_argument("--num-predict", type=int, default=500)
    parser.add_argument("--agent-concurrency", type=int, default=2)
    parser.add_argument("--case-concurrency", type=int, default=2)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    backend = _build_backend(args)
    hint_provider = None
    if args.hint == "biolinkbert":
        hint_provider = build_biolinkbert_hint_from_settings()
        if not hint_provider.available:
            raise SystemExit(
                f"BioLinkBERT classifier unavailable: {hint_provider.load_error}"
            )
    summary = asyncio.run(
        generate_file(
            source_file=args.source_file,
            output_file=args.output_file,
            backend=backend,
            hint_provider=hint_provider,
            agent_concurrency=args.agent_concurrency,
            case_concurrency=args.case_concurrency,
            limit=args.limit,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
