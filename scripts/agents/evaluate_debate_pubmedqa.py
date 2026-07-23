#!/usr/bin/env python3
"""Benchmark multi-agent debate on PubMedQA (yes/no/maybe), without a supervisor.

Modes:
  --backend mock|ollama     4-agent debate + majority vote
  --backend biolinkbert     BioLinkBERT-large classifier only (seed47 from .env)
  --hint biolinkbert        inject classifier label into each agent prompt
  --aggregate-with-biolinkbert
                            include BioLinkBERT vote in the final majority
  --fast                    compact prompts + lower num_predict (does NOT enable early-exit)
  --aggregate-mode bert_gate|bert_weighted|majority
                            how to combine panel + BioLinkBERT (default: bert_gate)

Examples:
  # Quality run on balanced90 (no early-exit; BERT-gate aggregation)
  .venv/bin/python scripts/agents/evaluate_debate_pubmedqa.py \\
    --backend ollama --hint biolinkbert --aggregate-with-biolinkbert \\
    --aggregate-mode bert_gate --rounds 2 --limit 90 --resume \\
    --num-predict 400 --agent-concurrency 1 \\
    --label debate_balanced90_ollama_r2_bertgate
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import DebateOrchestrator, MockInferenceBackend, build_default_agents, labels_unanimous
from app.agents.aggregation import (
    aggregate_pubmedqa_decision,
    final_labels_by_agent,
    majority_vote,
    opinion_label,
)
from app.agents.backends import (
    BioLinkBERTHintProvider,
    NullEvidenceHint,
    OllamaInferenceBackend,
    build_biolinkbert_hint_from_settings,
    hint_as_clinical_opinion,
)
from app.agents.models import AgentRoundOpinion
from app.rag.models import RetrievedDocument

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "data"
    / "benchmarks"
    / "pubmedqa"
    / "official_pqal_test"
    / "quick"
    / "balanced90.json"
)
DEFAULT_CORPUS = (
    PROJECT_ROOT / "data" / "benchmarks" / "pubmedqa" / "official_pqal_test" / "corpus.json"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "debate"


@dataclass
class DebateCaseResult:
    id: str
    expected_label: str
    predicted_label: str | None
    vote_share: dict[str, float]
    agent_labels: dict[str, str | None]
    round1_vote_label: str | None
    biolinkbert_label: str | None = None
    biolinkbert_confidence: float | None = None
    label_pass: bool = False
    round1_pass: bool = False
    biolinkbert_pass: bool | None = None
    unanimous_final: bool = False
    early_exit: bool = False
    rounds_run: int = 0
    aggregation_rule: str = ""
    latency_ms: float = 0.0
    final_opinions: list[dict[str, Any]] = field(default_factory=list)


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.dataset)
    if args.offset:
        cases = cases[max(args.offset, 0) :]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    corpus = _load_corpus(args.corpus)

    hint_provider: BioLinkBERTHintProvider | NullEvidenceHint = NullEvidenceHint()
    need_classifier = (
        args.backend == "biolinkbert"
        or args.hint == "biolinkbert"
        or args.aggregate_with_biolinkbert
    )
    if need_classifier:
        hint_provider = build_biolinkbert_hint_from_settings()
        if not hint_provider.available:
            raise SystemExit(f"BioLinkBERT classifier unavailable: {hint_provider.load_error}")
        print(f"Loaded BioLinkBERT classifier: {hint_provider.model_path}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or _default_label(args)
    json_path = report_dir / f"{label}.json"
    md_path = report_dir / f"{label}.md"
    checkpoint_path = report_dir / f"{label}.checkpoint.jsonl"

    prior_results = _load_checkpoint(checkpoint_path) if args.resume else {}
    if prior_results:
        print(f"Resuming with {len(prior_results)} completed cases from {checkpoint_path}")

    if args.backend == "biolinkbert":
        results = _evaluate_biolinkbert_only(
            hint_provider,  # type: ignore[arg-type]
            cases,
            corpus,
            prior_results=prior_results,
            checkpoint_path=checkpoint_path if args.resume else None,
        )
        rounds = 0
        aggregation = "biolinkbert_only"
        architecture = "biolinkbert_only"
        early_exit_rate = 0.0
    else:
        backend = _build_backend(
            args.backend,
            fast=args.fast,
            num_predict=args.num_predict,
        )
        agents = build_default_agents(
            backend,
            hint_provider=hint_provider if args.hint == "biolinkbert" else NullEvidenceHint(),
            task_mode="pubmedqa",
            compact=args.fast or args.compact,
        )
        bert_conf_threshold = args.early_exit_bert_confidence

        def _should_early_exit(round_number: int, round_opinions: list[AgentRoundOpinion]) -> bool:
            if not args.early_exit:
                return False
            if round_number < 1:
                return False
            if not labels_unanimous(round_opinions):
                return False
            if isinstance(hint_provider, BioLinkBERTHintProvider) and hint_provider.last_hint:
                panel_label = opinion_label(round_opinions[0].opinion)
                hint = hint_provider.last_hint
                return (
                    panel_label == hint.label
                    and hint.confidence >= bert_conf_threshold
                )
            return True

        orchestrator = DebateOrchestrator(
            agents,
            rounds=args.rounds,
            early_exit=_should_early_exit if args.early_exit else None,
            agent_concurrency=args.agent_concurrency,
        )
        results = asyncio.run(
            _evaluate_debate(
                orchestrator,
                cases,
                corpus,
                hint_provider=hint_provider if need_classifier else None,
                aggregate_with_biolinkbert=args.aggregate_with_biolinkbert,
                aggregate_mode=args.aggregate_mode,
                bert_gate_confidence=args.bert_gate_confidence,
                bert_vote_weight=args.bert_vote_weight,
                prior_results=prior_results,
                checkpoint_path=checkpoint_path if args.resume else None,
            )
        )
        rounds = args.rounds
        aggregation = (
            f"{args.aggregate_mode}+biolinkbert"
            if args.aggregate_with_biolinkbert
            else "confidence_weighted_majority_vote"
        )
        architecture = DebateOrchestrator.ARCHITECTURE
        early_exit_rate = (
            sum(1 for item in results if item.early_exit) / len(results) if results else 0.0
        )

    summary = _summarize(
        results,
        backend=args.backend,
        rounds=rounds,
        dataset=str(args.dataset),
        hint=args.hint,
        aggregation=aggregation,
        architecture=architecture,
        fast=args.fast,
        early_exit_rate=early_exit_rate,
    )
    payload = {"summary": summary, "cases": [asdict(item) for item in results]}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_report(summary, results), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"label_accuracy={summary['label_accuracy']:.3f} "
        f"biolinkbert_accuracy={summary.get('biolinkbert_accuracy')} "
        f"early_exit_rate={summary.get('early_exit_rate')} "
        f"mean_latency_ms={summary['mean_latency_ms']:.1f}"
    )


def _default_label(args: argparse.Namespace) -> str:
    parts = [f"debate_pubmedqa_{args.backend}"]
    if args.backend != "biolinkbert":
        parts.append(f"r{args.rounds}")
    if args.hint == "biolinkbert":
        parts.append("hint")
    if args.aggregate_with_biolinkbert:
        parts.append(args.aggregate_mode)
    if args.fast:
        parts.append("fast")
    return "_".join(parts)


def _load_checkpoint(path: Path) -> dict[str, DebateCaseResult]:
    if not path.exists():
        return {}
    loaded: dict[str, DebateCaseResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        loaded[data["id"]] = DebateCaseResult(**data)
    return loaded


def _append_checkpoint(path: Path, result: DebateCaseResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")


def _evaluate_biolinkbert_only(
    hint_provider: BioLinkBERTHintProvider,
    cases: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    *,
    prior_results: dict[str, DebateCaseResult],
    checkpoint_path: Path | None,
) -> list[DebateCaseResult]:
    results: list[DebateCaseResult] = []
    for index, case in enumerate(cases, start=1):
        if case["id"] in prior_results:
            results.append(prior_results[case["id"]])
            print(f"[{index}/{len(cases)}] SKIP id={case['id']} (checkpoint)")
            continue
        started = perf_counter()
        docs = _case_documents(case, corpus)
        hint_provider.set_case(question=case["question"], documents=docs)
        hint = hint_provider.predict_current()
        latency_ms = (perf_counter() - started) * 1000.0
        predicted = hint.label if hint else None
        result = DebateCaseResult(
            id=case["id"],
            expected_label=case["expected_label"],
            predicted_label=predicted,
            vote_share={predicted: 1.0} if predicted else {},
            agent_labels={"biolinkbert": predicted},
            round1_vote_label=predicted,
            biolinkbert_label=predicted,
            biolinkbert_confidence=hint.confidence if hint else None,
            label_pass=predicted == case["expected_label"],
            round1_pass=predicted == case["expected_label"],
            biolinkbert_pass=predicted == case["expected_label"] if predicted else False,
            unanimous_final=True,
            early_exit=False,
            rounds_run=0,
            aggregation_rule="biolinkbert_only",
            latency_ms=latency_ms,
            final_opinions=[],
        )
        results.append(result)
        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, result)
        status = "PASS" if result.label_pass else "FAIL"
        print(
            f"[{index}/{len(cases)}] {status} id={result.id} "
            f"exp={result.expected_label} pred={result.predicted_label} "
            f"conf={result.biolinkbert_confidence} ({latency_ms:.0f} ms)"
        )
        hint_provider.clear_case()
    return results


async def _evaluate_debate(
    orchestrator: DebateOrchestrator,
    cases: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    *,
    hint_provider: BioLinkBERTHintProvider | None,
    aggregate_with_biolinkbert: bool,
    aggregate_mode: str,
    bert_gate_confidence: float,
    bert_vote_weight: float,
    prior_results: dict[str, DebateCaseResult],
    checkpoint_path: Path | None,
) -> list[DebateCaseResult]:
    results: list[DebateCaseResult] = []
    for index, case in enumerate(cases, start=1):
        if case["id"] in prior_results:
            results.append(prior_results[case["id"]])
            print(f"[{index}/{len(cases)}] SKIP id={case['id']} (checkpoint)")
            continue
        started = perf_counter()
        docs = _case_documents(case, corpus)
        if hint_provider is not None:
            hint_provider.set_case(question=case["question"], documents=docs)
            hint_provider.predict_current()
        patient_case = _build_patient_case(case, corpus)
        debate = await orchestrator.run(patient_case)
        hint = hint_provider.last_hint if hint_provider is not None else None

        agent_opinions = [entry.opinion for entry in debate.final_opinions]
        bert_opinion = hint_as_clinical_opinion(hint) if (aggregate_with_biolinkbert and hint) else None
        predicted, share, rule = aggregate_pubmedqa_decision(
            agent_opinions,
            bert_opinion=bert_opinion,
            mode=aggregate_mode if aggregate_with_biolinkbert else "majority",
            bert_gate_confidence=bert_gate_confidence,
            bert_vote_weight=bert_vote_weight,
        )
        round1_opinions = [entry.opinion for entry in debate.rounds[0]]
        round1_label, _ = majority_vote(round1_opinions)
        agent_labels = final_labels_by_agent(debate.final_opinions)
        if hint is not None:
            agent_labels["biolinkbert"] = hint.label
        label_values = [label for label in agent_labels.values() if label is not None]
        latency_ms = (perf_counter() - started) * 1000.0
        early_exit = len(debate.rounds) < orchestrator.rounds

        result = DebateCaseResult(
            id=case["id"],
            expected_label=case["expected_label"],
            predicted_label=predicted,
            vote_share=share,
            agent_labels=agent_labels,
            round1_vote_label=round1_label,
            biolinkbert_label=hint.label if hint else None,
            biolinkbert_confidence=hint.confidence if hint else None,
            label_pass=predicted == case["expected_label"],
            round1_pass=round1_label == case["expected_label"],
            biolinkbert_pass=(hint.label == case["expected_label"]) if hint else None,
            unanimous_final=len(set(label_values)) == 1 and bool(label_values),
            early_exit=early_exit,
            rounds_run=len(debate.rounds),
            aggregation_rule=rule,
            latency_ms=latency_ms,
            final_opinions=[entry.model_dump() for entry in debate.final_opinions],
        )
        results.append(result)
        if checkpoint_path is not None:
            _append_checkpoint(checkpoint_path, result)
        status = "PASS" if result.label_pass else "FAIL"
        print(
            f"[{index}/{len(cases)}] {status} id={result.id} "
            f"exp={result.expected_label} pred={result.predicted_label} "
            f"bert={result.biolinkbert_label} rule={rule} "
            f"r1={result.round1_vote_label} rounds={result.rounds_run} "
            f"({latency_ms:.0f} ms)"
        )
        if hint_provider is not None:
            hint_provider.clear_case()
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--backend",
        choices=("mock", "ollama", "biolinkbert"),
        default="mock",
        help="mock/ollama = multi-agent debate; biolinkbert = classifier-only baseline",
    )
    parser.add_argument(
        "--hint",
        choices=("none", "biolinkbert"),
        default="none",
        help="Inject BioLinkBERT yes/no/maybe hint into agent prompts",
    )
    parser.add_argument(
        "--aggregate-with-biolinkbert",
        action="store_true",
        help="Include BioLinkBERT in the final aggregation",
    )
    parser.add_argument(
        "--aggregate-mode",
        choices=("majority", "bert_weighted", "bert_gate"),
        default="bert_gate",
        help="How to combine panel + BioLinkBERT (bert_gate recommended)",
    )
    parser.add_argument(
        "--bert-gate-confidence",
        type=float,
        default=0.90,
        help="Min BioLinkBERT confidence to trust yes/no under bert_gate",
    )
    parser.add_argument(
        "--bert-vote-weight",
        type=float,
        default=3.0,
        help="BioLinkBERT weight in bert_weighted / uncertain bert_gate path",
    )
    parser.add_argument("--rounds", type=int, default=3, choices=(2, 3))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N cases")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Compact prompts + lower num_predict (does NOT enable early-exit)",
    )
    parser.add_argument("--compact", action="store_true", help="Compact PubMedQA prompts only")
    parser.add_argument(
        "--early-exit",
        action="store_true",
        help="Optional: skip later rounds when panel is unanimous (+ BERT agree if present)",
    )
    parser.add_argument(
        "--early-exit-bert-confidence",
        type=float,
        default=0.85,
        help="Min BioLinkBERT confidence required for early-exit agreement",
    )
    parser.add_argument(
        "--agent-concurrency",
        type=int,
        default=1,
        help=(
            "Max concurrent Ollama agent calls in round 1 (the independent-opinion "
            "round). Round 2+ are round-robin turns and always run sequentially, "
            "since each turn depends on the previous one's output."
        ),
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Override Ollama num_predict (default: 220 with --fast, else settings)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint JSONL next to the report label",
    )
    return parser.parse_args()


def _build_backend(
    name: str,
    *,
    fast: bool = False,
    num_predict: int | None = None,
) -> MockInferenceBackend | OllamaInferenceBackend:
    if name == "mock":
        return MockInferenceBackend()
    from app.core.config import get_settings
    from app.providers.ollama import OllamaProvider

    settings = get_settings()
    if num_predict is not None:
        predict = num_predict
    elif fast:
        predict = 220
    else:
        # Do NOT inflate above .env — previous max(..., 800) made runs much slower.
        predict = settings.ollama_num_predict
    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        num_predict=predict,
        num_ctx=min(max(settings.ollama_num_ctx, 2048), 4096),
    )
    print(f"Ollama model={settings.default_model} num_predict={predict}")
    return OllamaInferenceBackend(provider, model=settings.default_model, temperature=0.1)


def _load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset at {path}")
    cases: list[dict[str, Any]] = []
    for item in data:
        expected = str(item.get("expected_label", "")).strip().lower()
        if expected not in {"yes", "no", "maybe"}:
            continue
        cases.append(
            {
                "id": str(item["id"]),
                "question": str(item.get("benchmark_question") or item.get("question") or ""),
                "expected_label": expected,
                "relevant_document_ids": [str(x) for x in item.get("relevant_document_ids", [])],
            }
        )
    return cases


def _load_corpus(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return {str(doc["id"]): doc for doc in data if isinstance(doc, dict) and "id" in doc}
    if isinstance(data, dict):
        return {str(key): value for key, value in data.items()}
    raise ValueError(f"Unsupported corpus format: {path}")


def _case_documents(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> list[RetrievedDocument]:
    documents: list[RetrievedDocument] = []
    for doc_id in case["relevant_document_ids"]:
        doc = corpus.get(doc_id)
        if not doc:
            continue
        documents.append(
            RetrievedDocument(
                id=doc_id,
                title=str(doc.get("title") or ""),
                content=str(doc.get("content") or ""),
                source=str(doc.get("source") or "pubmedqa"),
                score=float(doc.get("score") or 1.0),
                metadata=dict(doc.get("metadata") or {}),
            )
        )
    return documents


def _build_patient_case(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> str:
    evidence_blocks: list[str] = []
    for doc in _case_documents(case, corpus):
        # Truncate very long abstracts for faster LLM prompts.
        content = doc.content
        if len(content) > 2800:
            content = content[:2800] + "…"
        evidence_blocks.append(f"SOURCE {doc.id}\nTitle: {doc.title}\n{content}".strip())
    if not evidence_blocks:
        evidence_blocks.append("(No corpus evidence found for this case.)")
    return (
        "Task: Answer yes/no/maybe from the evidence only.\n\n"
        f"RESEARCH QUESTION:\n{case['question']}\n\n"
        "EVIDENCE:\n" + "\n\n".join(evidence_blocks)
    )


def _summarize(
    results: list[DebateCaseResult],
    *,
    backend: str,
    rounds: int,
    dataset: str,
    hint: str,
    aggregation: str,
    architecture: str,
    fast: bool,
    early_exit_rate: float,
) -> dict[str, Any]:
    n = len(results) or 1
    by_label: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_label.setdefault(result.expected_label, {"support": 0, "correct": 0})
        bucket["support"] += 1
        if result.label_pass:
            bucket["correct"] += 1

    pred_counts = Counter(result.predicted_label or "none" for result in results)
    agent_correct: dict[str, list[bool]] = {}
    for result in results:
        for agent_id, label in result.agent_labels.items():
            agent_correct.setdefault(agent_id, []).append(label == result.expected_label)

    bert_flags = [r.biolinkbert_pass for r in results if r.biolinkbert_pass is not None]
    return {
        "dataset": dataset,
        "backend": backend,
        "hint": hint,
        "rounds": rounds,
        "fast": fast,
        "cases": len(results),
        "label_accuracy": sum(1 for r in results if r.label_pass) / n,
        "round1_accuracy": sum(1 for r in results if r.round1_pass) / n,
        "biolinkbert_accuracy": (sum(1 for flag in bert_flags if flag) / len(bert_flags)) if bert_flags else None,
        "unanimous_rate": sum(1 for r in results if r.unanimous_final) / n,
        "early_exit_rate": early_exit_rate,
        "mean_latency_ms": mean([r.latency_ms for r in results]) if results else 0.0,
        "predicted_label_counts": dict(pred_counts),
        "per_label_accuracy": {
            label: (stats["correct"] / stats["support"] if stats["support"] else 0.0)
            for label, stats in sorted(by_label.items())
        },
        "per_agent_accuracy": {
            agent_id: (sum(flags) / len(flags) if flags else 0.0)
            for agent_id, flags in sorted(agent_correct.items())
        },
        "aggregation": aggregation,
        "architecture": architecture,
        "supervisor": architecture not in {"round_robin_no_supervisor", "biolinkbert_only"},
    }


def _markdown_report(summary: dict[str, Any], results: list[DebateCaseResult]) -> str:
    lines = [
        "# Multi-Agent Debate PubMedQA Benchmark",
        "",
        "## Summary",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Backend: `{summary['backend']}`",
        f"- Hint: `{summary['hint']}`",
        f"- Rounds: {summary['rounds']}",
        f"- Fast mode: {summary['fast']}",
        f"- Cases: {summary['cases']}",
        f"- Label accuracy: {summary['label_accuracy']:.3f}",
        f"- Round-1 accuracy: {summary['round1_accuracy']:.3f}",
        f"- BioLinkBERT accuracy: {summary['biolinkbert_accuracy']}",
        f"- Early-exit rate: {summary['early_exit_rate']:.3f}",
        f"- Unanimous final rate: {summary['unanimous_rate']:.3f}",
        f"- Mean latency: {summary['mean_latency_ms']:.1f} ms",
        f"- Architecture: `{summary['architecture']}` (supervisor: {summary['supervisor']})",
        f"- Aggregation: `{summary['aggregation']}`",
        "",
        "## Per-label accuracy",
        "",
        "| Label | Accuracy |",
        "|---|---:|",
    ]
    for label, acc in summary["per_label_accuracy"].items():
        lines.append(f"| {label} | {acc:.3f} |")
    lines.extend(
        [
            "",
            "## Per-agent / BioLinkBERT accuracy",
            "",
            "| Agent | Accuracy |",
            "|---|---:|",
        ]
    )
    for agent_id, acc in summary["per_agent_accuracy"].items():
        lines.append(f"| {agent_id} | {acc:.3f} |")
    lines.extend(["", "## Predicted label counts", ""])
    for label, count in summary["predicted_label_counts"].items():
        lines.append(f"- `{label}`: {count}")
    lines.extend(["", "## Cases", ""])
    for result in results:
        mark = "ok" if result.label_pass else "FAIL"
        extra = " early_exit" if result.early_exit else ""
        lines.append(
            f"- `{result.id}` [{mark}] expected={result.expected_label} "
            f"pred={result.predicted_label} bert={result.biolinkbert_label} "
            f"r1={result.round1_vote_label} rounds={result.rounds_run}{extra}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
