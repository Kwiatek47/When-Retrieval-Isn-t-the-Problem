#!/usr/bin/env python3
"""Self-consistency baseline for PubMedQA: N independent samples + majority vote.

Control arm for the paper: does debate buy anything over compute-matched
repeated sampling with frozen evidence? Samples use the same ClinicalAgent
round-1 prompt (persona ``generalist``, no peer context).

Cost matching: run the debate arm first (on a build that records LLM usage),
then pass its report to ``--match-cost-report``. Existing debate reports from
before usage tracking have no cost block — pass ``--samples N`` instead.

This copy is adapted to the current tree (no prompt_versioning / lockfiles).
The sampling loop matches ``origin/feat/paper-baselines``.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import majority_vote, opinion_label
from app.agents.backends import (
    BioLinkBERTHintProvider,
    EvidenceHint,
    NullEvidenceHint,
    build_biolinkbert_hint_from_settings,
)
from app.agents.models import AgentRoundOpinion, ClinicalOpinion
from app.agents.uncertainty import _label_entropy
from app.core.usage import start_usage_scope
from scripts.agents.evaluate_debate_pubmedqa import (
    DEFAULT_CORPUS,
    DEFAULT_DATASET,
    DEFAULT_REPORT_DIR,
    DebateCaseResult,
    _append_checkpoint,
    _build_backend,
    _build_patient_case,
    _case_documents,
    _load_cases,
    _load_checkpoint,
    _load_corpus,
    _markdown_report,
    _summarize,
    _summarize_cost,
    report_has_measured_cost,
)

SAMPLING_PERSONA = "generalist"


class _StaticHintProvider:
    """Freeze a BioLinkBERT hint so every sample of a case sees the same signal."""

    def __init__(self, hint: EvidenceHint | None) -> None:
        self._hint = hint

    def get_hint(self, patient_case: str) -> EvidenceHint | None:
        _ = patient_case
        return self._hint


def main() -> None:
    args = _parse_args()
    cases = _load_cases(args.dataset)
    if args.offset:
        cases = cases[max(args.offset, 0) :]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    corpus = _load_corpus(args.corpus)

    samples = _resolve_sample_count(args)

    hint_provider: BioLinkBERTHintProvider | None = None
    if args.hint == "biolinkbert" or args.record_biolinkbert:
        provider = build_biolinkbert_hint_from_settings()
        if not provider.available:
            raise SystemExit(f"BioLinkBERT classifier unavailable: {provider.load_error}")
        print(f"Loaded BioLinkBERT classifier: {provider.model_path}")
        hint_provider = provider

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"self_consistency_pubmedqa_{args.backend}_n{samples}"
    json_path = report_dir / f"{label}.json"
    md_path = report_dir / f"{label}.md"
    checkpoint_path = report_dir / f"{label}.checkpoint.jsonl"

    prior_results = _load_checkpoint(checkpoint_path) if args.resume else {}
    if prior_results:
        print(f"Resuming with {len(prior_results)} completed cases from {checkpoint_path}")
        _reject_mixed_sample_counts(prior_results, samples=samples, checkpoint_path=checkpoint_path)

    print(
        f"Self-consistency: {samples} sample(s)/case, temperature={args.temperature}, "
        f"persona={SAMPLING_PERSONA}, hint={args.hint}"
    )

    backend = _build_backend(
        args.backend,
        fast=args.fast,
        num_predict=args.num_predict,
    )

    def _agent_factory(case_hint_provider: Any, case_index: int) -> ClinicalAgent:
        _ = case_index
        return ClinicalAgent(
            agent_id=SAMPLING_PERSONA,
            persona=SAMPLING_PERSONA,
            backend=backend,
            hint_provider=case_hint_provider,
            temperature=args.temperature,
            task_mode="pubmedqa",
            compact=args.fast or args.compact,
        )

    results = asyncio.run(
        _evaluate_self_consistency(
            cases,
            corpus,
            agent_factory=_agent_factory,
            samples=samples,
            hint_provider=hint_provider,
            inject_hint=args.hint == "biolinkbert",
            case_concurrency=1,
            sample_concurrency=args.sample_concurrency,
            prior_results=prior_results,
            checkpoint_path=checkpoint_path if args.resume else None,
        )
    )

    summary = _summarize(
        results,
        backend=args.backend,
        rounds=1,
        dataset=str(args.dataset),
        hint=args.hint,
        aggregation=f"self_consistency_majority_n{samples}",
        architecture="self_consistency",
        fast=args.fast,
        early_exit_rate=0.0,
    )
    summary["arm"] = "self_consistency"
    summary["samples"] = samples
    summary["temperature"] = args.temperature
    summary["sampling_persona"] = SAMPLING_PERSONA
    summary["single_sample_accuracy"] = summary["round1_accuracy"]
    if args.match_cost_report:
        summary["cost_matched_from"] = str(args.match_cost_report)
    summary["cost"] = _summarize_cost(results)
    payload = {"summary": summary, "cases": [asdict(item) for item in results]}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_self_consistency_markdown(summary, results), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        f"label_accuracy={summary['label_accuracy']:.3f} "
        f"single_sample_accuracy={summary['single_sample_accuracy']:.3f} "
        f"biolinkbert_accuracy={summary.get('biolinkbert_accuracy')} "
        f"unanimous_rate={summary['unanimous_rate']:.3f}"
    )
    cost = summary.get("cost") or {}
    if cost.get("measured_cases"):
        print(
            f"mean_llm_calls_per_case={cost['mean_llm_calls_per_case']:.2f} "
            f"mean_total_tokens_per_case={cost['mean_total_tokens_per_case']:.0f} "
            f"measured_on={cost['measured_cases']}/{len(results)} cases"
        )
        _print_cost_match_verdict(args.match_cost_report, cost)


def _reject_mixed_sample_counts(
    prior_results: dict[str, DebateCaseResult],
    *,
    samples: int,
    checkpoint_path: Path,
) -> None:
    expected = f"self_consistency_majority_n{samples}"
    mismatched = sorted(
        {r.aggregation_rule for r in prior_results.values() if r.aggregation_rule != expected}
    )
    if not mismatched:
        return
    raise SystemExit(
        f"Checkpoint {checkpoint_path} was written with {', '.join(mismatched)}, "
        f"but this run would use {expected}. Use --label for a new arm."
    )


def _resolve_sample_count(args: argparse.Namespace) -> int:
    if not args.match_cost_report:
        return max(1, int(args.samples))

    path = Path(args.match_cost_report)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"--match-cost-report: no such report {path}") from exc
    if not report_has_measured_cost(report):
        raise SystemExit(
            f"--match-cost-report: {path} has no measured cost "
            "(need summary.cost.measured_cases > 0 and mean_llm_calls_per_case > 0). "
            "Re-run that debate arm so it records LLM calls (app/core/usage.py), "
            "or pass --samples N."
        )
    cost = (report.get("summary") or {}).get("cost") or {}
    mean_calls = cost.get("mean_llm_calls_per_case")
    samples = max(1, round(float(mean_calls)))
    print(
        f"Cost-matched N={samples} from {path} "
        f"(mean_llm_calls_per_case={float(mean_calls):.2f})"
    )
    return samples


def _print_cost_match_verdict(report_path: Path | None, cost: dict[str, Any]) -> None:
    if not report_path:
        return
    reference = (
        (json.loads(Path(report_path).read_text(encoding="utf-8")).get("summary") or {}).get("cost")
        or {}
    )
    ref_tokens = reference.get("mean_total_tokens_per_case")
    if not ref_tokens:
        return
    ratio = cost["mean_total_tokens_per_case"] / float(ref_tokens)
    print(
        f"Cost match vs {report_path}: calls "
        f"{cost['mean_llm_calls_per_case']:.2f} vs {reference.get('mean_llm_calls_per_case', 0):.2f}, "
        f"tokens {cost['mean_total_tokens_per_case']:.0f} vs {float(ref_tokens):.0f} "
        f"(ratio {ratio:.2f}×)"
    )


async def _evaluate_self_consistency(
    cases: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    *,
    agent_factory: Any,
    samples: int,
    hint_provider: BioLinkBERTHintProvider | None,
    inject_hint: bool,
    case_concurrency: int,
    sample_concurrency: int,
    prior_results: dict[str, DebateCaseResult],
    checkpoint_path: Path | None,
) -> list[DebateCaseResult]:
    _ = case_concurrency
    checkpoint_lock = asyncio.Lock()
    classifier_lock = asyncio.Lock()

    async def _run_case(index: int, case: dict[str, Any]) -> DebateCaseResult:
        if case["id"] in prior_results:
            print(f"[{index}/{len(cases)}] SKIP id={case['id']} (checkpoint)")
            return prior_results[case["id"]]

        usage = start_usage_scope()
        started = perf_counter()

        hint: EvidenceHint | None = None
        if hint_provider is not None:
            async with classifier_lock:
                hint_provider.set_case(question=case["question"], documents=_case_documents(case, corpus))
                hint = hint_provider.predict_current()
                hint_provider.clear_case()
        if inject_hint and hint is None:
            raise SystemExit("BioLinkBERT hint is required but classifier returned no prediction.")

        case_hint_provider = _StaticHintProvider(hint) if inject_hint else NullEvidenceHint()
        agent = agent_factory(case_hint_provider, index)
        patient_case = _build_patient_case(case, corpus)
        semaphore = asyncio.Semaphore(max(1, int(sample_concurrency)))

        async def _one_sample() -> ClinicalOpinion:
            async with semaphore:
                return await agent.generate_opinion(patient_case, context=None)

        opinions = list(await asyncio.gather(*[_one_sample() for _ in range(samples)]))
        predicted, share = majority_vote(opinions)
        sample_labels = [opinion_label(opinion) for opinion in opinions]
        clean_labels = [label for label in sample_labels if label]

        entries = [
            AgentRoundOpinion(
                agent_id=f"sample_{position:02d}",
                persona=SAMPLING_PERSONA,
                round=1,
                opinion=opinion,
            )
            for position, opinion in enumerate(opinions, start=1)
        ]
        first_label = sample_labels[0] if sample_labels else None
        entropy = _label_entropy(clean_labels) + 0.0
        latency_ms = (perf_counter() - started) * 1000.0

        result = DebateCaseResult(
            id=case["id"],
            expected_label=case["expected_label"],
            predicted_label=predicted,
            vote_share=share,
            agent_labels={entry.agent_id: label for entry, label in zip(entries, sample_labels)},
            round1_vote_label=first_label,
            biolinkbert_label=hint.label if hint else None,
            biolinkbert_confidence=hint.confidence if hint else None,
            label_pass=predicted == case["expected_label"],
            round1_pass=first_label == case["expected_label"],
            biolinkbert_pass=(hint.label == case["expected_label"]) if hint else None,
            unanimous_final=len(set(clean_labels)) == 1 and bool(clean_labels),
            early_exit=False,
            rounds_run=1,
            aggregation_rule=f"self_consistency_majority_n{samples}",
            latency_ms=latency_ms,
            final_opinions=[entry.model_dump() for entry in entries],
            history=[[entry.model_dump() for entry in entries]],
            uncertainty_score=entropy,
            uncertainty_signals={
                "sample_label_entropy": entropy,
                "sample_agreement": (max(share.values()) if share else 0.0),
                "sample_maybe_fraction": (
                    clean_labels.count("maybe") / len(clean_labels) if clean_labels else 0.0
                ),
                "mean_sample_confidence": (
                    sum(float(o.confidence_level) for o in opinions) / len(opinions)
                    if opinions
                    else 0.0
                ),
            },
            base_label=predicted,
            cost_measured=True,
            llm_calls=usage.llm_calls,
            failed_llm_calls=usage.failed_llm_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )
        if checkpoint_path is not None:
            async with checkpoint_lock:
                _append_checkpoint(checkpoint_path, result)
        status = "PASS" if result.label_pass else "FAIL"
        print(
            f"[{index}/{len(cases)}] {status} id={result.id} "
            f"exp={result.expected_label} pred={result.predicted_label} "
            f"n1={first_label} labels={','.join(label or '?' for label in sample_labels)} "
            f"calls={result.llm_calls} ({latency_ms:.0f} ms)"
        )
        return result

    results: list[DebateCaseResult] = []
    for idx, case in enumerate(cases, start=1):
        results.append(await _run_case(idx, case))
    return results


def _self_consistency_markdown(summary: dict[str, Any], results: list[DebateCaseResult]) -> str:
    body = _markdown_report(summary, results)
    header = "\n".join(
        [
            "# Self-Consistency PubMedQA Baseline",
            "",
            f"> Control arm: {summary['samples']} independent samples at temperature "
            f"{summary['temperature']}, persona `{summary['sampling_persona']}`, majority vote.",
        ]
    )
    return body.replace("# Multi-Agent Debate PubMedQA Benchmark", header, 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--backend", choices=("mock", "ollama"), default="mock")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--match-cost-report", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--hint", choices=("none", "biolinkbert"), default="none")
    parser.add_argument("--record-biolinkbert", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--num-predict", type=int, default=None)
    parser.add_argument("--sample-concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
