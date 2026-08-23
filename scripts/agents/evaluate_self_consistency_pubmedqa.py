#!/usr/bin/env python3
"""Self-consistency baseline for PubMedQA: N independent samples + majority vote.

This is the control arm for the paper's main question — does debate buy anything
over compute-matched repeated sampling with frozen evidence? Without it, the
debate numbers have nothing to be compared against.

The comparison is only fair if the two arms differ in exactly one thing (agents
seeing each other), so this script does not re-implement the pipeline: it imports
the case loader, the patient-case builder, the report dataclass and the summary
writer from ``evaluate_debate_pubmedqa.py``, and samples through the same
``ClinicalAgent`` with the same round-1 prompt (persona ``generalist``, no peer
context).

Cost matching: run the debate arm first, then pass its report to
``--match-cost-report`` to set N from the measured mean LLM calls per case.

One knob is deliberately *not* matched: debate agents run at ``ClinicalAgent``'s
default temperature 0.3, this arm defaults to 0.7, because sample diversity is
the mechanism being tested rather than an incidental setting. Run it again with
``--temperature 0.3`` to separate the two.

Free extras in the same report:
  - ``round1_*`` in the summary is the N=1 arm (the first sample of each case)
  - ``agent_labels`` keeps samples in order, so accuracy at N'=1,3,5… can be
    recomputed offline without re-running anything
  - ``uncertainty_score`` is the normalized entropy of the sample labels — the
    high-disagreement slice H2 is defined on

Examples:
  # 12 samples, cost-matched to a finished debate arm, two GPUs
  python3 scripts/agents/evaluate_self_consistency_pubmedqa.py \\
    --backend ollama --match-cost-report reports/debate/arm_clean.json \\
    --ollama-base-urls http://127.0.0.1:11434,http://127.0.0.1:11435 \\
    --case-concurrency 2 --num-predict 400 --resume --label arm_sc

  # Plumbing check without a GPU
  python3 scripts/agents/evaluate_self_consistency_pubmedqa.py \\
    --backend mock --samples 5 --limit 4 --label sc_smoke
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
    parse_ollama_base_urls,
    sticky_ollama_url,
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
    _StaticHintProvider,
    _summarize,
)

# The debate panel's first speaker; round 1 is exactly this agent with no context.
SAMPLING_PERSONA = "generalist"


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

    # Before the prompt snapshot, so a rejected resume leaves no artifacts behind.
    prior_results = _load_checkpoint(checkpoint_path) if args.resume else {}
    if prior_results:
        print(f"Resuming with {len(prior_results)} completed cases from {checkpoint_path}")
        _reject_mixed_sample_counts(prior_results, samples=samples, checkpoint_path=checkpoint_path)

    from app.agents.prompt_versioning import snapshot_prompts_for_run

    prompt_snapshot = snapshot_prompts_for_run(
        report_dir,
        run_label=label,
        script="scripts/agents/evaluate_self_consistency_pubmedqa.py",
        extra_meta={
            "dataset": str(args.dataset),
            "backend": args.backend,
            "arm": "self_consistency",
            "samples": samples,
            "temperature": args.temperature,
            "hint": args.hint,
            "persona": SAMPLING_PERSONA,
            "cost_matched_from": args.match_cost_report and str(args.match_cost_report),
        },
    )
    print(
        f"Prompt version={prompt_snapshot['prompt_version']} "
        f"sha256={prompt_snapshot['prompt_sha256'][:16]}… "
        f"snapshot={prompt_snapshot['snapshot_path']}"
    )

    from app.core.config import get_settings

    settings = get_settings()
    ollama_urls = parse_ollama_base_urls(args.ollama_base_urls, default=settings.ollama_base_url)
    if args.backend == "ollama":
        print(f"Ollama pool ({len(ollama_urls)}): {', '.join(ollama_urls)}")
    print(
        f"Self-consistency: {samples} sample(s)/case, temperature={args.temperature}, "
        f"persona={SAMPLING_PERSONA}, hint={args.hint}"
    )

    backend_cache: dict[str, Any] = {}

    def _agent_factory(case_hint_provider: Any, case_index: int) -> ClinicalAgent:
        base_url = sticky_ollama_url(ollama_urls, case_index)
        if base_url not in backend_cache:
            backend_cache[base_url] = _build_backend(
                args.backend,
                fast=args.fast,
                num_predict=args.num_predict,
                base_url=base_url,
                quiet=bool(backend_cache),
            )
        return ClinicalAgent(
            agent_id=SAMPLING_PERSONA,
            persona=SAMPLING_PERSONA,
            backend=backend_cache[base_url],
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
            case_concurrency=args.case_concurrency,
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
    summary["prompt_version"] = prompt_snapshot["prompt_version"]
    summary["prompt_sha256"] = prompt_snapshot["prompt_sha256"]
    summary["prompts_py_sha256"] = prompt_snapshot.get("prompts_py_sha256")
    summary["prompt_snapshot"] = prompt_snapshot.get("snapshot_path")
    payload = {
        "summary": summary,
        "prompt_versioning": {
            "prompt_version": prompt_snapshot["prompt_version"],
            "prompt_sha256": prompt_snapshot["prompt_sha256"],
            "prompts_py_sha256": prompt_snapshot.get("prompts_py_sha256"),
            "captured_at": prompt_snapshot.get("captured_at"),
            "snapshot_path": prompt_snapshot.get("snapshot_path"),
            "prompts_py_copy": prompt_snapshot.get("prompts_py_copy"),
            "registry_path": prompt_snapshot.get("registry_path"),
        },
        "cases": [asdict(item) for item in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_self_consistency_markdown(summary, results), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {prompt_snapshot['snapshot_path']}")
    print(
        f"label_accuracy={summary['label_accuracy']:.3f} "
        f"single_sample_accuracy={summary['single_sample_accuracy']:.3f} "
        f"biolinkbert_accuracy={summary.get('biolinkbert_accuracy')} "
        f"unanimous_rate={summary['unanimous_rate']:.3f} "
        f"mean_latency_ms={summary['mean_latency_ms']:.1f}"
    )
    cost = summary.get("cost") or {}
    if cost.get("measured_cases"):
        print(
            f"mean_llm_calls_per_case={cost['mean_llm_calls_per_case']:.2f} "
            f"mean_total_tokens_per_case={cost['mean_total_tokens_per_case']:.0f} "
            f"(prompt={cost['mean_prompt_tokens_per_case']:.0f} "
            f"completion={cost['mean_completion_tokens_per_case']:.0f}) "
            f"measured_on={cost['measured_cases']}/{len(results)} cases"
        )
        _print_cost_match_verdict(args.match_cost_report, cost)


def _reject_mixed_sample_counts(
    prior_results: dict[str, DebateCaseResult],
    *,
    samples: int,
    checkpoint_path: Path,
) -> None:
    """Refuse to finish an arm at a different N than it was started with.

    Half the cases at N=9 and half at N=12 is not an arm, it is two arms averaged
    together — and nothing downstream would show it, because the label file is the
    same. Cheaper to stop here than to discover it in the paper's cost table.
    """
    expected = f"self_consistency_majority_n{samples}"
    mismatched = sorted(
        {r.aggregation_rule for r in prior_results.values() if r.aggregation_rule != expected}
    )
    if not mismatched:
        return
    raise SystemExit(
        f"Checkpoint {checkpoint_path} was written with {', '.join(mismatched)}, "
        f"but this run would use {expected}. Mixing sample counts inside one arm "
        "silently corrupts both accuracy and cost. Use --label for a new arm, or "
        "delete the checkpoint to start over."
    )


def _resolve_sample_count(args: argparse.Namespace) -> int:
    """N from --samples, or from a finished arm's measured cost."""
    if not args.match_cost_report:
        return max(1, int(args.samples))

    path = Path(args.match_cost_report)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"--match-cost-report: no such report {path}") from exc
    cost = (report.get("summary") or {}).get("cost") or {}
    mean_calls = cost.get("mean_llm_calls_per_case")
    if not mean_calls:
        raise SystemExit(
            f"--match-cost-report: {path} has no measured cost. Re-run that arm on a "
            "build that counts LLM calls (see app/core/usage.py), or pass --samples N."
        )
    samples = max(1, round(float(mean_calls)))
    print(
        f"Cost-matched N={samples} from {path} "
        f"(mean_llm_calls_per_case={float(mean_calls):.2f}, "
        f"mean_total_tokens_per_case={cost.get('mean_total_tokens_per_case', 0):.0f})"
    )
    return samples


def _print_cost_match_verdict(report_path: Path | None, cost: dict[str, Any]) -> None:
    """Say plainly how close the two arms ended up on tokens, not just on calls."""
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
        f"(ratio {ratio:.2f}× — report this, matching calls does not match tokens)"
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
    case_concurrency = max(1, int(case_concurrency))
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
                # context=None is what makes this the control arm: every sample is
                # drawn independently, never conditioned on another sample.
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
        entropy = _label_entropy(clean_labels) + 0.0  # normalize -0.0 when all samples agree
        latency_ms = (perf_counter() - started) * 1000.0

        result = DebateCaseResult(
            id=case["id"],
            expected_label=case["expected_label"],
            predicted_label=predicted,
            vote_share=share,
            agent_labels={entry.agent_id: label for entry, label in zip(entries, sample_labels)},
            # The N=1 arm, for free: sample 1 is a single unaided query.
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
            consensus_mode="self_consistency",
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
            f"n1={first_label} bert={result.biolinkbert_label} "
            f"labels={','.join(label or '?' for label in sample_labels)} "
            f"entropy={entropy:.2f} calls={result.llm_calls} tok={result.total_tokens} "
            f"({latency_ms:.0f} ms)"
        )
        return result

    if case_concurrency == 1:
        results: list[DebateCaseResult] = []
        for idx, case in enumerate(cases, start=1):
            results.append(await _run_case(idx, case))
        return results

    sem = asyncio.Semaphore(case_concurrency)

    async def _guarded_run(index: int, case: dict[str, Any]) -> tuple[int, DebateCaseResult]:
        async with sem:
            return index, await _run_case(index, case)

    done = await asyncio.gather(
        *[asyncio.create_task(_guarded_run(index, case)) for index, case in enumerate(cases, start=1)]
    )
    done.sort(key=lambda item: item[0])
    return [result for _, result in done]


def _self_consistency_markdown(summary: dict[str, Any], results: list[DebateCaseResult]) -> str:
    """Debate report layout, retitled, with the sampling-specific reading notes."""
    body = _markdown_report(summary, results)
    header = "\n".join(
        [
            "# Self-Consistency PubMedQA Baseline",
            "",
            f"> Control arm for the debate benchmark: {summary['samples']} independent "
            f"samples at temperature {summary['temperature']}, persona "
            f"`{summary['sampling_persona']}`, majority vote. Same prompt, same frozen "
            "evidence and same report schema as round 1 of the debate — the only "
            "difference is that samples never see each other.",
            ">",
            "> Reading notes for the shared layout:",
            f"> - `Round-1 accuracy` is the **N=1 arm** ({summary['single_sample_accuracy']:.3f}): "
            "the first sample of each case, i.e. one unaided query.",
            "> - `Per-agent accuracy` is per **sample slot**, not per persona. Slot order "
            "is the draw order, so accuracy at smaller N can be recomputed offline.",
            "> - `Unanimous final rate` is the share of cases where all samples agreed.",
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
    parser.add_argument(
        "--backend",
        choices=("mock", "ollama"),
        default="mock",
        help="mock = offline plumbing check; ollama = real sampling",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="N samples per case (ignored when --match-cost-report is given)",
    )
    parser.add_argument(
        "--match-cost-report",
        type=Path,
        default=None,
        help=(
            "Set N from a finished debate report's measured mean LLM calls per case "
            "(e.g. reports/debate/arm_clean.json). This is what makes the comparison "
            "compute-matched instead of guessed."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature; the diversity across samples is the whole mechanism",
    )
    parser.add_argument(
        "--hint",
        choices=("none", "biolinkbert"),
        default="none",
        help="Inject the BioLinkBERT label into the prompt, mirroring the debate arm's --hint",
    )
    parser.add_argument(
        "--record-biolinkbert",
        action="store_true",
        help="Run the classifier for reference metrics only (no prompt, no vote)",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N cases")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--fast", action="store_true", help="Compact prompts + lower num_predict")
    parser.add_argument("--compact", action="store_true", help="Compact PubMedQA prompts only")
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Override Ollama num_predict; keep it identical to the debate arm",
    )
    parser.add_argument(
        "--case-concurrency",
        type=int,
        default=1,
        help="How many cases to process concurrently (one Ollama endpoint each)",
    )
    parser.add_argument(
        "--sample-concurrency",
        type=int,
        default=1,
        help=(
            "Max concurrent samples within a case. Mirrors --agent-concurrency in the "
            "debate arm; raising it changes latency, not cost."
        ),
    )
    parser.add_argument(
        "--ollama-base-urls",
        type=str,
        default=None,
        help="Comma-separated Ollama endpoints; cases are sticky-assigned round-robin",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint JSONL next to the report label",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
