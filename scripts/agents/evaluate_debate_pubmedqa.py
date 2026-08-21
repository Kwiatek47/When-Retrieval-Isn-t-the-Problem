#!/usr/bin/env python3
"""Benchmark multi-agent debate on PubMedQA (yes/no/maybe), without a supervisor.

Modes:
  --backend mock|ollama     4-agent debate + majority vote
  --backend biolinkbert     BioLinkBERT-large classifier only (seed47 from .env)
  --hint biolinkbert        inject classifier label into each agent prompt
  --director-hint biolinkbert|none
                            llm_director: pass classifier to Director (default) or withhold
                            (none = debate/abstract only; BERT still recorded in metrics)
  --aggregate-with-biolinkbert
                            include BioLinkBERT vote in the final majority
  --fast                    compact prompts + lower num_predict (does NOT enable early-exit)
  --aggregate-mode bert_gate|bert_weighted|majority|llm_director
                            how to combine panel + BioLinkBERT (default: bert_gate)
  --panel-maybe-veto unanimous|majority|off
                            panel maybe overrides BioLinkBERT under bert_gate (default: unanimous)
  --director-maybe-gate off|legacy
                            llm_director: prompt-only maybe rules (default off) vs legacy post-hoc gate

Examples:
  # Quality run on balanced90 (no early-exit; BERT-gate aggregation)
  .venv/bin/python scripts/agents/evaluate_debate_pubmedqa.py \\
    --backend ollama --hint biolinkbert --aggregate-with-biolinkbert \\
    --aggregate-mode bert_gate --rounds 2 --limit 90 --resume \\
    --num-predict 400 --agent-concurrency 1 \\
    --label debate_balanced90_ollama_r2_bertgate

  # FT Director without classifier label (debate-first ablation)
  python scripts/agents/evaluate_debate_pubmedqa.py \\
    --backend ollama --aggregate-mode llm_director --hint none \\
    --director-hint none --rounds 2 --no-supervisor-moderation \\
    --supervisor-model qwen2.5-supervisor-14b-stage1 \\
    --label stage1_director_no_bert_label
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field, fields
import json
import os
from pathlib import Path
import random
from statistics import mean
import sys
from time import perf_counter
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents import (
    DebateOrchestrator,
    MockInferenceBackend,
    build_default_agents,
    check_early_exit_asymmetric_veto,
)
from app.agents.aggregation import (
    aggregate_pubmedqa_decision,
    aggregate_with_llm_director,
    build_consensus_decision,
    final_labels_by_agent,
    majority_vote,
)
from app.agents.backends import (
    BioLinkBERTHintProvider,
    EvidenceHint,
    NullEvidenceHint,
    OllamaInferenceBackend,
    build_biolinkbert_hint_from_settings,
    hint_as_clinical_opinion,
    parse_ollama_base_urls,
    sticky_ollama_url,
)
from app.agents.models import AgentRoundOpinion
from app.agents.evidence_audit import audit_evidence
from app.agents.uncertainty import calibrate_threshold, compute_uncertainty, risk_coverage_curve
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
    history: list[list[dict[str, Any]]] = field(default_factory=list)
    uncertainty_score: float | None = None
    uncertainty_signals: dict[str, float] = field(default_factory=dict)
    base_label: str | None = None
    routed_to_maybe: bool = False
    audit_score: float | None = None
    audit_detail: dict[str, Any] = field(default_factory=dict)
    llm_director_rationale: str | None = None
    llm_director_consensus_type: str | None = None
    consensus_mode: str | None = None
    consensus_ranked_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    consensus_required_next_steps: list[str] = field(default_factory=list)
    safety_halted: bool = False
    safety_red_flag_reason: str | None = None
    exhausted_without_consensus: bool = False  # telemetry/log only; never overrides predicted_label


class _StaticHintProvider:
    """Per-case immutable hint provider for concurrent benchmark workers."""

    def __init__(self, hint: EvidenceHint | None) -> None:
        self._hint = hint

    def get_hint(self, patient_case: str) -> EvidenceHint | None:  # noqa: ARG002 - protocol compatibility
        return self._hint


def main() -> None:
    args = _parse_args()
    if args.no_supervisor_moderation:
        args.debate_mode = "peer"
    if args.max_rounds is None:
        args.max_rounds = args.rounds
    if args.adaptive_rounds and args.min_rounds > args.max_rounds:
        raise SystemExit("--min-rounds cannot exceed --max-rounds")
    if args.backend == "biolinkbert" and args.aggregate_mode == "llm_director":
        raise SystemExit("--aggregate-mode llm_director requires a debate backend (mock/ollama), not --backend biolinkbert")
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
        or args.aggregate_mode == "llm_director"
    )
    if need_classifier:
        hint_provider = build_biolinkbert_hint_from_settings()
        if not hint_provider.available:
            raise SystemExit(f"BioLinkBERT classifier unavailable: {hint_provider.load_error}")
        print(f"Loaded BioLinkBERT classifier: {hint_provider.model_path}")
    if args.aggregate_mode == "llm_director":
        print(f"Director hint={args.director_hint}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or _default_label(args)
    json_path = report_dir / f"{label}.json"
    md_path = report_dir / f"{label}.md"
    checkpoint_path = report_dir / f"{label}.checkpoint.jsonl"

    from app.agents.prompt_versioning import snapshot_prompts_for_run

    prompt_snapshot = snapshot_prompts_for_run(
        report_dir,
        run_label=label,
        script="scripts/agents/evaluate_debate_pubmedqa.py",
        extra_meta={
            "dataset": str(args.dataset),
            "backend": args.backend,
            "aggregate_mode": args.aggregate_mode,
            "supervisor_model": args.supervisor_model,
            "hint": args.hint,
            "rounds": args.rounds,
            "debate_mode": args.debate_mode,
            "blind_critic": args.blind_critic,
            "director_maybe_gate": getattr(args, "director_maybe_gate", "off"),
        },
    )
    print(
        f"Prompt version={prompt_snapshot['prompt_version']} "
        f"sha256={prompt_snapshot['prompt_sha256'][:16]}… "
        f"snapshot={prompt_snapshot['snapshot_path']}"
    )

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
        exhausted_without_consensus_rate = 0.0
    else:
        from app.core.config import get_settings

        settings = get_settings()
        ollama_urls = parse_ollama_base_urls(
            args.ollama_base_urls,
            default=settings.ollama_base_url,
        )
        if args.backend == "ollama":
            print(f"Ollama pool ({len(ollama_urls)}): {', '.join(ollama_urls)}")

        backend_cache: dict[tuple[str, str, str, str], MockInferenceBackend | OllamaInferenceBackend] = {}

        def _cached_backend(
            *,
            base_url: str,
            model: str | None = None,
            num_predict: int | None = None,
            quiet: bool = True,
        ) -> MockInferenceBackend | OllamaInferenceBackend:
            model_name = model or settings.default_model
            predict_key = (
                str(num_predict)
                if num_predict is not None
                else str(args.num_predict)
            )
            key = (base_url, model_name, predict_key, str(bool(args.fast)))
            if key not in backend_cache:
                backend_cache[key] = _build_backend(
                    args.backend,
                    fast=args.fast,
                    num_predict=num_predict if num_predict is not None else args.num_predict,
                    model=model,
                    base_url=base_url,
                    quiet=quiet,
                )
            return backend_cache[key]

        if args.supervisor_model and args.backend != "ollama":
            raise SystemExit("--supervisor-model requires --backend ollama")
        if args.supervisor_base_url and args.backend != "ollama":
            raise SystemExit("--supervisor-base-url requires --backend ollama")
        if args.supervisor_model:
            print(f"Supervisor model={args.supervisor_model}")
        if args.supervisor_base_url:
            print(f"Supervisor url={args.supervisor_base_url}")

        def _build_orchestrator(
            case_hint_provider: Any,
            case_index: int = 1,
        ) -> DebateOrchestrator:
            base_url = sticky_ollama_url(ollama_urls, case_index)
            backend = _cached_backend(base_url=base_url)
            supervisor_backend = None
            if args.supervisor_model:
                supervisor_backend = _cached_backend(
                    base_url=(args.supervisor_base_url or base_url).rstrip("/"),
                    model=args.supervisor_model,
                )
            agents = build_default_agents(
                backend,
                hint_provider=case_hint_provider,
                task_mode="pubmedqa",
                compact=args.fast or args.compact,
            )
            def _should_early_exit(
                round_number: int,
                round_opinions: list[AgentRoundOpinion],
                patient_case: str,
            ) -> bool:
                if not args.early_exit or round_number < 1:
                    return False
                # Option A: asymmetric maybe-veto + binary unanimity + keyword filter
                # (no BERT gate).
                return check_early_exit_asymmetric_veto(
                    round_opinions,
                    patient_case=patient_case,
                )

            return DebateOrchestrator(
                agents,
                rounds=args.rounds,
                min_rounds=args.min_rounds,
                max_rounds=args.max_rounds,
                adaptive_rounds=args.adaptive_rounds,
                conflict_entropy_threshold=args.conflict_entropy_threshold,
                debate_mode=args.debate_mode,
                early_exit=_should_early_exit if args.early_exit else None,
                agent_concurrency=args.agent_concurrency,
                blind_critic=args.blind_critic,
                safety_red_flag=args.safety_red_flag,
                peer_context=args.peer_context,
                supervisor_fail=args.supervisor_fail,
                supervisor_backend=supervisor_backend,
            )

        audit_backend = None
        if args.audit_model:
            audit_backend = _cached_backend(
                base_url=ollama_urls[0],
                model=args.audit_model,
                num_predict=400,
                quiet=False,
            )
            print(f"Evidence-audit backend enabled: {args.audit_model}")
        print(f"Debate mode={args.debate_mode}")
        print(f"Blind critic={args.blind_critic}")
        print(f"Safety red flag={args.safety_red_flag}")
        print(f"Peer context={args.peer_context}")
        print(f"Supervisor fail={args.supervisor_fail}")
        if args.adaptive_rounds:
            print(
                f"Adaptive rounds enabled: min={args.min_rounds} "
                f"max={args.max_rounds} entropy_threshold={args.conflict_entropy_threshold}"
            )
        results = asyncio.run(
            _evaluate_debate(
                cases,
                corpus,
                orchestrator_factory=_build_orchestrator,
                hint_provider=hint_provider if isinstance(hint_provider, BioLinkBERTHintProvider) else None,
                inject_hint=args.hint == "biolinkbert",
                director_hint=args.director_hint,
                aggregate_with_biolinkbert=args.aggregate_with_biolinkbert,
                aggregate_mode=args.aggregate_mode,
                bert_gate_confidence=args.bert_gate_confidence,
                bert_vote_weight=args.bert_vote_weight,
                panel_maybe_veto=args.panel_maybe_veto,
                director_maybe_gate=args.director_maybe_gate,
                safety_red_flag=args.safety_red_flag,
                case_concurrency=args.case_concurrency,
                prior_results=prior_results,
                checkpoint_path=checkpoint_path if args.resume else None,
                audit_backend=audit_backend,
            )
        )
        rounds = args.rounds
        if args.aggregate_mode == "llm_director":
            aggregation = "llm_director"
        else:
            aggregation = (
                f"{args.aggregate_mode}+biolinkbert"
                if args.aggregate_with_biolinkbert
                else "confidence_weighted_majority_vote"
            )
        architecture = DebateOrchestrator.ARCHITECTURE
        early_exit_rate = (
            sum(1 for item in results if item.early_exit) / len(results) if results else 0.0
        )
        exhausted_without_consensus_rate = (
            sum(1 for item in results if item.exhausted_without_consensus) / len(results)
            if results
            else 0.0
        )

    if args.uncertainty_route and args.backend != "biolinkbert":
        _apply_uncertainty_routing(
            results,
            split=args.uncertainty_split,
            objective=args.uncertainty_objective,
            seed=args.uncertainty_seed,
            fixed_threshold=args.uncertainty_threshold,
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
        exhausted_without_consensus_rate=exhausted_without_consensus_rate,
    )
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
    md_path.write_text(_markdown_report(summary, results), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {prompt_snapshot['snapshot_path']}")
    print(
        f"label_accuracy={summary['label_accuracy']:.3f} "
        f"biolinkbert_accuracy={summary.get('biolinkbert_accuracy')} "
        f"early_exit_rate={summary.get('early_exit_rate')} "
        f"exhausted_without_consensus_rate={summary.get('exhausted_without_consensus_rate')} "
        f"prompt_version={summary.get('prompt_version')} "
        f"mean_latency_ms={summary['mean_latency_ms']:.1f}"
    )


def _apply_uncertainty_routing(
    results: list[DebateCaseResult],
    *,
    split: float,
    objective: str,
    seed: int,
    fixed_threshold: float | None,
) -> None:
    """Calibrate an uncertainty threshold on a held-out split and route to `maybe`.

    Cases with ``uncertainty_score >= threshold`` are re-labeled ``maybe``; the
    rest keep their base (panel+classifier) label. To avoid data leakage the
    threshold is fit only on the calibration half and applied to the reporting
    half; the calibration half is left on its base label so reported metrics are
    honest. If ``fixed_threshold`` is given, calibration is skipped.
    """
    scored = [r for r in results if r.uncertainty_score is not None]
    if not scored:
        print("Uncertainty routing skipped: no uncertainty scores present.")
        return

    for r in results:
        if r.base_label is None:
            r.base_label = r.predicted_label

    if fixed_threshold is not None:
        threshold = fixed_threshold
        calib_ids: set[str] = set()
        metrics: dict[str, float] = {"threshold": threshold, "source": "fixed"}
        report = scored
    else:
        rng = random.Random(seed)
        by_label: dict[str, list[DebateCaseResult]] = {}
        for r in scored:
            by_label.setdefault(r.expected_label, []).append(r)
        calib: list[DebateCaseResult] = []
        report = []
        for label, group in by_label.items():
            group = list(group)
            rng.shuffle(group)
            cut = int(round(len(group) * split))
            calib.extend(group[:cut])
            report.extend(group[cut:])
        calib_ids = {r.id for r in calib}
        threshold, metrics = calibrate_threshold(
            [r.uncertainty_score for r in calib],
            [r.expected_label for r in calib],
            objective=objective,
            base_labels=[(r.base_label or r.predicted_label) for r in calib],
        )
        metrics["source"] = "calibrated"
        metrics["calib_n"] = len(calib)
        metrics["report_n"] = len(report)
        print(
            f"Uncertainty routing: calibrated threshold={threshold:.3f} on {len(calib)} cases "
            f"(objective={objective}); reporting on {len(report)} held-out cases."
        )

    report_ids = {r.id for r in report}
    for r in results:
        r.routed_to_maybe = False
        # Only route (and thus report metrics for) held-out cases; leave the
        # calibration cases on their base label to keep the split leakage-free.
        if fixed_threshold is None and r.id not in report_ids:
            r.predicted_label = r.base_label
            r.label_pass = r.predicted_label == r.expected_label
            continue
        if r.uncertainty_score is not None and r.uncertainty_score >= threshold:
            r.predicted_label = "maybe"
            r.routed_to_maybe = True
            r.aggregation_rule = f"{r.aggregation_rule}->uncertainty_route"
        else:
            r.predicted_label = r.base_label
        r.label_pass = r.predicted_label == r.expected_label

    _UNCERTAINTY_ROUTING_META.update(
        {
            "threshold": threshold,
            "objective": objective,
            "split": split,
            "seed": seed,
            "calibration_metrics": metrics,
            "report_ids": sorted(report_ids),
            "calib_ids": sorted(calib_ids),
        }
    )

    # Risk-coverage / AURC on the reporting split (selective-prediction view:
    # routing to maybe == abstaining from a definitive yes/no answer).
    rc_pool = [r for r in results if (not report_ids) or r.id in report_ids]
    if rc_pool:
        rc = risk_coverage_curve(
            [r.uncertainty_score or 0.0 for r in rc_pool],
            [(r.base_label or r.predicted_label) for r in rc_pool],
            [r.expected_label for r in rc_pool],
        )
        _UNCERTAINTY_ROUTING_META["risk_coverage"] = rc


_UNCERTAINTY_ROUTING_META: dict[str, Any] = {}


def _default_label(args: argparse.Namespace) -> str:
    parts = [f"debate_pubmedqa_{args.backend}"]
    if args.backend != "biolinkbert":
        parts.append(f"r{args.rounds}")
    if getattr(args, "case_concurrency", 1) > 1:
        parts.append(f"cc{args.case_concurrency}")
    if getattr(args, "debate_mode", "moderated") != "moderated":
        parts.append(str(args.debate_mode))
    if getattr(args, "adaptive_rounds", False):
        parts.append(f"adapt{args.min_rounds}-{args.max_rounds}")
    urls = getattr(args, "ollama_base_urls", None)
    if urls and "," in str(urls):
        parts.append(f"ollama{str(urls).count(',') + 1}")
    if args.hint == "biolinkbert":
        parts.append("hint")
    if args.aggregate_with_biolinkbert:
        parts.append(args.aggregate_mode)
    if args.aggregate_mode == "llm_director":
        parts.append("llm_director")
        if getattr(args, "director_hint", "biolinkbert") == "none":
            parts.append("nodirhint")
    if args.supervisor_model:
        parts.append(f"sup_{args.supervisor_model.replace(':', '').replace('.', '')}")
    if getattr(args, "director_maybe_gate", "off") == "legacy":
        parts.append("dir_gate_legacy")
    blind = getattr(args, "blind_critic", "all-rounds")
    if blind and blind != "all-rounds":
        parts.append(f"blind_{blind.replace('-', '')}")
    safety_rf = getattr(args, "safety_red_flag", "halt")
    if safety_rf and safety_rf != "halt":
        parts.append(f"safety_{safety_rf.replace('-', '_')}")
    peer_ctx = getattr(args, "peer_context", "nl")
    if peer_ctx and peer_ctx != "nl":
        parts.append(f"peer_{peer_ctx.replace('-', '_')}")
    sup_fail = getattr(args, "supervisor_fail", "peer-round")
    if sup_fail and sup_fail != "peer-round":
        parts.append(f"supfail_{sup_fail.replace('-', '_')}")
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
        known = {f.name for f in fields(DebateCaseResult)}
        filtered = {k: v for k, v in data.items() if k in known}
        loaded[data["id"]] = DebateCaseResult(**filtered)
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
    cases: list[dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    *,
    orchestrator_factory: Callable[[Any, int], DebateOrchestrator],
    hint_provider: BioLinkBERTHintProvider | None,
    inject_hint: bool,
    director_hint: str,
    aggregate_with_biolinkbert: bool,
    aggregate_mode: str,
    bert_gate_confidence: float,
    bert_vote_weight: float,
    panel_maybe_veto: str,
    director_maybe_gate: str,
    safety_red_flag: str,
    case_concurrency: int,
    prior_results: dict[str, DebateCaseResult],
    checkpoint_path: Path | None,
    audit_backend: Any = None,
) -> list[DebateCaseResult]:
    case_concurrency = max(1, int(case_concurrency))
    checkpoint_lock = asyncio.Lock()
    classifier_lock = asyncio.Lock()
    director_hint_mode = (director_hint or "biolinkbert").strip().lower()

    async def _run_case(index: int, case: dict[str, Any]) -> DebateCaseResult:
        if case["id"] in prior_results:
            print(f"[{index}/{len(cases)}] SKIP id={case['id']} (checkpoint)")
            return prior_results[case["id"]]

        started = perf_counter()
        docs = _case_documents(case, corpus)

        hint: EvidenceHint | None = None
        if hint_provider is not None:
            async with classifier_lock:
                hint_provider.set_case(question=case["question"], documents=docs)
                hint = hint_provider.predict_current()
                hint_provider.clear_case()

        if inject_hint and hint is None and hint_provider is not None:
            raise SystemExit("BioLinkBERT hint is required but classifier returned no prediction.")

        if (
            aggregate_mode == "llm_director"
            and director_hint_mode == "biolinkbert"
            and hint is None
        ):
            raise SystemExit(
                "--aggregate-mode llm_director with --director-hint biolinkbert "
                "requires a BioLinkBERT prediction"
            )

        case_hint_provider = (
            _StaticHintProvider(hint) if inject_hint else NullEvidenceHint()
        )
        orchestrator = orchestrator_factory(case_hint_provider, index)
        patient_case = _build_patient_case(case, corpus)
        debate = await orchestrator.run(patient_case)

        llm_director_rationale: str | None = None
        llm_director_consensus_type: str | None = None

        if debate.safety_halted and safety_red_flag == "escalate-label":
            predicted = "maybe"
            share = {lab: (1.0 if lab == "maybe" else 0.0) for lab in ("yes", "no", "maybe")}
            rule = "safety_escalation"
            llm_director_consensus_type = "escalation"
            llm_director_rationale = debate.safety_red_flag_reason
        elif aggregate_mode == "llm_director":
            if director_hint_mode == "biolinkbert":
                assert hint is not None
                biolinkbert_hint_text = json.dumps(asdict(hint), ensure_ascii=False)
            else:
                biolinkbert_hint_text = (
                    "No classifier label provided. Decide only from the abstract "
                    "(patient_case) and the debate transcript. Do not invent a "
                    "BioLinkBERT prediction."
                )
            supervisor = getattr(orchestrator, "supervisor", None)
            if supervisor is None:
                raise SystemExit("SupervisorAgent missing from DebateOrchestrator instance")

            predicted = await aggregate_with_llm_director(
                patient_case=patient_case,
                debate_history=debate.rounds,
                biolinkbert_hint=biolinkbert_hint_text,
                supervisor=supervisor,
                shared_report=debate.shared_report,
                director_maybe_gate=director_maybe_gate,  # type: ignore[arg-type]
            )
            director_output = getattr(supervisor, "last_director_output", None)
            if director_output is not None:
                llm_director_rationale = director_output.rationale
                llm_director_consensus_type = director_output.consensus_type
            share = {lab: (1.0 if lab == predicted else 0.0) for lab in ("yes", "no", "maybe")}
            rule = "llm_director"
        else:
            agent_opinions = [entry.opinion for entry in debate.final_opinions]
            bert_opinion = (
                hint_as_clinical_opinion(hint)
                if (aggregate_with_biolinkbert and hint)
                else None
            )
            predicted, share, rule = aggregate_pubmedqa_decision(
                agent_opinions,
                bert_opinion=bert_opinion,
                mode=aggregate_mode if aggregate_with_biolinkbert else "majority",
                bert_gate_confidence=bert_gate_confidence,
                bert_vote_weight=bert_vote_weight,
                panel_maybe_veto=panel_maybe_veto,  # type: ignore[arg-type]
            )

        audit_result = None
        if audit_backend is not None:
            evidence_text = "\n\n".join(f"{d.title}\n{d.content}".strip() for d in docs)
            audit_result = await audit_evidence(
                audit_backend, question=case["question"], evidence=evidence_text
            )

        signals = compute_uncertainty(
            debate.rounds,
            debate.final_opinions,
            bert_label=hint.label if hint else None,
            bert_confidence=hint.confidence if hint else None,
            audit_score=audit_result.audit_score if audit_result is not None else None,
        )
        round1_opinions = [entry.opinion for entry in debate.rounds[0]]
        round1_label, _ = majority_vote(round1_opinions)
        agent_labels = final_labels_by_agent(debate.final_opinions)
        if hint is not None:
            agent_labels["biolinkbert"] = hint.label
        label_values = [label for label in agent_labels.values() if label is not None]
        latency_ms = (perf_counter() - started) * 1000.0
        early_exit = len(debate.rounds) < orchestrator.rounds

        agent_opinions_for_consensus = [entry.opinion for entry in debate.final_opinions]
        safety_blocked = bool(debate.safety_halted) or any(
            entry.opinion.safety_opinion is not None
            and not entry.opinion.safety_opinion.safety_passed
            and entry.opinion.safety_opinion.immediate_intervention_required
            for entry in debate.final_opinions
        )
        if rule == "safety_escalation" or (
            aggregate_mode == "llm_director" and llm_director_consensus_type
        ):
            consensus_mode = llm_director_consensus_type or "escalation"
            consensus_ranked_hypotheses: list[dict[str, Any]] = []
            consensus_required_next_steps: list[str] = (
                [debate.safety_red_flag_reason]
                if debate.safety_red_flag_reason
                else []
            )
        else:
            consensus = build_consensus_decision(
                agent_opinions_for_consensus,
                predicted_label=predicted,
                vote_share=share,
                safety_blocked=safety_blocked,
            )
            consensus_mode = consensus.mode
            consensus_ranked_hypotheses = [
                item.model_dump() for item in consensus.ranked_hypotheses
            ]
            consensus_required_next_steps = consensus.required_next_steps

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
            history=[
                [entry.model_dump() for entry in round_entries]
                for round_entries in debate.rounds
            ],
            uncertainty_score=signals.score,
            uncertainty_signals=signals.as_dict(),
            base_label=predicted,
            audit_score=audit_result.audit_score if audit_result is not None else None,
            audit_detail=audit_result.as_dict() if audit_result is not None else {},
            llm_director_rationale=llm_director_rationale,
            llm_director_consensus_type=llm_director_consensus_type,
            consensus_mode=consensus_mode,
            consensus_ranked_hypotheses=consensus_ranked_hypotheses,
            consensus_required_next_steps=consensus_required_next_steps,
            safety_halted=bool(debate.safety_halted),
            safety_red_flag_reason=debate.safety_red_flag_reason,
            exhausted_without_consensus=bool(debate.exhausted_without_consensus),
        )
        if checkpoint_path is not None:
            async with checkpoint_lock:
                _append_checkpoint(checkpoint_path, result)
        status = "PASS" if result.label_pass else "FAIL"
        print(
            f"[{index}/{len(cases)}] {status} id={result.id} "
            f"exp={result.expected_label} pred={result.predicted_label} "
            f"bert={result.biolinkbert_label} rule={rule} "
            f"r1={result.round1_vote_label} rounds={result.rounds_run} "
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

    tasks = [
        asyncio.create_task(_guarded_run(index, case))
        for index, case in enumerate(cases, start=1)
    ]
    done = await asyncio.gather(*tasks)
    done.sort(key=lambda item: item[0])
    return [result for _, result in done]


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
        "--director-hint",
        choices=("biolinkbert", "none"),
        default="biolinkbert",
        help=(
            "llm_director only: pass BioLinkBERT label/confidence to the Director "
            "(biolinkbert, default) or withhold it (none). Classifier is still run for report metrics."
        ),
    )
    parser.add_argument(
        "--aggregate-with-biolinkbert",
        action="store_true",
        help="Include BioLinkBERT in the final aggregation",
    )
    parser.add_argument(
        "--aggregate-mode",
        choices=("majority", "bert_weighted", "bert_gate", "llm_director"),
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
        "--panel-maybe-veto",
        choices=("unanimous", "majority", "off"),
        default="unanimous",
        help=(
            "Under bert_gate: panel maybe overrides BioLinkBERT when unanimous "
            "(default), majority (>50%%), or off (legacy keep-BERT behavior)"
        ),
    )
    parser.add_argument(
        "--director-maybe-gate",
        choices=("off", "legacy"),
        default="off",
        help=(
            "Under llm_director: off (default) uses prompt-only maybe rules so "
            "label and rationale stay consistent; legacy re-enables post-hoc "
            "apply_maybe_director_gate override (ablation)"
        ),
    )
    parser.add_argument(
        "--bert-vote-weight",
        type=float,
        default=3.0,
        help="BioLinkBERT weight in bert_weighted / uncertain bert_gate path",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        choices=(2, 3, 4, 5),
        help="Fixed round count when --adaptive-rounds is off; also default max for adaptive mode",
    )
    parser.add_argument(
        "--adaptive-rounds",
        action="store_true",
        help=(
            "Escalate beyond --min-rounds while panel conflict remains "
            "(entropy / contradictions / non-unanimous), up to --max-rounds"
        ),
    )
    parser.add_argument(
        "--min-rounds",
        type=int,
        default=2,
        choices=(2, 3, 4, 5),
        help="Minimum rounds when --adaptive-rounds is enabled (default: 2)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        choices=(2, 3, 4, 5),
        help="Maximum rounds when --adaptive-rounds is enabled (default: --rounds)",
    )
    parser.add_argument(
        "--conflict-entropy-threshold",
        type=float,
        default=0.35,
        help="Continue adaptive debate when normalized label entropy exceeds this value",
    )
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
        help=(
            "Skip later rounds when panel is unanimously yes/no, "
            "uncertainty_advocate does not veto (maybe / confidence < 0.65), "
            "and the abstract has no inconclusiveness cue phrases. "
            "Does not require BioLinkBERT agreement."
        ),
    )
    parser.add_argument(
        "--early-exit-bert-confidence",
        type=float,
        default=0.85,
        help="Deprecated/unused: early-exit no longer gates on BioLinkBERT (kept for CLI compat)",
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
        "--case-concurrency",
        type=int,
        default=1,
        help=(
            "How many benchmark cases to process concurrently. "
            "Each case runs its own debate orchestrator."
        ),
    )
    parser.add_argument(
        "--debate-mode",
        choices=("moderated", "peer", "hybrid"),
        default="moderated",
        help=(
            "Round 2+ debate style: moderated=supervisor-only (default), "
            "peer=round-robin peer critique without supervisor, "
            "hybrid=supervisor instructions plus round-robin peer context"
        ),
    )
    parser.add_argument(
        "--blind-critic",
        choices=("all-rounds", "r1-only", "off"),
        default="all-rounds",
        help=(
            "BioLinkBERT hint policy for uncertainty_advocate: "
            "all-rounds (default) never shows hint; r1-only hides in round 1 only; "
            "off shows hint from round 1"
        ),
    )
    parser.add_argument(
        "--safety-red-flag",
        choices=("halt", "escalate-label", "defer"),
        default="halt",
        help=(
            "On safety_officer critical red flag: halt debate immediately (default), "
            "escalate-label also forces predicted=maybe, or defer injects RED FLAG "
            "into the next round only (legacy)"
        ),
    )
    parser.add_argument(
        "--peer-context",
        choices=("nl", "compact-json", "full-json"),
        default="nl",
        help=(
            "How peer/supervisor opinions are injected into prompts: "
            "nl (default) natural-language summaries, compact-json short JSON, "
            "full-json full opinion dumps"
        ),
    )
    parser.add_argument(
        "--supervisor-fail",
        choices=("peer-round", "peer-rest", "empty-defer"),
        default="peer-round",
        help=(
            "When supervisor moderation JSON fails after retry: peer-round "
            "(default) runs that round as peer debate; peer-rest keeps peer for "
            "the rest of the case; empty-defer injects legacy empty moderation"
        ),
    )
    parser.add_argument(
        "--no-supervisor-moderation",
        action="store_true",
        help="Alias for --debate-mode peer (pure peer debate, no supervisor moderation)",
    )
    parser.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="Override Ollama num_predict (default: 220 with --fast, else settings)",
    )
    parser.add_argument(
        "--supervisor-model",
        type=str,
        default=None,
        help=(
            "Ollama model for the supervisor only (moderation + llm_director). "
            "Debate agents keep OLLAMA_MODEL / default backend model."
        ),
    )
    parser.add_argument(
        "--supervisor-base-url",
        type=str,
        default=None,
        help=(
            "Optional dedicated Ollama endpoint for the supervisor only, e.g. "
            "'http://127.0.0.1:11437'. Use this to pin the 14b supervisor/director "
            "to a separate GPU from the 7b agent pool."
        ),
    )
    parser.add_argument(
        "--ollama-base-urls",
        type=str,
        default=None,
        help=(
            "Comma-separated Ollama endpoints for multi-GPU load balancing, e.g. "
            "'http://127.0.0.1:11434,http://127.0.0.1:11435,http://127.0.0.1:11436'. "
            "Cases are sticky-assigned round-robin across URLs. "
            "Defaults to OLLAMA_BASE_URL."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint JSONL next to the report label",
    )
    parser.add_argument(
        "--uncertainty-route",
        action="store_true",
        help="Route high-uncertainty cases to `maybe` using the debate uncertainty score",
    )
    parser.add_argument(
        "--uncertainty-split",
        type=float,
        default=0.5,
        help="Fraction of cases used to calibrate the uncertainty threshold (rest is reported)",
    )
    parser.add_argument(
        "--uncertainty-objective",
        choices=("accuracy", "macro_f1", "balanced_acc", "maybe_f1"),
        default="accuracy",
        help="Objective maximized when calibrating the uncertainty threshold (3-class)",
    )
    parser.add_argument(
        "--uncertainty-seed",
        type=int,
        default=47,
        help="Seed for the deterministic stratified calibration/report split",
    )
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=None,
        help="Skip calibration and route with this fixed uncertainty threshold",
    )
    parser.add_argument(
        "--audit-model",
        type=str,
        default=None,
        help="If set, run the NLI evidence condition-audit per case with this Ollama model "
        "and feed audit_score into the uncertainty signals (e.g. qwen2.5:14b, deepseek-r1:14b)",
    )
    return parser.parse_args()


def _build_backend(
    name: str,
    *,
    fast: bool = False,
    num_predict: int | None = None,
    model: str | None = None,
    base_url: str | None = None,
    quiet: bool = False,
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
    resolved_url = (base_url or settings.ollama_base_url).rstrip("/")
    provider = OllamaProvider(
        base_url=resolved_url,
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        num_predict=predict,
        num_ctx=min(max(settings.ollama_num_ctx, 4096), 8192),
    )
    model_name = model or settings.default_model
    if not quiet:
        print(f"Ollama model={model_name} num_predict={predict} url={resolved_url}")
    return OllamaInferenceBackend(provider, model=model_name, temperature=0.1)


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
    exhausted_without_consensus_rate: float = 0.0,
) -> dict[str, Any]:
    # When uncertainty routing is active, report metrics on the held-out split
    # only (the calibration split is excluded to keep the numbers leakage-free).
    routing_meta = dict(_UNCERTAINTY_ROUTING_META)
    report_ids = set(routing_meta.get("report_ids") or [])
    if report_ids:
        results = [r for r in results if r.id in report_ids]

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
        "exhausted_without_consensus_rate": exhausted_without_consensus_rate,
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
        "supervisor": architecture not in {"peer_round_robin", "biolinkbert_only"},
        "uncertainty_routing": (
            {
                **routing_meta,
                "reported_on": "held_out_split" if report_ids else "all",
                "routed_to_maybe_count": sum(1 for r in results if r.routed_to_maybe),
                "base_label_accuracy": (
                    sum(1 for r in results if (r.base_label or r.predicted_label) == r.expected_label) / n
                ),
            }
            if routing_meta
            else None
        ),
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
        f"- Exhausted-without-consensus rate (telemetry only): {summary.get('exhausted_without_consensus_rate', 0.0):.3f}",
        f"- Unanimous final rate: {summary['unanimous_rate']:.3f}",
        f"- Mean latency: {summary['mean_latency_ms']:.1f} ms",
        f"- Architecture: `{summary['architecture']}` (supervisor: {summary['supervisor']})",
        f"- Aggregation: `{summary['aggregation']}`",
        f"- Prompt version: `{summary.get('prompt_version')}`",
        f"- Prompt sha256: `{summary.get('prompt_sha256')}`",
        f"- Prompt snapshot: `{summary.get('prompt_snapshot')}`",
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
        if result.exhausted_without_consensus:
            extra += " exhausted_conflict"
        lines.append(
            f"- `{result.id}` [{mark}] expected={result.expected_label} "
            f"pred={result.predicted_label} bert={result.biolinkbert_label} "
            f"r1={result.round1_vote_label} rounds={result.rounds_run}{extra}"
        )
        if result.llm_director_consensus_type or result.llm_director_rationale:
            lines.append(
                f"  llm_director: consensus_type={result.llm_director_consensus_type} rationale={result.llm_director_rationale}"
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
