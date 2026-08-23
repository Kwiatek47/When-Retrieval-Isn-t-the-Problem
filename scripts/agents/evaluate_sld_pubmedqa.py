#!/usr/bin/env python3
"""Benchmark runner for MAS-Supervisor v2 (Supervised Ledger Debate).

``--backend mock`` uses a schema-aware ``MockSLDBackend`` (dispatches on each
prompt's distinctive role/preamble text, grounds citations in real sentence
IDs parsed back out of the prompt) for offline smoke testing — unlike the
legacy ``MockInferenceBackend`` (app/agents/backends.py), which only emits
``ClinicalOpinion``-shaped JSON and is not compatible with SLD's per-persona
schemas.

``--backend ollama`` runs the real pipeline. ``--arm`` selects a point on the
design doc's experimental ladder (§7):
  L0  single direct call, no pipeline at all           (lower baseline)
  L1  BioLinkBERT alone, 0 LLM calls                    (target: ~0.720)
  L3  R1 + Moderator + rule, no round 2                 (extraction value)
  L4  R1 + R2 without the ledger (raw peer notes)        (ledger's marginal value)
  L5  full pipeline, self-consistency Director           (main arm)
  L6  L5 but verdict_source=llm (Director's own label, not the rule table)
  L8  L5 + BioLinkBERT fusion (ledger_gate_fusion)
L2 (cost-matched self-consistency) and L7 (legacy hybrid+majority) are run via
``evaluate_self_consistency_pubmedqa.py`` / ``evaluate_debate_pubmedqa.py``
respectively — they don't go through the SLD pipeline at all, so they don't
belong in this script.

Dataset/corpus loading and checkpoint/resume plumbing are adapted from
``evaluate_debate_pubmedqa.py``, per the design doc's reuse plan.

Usage:
  # Offline smoke test
  python3 scripts/agents/evaluate_sld_pubmedqa.py --backend mock \\
    --dataset data/benchmarks/pubmedqa/official_pqal_test/quick/balanced90.json \\
    --limit 10 --label sld_smoke

  # Real run against Ollama (qwen2.5:7b), arm L5, dev-90
  python3 scripts/agents/evaluate_sld_pubmedqa.py --backend ollama --arm L5 \\
    --dataset data/benchmarks/pubmedqa/eval_pubmedqa_strict_90.json \\
    --corpus data/benchmarks/pubmedqa/pubmedqa_strict_corpus_90.json \\
    --case-concurrency 2 --resume --label sld_L5_dev90
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.sld import panel  # noqa: E402
from app.agents.sld.ledger import SLDResult  # noqa: E402
from app.agents.sld.pipeline import SLDCase, SLDPipeline  # noqa: E402
from app.agents.sld.segmentation import classify_question_type  # noqa: E402
from app.schemas import ChatMessage  # noqa: E402

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
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "sld"

# Native SLDPipeline arms only; L0/L1 bypass the pipeline entirely (see module
# docstring), L2/L7 live in other scripts.
ARM_PIPELINE_KWARGS: dict[str, dict[str, Any]] = {
    "L3": {"run_round_two": False},
    "L4": {"show_ledger_in_r2": False},
    "L5": {},
    "L6": {"verdict_source": "llm"},
    "L8": {"fuse_biolinkbert": True},
}
NATIVE_ARMS = tuple(ARM_PIPELINE_KWARGS)
ALL_ARMS = ("L0", "L1", *NATIVE_ARMS)

_SENTENCE_LINE_RE = re.compile(r"^(S\d+)(?:\s*\[\w+\])?:\s*(.+)$", re.MULTILINE)


def _sentences_from_prompt(prompt: str) -> dict[str, str]:
    return {match.group(1): match.group(2).strip() for match in _SENTENCE_LINE_RE.finditer(prompt)}


class MockSLDBackend:
    """Deterministic, schema-aware offline backend for the SLD pipeline.

    Grounds every claim it emits in a real sentence parsed out of the prompt
    (so the verification gate sees genuinely grounded citations, not just
    fallbacks) unless ``hallucinate=True``, in which case it cites a
    sentence_id one past the real range — proving the gate actually rejects
    fabricated citations in a live pipeline run, not just in isolated tests.
    """

    def __init__(self, *, hallucinate: bool = False) -> None:
        self.hallucinate = hallucinate
        self.call_count = 0

    async def complete(
        self, messages: list[ChatMessage], *, temperature: float = 0.3, num_predict: int | None = None
    ) -> str:
        self.call_count += 1
        prompt = messages[-1].content
        sentences = _sentences_from_prompt(prompt)
        ids = sorted(sentences, key=lambda sid: int(sid[1:])) or ["S1"]
        cite_id = ids[0]
        cite_text = sentences.get(cite_id, "the reported finding")
        bad_id = f"S{int(cite_id[1:]) + 1000}"  # never a real sentence id
        citation_ids = [bad_id] if self.hallucinate else [cite_id]
        claim_text = "completely unrelated fabricated content" if self.hallucinate else cite_text[:80]

        if "ROLE: question_framer." in prompt:
            return json.dumps(
                {
                    "persona": "question_framer",
                    "agent_id": "question_framer",
                    "target_population": {"text": claim_text, "sentence_ids": citation_ids},
                    "target_exposure": {"text": claim_text, "sentence_ids": citation_ids},
                    "target_outcome": {"text": claim_text, "sentence_ids": citation_ids},
                    "question_type": "utility",
                    "yes_requires": "a clear positive finding",
                    "no_requires": "a clear negative finding",
                }
            )
        if "ROLE: findings_auditor." in prompt:
            return json.dumps(
                {
                    "persona": "findings_auditor",
                    "agent_id": "findings_auditor",
                    "primary_endpoint": {"text": claim_text, "sentence_ids": citation_ids},
                    "direction": "positive",
                    "significance": {"text": claim_text, "sentence_ids": citation_ids},
                    "effect_magnitude": {"text": claim_text, "sentence_ids": citation_ids},
                }
            )
        if "ROLE: gap_auditor." in prompt:
            return json.dumps(
                {
                    "persona": "gap_auditor",
                    "agent_id": "gap_auditor",
                    "gaps": [
                        {
                            "gap_type": "underpowered",
                            "description": claim_text,
                            "sentence_ids": citation_ids,
                        }
                    ],
                }
            )
        if "ROLE: conclusion_reconstructor." in prompt:
            return json.dumps(
                {
                    "persona": "conclusion_reconstructor",
                    "agent_id": "conclusion_reconstructor",
                    "reconstructed_conclusion": {"text": claim_text, "sentence_ids": citation_ids},
                    "direction": "positive",
                    "strength": "qualified",
                }
            )
        if "You are the Supervisor moderating a panel" in prompt:
            return json.dumps({"conflicts": [], "open_questions": [], "round_instructions": []})
        if "You are the same analyst from round 1" in prompt:
            return json.dumps(
                {
                    "agent_id": "mock",
                    "label": "yes",
                    "rationale": claim_text,
                    "citations": citation_ids,
                    "complement": None,
                    "self_audit": None,
                }
            )
        if "You are the Director making the final structured verdict" in prompt:
            return json.dumps(
                {
                    "question_answered_by_endpoint": True,
                    "direction_determinate": True,
                    "findings_statistically_supported": True,
                    "conclusion_would_be_hedged": False,
                    "direction": "positive",
                    "label": "yes",
                    "rationale": claim_text,
                    "citations": citation_ids,
                }
            )
        if "Answer yes, no, or maybe" in prompt:  # L0
            return json.dumps({"label": "maybe", "rationale": claim_text})
        raise AssertionError(f"MockSLDBackend received an unrecognized prompt: {prompt[:200]!r}")


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


def _case_abstract_raw(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> str:
    doc_ids = case.get("relevant_document_ids") or []
    for doc_id in doc_ids:
        doc = corpus.get(doc_id)
        if doc:
            return str(doc.get("content") or "")
    return ""


def _case_documents(case: dict[str, Any], corpus: dict[str, dict[str, Any]]) -> list[Any]:
    from app.rag.models import RetrievedDocument

    documents = []
    for doc_id in case.get("relevant_document_ids") or []:
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


def _load_checkpoint(path: Path) -> dict[str, SLDResult]:
    if not path.exists():
        return {}
    loaded: dict[str, SLDResult] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        result = SLDResult.model_validate(data)
        loaded[result.case_id] = result
    return loaded


def _append_checkpoint(path: Path, result: SLDResult) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(result.model_dump_json() + "\n")


# --- L0: single direct call, no pipeline -------------------------------------


class _SingleCallAnswer(BaseModel):
    label: str = Field(description="yes, no, or maybe")
    rationale: str = ""


async def _run_l0_case(case: dict[str, Any], corpus: dict[str, dict[str, Any]], backend: Any) -> SLDResult:
    abstract = _case_abstract_raw(case, corpus)
    prompt = (
        "Answer yes, no, or maybe based on the abstract only.\n\n"
        f"QUESTION:\n{case['question']}\n\nABSTRACT:\n{abstract}\n\n"
        'Produce: {"label": "yes"|"no"|"maybe", "rationale": "..."}'
    )
    answer = await panel.call_structured_llm(
        backend,
        user_prompt=prompt,
        model_cls=_SingleCallAnswer,
        fallback=_SingleCallAnswer(label="maybe", rationale="fallback"),
        temperature=0.2,
        label="L0",
    )
    label = answer.label.strip().lower()
    if label not in ("yes", "no", "maybe"):
        label = "maybe"
    return SLDResult(
        case_id=case["id"],
        question=case["question"],
        expected_label=case["expected_label"],
        question_type=classify_question_type(case["question"]),
        predicted_label=label,  # type: ignore[arg-type]
        rule_name="single_call",
    )


# --- L1: BioLinkBERT alone, 0 LLM ---------------------------------------------


async def _run_l1_case(
    case: dict[str, Any],
    corpus: dict[str, dict[str, Any]],
    hint_provider: Any,
    lock: asyncio.Lock,
) -> SLDResult:
    documents = _case_documents(case, corpus)
    async with lock:
        hint_provider.set_case(question=case["question"], documents=documents)
        hint = hint_provider.predict_current()
        hint_provider.clear_case()
    label = hint.label if hint else "maybe"
    if label not in ("yes", "no", "maybe"):
        label = "maybe"
    return SLDResult(
        case_id=case["id"],
        question=case["question"],
        expected_label=case["expected_label"],
        question_type=classify_question_type(case["question"]),
        predicted_label=label,  # type: ignore[arg-type]
        rule_name="biolinkbert_only",
    )


# --- Metrics -------------------------------------------------------------------


def _macro_f1(results: list[SLDResult]) -> tuple[float, dict[str, float]]:
    labels = ("yes", "no", "maybe")
    per_label_f1: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for r in results if r.predicted_label == label and r.expected_label == label)
        fp = sum(1 for r in results if r.predicted_label == label and r.expected_label != label)
        fn = sum(1 for r in results if r.predicted_label != label and r.expected_label == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_label_f1[label] = f1
    macro = sum(per_label_f1.values()) / len(labels)
    return macro, per_label_f1


def _summarize(results: list[SLDResult]) -> dict[str, Any]:
    n = len(results) or 1
    correct = sum(1 for r in results if r.predicted_label == r.expected_label)
    macro_f1, per_label_f1 = _macro_f1(results)
    rule_counts = Counter(r.rule_name or "none" for r in results)
    pred_counts = Counter(r.predicted_label or "none" for r in results)

    # Two signals aimed squarely at the failure mode that sank the legacy
    # debate arm (design doc, memory `pubmedqa-debata-jest-atrapa`): a panel
    # that never actually disagrees, or a Moderator that never finds a real
    # conflict, is a classifier wearing a debate costume. Both are 0 when a
    # case never had a ledger/round 2 (e.g. L0/L1), so they're computed only
    # over cases where the signal could exist.
    with_ledger = [r for r in results if r.ledger is not None]
    conflict_rate = (
        sum(1 for r in with_ledger if r.ledger.conflicts) / len(with_ledger) if with_ledger else None
    )
    with_r2 = [r for r in results if r.panel_r2_verified]
    r2_unanimous_rate = (
        sum(1 for r in with_r2 if len({o.label for o in r.panel_r2_verified}) == 1) / len(with_r2)
        if with_r2
        else None
    )
    return {
        "cases": len(results),
        "label_accuracy": correct / n,
        "macro_f1": macro_f1,
        "per_label_f1": per_label_f1,
        "mean_grounding_score_r1": sum(r.grounding_score_r1 for r in results) / n,
        "mean_grounding_score_r2": sum(r.grounding_score_r2 for r in results) / n,
        "total_dropped_claims_r1": sum(len(r.dropped_claims_r1) for r in results),
        "total_dropped_claims_r2": sum(len(r.dropped_claims_r2) for r in results),
        "ledger_conflict_rate": conflict_rate,
        "round2_unanimous_rate": r2_unanimous_rate,
        "predicted_label_counts": dict(pred_counts),
        "rule_name_counts": dict(rule_counts),
    }


def _markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Supervised Ledger Debate — SLD v2 benchmark",
        "",
        f"- Arm: `{summary.get('arm')}`",
        f"- Dataset: `{summary.get('dataset')}`",
        f"- Cases: {summary['cases']}",
        f"- Label accuracy: {summary['label_accuracy']:.3f}",
        f"- Macro-F1: {summary['macro_f1']:.3f}",
        f"- Mean grounding score (R1): {summary['mean_grounding_score_r1']:.3f}",
        f"- Mean grounding score (R2): {summary['mean_grounding_score_r2']:.3f}",
        f"- Dropped claims (R1 / R2): {summary['total_dropped_claims_r1']} / {summary['total_dropped_claims_r2']}",
    ]
    conflict_rate = summary.get("ledger_conflict_rate")
    unanimous_rate = summary.get("round2_unanimous_rate")
    lines.append(
        f"- Ledger conflict rate: {conflict_rate:.3f}" if conflict_rate is not None else "- Ledger conflict rate: n/a (no ledger in this arm)"
    )
    lines.append(
        f"- Round 2 unanimous rate: {unanimous_rate:.3f} "
        "(high + low accuracy = likely rubber-stamping, see memory `pubmedqa-debata-jest-atrapa`)"
        if unanimous_rate is not None
        else "- Round 2 unanimous rate: n/a (no round 2 in this arm)"
    )
    lines += [
        "",
        "## Per-label F1",
        "",
        "| Label | F1 |",
        "|---|---:|",
    ]
    for label, f1 in summary["per_label_f1"].items():
        lines.append(f"| {label} | {f1:.3f} |")
    lines += [
        "",
        "## Predicted label counts",
        "",
        "| Label | Count |",
        "|---|---:|",
    ]
    for label, count in summary["predicted_label_counts"].items():
        lines.append(f"| {label} | {count} |")
    lines += [
        "",
        "## Rule name counts",
        "",
        "| Rule | Count |",
        "|---|---:|",
    ]
    for rule, count in summary["rule_name_counts"].items():
        lines.append(f"| {rule} | {count} |")
    return "\n".join(lines) + "\n"


def _build_ollama_backend(args: argparse.Namespace, base_url: str) -> Any:
    from app.agents.backends import OllamaInferenceBackend
    from app.core.config import get_settings
    from app.providers.ollama import OllamaProvider

    settings = get_settings()
    model_name = args.model or settings.default_model
    provider = OllamaProvider(
        base_url=base_url,
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
        # SLD prompts are budgeted to ~1600 tok in, compact JSON out — the
        # legacy debate script needs 8192 for its unbounded Director prompt
        # (design doc P4); SLD does not.
        num_ctx=args.num_ctx,
    )
    print(f"Ollama model={model_name} num_ctx={args.num_ctx} num_predict={args.num_predict} url={base_url}")
    return OllamaInferenceBackend(provider, model=model_name, temperature=0.3)


async def _run(args: argparse.Namespace) -> None:
    cases = _load_cases(args.dataset)
    if args.offset:
        cases = cases[max(args.offset, 0) :]
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    corpus = _load_corpus(args.corpus)

    # A cluster with several independent Ollama hosts/GPUs: each case sticks
    # to one URL (deterministic by case index) so retries/resume stay on the
    # same host instead of round-robining mid-case.
    if args.backend == "mock":
        base_urls = ["mock"]
        backend_by_url: dict[str, Any] = {"mock": MockSLDBackend(hallucinate=args.hallucinate)}
    else:
        from app.agents.backends import parse_ollama_base_urls
        from app.core.config import get_settings

        base_urls = parse_ollama_base_urls(args.ollama_base_urls, default=get_settings().ollama_base_url)
        backend_by_url = {url: _build_ollama_backend(args, url) for url in base_urls}

    def _backend_for(index: int) -> Any:
        from app.agents.backends import sticky_ollama_url

        return backend_by_url[sticky_ollama_url(base_urls, index)]

    needs_biolinkbert = args.arm in ("L1", "L8")
    hint_provider = None
    classifier_lock = asyncio.Lock()
    if needs_biolinkbert:
        from app.agents.backends import build_biolinkbert_hint_from_settings

        hint_provider = build_biolinkbert_hint_from_settings()
        if not hint_provider.available:
            raise SystemExit(
                f"--arm {args.arm} needs the BioLinkBERT classifier, but it isn't available "
                f"({hint_provider.load_error})"
            )

    personas = panel.R1_PERSONAS
    if args.panel_size == 3:
        # gap_auditor dropped: its output feeds only `gaps`/round_instructions
        # (supplementary signal), unlike the other three which each feed a
        # field compose_label's rule table reads directly.
        personas = tuple(p for p in panel.R1_PERSONAS if p != "gap_auditor")

    pipeline_by_url: dict[str, SLDPipeline] = {}
    if args.arm in ARM_PIPELINE_KWARGS:
        pipeline_by_url = {
            url: SLDPipeline(
                backend=backend_by_url[url],
                director_samples=args.director_samples,
                concurrency=args.agent_concurrency,
                r1_num_predict=args.num_predict,
                r2_num_predict=args.num_predict,
                moderator_num_predict=args.num_predict,
                director_num_predict=args.num_predict,
                personas=personas,
                verification_enabled=not args.no_verification_gate,
                use_stats_profile=not args.no_stats_profile,
                label_blind=not args.not_label_blind,
                **ARM_PIPELINE_KWARGS[args.arm],
            )
            for url in base_urls
        }

    def _pipeline_for(index: int) -> SLDPipeline:
        from app.agents.backends import sticky_ollama_url

        return pipeline_by_url[sticky_ollama_url(base_urls, index)]

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = report_dir / f"{args.label}.checkpoint.jsonl"
    prior_results = _load_checkpoint(checkpoint_path) if args.resume else {}

    checkpoint_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, args.case_concurrency))
    results_by_id: dict[str, SLDResult] = dict(prior_results)

    async def _run_one(index: int, case: dict[str, Any]) -> None:
        if case["id"] in prior_results:
            print(f"[{index}/{len(cases)}] SKIP id={case['id']} (checkpoint)")
            return
        async with semaphore:
            started = time.perf_counter()
            if args.arm == "L0":
                result = await _run_l0_case(case, corpus, _backend_for(index))
            elif args.arm == "L1":
                result = await _run_l1_case(case, corpus, hint_provider, classifier_lock)
            else:
                pipeline = _pipeline_for(index)
                biolinkbert_label = None
                if needs_biolinkbert:
                    documents = _case_documents(case, corpus)
                    async with classifier_lock:
                        hint_provider.set_case(question=case["question"], documents=documents)
                        hint = hint_provider.predict_current()
                        hint_provider.clear_case()
                    biolinkbert_label = hint.label if hint else None
                sld_case = SLDCase(
                    case_id=case["id"],
                    question=case["question"],
                    abstract_raw=_case_abstract_raw(case, corpus),
                    expected_label=case["expected_label"],
                    biolinkbert_label=biolinkbert_label,
                )
                result = await pipeline.run(sld_case)
            latency_ms = (time.perf_counter() - started) * 1000.0
            async with checkpoint_lock:
                _append_checkpoint(checkpoint_path, result)
            results_by_id[result.case_id] = result
            status = "PASS" if result.predicted_label == result.expected_label else "FAIL"
            print(
                f"[{index}/{len(cases)}] {status} id={result.case_id} "
                f"exp={result.expected_label} pred={result.predicted_label} rule={result.rule_name} "
                f"grounding_r1={result.grounding_score_r1:.2f} grounding_r2={result.grounding_score_r2:.2f} "
                f"({latency_ms:.0f} ms)"
            )

    await asyncio.gather(*[_run_one(i, case) for i, case in enumerate(cases, start=1)])
    results = [results_by_id[case["id"]] for case in cases if case["id"] in results_by_id]

    summary = _summarize(results)
    summary["arm"] = args.arm
    summary["dataset"] = str(args.dataset)
    json_path = report_dir / f"{args.label}.json"
    json_path.write_text(
        json.dumps(
            {"summary": summary, "cases": [json.loads(r.model_dump_json()) for r in results]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    md_path = report_dir / f"{args.label}.md"
    md_path.write_text(_markdown_report(summary), encoding="utf-8")
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("mock", "ollama"), default="mock")
    parser.add_argument("--arm", choices=ALL_ARMS, default="L5")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--label", type=str, default="sld_run")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--director-samples", type=int, default=3)
    parser.add_argument("--case-concurrency", type=int, default=1)
    parser.add_argument("--agent-concurrency", type=int, default=4)
    parser.add_argument("--model", type=str, default=None, help="Ollama model (default from settings/.env)")
    parser.add_argument(
        "--ollama-base-urls",
        type=str,
        default=None,
        help="Comma-separated Ollama base URLs for a multi-host/multi-GPU cluster "
        "(e.g. http://10.0.0.1:11434,http://10.0.0.2:11434). Each case sticks to one "
        "URL by case index (app.agents.backends.sticky_ollama_url), matching "
        "evaluate_debate_pubmedqa.py's convention. Default: single URL from settings/.env.",
    )
    parser.add_argument("--num-predict", type=int, default=500)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument(
        "--panel-size",
        type=int,
        choices=(3, 4),
        default=4,
        help="Ablation (f): 3 drops gap_auditor from both rounds (design doc §7)",
    )
    parser.add_argument(
        "--no-verification-gate",
        action="store_true",
        help="Ablation (a): let unverified/hallucinated claims flow through to the ledger "
        "and prompts. grounding_score/dropped_claims are still computed and reported either way.",
    )
    parser.add_argument(
        "--no-stats-profile",
        action="store_true",
        help="Ablation (b): withhold the regex stats_profile block from R1 prompts.",
    )
    parser.add_argument(
        "--not-label-blind",
        action="store_true",
        help="Ablation (d): reveal the eventual yes/no/maybe task to R1 personas "
        "instead of withholding it (measures the cost of the label prior this normally avoids).",
    )
    parser.add_argument(
        "--hallucinate",
        action="store_true",
        help="Mock backend only: cite a nonexistent sentence_id in every field, "
        "to demonstrate the verification gate firing in a live pipeline run.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
