"""Round 1 (label-blind panel) and Round 2 (cooperative debate) execution.

Concurrency pattern follows ``DebateOrchestrator._run_independent_round``
(``app/agents/orchestrator.py:315``): a bounded ``asyncio.Semaphore`` plus
``asyncio.gather``, all agents/personas running against the *same* prior-round
state (never a same-round peer draft — see design doc P3, false consensus).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from app.agents.backends import InferenceBackend, _extract_json_object
from app.agents.sld.ledger import (
    ConclusionReconstructorContribution,
    EvidenceLedger,
    FindingsAuditorContribution,
    GapAuditorContribution,
    NeutralContribution,
    PanelContribution,
    QuestionFramerContribution,
    RoundTwoOpinion,
)
from app.agents.sld.prompts import (
    SYSTEM_JSON_ONLY,
    _TRUNCATED_FIELD_CHARS,
    build_conclusion_reconstructor_prompt,
    build_findings_auditor_prompt,
    build_gap_auditor_prompt,
    build_neutral_prompt,
    build_question_framer_prompt,
    build_round_two_peer_prompt,
    build_round_two_prompt,
    render_contribution,
    render_ledger,
)
from app.agents.sld.segmentation import StatsProfile
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)

R1_PERSONAS: tuple[str, ...] = (
    "question_framer",
    "findings_auditor",
    "gap_auditor",
    "conclusion_reconstructor",
)

T = TypeVar("T", bound=BaseModel)


async def _complete_or_none(
    backend: InferenceBackend,
    messages: list[ChatMessage],
    temperature: float,
    num_predict: int | None,
) -> str | None:
    try:
        raw = await backend.complete(messages, temperature=temperature, num_predict=num_predict)
        return (raw or "").strip() or None
    except Exception:
        logger.warning("SLD backend call failed.", exc_info=True)
        return None


def _try_parse(raw: str | None, model_cls: type[T]) -> T | None:
    if raw is None:
        return None
    try:
        obj_text = _extract_json_object(raw)
        data = json.loads(obj_text)
        if not isinstance(data, dict):
            return None
        return model_cls.model_validate(data)
    except (json.JSONDecodeError, ValidationError, ValueError):
        return None


async def call_structured_llm(
    backend: InferenceBackend,
    *,
    user_prompt: str,
    model_cls: type[T],
    fallback: T,
    temperature: float = 0.3,
    num_predict: int | None = None,
    label: str = "",
) -> T:
    """Call the backend, parse the JSON response into ``model_cls``, one retry.

    Never raises: an empty/invalid response after the repair retry degrades to
    ``fallback`` so one flaky call can't crash the whole panel round (same
    contract as ``ClinicalAgent.generate_opinion``).
    """
    messages = [
        ChatMessage(role="system", content=SYSTEM_JSON_ONLY),
        ChatMessage(role="user", content=user_prompt),
    ]
    raw = await _complete_or_none(backend, messages, temperature, num_predict)
    parsed = _try_parse(raw, model_cls)
    if parsed is not None:
        return parsed

    repair_messages = [
        messages[0],
        ChatMessage(
            role="user",
            content=user_prompt
            + "\n\nYour previous reply was empty or invalid JSON. Return ONLY valid JSON matching the schema.",
        ),
    ]
    raw_retry = await _complete_or_none(backend, repair_messages, 0.0, num_predict)
    parsed = _try_parse(raw_retry, model_cls)
    if parsed is not None:
        return parsed

    logger.warning("%s: falling back after empty/invalid JSON (both attempts).", label or model_cls.__name__)
    return fallback


# --- Fallback instances --------------------------------------------------------
# Deliberately carry no citations (empty sentence_ids / None claims), so the
# verification gate has nothing ungrounded to reject — a failed call degrades
# to "this persona said nothing useful", not to a fabricated-looking claim.


def _fallback_question_framer(agent_id: str) -> QuestionFramerContribution:
    return QuestionFramerContribution(
        agent_id=agent_id,
        question_type="association",
        yes_requires="",
        no_requires="",
    )


def _fallback_findings_auditor(agent_id: str) -> FindingsAuditorContribution:
    return FindingsAuditorContribution(agent_id=agent_id, direction="none")


def _fallback_gap_auditor(agent_id: str) -> GapAuditorContribution:
    return GapAuditorContribution(agent_id=agent_id, gaps=[])


def _fallback_conclusion_reconstructor(agent_id: str) -> ConclusionReconstructorContribution:
    return ConclusionReconstructorContribution(agent_id=agent_id, direction="none", strength="speculative")


def _fallback_neutral(agent_id: str) -> NeutralContribution:
    return NeutralContribution(
        agent_id=agent_id,
        question_type="association",
        direction="none",
        conclusion_direction="none",
        conclusion_strength="speculative",
    )


def _fallback_round_two(agent_id: str) -> RoundTwoOpinion:
    return RoundTwoOpinion(
        agent_id=agent_id,
        label="maybe",
        rationale="Fallback after empty/invalid model output; discount this opinion.",
        citations=[],
    )


# --- Round 1 -------------------------------------------------------------------


async def run_round_one(
    *,
    question: str,
    sentences: dict[str, str],
    section_tags: dict[str, str],
    stats_profile: StatsProfile,
    backend: InferenceBackend,
    personas: tuple[str, ...] = R1_PERSONAS,
    concurrency: int = 4,
    temperature: float = 0.3,
    num_predict: int | None = None,
    label_blind: bool = True,
) -> list[PanelContribution]:
    """Every persona answers concurrently, seeing no peer output.

    ``label_blind=False`` is ablation (d) (design doc §7): reveals the
    eventual yes/no/maybe task to R1 personas instead of withholding it,
    to measure the cost of the label prior this normally avoids (P2).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(persona: str) -> PanelContribution:
        async with semaphore:
            if persona == "question_framer":
                prompt = build_question_framer_prompt(question, sentences, label_blind=label_blind)
                return await call_structured_llm(
                    backend,
                    user_prompt=prompt,
                    model_cls=QuestionFramerContribution,
                    fallback=_fallback_question_framer(persona),
                    temperature=temperature,
                    num_predict=num_predict,
                    label=persona,
                )
            if persona == "findings_auditor":
                prompt = build_findings_auditor_prompt(
                    question, sentences, section_tags, stats_profile, label_blind=label_blind
                )
                return await call_structured_llm(
                    backend,
                    user_prompt=prompt,
                    model_cls=FindingsAuditorContribution,
                    fallback=_fallback_findings_auditor(persona),
                    temperature=temperature,
                    num_predict=num_predict,
                    label=persona,
                )
            if persona == "gap_auditor":
                prompt = build_gap_auditor_prompt(question, sentences, stats_profile, label_blind=label_blind)
                return await call_structured_llm(
                    backend,
                    user_prompt=prompt,
                    model_cls=GapAuditorContribution,
                    fallback=_fallback_gap_auditor(persona),
                    temperature=temperature,
                    num_predict=num_predict,
                    label=persona,
                )
            if persona == "conclusion_reconstructor":
                prompt = build_conclusion_reconstructor_prompt(question, sentences, label_blind=label_blind)
                return await call_structured_llm(
                    backend,
                    user_prompt=prompt,
                    model_cls=ConclusionReconstructorContribution,
                    fallback=_fallback_conclusion_reconstructor(persona),
                    temperature=temperature,
                    num_predict=num_predict,
                    label=persona,
                )
            raise ValueError(f"Unknown R1 persona: {persona!r}")

    return list(await asyncio.gather(*[_one(persona) for persona in personas]))


async def run_round_one_neutral(
    *,
    question: str,
    sentences: dict[str, str],
    section_tags: dict[str, str],
    stats_profile: StatsProfile,
    backend: InferenceBackend,
    count: int = 4,
    concurrency: int = 4,
    temperature: float = 0.3,
    num_predict: int | None = None,
    label_blind: bool = True,
) -> list[NeutralContribution]:
    """Ablation (c) (design doc §7c): ``count`` identical agents each attempt
    the full extraction task (all four specialized personas' fields at
    once), instead of one specialized persona each."""
    semaphore = asyncio.Semaphore(concurrency)

    async def _one(agent_id: str) -> NeutralContribution:
        async with semaphore:
            prompt = build_neutral_prompt(
                question, sentences, section_tags, stats_profile, label_blind=label_blind
            )
            return await call_structured_llm(
                backend,
                user_prompt=prompt,
                model_cls=NeutralContribution,
                fallback=_fallback_neutral(agent_id),
                temperature=temperature,
                num_predict=num_predict,
                label=agent_id,
            )

    agent_ids = [f"neutral_{i + 1}" for i in range(max(1, count))]
    return list(await asyncio.gather(*[_one(agent_id) for agent_id in agent_ids]))


# --- Round 2 -------------------------------------------------------------------


async def run_round_two(
    *,
    question: str,
    sentences: dict[str, str],
    own_contributions: dict[str, PanelContribution],
    round_instructions: list[str],
    backend: InferenceBackend,
    ledger: EvidenceLedger | None = None,
    concurrency: int = 4,
    temperature: float = 0.3,
    num_predict: int | None = None,
) -> list[RoundTwoOpinion]:
    """Every R1 persona now commits to a label.

    When ``ledger`` is given, each agent reads the shared, Supervisor-verified
    ledger plus only its own round-1 note (never a peer's directly — the
    ledger is the shared state, design doc P3). When ``ledger`` is ``None``
    (the L4 ablation arm), agents instead see raw, unverified peer round-1
    notes directly — same round/call count, no ledger, isolating its value.
    """
    semaphore = asyncio.Semaphore(concurrency)
    ledger_rendering = render_ledger(ledger) if ledger is not None else None

    async def _one(agent_id: str, own_contribution: PanelContribution) -> RoundTwoOpinion:
        async with semaphore:
            if ledger_rendering is not None:
                prompt = build_round_two_prompt(
                    question, sentences, ledger_rendering, own_contribution, round_instructions
                )
            else:
                peer_notes = "\n\n".join(
                    render_contribution(contribution, max_field_chars=_TRUNCATED_FIELD_CHARS)
                    for peer_id, contribution in own_contributions.items()
                    if peer_id != agent_id
                )
                prompt = build_round_two_peer_prompt(
                    question,
                    sentences,
                    peer_notes or "(no peer notes)",
                    own_contribution,
                    round_instructions,
                )
            return await call_structured_llm(
                backend,
                user_prompt=prompt,
                model_cls=RoundTwoOpinion,
                fallback=_fallback_round_two(agent_id),
                temperature=temperature,
                num_predict=num_predict,
                label=agent_id,
            )

    return list(
        await asyncio.gather(
            *[_one(agent_id, contribution) for agent_id, contribution in own_contributions.items()]
        )
    )
