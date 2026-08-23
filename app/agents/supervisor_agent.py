"""Supervisor agent for moderation + final decision synthesis."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agents.backends import InferenceBackend
from app.agents.models import (
    SupervisorDirectorOutput,
    SupervisorModerationOutput,
)
from app.agents.prompts import (
    SUPERVISOR_DIRECTOR_PROMPT,
    SUPERVISOR_MODERATOR_PROMPT,
    PeerContextMode,
    format_opinions_for_supervisor,
)
from app.schemas import ChatMessage

logger = logging.getLogger(__name__)


def _json_extract_object(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip fenced code blocks like ```json ... ```
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try to extract the first JSON object present in the text.
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        return match.group(0)
    return text


def _safe_json_loads(raw: str) -> dict[str, Any]:
    obj_text = _json_extract_object(raw)
    data = json.loads(obj_text)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object, got {type(data).__name__}")
    return data


def _normalize_author_conclusion(value: Any) -> str:
    text = str(value or "unclear").strip().lower()
    if text in {"yes", "no", "maybe", "unclear"}:
        return text
    return "unclear"


class SupervisorAgent:
    """LLM-powered supervisor that moderates rounds and synthesizes a final decision."""

    def __init__(
        self,
        *,
        backend: InferenceBackend,
        temperature: float = 0.2,
        peer_context: PeerContextMode = "nl",
    ) -> None:
        self.backend = backend
        self.temperature = temperature
        self.peer_context: PeerContextMode = (peer_context or "nl")  # type: ignore[assignment]
        self.last_moderation_output: SupervisorModerationOutput | None = None
        self.last_moderation_failed: bool = False
        self.last_director_output: SupervisorDirectorOutput | None = None

    async def moderate_round(
        self,
        patient_case: str,
        agents_opinions: dict[str, Any] | list[Any],
    ) -> SupervisorModerationOutput:
        rendered = format_opinions_for_supervisor(
            agents_opinions,
            peer_context=self.peer_context,
        )
        prompt = SUPERVISOR_MODERATOR_PROMPT.format(
            patient_case=patient_case,
            previous_round_opinions=rendered,
        )
        messages = [
            ChatMessage(role="system", content="Return only valid JSON matching the requested schema."),
            ChatMessage(role="user", content=prompt),
        ]

        raw = await self._complete_with_repair(messages, self.temperature)
        try:
            data = _safe_json_loads(raw)
            data["author_conclusion"] = _normalize_author_conclusion(
                data.get("author_conclusion")
            )
            if not isinstance(data.get("residual_uncertainty"), list):
                data["residual_uncertainty"] = []
            data.setdefault("primary_endpoint_result", "")
            output = SupervisorModerationOutput.model_validate(data)
            self.last_moderation_output = output
            self.last_moderation_failed = False
            return output
        except Exception:
            logger.warning("Failed to parse SupervisorModerationOutput; using fallback.", exc_info=True)
            output = SupervisorModerationOutput(
                agreements=[],
                contradictions=[],
                round_instructions=[
                    "Please re-run moderation: keep only evidence-grounded agreements/contradictions and output valid JSON."
                ],
                primary_endpoint_result="",
                author_conclusion="unclear",
                residual_uncertainty=[],
            )
            self.last_moderation_output = output
            self.last_moderation_failed = True
            return output

    async def synthesize_decision(
        self,
        patient_case: str,
        debate_transcript: str,
        biolinkbert_hint: str,
        *,
        shared_report: str | None = None,
        debate_brief: str | None = None,
        panel_vote_summary: str | None = None,
    ) -> SupervisorDirectorOutput:
        schema = SupervisorDirectorOutput.model_json_schema()
        # Full multi-round conflict transcript is the primary Director input.
        # ``debate_brief`` remains for backward-compatible callers only.
        transcript = (debate_transcript or "").strip() or (debate_brief or "")
        prompt = SUPERVISOR_DIRECTOR_PROMPT.format(
            patient_case=patient_case,
            full_debate_transcript=transcript,
            panel_vote_summary=(panel_vote_summary or "").strip()
            or "(panel vote summary unavailable)",
            biolinkbert_hint=biolinkbert_hint,
        )

        # Add explicit Pydantic schema for higher compliance.
        prompt = (
            prompt
            + "\n\nPydantic JSON schema (Director):\n"
            + json.dumps(schema, ensure_ascii=False)
        )

        messages = [
            ChatMessage(role="system", content="Return only valid JSON matching the requested schema."),
            ChatMessage(role="user", content=prompt),
        ]

        raw = await self._complete_with_repair(messages, self.temperature)
        try:
            data = _safe_json_loads(raw)
            # Defaults for models that omit optional / newer schema fields.
            data.setdefault("debate_conflict_level", "medium")
            data.setdefault("conclusiveness_score", 5)
            if not isinstance(data.get("unresolved_contradictions"), list):
                data["unresolved_contradictions"] = []
            data.setdefault("primary_endpoint_answers_question", True)
            data.setdefault("findings_decisive_for_question", True)
            data.setdefault("authors_state_uncertainty", False)
            conflict = str(data.get("debate_conflict_level") or "medium").strip().lower()
            if conflict not in {"low", "medium", "high"}:
                conflict = "medium"
            data["debate_conflict_level"] = conflict
            try:
                score = int(data.get("conclusiveness_score", 5))
            except (TypeError, ValueError):
                score = 5
            data["conclusiveness_score"] = min(10, max(1, score))
            coverage = str(data.get("question_coverage") or "full").strip().lower()
            if coverage not in {"full", "partial", "none"}:
                coverage = "full"
            data["question_coverage"] = coverage
            # An off-vocabulary consensus_type used to fail validation and discard the
            # whole output, silently forcing the fallback "maybe". A usable final_label
            # must not be thrown away over a descriptive field.
            consensus = str(data.get("consensus_type") or "").strip().lower()
            if consensus not in {"consensus", "differential", "escalation"}:
                consensus = "differential"
            data["consensus_type"] = consensus
            output = SupervisorDirectorOutput.model_validate(data)
            self.last_director_output = output
            return output
        except Exception:
            logger.warning("Failed to parse SupervisorDirectorOutput; using fallback.", exc_info=True)
            output = SupervisorDirectorOutput(
                final_label="maybe",
                consensus_type="escalation",
                rationale="Supervisor failed to produce valid output; defaulting to conservative 'maybe'.",
                debate_conflict_level="high",
                conclusiveness_score=3,
                primary_endpoint_answers_question=False,
                findings_decisive_for_question=False,
                authors_state_uncertainty=True,
                question_coverage="none",
            )
            self.last_director_output = output
            return output

    async def _complete_with_repair(self, messages: list[ChatMessage], temperature: float) -> str:
        try:
            raw = await self.backend.complete(messages, temperature=temperature)
            if (raw or "").strip():
                return raw
            logger.warning("Supervisor returned empty response; retrying with repair.")
        except Exception:
            logger.warning("Supervisor backend call failed; retry with temperature=0.0.", exc_info=True)

        repair = [
            messages[0],
            ChatMessage(
                role="user",
                content=messages[1].content
                + "\n\nYour previous reply was empty or invalid. Return ONLY valid JSON, no markdown, no commentary.",
            ),
        ]
        return await self.backend.complete(repair, temperature=0.0)
