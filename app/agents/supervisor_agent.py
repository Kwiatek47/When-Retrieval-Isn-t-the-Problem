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
from app.agents.prompts import SUPERVISOR_DIRECTOR_PROMPT, SUPERVISOR_MODERATOR_PROMPT
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


class SupervisorAgent:
    """LLM-powered supervisor that moderates rounds and synthesizes a final decision."""

    def __init__(
        self,
        *,
        backend: InferenceBackend,
        temperature: float = 0.2,
    ) -> None:
        self.backend = backend
        self.temperature = temperature

    async def moderate_round(
        self,
        patient_case: str,
        agents_opinions: dict[str, Any],
    ) -> SupervisorModerationOutput:
        prompt = SUPERVISOR_MODERATOR_PROMPT.format(
            patient_case=patient_case,
            previous_round_opinions=json.dumps(agents_opinions, ensure_ascii=False),
        )
        messages = [
            ChatMessage(role="system", content="Return only valid JSON matching the requested schema."),
            ChatMessage(role="user", content=prompt),
        ]

        raw = await self._complete_with_repair(messages, self.temperature)
        try:
            data = _safe_json_loads(raw)
            return SupervisorModerationOutput.model_validate(data)
        except Exception:
            logger.warning("Failed to parse SupervisorModerationOutput; using fallback.", exc_info=True)
            return SupervisorModerationOutput(
                agreements=[],
                contradictions=[],
                round_instructions=[
                    "Please re-run moderation: keep only evidence-grounded agreements/contradictions and output valid JSON."
                ],
            )

    async def synthesize_decision(
        self,
        patient_case: str,
        debate_transcript: str,
        biolinkbert_hint: str,
    ) -> SupervisorDirectorOutput:
        schema = SupervisorDirectorOutput.model_json_schema()
        prompt = SUPERVISOR_DIRECTOR_PROMPT.format(
            patient_case=patient_case,
            full_debate_transcript=debate_transcript,
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
            return SupervisorDirectorOutput.model_validate(data)
        except Exception:
            logger.warning("Failed to parse SupervisorDirectorOutput; using fallback.", exc_info=True)
            return SupervisorDirectorOutput(
                final_label="maybe",
                consensus_type="escalation",
                rationale="Supervisor failed to produce valid output; defaulting to conservative 'maybe'.",
            )

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

