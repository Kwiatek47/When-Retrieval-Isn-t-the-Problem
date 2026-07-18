"""Clinical debate agent that produces structured ClinicalOpinion JSON."""

from __future__ import annotations

import logging

from app.agents.backends import (
    EvidenceHintProvider,
    InferenceBackend,
    NullEvidenceHint,
    parse_clinical_opinion_json,
)
from app.agents.models import ClinicalOpinion
from app.agents.prompts import build_messages

logger = logging.getLogger(__name__)


class ClinicalAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        persona: str,
        backend: InferenceBackend,
        hint_provider: EvidenceHintProvider | None = None,
        temperature: float = 0.3,
    ) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.backend = backend
        self.hint_provider: EvidenceHintProvider = hint_provider or NullEvidenceHint()
        self.temperature = temperature

    async def generate_opinion(
        self,
        patient_case: str,
        context: list[ClinicalOpinion] | None = None,
    ) -> ClinicalOpinion:
        """
        Generate a structured clinical opinion.

        Round 1: pass empty/None context for an independent opinion.
        Later rounds: pass peer ClinicalOpinion objects to critique and revise.
        """
        hint = self.hint_provider.get_hint(patient_case)
        messages = build_messages(
            agent_id=self.agent_id,
            persona=self.persona,
            patient_case=patient_case,
            context=context,
            evidence_hint=hint,
        )
        raw = await self.backend.complete(messages, temperature=self.temperature)
        try:
            return parse_clinical_opinion_json(raw)
        except Exception:
            logger.warning(
                "Agent %s returned invalid ClinicalOpinion JSON; retrying once.",
                self.agent_id,
                exc_info=True,
            )
            repair_messages = build_messages(
                agent_id=self.agent_id,
                persona=self.persona,
                patient_case=patient_case,
                context=context,
                evidence_hint=hint,
                repair=True,
            )
            raw_retry = await self.backend.complete(repair_messages, temperature=0.0)
            return parse_clinical_opinion_json(raw_retry)
