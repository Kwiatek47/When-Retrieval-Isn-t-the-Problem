"""Clinical debate agent that produces structured ClinicalOpinion JSON."""

from __future__ import annotations

import logging

from app.agents.backends import (
    EvidenceHintProvider,
    InferenceBackend,
    NullEvidenceHint,
    fallback_clinical_opinion,
    parse_clinical_opinion_json,
)
from app.agents.models import AgentRoundOpinion, ClinicalOpinion
from app.agents.prompts import build_messages
from app.schemas import ChatMessage

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
        task_mode: str = "clinical",
        compact: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.backend = backend
        self.hint_provider: EvidenceHintProvider = hint_provider or NullEvidenceHint()
        self.temperature = temperature
        self.task_mode = task_mode
        self.compact = compact

    async def generate_opinion(
        self,
        patient_case: str,
        context: list[AgentRoundOpinion] | None = None,
        *,
        anonymize: bool = False,
        info_requests: list[str] | None = None,
        partial_evidence: bool = False,
        shuffle_seed: int | None = None,
    ) -> ClinicalOpinion:
        """
        Generate a structured clinical opinion.

        Round 1: pass empty/None context for an independent opinion.
        Later rounds: pass peer AgentRoundOpinion entries (previous round's
        finals plus anyone who has already spoken this round) to critique
        and revise.

        The keyword-only flags are used by the supervisor architectures:
        `anonymize` strips peer identities from the transcript, `partial_evidence`
        tells the agent it holds only one segment of the case, and `info_requests`
        carries peers' questions routed to this agent. Defaults reproduce the
        baseline debate exactly.

        Never raises: backend failures (timeout, connection error, invalid
        JSON after one repair attempt) degrade to a low-confidence fallback
        opinion so a single flaky agent cannot crash the whole debate round.
        """
        hint = self.hint_provider.get_hint(patient_case)
        prompt_kwargs = {
            "anonymize": anonymize,
            "info_requests": info_requests,
            "partial_evidence": partial_evidence,
            "shuffle_seed": shuffle_seed,
        }
        messages = build_messages(
            agent_id=self.agent_id,
            persona=self.persona,
            patient_case=patient_case,
            context=context,
            evidence_hint=hint,
            task_mode=self.task_mode,
            compact=self.compact,
            **prompt_kwargs,
        )
        raw = await self._complete_or_none(messages, self.temperature)
        if raw is not None:
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
            task_mode=self.task_mode,
            compact=self.compact,
            **prompt_kwargs,
        )
        raw_retry = await self._complete_or_none(repair_messages, 0.0)
        if raw_retry is not None:
            try:
                return parse_clinical_opinion_json(raw_retry)
            except Exception:
                logger.warning(
                    "Agent %s retry also failed; using fallback opinion.",
                    self.agent_id,
                    exc_info=True,
                )

        fallback_label = hint.label if hint is not None else "maybe"
        return fallback_clinical_opinion(
            label=fallback_label,
            reason=f"Fallback for agent `{self.agent_id}` after invalid JSON or backend error.",
        )

    async def _complete_or_none(
        self, messages: list[ChatMessage], temperature: float
    ) -> str | None:
        """Run the backend, swallowing any exception (timeout, connection error, ...)."""
        try:
            return await self.backend.complete(messages, temperature=temperature)
        except Exception:
            logger.warning(
                "Agent %s backend call failed.",
                self.agent_id,
                exc_info=True,
            )
            return None
