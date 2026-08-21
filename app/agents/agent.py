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
from app.agents.prompts import PeerContextMode  # re-export type for callers
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
        peer_context: PeerContextMode = "nl",
    ) -> None:
        self.agent_id = agent_id
        self.persona = persona
        self.backend = backend
        self.hint_provider: EvidenceHintProvider = hint_provider or NullEvidenceHint()
        self.temperature = temperature
        self.task_mode = task_mode
        self.compact = compact
        self.peer_context: PeerContextMode = (peer_context or "nl")  # type: ignore[assignment]

    async def generate_opinion(
        self,
        patient_case: str,
        context: list[AgentRoundOpinion] | None = None,
        *,
        include_evidence_hint: bool = True,
    ) -> ClinicalOpinion:
        """
        Generate a structured clinical opinion.

        Round 1: pass empty/None context for an independent opinion.
        Later rounds: pass peer AgentRoundOpinion entries (previous round's
        finals plus anyone who has already spoken this round) to critique
        and revise.

        Set ``include_evidence_hint=False`` to hide BioLinkBERT (e.g. blind
        ``uncertainty_advocate`` when ``blind_critic`` is enabled).

        Never raises: backend failures (timeout, connection error, invalid
        JSON after one repair attempt) degrade to a low-confidence fallback
        opinion so a single flaky agent cannot crash the whole debate round.
        """
        hint = (
            self.hint_provider.get_hint(patient_case)
            if include_evidence_hint
            else None
        )
        messages = build_messages(
            agent_id=self.agent_id,
            persona=self.persona,
            patient_case=patient_case,
            context=context,
            evidence_hint=hint,
            task_mode=self.task_mode,
            compact=self.compact,
            peer_context=self.peer_context,
        )
        raw = await self._complete_or_none(messages, self.temperature)
        opinion = self._try_parse(raw)
        if opinion is not None:
            return opinion

        # Repair: force compact PubMedQA schema to reduce empty/truncated JSON under load.
        repair_compact = self.compact or (self.task_mode or "").strip().lower() == "pubmedqa"
        repair_messages = build_messages(
            agent_id=self.agent_id,
            persona=self.persona,
            patient_case=patient_case,
            context=context,
            evidence_hint=hint,
            repair=True,
            task_mode=self.task_mode,
            compact=repair_compact,
            peer_context=self.peer_context,
        )
        raw_retry = await self._complete_or_none(repair_messages, 0.0)
        opinion = self._try_parse(raw_retry, retry=True)
        if opinion is not None:
            return opinion

        # Prefer not to invent a clinical label from thin air when the blind critic
        # has no hint — keep maybe at very low confidence so the director can discount it.
        fallback_label = hint.label if hint is not None else "maybe"
        return fallback_clinical_opinion(
            label=fallback_label,
            reason=(
                f"Fallback for agent `{self.agent_id}` after empty/invalid JSON "
                "or backend error (discount this opinion)."
            ),
        )

    def _try_parse(self, raw: str | None, *, retry: bool = False) -> ClinicalOpinion | None:
        if raw is None or not str(raw).strip():
            logger.warning(
                "Agent %s received empty model response%s.",
                self.agent_id,
                " on retry" if retry else "",
            )
            return None
        try:
            return parse_clinical_opinion_json(raw)
        except Exception:
            preview = str(raw).replace("\n", "\\n")[:180]
            logger.warning(
                "Agent %s returned invalid ClinicalOpinion JSON%s; preview=%r",
                self.agent_id,
                "; retry failed, using fallback" if retry else "; retrying once",
                preview,
                exc_info=True,
            )
            return None

    async def _complete_or_none(
        self, messages: list[ChatMessage], temperature: float
    ) -> str | None:
        """Run the backend, swallowing any exception (timeout, connection error, ...)."""
        try:
            content = await self.backend.complete(messages, temperature=temperature)
            return (content or "").strip() or None
        except Exception:
            logger.warning(
                "Agent %s backend call failed.",
                self.agent_id,
                exc_info=True,
            )
            return None
