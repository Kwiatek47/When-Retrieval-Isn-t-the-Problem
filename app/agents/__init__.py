"""Multi-agent clinical debate package (MAC / MedAgent-inspired, no supervisor)."""

from app.agents.agent import ClinicalAgent
from app.agents.backends import (
    BioLinkBERTHintProvider,
    EvidenceHint,
    EvidenceHintProvider,
    InferenceBackend,
    MockInferenceBackend,
    NullEvidenceHint,
    OllamaInferenceBackend,
)
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult
from app.agents.orchestrator import DEFAULT_PERSONAS, DebateOrchestrator

__all__ = [
    "AgentRoundOpinion",
    "BioLinkBERTHintProvider",
    "ClinicalAgent",
    "ClinicalOpinion",
    "DebateOrchestrator",
    "DebateResult",
    "DEFAULT_PERSONAS",
    "EvidenceHint",
    "EvidenceHintProvider",
    "InferenceBackend",
    "MockInferenceBackend",
    "NullEvidenceHint",
    "OllamaInferenceBackend",
    "build_default_agents",
]


def build_default_agents(
    backend: InferenceBackend,
    *,
    hint_provider: EvidenceHintProvider | None = None,
) -> list[ClinicalAgent]:
    """Create the standard 4-persona debate panel sharing one inference backend."""
    return [
        ClinicalAgent(
            agent_id=agent_id,
            persona=persona,
            backend=backend,
            hint_provider=hint_provider,
        )
        for agent_id, persona in DEFAULT_PERSONAS
    ]
