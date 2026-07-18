"""Multi-agent clinical debate package (MAC / MedAgent-inspired, no supervisor)."""

from app.agents.aggregation import (
    aggregate_pubmedqa_decision,
    extract_label,
    majority_vote,
    opinion_label,
)
from app.agents.agent import ClinicalAgent
from app.agents.backends import (
    BioLinkBERTHintProvider,
    EvidenceHint,
    EvidenceHintProvider,
    InferenceBackend,
    MockInferenceBackend,
    NullEvidenceHint,
    OllamaInferenceBackend,
    build_biolinkbert_hint_from_settings,
    hint_as_clinical_opinion,
)
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult
from app.agents.orchestrator import DEFAULT_PERSONAS, DebateOrchestrator, labels_unanimous

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
    "aggregate_pubmedqa_decision",
    "build_biolinkbert_hint_from_settings",
    "build_default_agents",
    "extract_label",
    "hint_as_clinical_opinion",
    "labels_unanimous",
    "majority_vote",
    "opinion_label",
]


def build_default_agents(
    backend: InferenceBackend,
    *,
    hint_provider: EvidenceHintProvider | None = None,
    task_mode: str = "clinical",
    compact: bool = False,
) -> list[ClinicalAgent]:
    """Create the standard 4-persona debate panel sharing one inference backend."""
    return [
        ClinicalAgent(
            agent_id=agent_id,
            persona=persona,
            backend=backend,
            hint_provider=hint_provider,
            task_mode=task_mode,
            compact=compact,
        )
        for agent_id, persona in DEFAULT_PERSONAS
    ]
