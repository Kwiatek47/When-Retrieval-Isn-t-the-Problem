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
from app.agents.partitioning import ContextPartition, partition_patient_case
from app.agents.supervisor import SupervisorAgent, SupervisorVerdict, eligibility_gate
from app.agents.supervisor_orchestrator import SupervisorOrchestrator
from app.agents.token_meter import CountingBackend, UsageSnapshot, total_usage
from app.agents.orchestrator import (
    DEFAULT_PERSONAS,
    NEUTRAL_PERSONAS,
    PUBMEDQA_PERSONAS,
    DebateOrchestrator,
    labels_unanimous,
)

__all__ = [
    "AgentRoundOpinion",
    "BioLinkBERTHintProvider",
    "ClinicalAgent",
    "ClinicalOpinion",
    "ContextPartition",
    "CountingBackend",
    "DebateOrchestrator",
    "DebateResult",
    "SupervisorAgent",
    "SupervisorOrchestrator",
    "SupervisorVerdict",
    "UsageSnapshot",
    "eligibility_gate",
    "partition_patient_case",
    "total_usage",
    "DEFAULT_PERSONAS",
    "NEUTRAL_PERSONAS",
    "PUBMEDQA_PERSONAS",
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
    neutral: bool = False,
) -> list[ClinicalAgent]:
    """Create the standard debate panel sharing one inference backend.

    For PubMedQA we swap the clinical `safety_officer` for an
    `uncertainty_advocate` persona that actively argues for inconclusive
    evidence, to counter the silent-agreement collapse on the `maybe` class.

    With `neutral=True` the panel becomes four identical role-free analysts
    instead. That is the starting point of the supervisor architectures, which
    differentiate agents by the evidence they can see rather than by an assigned
    persona; it also isolates how much of any effect the personas alone explain.
    """
    if neutral:
        personas = NEUTRAL_PERSONAS
    elif (task_mode or "").strip().lower() == "pubmedqa":
        personas = PUBMEDQA_PERSONAS
    else:
        personas = DEFAULT_PERSONAS
    return [
        ClinicalAgent(
            agent_id=agent_id,
            persona=persona,
            backend=backend,
            hint_provider=hint_provider,
            task_mode=task_mode,
            compact=compact,
        )
        for agent_id, persona in personas
    ]
