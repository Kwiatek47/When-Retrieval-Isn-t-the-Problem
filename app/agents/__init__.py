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
    parse_ollama_base_urls,
    sticky_ollama_url,
)
from app.agents.models import AgentRoundOpinion, ClinicalOpinion, DebateResult
from app.agents.orchestrator import (
    DEFAULT_PERSONAS,
    PUBMEDQA_PERSONAS,
    DebateOrchestrator,
    abstract_suggests_inconclusive,
    check_early_exit_asymmetric_veto,
    labels_unanimous,
    should_continue_debate,
)
from app.agents.supervisor_agent import SupervisorAgent

__all__ = [
    "AgentRoundOpinion",
    "BioLinkBERTHintProvider",
    "ClinicalAgent",
    "ClinicalOpinion",
    "DebateOrchestrator",
    "DebateResult",
    "DEFAULT_PERSONAS",
    "PUBMEDQA_PERSONAS",
    "EvidenceHint",
    "EvidenceHintProvider",
    "InferenceBackend",
    "MockInferenceBackend",
    "NullEvidenceHint",
    "OllamaInferenceBackend",
    "aggregate_pubmedqa_decision",
    "abstract_suggests_inconclusive",
    "build_biolinkbert_hint_from_settings",
    "build_default_agents",
    "check_early_exit_asymmetric_veto",
    "extract_label",
    "hint_as_clinical_opinion",
    "labels_unanimous",
    "majority_vote",
    "opinion_label",
    "parse_ollama_base_urls",
    "should_continue_debate",
    "sticky_ollama_url",
    "SupervisorAgent",
]


def build_default_agents(
    backend: InferenceBackend,
    *,
    hint_provider: EvidenceHintProvider | None = None,
    task_mode: str = "clinical",
    compact: bool = False,
) -> list[ClinicalAgent]:
    """Create the standard debate panel sharing one inference backend.

    For PubMedQA we swap the clinical `safety_officer` for an
    `uncertainty_advocate` persona that actively argues for inconclusive
    evidence, to counter the silent-agreement collapse on the `maybe` class.
    """
    personas = PUBMEDQA_PERSONAS if (task_mode or "").strip().lower() == "pubmedqa" else DEFAULT_PERSONAS
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
