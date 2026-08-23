"""Typed schema for the Supervised Ledger Debate pipeline (Pydantic v2).

Every claim in this module carries ``sentence_ids`` pointing back into the
``S1..Sn`` dictionary produced by :mod:`app.agents.sld.segmentation`. That is
the citation contract :mod:`app.agents.sld.verify` checks: a claim whose
``sentence_ids`` don't resolve to real sentences (or whose text isn't actually
supported by them) gets dropped before it ever reaches the Supervisor or the
Director.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from app.agents.sld.segmentation import QuestionType

Direction = Literal["positive", "negative", "none"]
ConclusionStrength = Literal["definitive", "qualified", "speculative"]
GapType = Literal[
    "surrogate_outcome",
    "subgroup_only",
    "association_not_utility",
    "no_comparator",
    "underpowered",
    "contradictory_endpoints",
    "weak_discrimination",
]


class Claim(BaseModel):
    """An atomic assertion grounded in specific sentences of the abstract."""

    text: str
    sentence_ids: list[str] = Field(default_factory=list)


class Gap(BaseModel):
    """A typed evidentiary gap raised by the gap_auditor persona."""

    gap_type: GapType
    description: str
    sentence_ids: list[str] = Field(default_factory=list)


# --- Round 1: label-blind panel contributions -------------------------------
# One model per persona so each R1 LLM call has a narrow, persona-specific
# schema instead of one form with mostly-empty optional fields.


class QuestionFramerContribution(BaseModel):
    persona: Literal["question_framer"] = "question_framer"
    agent_id: str
    # Claim fields are optional so verify.py can null out an individual
    # fabricated citation without invalidating the rest of the contribution.
    target_population: Claim | None = None
    target_exposure: Claim | None = None
    target_outcome: Claim | None = None
    question_type: QuestionType
    yes_requires: str
    no_requires: str


class FindingsAuditorContribution(BaseModel):
    """Restricted to RESULTS-tagged sentences; enforced by verify.py, not here."""

    persona: Literal["findings_auditor"] = "findings_auditor"
    agent_id: str
    primary_endpoint: Claim | None = None
    direction: Direction
    significance: Claim | None = None
    effect_magnitude: Claim | None = None


class GapAuditorContribution(BaseModel):
    """An empty ``gaps`` list is a valid, meaningful answer (no gaps found)."""

    persona: Literal["gap_auditor"] = "gap_auditor"
    agent_id: str
    gaps: list[Gap] = Field(default_factory=list)


class ConclusionReconstructorContribution(BaseModel):
    """Reconstructs the missing CONCLUSIONS sentence direction/strength only —
    never a yes/no/maybe label (see P6 in the design doc: PubMedQA abstracts in
    this corpus never contain the authors' actual conclusion sentence)."""

    persona: Literal["conclusion_reconstructor"] = "conclusion_reconstructor"
    agent_id: str
    reconstructed_conclusion: Claim | None = None
    direction: Direction
    strength: ConclusionStrength


PanelContribution = Annotated[
    Union[
        QuestionFramerContribution,
        FindingsAuditorContribution,
        GapAuditorContribution,
        ConclusionReconstructorContribution,
    ],
    Field(discriminator="persona"),
]


# --- Supervisor / Moderator output -------------------------------------------


class LedgerConflict(BaseModel):
    """A disagreement between two or more R1 contributions, cited by sentence."""

    description: str
    agent_ids: list[str] = Field(default_factory=list)
    sentence_ids: list[str] = Field(default_factory=list)


class EvidenceLedger(BaseModel):
    """The typed blackboard the Supervisor/Moderator writes and Round 2 reads.

    Fields are optional because a persona's contribution may itself have been
    entirely dropped by the verification gate (see verify.py); an empty ledger
    field just means Round 2 gets a ``round_instructions`` entry asking for it.
    """

    target_population: Claim | None = None
    target_exposure: Claim | None = None
    target_outcome: Claim | None = None
    question_type: QuestionType | None = None
    primary_endpoint: Claim | None = None
    direction: Direction | None = None
    significance: Claim | None = None
    effect_magnitude: Claim | None = None
    reconstructed_conclusion: Claim | None = None
    conclusion_direction: Direction | None = None
    conclusion_strength: ConclusionStrength | None = None
    gaps: list[Gap] = Field(default_factory=list)
    conflicts: list[LedgerConflict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    round_instructions: list[str] = Field(default_factory=list)


# --- Round 2: cooperative debate on the ledger -------------------------------


class RoundTwoOpinion(BaseModel):
    """Cooperative (not adversarial) R2 output: no "attack the peers" framing."""

    agent_id: str
    label: Literal["yes", "no", "maybe"]
    rationale: str
    citations: list[str] = Field(default_factory=list)
    complement: str | None = Field(
        default=None, description="A fact from the ledger the panel omitted."
    )
    self_audit: str | None = Field(
        default=None, description="One plausible failure mode of this agent's own reading."
    )


# --- Supervisor / Director verdict -------------------------------------------


class DirectorVerdict(BaseModel):
    question_answered_by_endpoint: bool
    direction_determinate: bool
    findings_statistically_supported: bool
    conclusion_would_be_hedged: bool
    direction: Direction
    label: Literal["yes", "no", "maybe"]
    rationale: str
    citations: list[str] = Field(default_factory=list)


# --- Final per-case result ----------------------------------------------------


class SLDResult(BaseModel):
    """Everything produced for one PubMedQA case, kept for audit and ablation."""

    case_id: str
    question: str
    expected_label: str | None = None
    question_type: QuestionType | None = None
    sentences: dict[str, str] = Field(default_factory=dict)

    panel_r1_raw: list[PanelContribution] = Field(default_factory=list)
    panel_r1_verified: list[PanelContribution] = Field(default_factory=list)
    grounding_score_r1: float = 1.0
    dropped_claims_r1: list[str] = Field(default_factory=list)

    ledger: EvidenceLedger | None = None

    panel_r2_raw: list[RoundTwoOpinion] = Field(default_factory=list)
    panel_r2_verified: list[RoundTwoOpinion] = Field(default_factory=list)
    grounding_score_r2: float = 1.0
    dropped_claims_r2: list[str] = Field(default_factory=list)

    director_verdict: DirectorVerdict | None = None
    predicted_label: Literal["yes", "no", "maybe"] | None = None
    rule_name: str | None = None
