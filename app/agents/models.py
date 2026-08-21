"""Pydantic models for multi-agent clinical debate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SharedDebateReport(BaseModel):
    """MedAgents-style shared report produced by the supervisor moderator."""

    primary_endpoint_result: str = Field(
        default="",
        description="What the abstract's primary endpoint/result actually showed.",
    )
    author_conclusion: Literal["yes", "no", "maybe", "unclear"] = "unclear"
    residual_uncertainty: list[str] = Field(default_factory=list)
    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    round_instructions: list[str] = Field(default_factory=list)


class SupervisorModerationOutput(BaseModel):
    """Supervisor output for moderating peer opinions into next-round instructions."""

    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    round_instructions: list[str] = Field(default_factory=list)
    # Optional MedAgents-style shared report fields (filled when available).
    primary_endpoint_result: str = Field(default="")
    author_conclusion: Literal["yes", "no", "maybe", "unclear"] = "unclear"
    residual_uncertainty: list[str] = Field(default_factory=list)

    def as_shared_report(self) -> SharedDebateReport:
        return SharedDebateReport(
            primary_endpoint_result=self.primary_endpoint_result,
            author_conclusion=self.author_conclusion,
            residual_uncertainty=list(self.residual_uncertainty),
            agreements=list(self.agreements),
            contradictions=list(self.contradictions),
            round_instructions=list(self.round_instructions),
        )


class SupervisorDirectorOutput(BaseModel):
    """Supervisor director output with final label decision (PubMedQA-compatible)."""

    # --- Chain of Thought Scoring Fields ---
    debate_conflict_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Assess the level of conflict between agents across rounds.",
    )
    conclusiveness_score: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Rate the conclusiveness of the primary evidence from 1 to 10.",
    )
    unresolved_contradictions: list[str] = Field(default_factory=list)
    
    # --- Final Output Fields ---
    final_label: Literal["yes", "no", "maybe"]
    consensus_type: Literal["consensus", "differential", "escalation"]
    rationale: str = Field(default="")
    
    # Maybe-aware gate checklist (director must answer before locking yes/no).
    primary_endpoint_answers_question: bool = True
    findings_decisive_for_question: bool = True
    authors_state_uncertainty: bool = False
    # How completely the abstract settles the research question as written.
    question_coverage: Literal["full", "partial", "none"] = "full"


class RankedHypothesis(BaseModel):
    """One ranked outcome hypothesis for differential-consensus mode."""

    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class ConsensusDecision(BaseModel):
    """Structured final decision beyond a single yes/no/maybe label."""

    mode: Literal["consensus", "differential", "escalation"]
    final_label: Literal["yes", "no", "maybe"] | None = None
    ranked_hypotheses: list[RankedHypothesis] = Field(default_factory=list)
    required_next_steps: list[str] = Field(default_factory=list)
    grounding_score: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_blocked: bool = False
    rationale: str = Field(default="")


class SafetyOpinion(BaseModel):
    """Specialized output for the `safety_officer` persona."""

    safety_passed: bool
    red_flags_detected: list[str] = Field(default_factory=list)
    immediate_intervention_required: bool
    reasoning: str = Field(default="")


class ClinicalOpinion(BaseModel):
    """Structured clinical opinion returned by every debate agent."""

    top_1_diagnosis: str = Field(..., min_length=1)
    evidence_conclusiveness: str = Field(default="")
    top_3_differential_diagnoses: list[str] = Field(..., min_length=1, max_length=3)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    required_further_tests: list[str] = Field(default_factory=list)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    missing_information: str = Field(default="")
    # Optional structured safety assessment; primarily expected from safety_officer.
    safety_opinion: SafetyOpinion | None = None


class AgentRoundOpinion(BaseModel):
    """Opinion wrapped with orchestration metadata (agent identity + round)."""

    agent_id: str
    persona: str
    round: int = Field(..., ge=1)
    opinion: ClinicalOpinion


class DebateResult(BaseModel):
    """Full debate history and final opinions after N rounds (optionally includes supervisor outputs)."""

    patient_case: str
    rounds: list[list[AgentRoundOpinion]]
    final_opinions: list[AgentRoundOpinion]
    supervisor_moderation: list[SupervisorModerationOutput] = Field(default_factory=list)
    supervisor_director_output: SupervisorDirectorOutput | None = None
    shared_report: SharedDebateReport | None = None
    safety_halted: bool = False
    safety_red_flag_reason: str | None = None
    exhausted_without_consensus: bool = False  # telemetry only; never overrides labels
