"""Pydantic models for multi-agent clinical debate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SupervisorModerationOutput(BaseModel):
    """Supervisor output for moderating peer opinions into next-round instructions."""

    agreements: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    round_instructions: list[str] = Field(default_factory=list)


class SupervisorDirectorOutput(BaseModel):
    """Supervisor director output with final label decision (PubMedQA-compatible)."""

    final_label: Literal["yes", "no", "maybe"]
    consensus_type: Literal["consensus", "differential", "escalation"]
    rationale: str = Field(default="")


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
