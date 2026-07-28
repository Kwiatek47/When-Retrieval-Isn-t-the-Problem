"""Pydantic models for multi-agent clinical debate."""

from __future__ import annotations

from pydantic import BaseModel, Field


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
    information_requests: list[str] = Field(default_factory=list)
    """Questions this agent needs answered by peers holding other evidence.

    Only populated under information asymmetry, where an agent sees a single
    segment of the case and must ask for what it cannot see (InfoNav). Stays
    empty in the shared-context architectures.
    """


class AgentRoundOpinion(BaseModel):
    """Opinion wrapped with orchestration metadata (agent identity + round)."""

    agent_id: str
    persona: str
    round: int = Field(..., ge=1)
    opinion: ClinicalOpinion
    segment_id: str = Field(default="")
    """Which evidence segment the agent could see, "" when it saw everything."""


class DebateResult(BaseModel):
    """Full debate history and final opinions after N rounds (no supervisor)."""

    patient_case: str
    rounds: list[list[AgentRoundOpinion]]
    final_opinions: list[AgentRoundOpinion]
