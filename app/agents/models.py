"""Pydantic models for multi-agent clinical debate."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClinicalOpinion(BaseModel):
    """Structured clinical opinion returned by every debate agent."""

    top_1_diagnosis: str = Field(..., min_length=1)
    top_3_differential_diagnoses: list[str] = Field(..., min_length=1, max_length=3)
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    required_further_tests: list[str] = Field(default_factory=list)
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    sources_used: list[str] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    missing_information: str = Field(default="")


class AgentRoundOpinion(BaseModel):
    """Opinion wrapped with orchestration metadata (agent identity + round)."""

    agent_id: str
    persona: str
    round: int = Field(..., ge=1)
    opinion: ClinicalOpinion


class DebateResult(BaseModel):
    """Full debate history and final opinions after N rounds (no supervisor)."""

    patient_case: str
    rounds: list[list[AgentRoundOpinion]]
    final_opinions: list[AgentRoundOpinion]
