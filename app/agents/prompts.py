"""Prompt builders for multi-agent clinical debate (inject point for LLM / BioLinkBERT hint)."""

from __future__ import annotations

import json

from app.agents.backends import EvidenceHint
from app.agents.models import ClinicalOpinion
from app.schemas import ChatMessage

CLINICAL_OPINION_SCHEMA = """
Return ONLY a single JSON object with exactly these fields:
- top_1_diagnosis (string)
- top_3_differential_diagnoses (array of 1-3 strings)
- pros (array of strings)
- cons (array of strings)
- required_further_tests (array of strings)
- confidence_level (number between 0.0 and 1.0)
- sources_used (array of strings)
- red_flags (array of strings)
- missing_information (string)
No markdown fences, no commentary outside JSON.
""".strip()


PERSONA_INSTRUCTIONS: dict[str, str] = {
    "generalist": (
        "You are a broad clinical generalist. Prioritize the most likely common diagnoses "
        "and a practical initial workup."
    ),
    "evidence_skeptic": (
        "You are an evidence-skeptical clinician. Challenge weak causal leaps, demand "
        "stronger supporting findings, and keep confidence conservative."
    ),
    "differential_expander": (
        "You expand the differential. Explicitly consider less common but plausible "
        "and serious alternatives that others may underweight."
    ),
    "safety_officer": (
        "You focus on patient safety: life-threatening red flags, can't-miss diagnoses, "
        "and urgent tests that rule out catastrophic causes."
    ),
}


def build_messages(
    *,
    agent_id: str,
    persona: str,
    patient_case: str,
    context: list[ClinicalOpinion] | None = None,
    evidence_hint: EvidenceHint | None = None,
    repair: bool = False,
) -> list[ChatMessage]:
    """Build chat messages for independent (round 1) or critique (round 2+) opinion generation."""
    persona_text = PERSONA_INSTRUCTIONS.get(persona, PERSONA_INSTRUCTIONS["generalist"])
    system = (
        f"You are clinical debate agent `{agent_id}` with persona `{persona}`.\n"
        f"agent_id={agent_id}\n"
        f"{persona_text}\n\n"
        f"{CLINICAL_OPINION_SCHEMA}"
    )
    if repair:
        system += (
            "\n\nYour previous reply was invalid JSON. "
            "Respond again with ONLY a valid JSON object matching the schema."
        )

    parts = [f"PATIENT CASE:\n{patient_case.strip()}"]

    # Injection point for BioLinkBERT / classifier signal (not a substitute for LLM reasoning).
    if evidence_hint is not None:
        parts.append(
            "EVIDENCE CLASSIFIER HINT (BioLinkBERT-style yes/no/maybe signal; use critically):\n"
            f"- label: {evidence_hint.label}\n"
            f"- confidence: {evidence_hint.confidence:.3f}\n"
            f"- model: {evidence_hint.model_path or 'unspecified'}"
        )

    if context:
        serialized = [
            json.loads(opinion.model_dump_json()) for opinion in context
        ]
        parts.append(
            "PEER OPINIONS FROM THE PREVIOUS ROUND (critique weak arguments, "
            "update hypotheses, and resolve contradictions where possible):\n"
            f"{json.dumps(serialized, ensure_ascii=False, indent=2)}"
        )
        parts.append(
            "Produce an UPDATED ClinicalOpinion that reflects what you accept, "
            "reject, or still find uncertain after reviewing peers."
        )
    else:
        parts.append(
            "This is an independent first-round opinion. Do not assume peer input. "
            "Reason only from the patient case (and classifier hint if present)."
        )

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content="\n\n".join(parts)),
    ]
