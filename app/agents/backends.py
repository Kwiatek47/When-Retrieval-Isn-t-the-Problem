"""Inference and evidence-hint backends for clinical debate agents."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from app.agents.models import ClinicalOpinion
from app.schemas import ChatMessage


@dataclass(frozen=True)
class EvidenceHint:
    """Optional classifier signal (e.g. BioLinkBERT yes/no/maybe) for prompts."""

    label: str
    confidence: float
    model_path: str = ""


class InferenceBackend(Protocol):
    """Pluggable text completion used by ClinicalAgent (swap mock / Ollama / vLLM)."""

    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
        ...


class EvidenceHintProvider(Protocol):
    """Optional evidence signal injected into prompts; not a ClinicalOpinion generator."""

    def get_hint(self, patient_case: str) -> EvidenceHint | None:
        ...


class NullEvidenceHint:
    """No-op hint provider (default)."""

    def get_hint(self, patient_case: str) -> EvidenceHint | None:
        return None


class BioLinkBERTHintProvider:
    """
    Placeholder for BioLinkBERT-large evidence classifier hints.

    BioLinkBERT in this project is a yes/no/maybe sequence classifier, not a
    generative model. Wire an existing EvidenceClassifier instance when ready;
    until then this returns None so debate still runs on the LLM/mock backend.
    """

    def __init__(self, classifier: Any | None = None) -> None:
        self._classifier = classifier

    def get_hint(self, patient_case: str) -> EvidenceHint | None:
        if self._classifier is None:
            return None
        if not getattr(self._classifier, "available", False):
            return None
        # Placeholder: EvidenceClassifier.predict expects question + source_documents.
        # Full RAG wiring is out of scope for this skeleton.
        _ = patient_case
        return None


class MockInferenceBackend:
    """Deterministic offline backend for demos and unit tests."""

    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
        _ = temperature
        system = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in messages if m.role == "user"), "")
        agent_id = _extract_between(system, "agent_id=", "\n") or "agent"
        has_context = "PEER OPINIONS" in user or "peer opinions" in user.lower()
        opinion = _mock_opinion(agent_id=agent_id, revised=has_context)
        return opinion.model_dump_json()


class OllamaInferenceBackend:
    """LLM backend wrapping the project's Ollama LLMProvider."""

    def __init__(
        self,
        provider: Any,
        *,
        model: str,
        temperature: float = 0.3,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature

    async def complete(self, messages: list[ChatMessage], *, temperature: float | None = None) -> str:
        response = await self._provider.chat(
            model=self._model,
            messages=messages,
            temperature=self._temperature if temperature is None else temperature,
        )
        return response.message.content


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    rest = text.split(start, 1)[1]
    return rest.split(end, 1)[0].strip()


def _mock_opinion(*, agent_id: str, revised: bool) -> ClinicalOpinion:
    persona_bias = {
        "generalist": "Community-acquired pneumonia",
        "evidence_skeptic": "Viral lower respiratory tract infection",
        "differential_expander": "Pulmonary embolism",
        "safety_officer": "Acute coronary syndrome (atypical)",
    }
    top = persona_bias.get(agent_id, "Undifferentiated acute illness")
    suffix = " (revised after peer critique)" if revised else ""
    differentials = [
        top,
        "Acute bronchitis",
        "Heart failure exacerbation",
    ][:3]
    return ClinicalOpinion(
        top_1_diagnosis=f"{top}{suffix}",
        top_3_differential_diagnoses=differentials,
        pros=[
            f"[{agent_id}] Presentation is consistent with {top}.",
            "Fever and productive cough support an infectious process.",
        ],
        cons=[
            f"[{agent_id}] Incomplete vitals and imaging limit certainty.",
            "Cannot exclude cardiac or thromboembolic causes yet.",
        ],
        required_further_tests=[
            "Chest X-ray",
            "CBC with differential",
            "ECG",
            "D-dimer if PE pretest probability is intermediate+",
        ],
        confidence_level=0.55 if revised else 0.45,
        sources_used=["mock://local-guideline", "mock://peer-debate" if revised else "mock://case-only"],
        red_flags=[
            "Hypoxia / respiratory distress",
            "Chest pain with diaphoresis",
            "Hemodynamic instability",
        ],
        missing_information=(
            "Need SpO2, BP, heart rate, chest imaging, and medication list "
            "before committing to a final working diagnosis."
        ),
    )


def parse_clinical_opinion_json(raw: str) -> ClinicalOpinion:
    """Parse model output into ClinicalOpinion, tolerating fenced JSON."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    return ClinicalOpinion.model_validate(data)
