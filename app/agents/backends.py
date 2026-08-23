"""Inference and evidence-hint backends for clinical debate agents."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

from app.agents.models import ClinicalOpinion, SafetyOpinion
from app.rag.models import RetrievedDocument
from app.schemas import ChatMessage


@dataclass(frozen=True)
class EvidenceHint:
    """Optional classifier signal (e.g. BioLinkBERT yes/no/maybe) for prompts."""

    label: str
    confidence: float
    model_path: str = ""
    probabilities: dict[str, float] | None = None


class InferenceBackend(Protocol):
    """Pluggable text completion used by ClinicalAgent (swap mock / Ollama / vLLM)."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        num_predict: int | None = None,
    ) -> str:
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
    BioLinkBERT / EvidenceClassifier yes/no/maybe hint for PubMedQA-style cases.

    This is a sequence classifier, not a generative model: it supplies a label+confidence
    that debate agents can see in the prompt (and that the eval script can use as baseline).
    """

    def __init__(self, classifier: Any | None = None) -> None:
        self._classifier = classifier
        self._question: str | None = None
        self._documents: list[RetrievedDocument] | None = None
        self.last_hint: EvidenceHint | None = None

    @property
    def available(self) -> bool:
        return bool(self._classifier is not None and getattr(self._classifier, "available", False))

    @property
    def load_error(self) -> str:
        if self._classifier is None:
            return "No classifier attached."
        return str(getattr(self._classifier, "load_error", "") or "")

    @property
    def model_path(self) -> str:
        if self._classifier is None:
            return ""
        return str(getattr(self._classifier, "model_path", "") or "")

    def set_case(self, *, question: str, documents: list[RetrievedDocument]) -> None:
        """Bind the current PubMedQA question + evidence docs (preferred over parsing)."""
        self._question = question.strip()
        self._documents = list(documents)

    def clear_case(self) -> None:
        self._question = None
        self._documents = None

    def predict_current(self) -> EvidenceHint | None:
        """Run classifier on the case bound via set_case / last get_hint parse."""
        if self._classifier is None or not getattr(self._classifier, "available", False):
            return None
        question = self._question or ""
        documents = self._documents or []
        if not question or not documents:
            return None
        prediction = self._classifier.predict(question=question, source_documents=documents)
        if prediction is None:
            return None
        hint = EvidenceHint(
            label=prediction.label,
            confidence=float(prediction.confidence),
            model_path=str(prediction.model_path),
            probabilities=dict(prediction.probabilities),
        )
        self.last_hint = hint
        return hint

    def get_hint(self, patient_case: str) -> EvidenceHint | None:
        if self._classifier is None:
            return None
        if not getattr(self._classifier, "available", False):
            return None
        if not self._question or not self._documents:
            parsed_question, parsed_docs = parse_pubmedqa_patient_case(patient_case)
            if parsed_question and parsed_docs:
                self._question = parsed_question
                self._documents = parsed_docs
        return self.predict_current()


def build_biolinkbert_hint_from_settings() -> BioLinkBERTHintProvider:
    """Construct hint provider from RAG_EVIDENCE_CLASSIFIER_* settings (seed47 path in .env)."""
    from app.core.config import get_settings
    from app.rag.evidence_classifier import EvidenceClassifier

    settings = get_settings()
    classifier = EvidenceClassifier(
        enabled=True,
        model_path=settings.rag_evidence_classifier_model_path,
        temperature_path=settings.rag_evidence_classifier_temperature_path,
        max_length=settings.rag_evidence_classifier_max_length,
        max_sources=settings.rag_evidence_judge_max_sources,
        device=settings.rag_evidence_classifier_device,
        min_macro_f1=settings.rag_evidence_classifier_min_macro_f1,
        min_per_label_accuracy=settings.rag_evidence_classifier_min_per_label_accuracy,
    )
    return BioLinkBERTHintProvider(classifier)


def parse_pubmedqa_patient_case(patient_case: str) -> tuple[str, list[RetrievedDocument]]:
    """Extract question + evidence documents from the eval script case format."""
    text = patient_case.strip()
    question = ""
    evidence = ""
    question_match = re.search(
        r"RESEARCH QUESTION:\s*(.*?)\s*EVIDENCE:\s*(.*)\Z",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if question_match:
        question = question_match.group(1).strip()
        evidence = question_match.group(2).strip()
    else:
        evidence = text

    documents: list[RetrievedDocument] = []
    blocks = re.split(r"(?=\nSOURCE |\ASOURCE )", evidence)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        source_match = re.match(r"SOURCE\s+(\S+)\s*(.*)\Z", block, flags=re.DOTALL)
        if not source_match:
            documents.append(
                RetrievedDocument(
                    id="evidence-0",
                    title="",
                    content=block,
                    source="pubmedqa",
                    score=1.0,
                )
            )
            continue
        doc_id = source_match.group(1).strip()
        body = source_match.group(2).strip()
        title = ""
        title_match = re.match(r"Title:\s*(.*?)\n(.*)\Z", body, flags=re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            body = title_match.group(2).strip()
        documents.append(
            RetrievedDocument(
                id=doc_id,
                title=title,
                content=body,
                source="pubmedqa",
                score=1.0,
            )
        )
    return question, documents


def hint_as_clinical_opinion(hint: EvidenceHint) -> ClinicalOpinion:
    """Represent a BioLinkBERT prediction as a ClinicalOpinion for majority aggregation."""
    ordered = ["yes", "no", "maybe"]
    if hint.probabilities:
        ordered = sorted(hint.probabilities.keys(), key=lambda label: hint.probabilities.get(label, 0.0), reverse=True)
    return ClinicalOpinion(
        top_1_diagnosis=hint.label,
        top_3_differential_diagnoses=ordered[:3] or [hint.label],
        pros=[f"BioLinkBERT predicted {hint.label} (confidence={hint.confidence:.3f})."],
        cons=["Classifier has no natural-language rationale beyond class probabilities."],
        required_further_tests=[],
        confidence_level=max(0.0, min(1.0, hint.confidence)),
        sources_used=[hint.model_path or "biolinkbert"],
        red_flags=[],
        missing_information="",
    )


class MockInferenceBackend:
    """Deterministic offline backend for demos and unit tests."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.3,
        num_predict: int | None = None,
    ) -> str:
        _ = temperature
        _ = num_predict
        system = next((m.content for m in messages if m.role == "system"), "")
        user = next((m.content for m in messages if m.role == "user"), "")
        agent_id = _extract_between(system, "agent_id=", "\n") or "agent"
        task_mode = _extract_between(system, "task_mode=", "\n") or "clinical"
        has_context = "PEER OPINIONS" in user or "peer opinions" in user.lower()
        if task_mode.strip().lower() == "pubmedqa":
            opinion = _mock_pubmedqa_opinion(
                agent_id=agent_id,
                case_text=user,
                revised=has_context,
            )
        else:
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
        max_retries: int = 2,
    ) -> None:
        self._provider = provider
        self._model = model
        self._temperature = temperature
        self._max_retries = max(0, int(max_retries))
        self.base_url = str(getattr(provider, "base_url", "") or "")

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        num_predict: int | None = None,
    ) -> str:
        temp = self._temperature if temperature is None else temperature
        last_error: Exception | None = None
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            try:
                response = await self._provider.chat(
                    model=self._model,
                    messages=messages,
                    temperature=temp if attempt == 0 else 0.0,
                    num_predict=num_predict,
                )
                content = (response.message.content or "").strip()
                if content:
                    return content
                last_error = RuntimeError("Ollama returned an empty response.")
            except Exception as exc:
                last_error = exc
            if attempt + 1 < attempts:
                await asyncio.sleep(0.35 * (attempt + 1))
        raise RuntimeError(
            f"Ollama complete failed after {attempts} attempt(s): {last_error}"
        ) from last_error


def parse_ollama_base_urls(raw: str | None, *, default: str = "http://localhost:11434") -> list[str]:
    """Parse comma-separated Ollama base URLs; empty input returns ``[default]``."""
    fallback = (default or "http://localhost:11434").strip().rstrip("/")
    if not raw or not str(raw).strip():
        return [fallback]
    urls = [part.strip().rstrip("/") for part in str(raw).split(",") if part.strip()]
    return urls or [fallback]


def sticky_ollama_url(urls: list[str], case_index: int) -> str:
    """Assign a case to a stable Ollama URL (1-based case index)."""
    if not urls:
        raise ValueError("urls must be non-empty")
    idx = max(case_index, 1) - 1
    return urls[idx % len(urls)]


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


def _mock_pubmedqa_opinion(*, agent_id: str, case_text: str, revised: bool) -> ClinicalOpinion:
    """Lightweight heuristic labels for offline PubMedQA debate plumbing tests."""
    text = case_text.lower()
    persona_prior = {
        "generalist": "yes",
        "evidence_skeptic": "maybe",
        "differential_expander": "no",
        "safety_officer": "maybe",
        "uncertainty_advocate": "maybe",
    }
    label = persona_prior.get(agent_id, "maybe")
    if any(token in text for token in ("no significant", "not associated", "failed to", "did not")):
        label = "no" if agent_id != "evidence_skeptic" else "maybe"
    elif any(token in text for token in ("significantly", "effective", "improved", "useful", "valuable")):
        label = "yes" if agent_id != "evidence_skeptic" else ("maybe" if not revised else "yes")
    elif any(token in text for token in ("inconclusive", "unclear", "limited evidence", "mixed")):
        label = "maybe"

    if revised and agent_id == "differential_expander" and "valuable" in text:
        label = "yes"

    return ClinicalOpinion(
        top_1_diagnosis=label,
        top_3_differential_diagnoses=["yes", "no", "maybe"],
        pros=[f"[{agent_id}] Abstract language leans toward '{label}'."],
        cons=[f"[{agent_id}] Mock backend; not a real literature judgment."],
        required_further_tests=["Larger RCT", "External validation"],
        confidence_level=0.62 if revised else 0.48,
        sources_used=["mock://abstract"],
        red_flags=[] if label != "maybe" else ["Residual uncertainty in abstract"],
        missing_information="Full methods / raw data not available in mock mode.",
    )


def _sanitize_clinical_opinion_json(text: str) -> str:
    """Best-effort fixes for common LLM JSON mistakes in pros/cons agent tags."""
    cleaned = text
    # @agent_id's -> [agent_id] (apostrophe breaks many model outputs)
    cleaned = re.sub(r"@(\w+)'s\b", r"[\1]", cleaned)
    cleaned = re.sub(r"@(\w+)\b", r"[\1]", cleaned)
    # Unescaped possessives inside JSON strings (e.g. calprotectin's).
    cleaned = re.sub(r"(\w)'s\b", r"\1s", cleaned)
    return cleaned


def _flatten_pro_con_item(item: Any) -> str:
    """Coerce structured criticism objects into JSON-safe strings."""
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        chunks: list[str] = []
        for key, value in item.items():
            tag = str(key).strip()
            if tag and not tag.startswith("["):
                tag = f"[{tag}]"
            text = str(value or "").strip()
            chunks.append(f"{tag} {text}".strip() if tag else text)
        return " ".join(chunk for chunk in chunks if chunk).strip()
    return str(item or "").strip()


def _coerce_string_list(value: Any, *, flatten_pro_con: bool = False) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = _flatten_pro_con_item(item) if flatten_pro_con else str(item or "").strip()
        if text:
            out.append(text)
    return out


def _recover_partial_clinical_json(text: str) -> dict[str, Any] | None:
    """Extract minimal fields from truncated / broken JSON (common under long debates)."""
    label_match = re.search(
        r'"top_1_diagnosis"\s*:\s*"(yes|no|maybe)"',
        text,
        flags=re.IGNORECASE,
    )
    if not label_match:
        return None
    label = label_match.group(1).lower()
    conf_match = re.search(r'"confidence_level"\s*:\s*([0-9.]+)', text)
    try:
        confidence = float(conf_match.group(1)) if conf_match else 0.5
    except ValueError:
        confidence = 0.5
    concl_match = re.search(
        r'"evidence_conclusiveness"\s*:\s*"(conclusive|inconclusive)"',
        text,
        flags=re.IGNORECASE,
    )
    conclusiveness = (
        concl_match.group(1).lower() if concl_match else "inconclusive"
    )
    return {
        "top_1_diagnosis": label,
        "evidence_conclusiveness": conclusiveness,
        "top_3_differential_diagnoses": ["yes", "no", "maybe"],
        "pros": [],
        "cons": [],
        "required_further_tests": [],
        "confidence_level": confidence,
        "sources_used": ["abstract"],
        "red_flags": [],
        "missing_information": "Recovered from truncated agent JSON.",
    }


def parse_clinical_opinion_json(raw: str) -> ClinicalOpinion:
    """Parse model output into ClinicalOpinion, tolerating fenced/partial JSON."""
    text = _extract_json_object(raw)
    if not text:
        raise ValueError("Empty model response; expected ClinicalOpinion JSON.")
    candidates = [text, _sanitize_clinical_opinion_json(text)]
    last_exc: json.JSONDecodeError | None = None
    data: dict[str, Any] | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                data = parsed
                break
        except json.JSONDecodeError as exc:
            last_exc = exc
    if data is None:
        recovered = _recover_partial_clinical_json(text)
        if recovered is None:
            recovered = _recover_partial_clinical_json(_sanitize_clinical_opinion_json(text))
        if recovered is not None:
            data = recovered
        else:
            assert last_exc is not None
            raise ValueError(f"Invalid ClinicalOpinion JSON: {last_exc}") from last_exc
    normalized = _normalize_clinical_opinion_payload(data)
    try:
        return ClinicalOpinion.model_validate(normalized)
    except Exception:
        recovered = _recover_partial_clinical_json(text)
        if recovered is None:
            recovered = _recover_partial_clinical_json(_sanitize_clinical_opinion_json(text))
        if recovered is not None:
            return ClinicalOpinion.model_validate(_normalize_clinical_opinion_payload(recovered))
        raise


def _extract_json_object(raw: str) -> str:
    """Strip fences / prose and return the first JSON object substring if present."""
    text = (raw or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return match.group(0).strip() if match else text


def _normalize_clinical_opinion_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM omissions so validation does not crash the debate."""
    payload = dict(data)

    # Special case: `safety_officer` may return SafetyOpinion-only JSON.
    # We adapt it into a ClinicalOpinion so downstream debate code and
    # the supervisor safety escalation logic can still read it.
    if "safety_passed" in payload or "immediate_intervention_required" in payload:
        red_flags_detected = payload.get("red_flags_detected") or []
        if isinstance(red_flags_detected, str):
            red_flags_detected = [red_flags_detected]
        if not isinstance(red_flags_detected, list):
            red_flags_detected = []
        red_flags_detected = [str(x).strip() for x in red_flags_detected if str(x).strip()]

        try:
            safety_opinion = SafetyOpinion(
                safety_passed=bool(payload.get("safety_passed")),
                red_flags_detected=red_flags_detected,
                immediate_intervention_required=bool(payload.get("immediate_intervention_required")),
                reasoning=str(payload.get("reasoning") or ""),
            )
        except Exception:
            safety_opinion = None

        if safety_opinion is not None:
            payload["safety_opinion"] = safety_opinion
        # Ensure ClinicalOpinion required shape exists.
        payload.setdefault("top_1_diagnosis", "maybe")
        payload.setdefault("top_3_differential_diagnoses", ["yes", "no", "maybe"])
        payload.setdefault("pros", [])
        payload.setdefault("cons", [])
        payload.setdefault("required_further_tests", [])
        payload.setdefault("confidence_level", 0.0)
        payload.setdefault("sources_used", [])
        # Mirror safety audit flags into ClinicalOpinion.red_flags.
        payload["red_flags"] = red_flags_detected
        payload.setdefault("missing_information", "")
    top = str(payload.get("top_1_diagnosis") or "").strip()
    if not top:
        # Sometimes models put the label only in differentials / free text.
        diffs = payload.get("top_3_differential_diagnoses") or []
        if isinstance(diffs, list) and diffs:
            top = str(diffs[0]).strip()
        elif isinstance(diffs, str) and diffs.strip():
            top = diffs.strip()
        else:
            top = "maybe"
        payload["top_1_diagnosis"] = top

    differentials = payload.get("top_3_differential_diagnoses")
    if isinstance(differentials, str):
        differentials = [differentials]
    if not isinstance(differentials, list):
        differentials = []
    cleaned = [str(item).strip() for item in differentials if str(item).strip()]
    if not cleaned:
        # PubMedQA-friendly default when the model leaves the list empty.
        if top.lower() in {"yes", "no", "maybe"}:
            cleaned = [top.lower(), "yes", "no", "maybe"]
            # unique preserve order
            seen: set[str] = set()
            cleaned = [x for x in cleaned if not (x in seen or seen.add(x))][:3]
        else:
            cleaned = [top]
    payload["top_3_differential_diagnoses"] = cleaned[:3]

    try:
        confidence = float(payload.get("confidence_level", 0.4))
    except (TypeError, ValueError):
        confidence = 0.4
    payload["confidence_level"] = min(max(confidence, 0.0), 1.0)

    for key in ("required_further_tests", "sources_used", "red_flags"):
        value = payload.get(key, [])
        if value is None:
            payload[key] = []
        elif isinstance(value, str):
            payload[key] = [value] if value.strip() else []
        elif not isinstance(value, list):
            payload[key] = []

    # Defense-round chain-of-thought key is prompt-only; strip before validation.
    payload.pop("internal_monologue", None)

    # Defense-round semantic keys → standard pros/cons before coercion.
    if "best_evidence_supporting_my_label" in payload:
        payload["pros"] = payload.pop("best_evidence_supporting_my_label")
    if "explicit_attack_on_opposing_peers" in payload:
        payload["cons"] = payload.pop("explicit_attack_on_opposing_peers")

    payload["pros"] = _coerce_string_list(payload.get("pros"), flatten_pro_con=True)
    payload["cons"] = _coerce_string_list(payload.get("cons"), flatten_pro_con=True)

    if payload.get("missing_information") is None:
        payload["missing_information"] = ""
    else:
        payload["missing_information"] = str(payload.get("missing_information") or "")

    return payload


def fallback_clinical_opinion(*, label: str = "maybe", reason: str = "") -> ClinicalOpinion:
    """Safe opinion used when the LLM repeatedly returns invalid JSON."""
    normalized = label.strip().lower() if label.strip().lower() in {"yes", "no", "maybe"} else "maybe"
    return ClinicalOpinion(
        top_1_diagnosis=normalized,
        top_3_differential_diagnoses=["yes", "no", "maybe"],
        pros=[reason or "Fallback opinion after invalid model JSON."],
        cons=["Model output could not be validated."],
        required_further_tests=[],
        confidence_level=0.25,
        sources_used=["fallback"],
        red_flags=[],
        missing_information="Invalid or incomplete structured output from the model.",
    )
