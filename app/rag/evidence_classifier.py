from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Any

from app.rag.models import RetrievedDocument


logger = logging.getLogger(__name__)

LABELS = ("yes", "no", "maybe")
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class EvidenceClassifierPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    rationale: str
    model_path: str


class EvidenceClassifier:
    """Lazy DeBERTa yes/no/maybe classifier for PubMedQA-style evidence decisions."""

    def __init__(
        self,
        *,
        enabled: bool,
        model_path: Path,
        temperature_path: Path | None = None,
        max_length: int = 512,
        max_sources: int = 1,
        device: str = "auto",
        min_macro_f1: float = 0.40,
        min_per_label_accuracy: float = 0.10,
    ) -> None:
        self.enabled = enabled
        self.model_path = model_path
        self.temperature_path = temperature_path
        self.max_length = max_length
        self.max_sources = max(max_sources, 1)
        self.device_name = device
        self.min_macro_f1 = min_macro_f1
        self.min_per_label_accuracy = min_per_label_accuracy
        self._loaded = False
        self._available = False
        self._load_error = ""
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._device: Any = None
        self._temperature = 1.0

    @property
    def available(self) -> bool:
        if not self.enabled:
            return False
        if not self._loaded:
            self._load()
        return self._available

    @property
    def load_error(self) -> str:
        if not self._loaded:
            self._load()
        return self._load_error

    def predict(
        self,
        *,
        question: str,
        source_documents: list[RetrievedDocument],
    ) -> EvidenceClassifierPrediction | None:
        if not self.available or not source_documents:
            return None

        evidence = _evidence_text(source_documents[: self.max_sources])
        if not question.strip() or not evidence.strip():
            return None

        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        encoded = self._tokenizer(
            question,
            evidence,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.no_grad():
            logits = self._model(**encoded).logits[0] / max(self._temperature, 1e-6)
            probabilities_tensor = self._torch.softmax(logits, dim=-1).detach().cpu()

        probabilities = {
            _label_for_index(index): float(probabilities_tensor[index])
            for index in range(min(len(probabilities_tensor), len(LABELS)))
        }
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        rationale = f"DeBERTa evidence classifier predicted {label} with confidence {confidence:.2f} [S1]."
        return EvidenceClassifierPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            rationale=rationale,
            model_path=str(self.model_path),
        )

    def _load(self) -> None:
        self._loaded = True
        if not self.enabled:
            self._load_error = "Evidence classifier disabled."
            return
        if not self.model_path.exists():
            self._load_error = f"Evidence classifier model path does not exist: {self.model_path}"
            logger.info(self._load_error)
            return
        quality_error = _quality_gate_error(
            metrics_path=self.model_path / "dev_metrics_calibrated.json",
            min_macro_f1=self.min_macro_f1,
            min_per_label_accuracy=self.min_per_label_accuracy,
        )
        if quality_error:
            self._load_error = quality_error
            logger.warning(self._load_error)
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - exercised only when optional deps are absent
            self._load_error = f"Evidence classifier dependencies are unavailable: {exc}"
            logger.warning(self._load_error)
            return

        try:
            self._torch = torch
            self._device = _resolve_device(torch, self.device_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self._model.to(self._device)
            self._model.eval()
            self._temperature = _load_temperature(self.temperature_path or (self.model_path / "calibration.json"))
            self._available = True
            logger.info(
                "Loaded evidence classifier model=%s device=%s temperature=%.4f",
                self.model_path,
                self._device,
                self._temperature,
            )
        except Exception as exc:  # pragma: no cover - model-load failures are environment dependent
            self._load_error = f"Evidence classifier failed to load: {exc}"
            logger.warning(self._load_error, exc_info=True)


def _resolve_device(torch: Any, requested: str) -> Any:
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_temperature(path: Path) -> float:
    if not path.exists():
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(float(data.get("temperature", 1.0)), 1e-6)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Could not read evidence classifier calibration from %s.", path, exc_info=True)
        return 1.0


def _quality_gate_error(
    *,
    metrics_path: Path,
    min_macro_f1: float,
    min_per_label_accuracy: float,
) -> str:
    if min_macro_f1 <= 0 and min_per_label_accuracy <= 0:
        return ""
    if not metrics_path.exists():
        return ""
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"Evidence classifier quality gate could not read metrics: {metrics_path}"

    macro_f1 = float(metrics.get("macro_f1", 0.0))
    if macro_f1 < min_macro_f1:
        return (
            "Evidence classifier disabled by quality gate: "
            f"dev macro_f1={macro_f1:.4f} < required {min_macro_f1:.4f}."
        )

    per_label = metrics.get("per_label", {})
    for label, label_metrics in per_label.items():
        support = int(label_metrics.get("support", 0))
        accuracy = float(label_metrics.get("accuracy", 0.0))
        if support > 0 and accuracy < min_per_label_accuracy:
            return (
                "Evidence classifier disabled by quality gate: "
                f"dev {label} accuracy={accuracy:.4f} < required {min_per_label_accuracy:.4f}."
            )
    return ""


def _label_for_index(index: int) -> str:
    return ID_TO_LABEL.get(index, f"label_{index}")


def _evidence_text(source_documents: list[RetrievedDocument]) -> str:
    parts = []
    for document in source_documents:
        content = re.sub(r"\s+", " ", document.content).strip()
        context_marker = "Abstract context:"
        if context_marker in content:
            content = content.split(context_marker, 1)[1].strip()
        parts.append(content)
    return " ".join(parts)
