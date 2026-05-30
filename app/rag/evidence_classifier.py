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
        self._decision_thresholds: dict[str, float] = {}

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
        label = _select_label(probabilities, thresholds=self._decision_thresholds)
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
        reference_paths = _quality_reference_paths(self.model_path)
        quality_error = _quality_gate_error(
            metrics_path=self.model_path / "dev_metrics_calibrated.json",
            fallback_metrics_path=self.model_path / "dev_metrics.json",
            reference_paths=reference_paths,
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
            self._temperature = _load_temperature(
                self.temperature_path or (self.model_path / "calibration.json"),
                reference_paths=reference_paths,
            )
            self._decision_thresholds = _load_decision_thresholds(
                self.model_path / "decision_thresholds.json",
                reference_paths=reference_paths,
            )
            self._available = True
            logger.info(
                "Loaded evidence classifier model=%s device=%s temperature=%.4f thresholds=%s",
                self.model_path,
                self._device,
                self._temperature,
                self._decision_thresholds or "argmax",
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


def _load_temperature(path: Path, *, reference_paths: list[Path] | None = None) -> float:
    if not path.exists():
        return 1.0
    if _is_stale(path, reference_paths or []):
        logger.warning("Ignoring stale evidence classifier calibration from %s.", path)
        return 1.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return max(float(data.get("temperature", 1.0)), 1e-6)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Could not read evidence classifier calibration from %s.", path, exc_info=True)
        return 1.0


def _load_decision_thresholds(path: Path, *, reference_paths: list[Path] | None = None) -> dict[str, float]:
    if not path.exists():
        return {}
    if _is_stale(path, reference_paths or []):
        logger.warning("Ignoring stale evidence classifier decision thresholds from %s.", path)
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_thresholds = data.get("thresholds", data)
        thresholds = {
            label: max(float(raw_thresholds[label]), 1e-6)
            for label in LABELS
            if label in raw_thresholds
        }
        return thresholds if len(thresholds) == len(LABELS) else {}
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("Could not read evidence classifier decision thresholds from %s.", path, exc_info=True)
        return {}


def _quality_gate_error(
    *,
    metrics_path: Path,
    fallback_metrics_path: Path | None = None,
    reference_paths: list[Path] | None = None,
    min_macro_f1: float,
    min_per_label_accuracy: float,
) -> str:
    if min_macro_f1 <= 0 and min_per_label_accuracy <= 0:
        return ""
    selected_metrics_path, stale_error = _select_quality_metrics_path(
        metrics_path=metrics_path,
        fallback_metrics_path=fallback_metrics_path,
        reference_paths=reference_paths or [],
    )
    if stale_error:
        return stale_error
    if selected_metrics_path is None:
        return ""
    try:
        metrics = json.loads(selected_metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"Evidence classifier quality gate could not read metrics: {selected_metrics_path}"

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


def _select_quality_metrics_path(
    *,
    metrics_path: Path,
    fallback_metrics_path: Path | None,
    reference_paths: list[Path],
) -> tuple[Path | None, str]:
    primary_exists = metrics_path.exists()
    fallback_exists = fallback_metrics_path is not None and fallback_metrics_path.exists()
    primary_stale = primary_exists and _is_stale(metrics_path, reference_paths)
    fallback_stale = fallback_exists and _is_stale(fallback_metrics_path, reference_paths)  # type: ignore[arg-type]

    if primary_exists and not primary_stale:
        return metrics_path, ""
    if fallback_exists and not fallback_stale:
        if primary_exists and primary_stale:
            logger.warning(
                "Evidence classifier calibrated metrics are stale: %s; using fresh metrics: %s.",
                metrics_path,
                fallback_metrics_path,
            )
        return fallback_metrics_path, ""  # type: ignore[return-value]
    if primary_stale or fallback_stale:
        stale_paths = [
            str(path)
            for path, stale in ((metrics_path, primary_stale), (fallback_metrics_path, fallback_stale))
            if path is not None and stale
        ]
        return (
            None,
            "Evidence classifier disabled by quality gate: stale metrics "
            f"{', '.join(stale_paths)} are older than the model artifacts.",
        )
    return None, ""


def _quality_reference_paths(model_path: Path) -> list[Path]:
    return [
        model_path / "model.safetensors",
        model_path / "pytorch_model.bin",
        model_path / "dev_metrics.json",
    ]


def _is_stale(path: Path, reference_paths: list[Path]) -> bool:
    if not path.exists():
        return False
    try:
        path_mtime = path.stat().st_mtime
    except OSError:
        return False
    for reference_path in reference_paths:
        if reference_path == path or not reference_path.exists():
            continue
        try:
            if reference_path.stat().st_mtime > path_mtime:
                return True
        except OSError:
            continue
    return False


def _select_label(probabilities: dict[str, float], *, thresholds: dict[str, float]) -> str:
    if not thresholds:
        return max(probabilities, key=probabilities.get)
    return max(
        probabilities,
        key=lambda label: probabilities[label] / max(thresholds.get(label, 1.0), 1e-6),
    )


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
