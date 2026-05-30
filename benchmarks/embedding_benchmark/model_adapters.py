from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from embedding_benchmark.common import ModelSpec


@dataclass(frozen=True)
class TextRecord:
    chunk_id: str
    title: str
    text: str


class EmbeddingAdapter(ABC):
    def __init__(self, spec: ModelSpec, *, device: str, precision: str) -> None:
        self.spec = spec
        self.device = device
        self.precision = precision

    @abstractmethod
    def encode_documents(self, records: list[TextRecord], *, batch_size: int) -> np.ndarray:
        """Return a float32 matrix with shape [len(records), dim]."""

    @abstractmethod
    def encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        """Return a float32 matrix with shape [len(queries), dim]."""

    def close(self) -> None:
        pass


class SentenceTransformerAdapter(EmbeddingAdapter):
    def __init__(self, spec: ModelSpec, *, device: str, precision: str) -> None:
        super().__init__(spec, device=device, precision=precision)
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {
            "device": device,
            "trust_remote_code": bool(spec.config.get("trust_remote_code", False)),
        }
        self.model = SentenceTransformer(str(spec.config["model"]), **kwargs)
        max_length = spec.config.get("max_length")
        if max_length is not None:
            self.model.max_seq_length = int(max_length)

        if precision == "fp16":
            self.model = self.model.half()
        elif precision not in {"fp32", "bf16"}:
            raise RuntimeError(f"Unsupported precision for SentenceTransformerAdapter: {precision}")

    def encode_documents(self, records: list[TextRecord], *, batch_size: int) -> np.ndarray:
        prefix = str(self.spec.config.get("document_prefix") or "")
        texts = [prefix + _document_text(record) for record in records]
        return self._encode(texts, batch_size=batch_size)

    def encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        prefix = str(self.spec.config.get("query_prefix") or "")
        texts = [prefix + query for query in queries]
        return self._encode(texts, batch_size=batch_size)

    def _encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=bool(self.spec.config.get("normalize", True)),
            show_progress_bar=False,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        _validate_embeddings(embeddings, expected_rows=len(texts), expected_dim=self.spec.dim, label=self.spec.slug)
        return embeddings

    def close(self) -> None:
        del self.model


class MedCPTAdapter(EmbeddingAdapter):
    def __init__(self, spec: ModelSpec, *, device: str, precision: str) -> None:
        super().__init__(spec, device=device, precision=precision)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.document_tokenizer = AutoTokenizer.from_pretrained(str(spec.config["document_model"]))
        self.query_tokenizer = AutoTokenizer.from_pretrained(str(spec.config["query_model"]))
        self.document_model = AutoModel.from_pretrained(str(spec.config["document_model"])).to(device)
        self.query_model = AutoModel.from_pretrained(str(spec.config["query_model"])).to(device)
        self.document_model.eval()
        self.query_model.eval()
        if precision == "fp16":
            self.document_model.half()
            self.query_model.half()
        elif precision != "fp32":
            raise RuntimeError(f"Unsupported precision for MedCPTAdapter: {precision}")

    def encode_documents(self, records: list[TextRecord], *, batch_size: int) -> np.ndarray:
        pairs = [[record.title, record.text] for record in records]
        return self._encode(
            tokenizer=self.document_tokenizer,
            model=self.document_model,
            inputs=pairs,
            max_length=int(self.spec.config.get("max_length") or 512),
            batch_size=batch_size,
            label=f"{self.spec.slug}:documents",
        )

    def encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        return self._encode(
            tokenizer=self.query_tokenizer,
            model=self.query_model,
            inputs=queries,
            max_length=int(self.spec.config.get("query_max_length") or 64),
            batch_size=batch_size,
            label=f"{self.spec.slug}:queries",
        )

    def _encode(
        self,
        *,
        tokenizer: Any,
        model: Any,
        inputs: list[str] | list[list[str]],
        max_length: int,
        batch_size: int,
        label: str,
    ) -> np.ndarray:
        arrays = []
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.torch.no_grad():
                output = model(**encoded).last_hidden_state[:, 0, :]
            if bool(self.spec.config.get("normalize", True)):
                output = self.torch.nn.functional.normalize(output, p=2, dim=1)
            arrays.append(output.detach().float().cpu().numpy())
        embeddings = np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
        _validate_embeddings(embeddings, expected_rows=len(inputs), expected_dim=self.spec.dim, label=label)
        return embeddings

    def close(self) -> None:
        del self.document_model
        del self.query_model


def create_adapter(spec: ModelSpec, *, device: str, precision: str) -> EmbeddingAdapter:
    if spec.family == "medcpt":
        return MedCPTAdapter(spec, device=device, precision=precision)
    if spec.family == "sentence_transformer":
        return SentenceTransformerAdapter(spec, device=device, precision=precision)
    raise RuntimeError(f"Unsupported model family for {spec.slug}: {spec.family}")


def _document_text(record: TextRecord) -> str:
    if record.title:
        return f"{record.title}\n\n{record.text}"
    return record.text


def _validate_embeddings(embeddings: np.ndarray, *, expected_rows: int, expected_dim: int, label: str) -> None:
    if embeddings.shape != (expected_rows, expected_dim):
        raise RuntimeError(f"{label}: expected embeddings shape {(expected_rows, expected_dim)}, got {embeddings.shape}.")
    if not np.isfinite(embeddings).all():
        raise RuntimeError(f"{label}: embeddings contain NaN or infinity.")
    zero_rows = np.linalg.norm(embeddings, axis=1) == 0
    if bool(zero_rows.any()):
        raise RuntimeError(f"{label}: embeddings contain {int(zero_rows.sum())} zero vectors.")

