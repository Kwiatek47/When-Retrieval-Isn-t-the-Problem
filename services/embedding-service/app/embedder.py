from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class DocumentInput:
    title: str
    text: str


class MedCPTEmbedder:
    """MedCPT embedding implementation for biomedical dense retrieval."""

    def __init__(
        self,
        *,
        model_name: str,
        query_model_name: str,
        document_model_name: str,
        dimension: int,
        query_max_length: int,
        document_max_length: int,
        batch_size: int,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.query_model_name = query_model_name
        self.document_model_name = document_model_name
        self.dimension = dimension
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.query_tokenizer = AutoTokenizer.from_pretrained(query_model_name)
        self.query_model = AutoModel.from_pretrained(query_model_name).to(self.device)
        self.query_model.eval()

        self.document_tokenizer = AutoTokenizer.from_pretrained(document_model_name)
        self.document_model = AutoModel.from_pretrained(document_model_name).to(self.device)
        self.document_model.eval()

    def embed_documents(self, documents: list[DocumentInput]) -> list[list[float]]:
        article_pairs = [[document.title, document.text] for document in documents]
        embeddings: list[list[float]] = []
        for start in range(0, len(article_pairs), self.batch_size):
            embeddings.extend(
                self._embed(
                    tokenizer=self.document_tokenizer,
                    model=self.document_model,
                    inputs=article_pairs[start : start + self.batch_size],
                    max_length=self.document_max_length,
                )
            )
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed(
            tokenizer=self.query_tokenizer,
            model=self.query_model,
            inputs=[text],
            max_length=self.query_max_length,
        )[0]

    def _embed(
        self,
        *,
        tokenizer: AutoTokenizer,
        model: AutoModel,
        inputs: list[str] | list[list[str]],
        max_length: int,
    ) -> list[list[float]]:
        encoded = tokenizer(
            inputs,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            embeddings = model(**encoded).last_hidden_state[:, 0, :]

        if embeddings.shape[1] != self.dimension:
            raise RuntimeError(
                f"Expected embedding dimension {self.dimension}, got {embeddings.shape[1]}."
            )

        return embeddings.cpu().tolist()
