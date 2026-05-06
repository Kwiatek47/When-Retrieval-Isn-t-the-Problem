from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class DocumentEmbeddingInput(BaseModel):
    text: str = Field(..., min_length=1)
    title: str = ""


class EmbedDocumentsRequest(BaseModel):
    documents: list[DocumentEmbeddingInput] | None = None
    texts: list[str] | None = None

    @model_validator(mode="after")
    def require_one_input_shape(self) -> "EmbedDocumentsRequest":
        if self.documents and self.texts:
            raise ValueError("Provide either documents or texts, not both.")
        if not self.documents and not self.texts:
            raise ValueError("Provide documents or texts.")
        return self

    def as_documents(self) -> list[DocumentEmbeddingInput]:
        if self.documents is not None:
            return self.documents
        return [DocumentEmbeddingInput(text=text) for text in self.texts or []]


class EmbedQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)


class HybridQueryRequest(BaseModel):
    text: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=100)


class EmbedResponse(BaseModel):
    model: str
    encoder: Literal["document", "query"]
    encoder_model: str
    dimension: int
    embeddings: list[list[float]]


class HybridQueryDocument(BaseModel):
    id: str
    title: str
    content: str
    source: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class HybridQueryResponse(BaseModel):
    model: str
    encoder: Literal["query"]
    encoder_model: str
    dimension: int
    collection: str
    vector_name: str
    documents: list[HybridQueryDocument]


class HealthResponse(BaseModel):
    status: str
    model: str
    query_model: str
    document_model: str
    dimension: int
    device: str
