from typing import Literal
from typing import Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default=get_settings().default_model, min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class Citation(BaseModel):
    id: str
    title: str
    source: str
    score: float
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievalInfo(BaseModel):
    enabled: bool
    status: Literal["skipped", "grounded", "no_sources"]
    provider: str
    query: str
    documents_count: int


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool
    citations: list[Citation] = Field(default_factory=list)
    retrieval: Optional[RetrievalInfo] = None
