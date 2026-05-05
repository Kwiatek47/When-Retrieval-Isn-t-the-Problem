from typing import Literal

from pydantic import BaseModel, Field

from app.core.config import get_settings


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default=get_settings().default_model, min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    prompt_version: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool
    request_id: str = ""
    prompt_version: str = ""
    timestamp: str = ""
    latency_ms: int = Field(default=0, ge=0)


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)
    model: str | None = Field(default=None, min_length=1)
    prompt_version: str | None = Field(default=None, min_length=1)


class FeedbackResponse(BaseModel):
    ok: bool = True
