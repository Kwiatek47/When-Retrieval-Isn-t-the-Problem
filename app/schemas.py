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


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool
