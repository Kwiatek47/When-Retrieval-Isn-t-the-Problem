from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal, Protocol

import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


SYSTEM_PROMPT = (
    "You are a professional medical assistant. Provide concise, helpful information. "
    "Always include a disclaimer that this is not medical advice."
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "medgemma")


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL, min_length=1)
    messages: list[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    model: str
    message: ChatMessage
    done: bool


class LLMProvider(Protocol):
    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
    ) -> ChatResponse:
        ...


class ProviderError(Exception):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class OllamaProvider:
    def __init__(self, base_url: str = OLLAMA_BASE_URL, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def chat(
        self,
        *,
        model: str,
        messages: list[ChatMessage],
        temperature: float,
    ) -> ChatResponse:
        payload = {
            "model": model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ProviderUnavailableError(f"Ollama is not reachable at {self.base_url}.") from exc
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise ProviderError(f"Ollama returned an error: {detail}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError("Ollama returned an invalid JSON response.") from exc

        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        if not content:
            raise ProviderError("Ollama returned an empty response.")

        return ChatResponse(
            model=str(data.get("model") or model),
            message=ChatMessage(role="assistant", content=content),
            done=bool(data.get("done", True)),
        )


provider = OllamaProvider()
app = FastAPI(title="MedChat", version="0.1.0")


def get_llm_provider() -> LLMProvider:
    return provider


def build_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    user_visible_messages = [message for message in messages if message.role != "system"]
    return [ChatMessage(role="system", content=SYSTEM_PROMPT), *user_visible_messages]


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    llm_provider: Annotated[LLMProvider, Depends(get_llm_provider)],
) -> ChatResponse:
    try:
        return await llm_provider.chat(
            model=request.model,
            messages=build_messages(request.messages),
            temperature=request.temperature,
        )
    except ProviderUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
