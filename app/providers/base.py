from typing import Protocol

from app.schemas import ChatMessage, ChatResponse


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
