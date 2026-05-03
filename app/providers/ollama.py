import httpx

from app.providers.base import ProviderError, ProviderUnavailableError
from app.schemas import ChatMessage, ChatResponse


class OllamaProvider:
    def __init__(self, base_url: str, timeout: float) -> None:
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
