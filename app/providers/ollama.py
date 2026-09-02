import httpx

from app.core.usage import record_llm_usage
from app.providers.base import ProviderError, ProviderUnavailableError
from app.schemas import ChatMessage, ChatResponse


class OllamaProvider:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        keep_alive: str = "30m",
        num_predict: int = 400,
        num_ctx: int = 2048,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self.num_ctx = num_ctx

    async def ping(self) -> dict:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                data = response.json()
        except httpx.ConnectError as exc:
            raise ProviderUnavailableError(
                f"Cannot connect to Ollama at {self.base_url} (connection refused)."
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnavailableError(
                f"Ollama at {self.base_url} did not respond to /api/tags within 5s."
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama /api/tags failed: {exc}") from exc

        models = [item.get("name") for item in data.get("models", []) if isinstance(item, dict)]
        return {"base_url": self.base_url, "models": [m for m in models if m]}

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
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }

        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
        except httpx.ConnectError as exc:
            record_llm_usage(model=model, failed=True)
            raise ProviderUnavailableError(
                f"Cannot connect to Ollama at {self.base_url} (connection refused). "
                "Is `ollama serve` running and bound to this address? "
                "Check with `curl {url}/api/tags`.".format(url=self.base_url)
            ) from exc
        except httpx.TimeoutException as exc:
            record_llm_usage(model=model, failed=True)
            raise ProviderUnavailableError(
                f"Ollama at {self.base_url} did not respond within {self.timeout:.0f}s. "
                "The model may still be loading or generating a long answer. "
                "Increase OLLAMA_TIMEOUT or cap OLLAMA_NUM_PREDICT."
            ) from exc
        except httpx.HTTPStatusError as exc:
            record_llm_usage(model=model, failed=True)
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise ProviderError(f"Ollama returned an error: {detail}") from exc
        except httpx.HTTPError as exc:
            record_llm_usage(model=model, failed=True)
            raise ProviderError(f"Ollama request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderError("Ollama returned an invalid JSON response.") from exc

        message = data.get("message") or {}
        content = str(message.get("content") or "").strip()
        if not content:
            record_llm_usage(model=model, failed=True)
            raise ProviderError("Ollama returned an empty response.")

        record_llm_usage(
            model=str(data.get("model") or model),
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )
        return ChatResponse(
            model=str(data.get("model") or model),
            message=ChatMessage(role="assistant", content=content),
            done=bool(data.get("done", True)),
        )
