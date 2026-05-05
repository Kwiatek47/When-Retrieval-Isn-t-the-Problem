from functools import lru_cache

from app.core.config import get_settings
from app.providers.base import LLMProvider
from app.providers.ollama import OllamaProvider
from app.services.telemetry_service import TelemetryLogger


@lru_cache
def get_ollama_provider() -> OllamaProvider:
    settings = get_settings()
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        timeout=settings.ollama_timeout,
    )


def get_llm_provider() -> LLMProvider:
    return get_ollama_provider()


@lru_cache
def get_telemetry_logger() -> TelemetryLogger:
    settings = get_settings()
    return TelemetryLogger(path=settings.telemetry_path)
