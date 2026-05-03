from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    app_title: str
    app_version: str
    default_model: str
    ollama_base_url: str
    ollama_timeout: float
    static_dir: Path
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_title="MedChat",
        app_version="0.1.0",
        default_model=os.getenv("OLLAMA_MODEL", "medgemma"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "60")),
        static_dir=PROJECT_ROOT / "static",
        system_prompt=(
            "You are a professional medical assistant. Provide concise, helpful information. "
            "Always include a disclaimer that this is not medical advice."
        ),
    )
