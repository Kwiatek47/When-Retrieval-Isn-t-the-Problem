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
    ollama_keep_alive: str
    ollama_num_predict: int
    ollama_num_ctx: int
    static_dir: Path
    telemetry_path: Path
    active_prompt_version: str
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_title="MedChat",
        app_version="0.1.0",
        default_model=os.getenv("OLLAMA_MODEL", "medgemma"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")),
        ollama_keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
        ollama_num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "400")),
        ollama_num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")),
        static_dir=PROJECT_ROOT / "static",
        telemetry_path=Path(os.getenv("TELEMETRY_PATH", str(PROJECT_ROOT / "data" / "telemetry" / "events.jsonl"))),
        active_prompt_version=os.getenv("PROMPT_VERSION", "v3"),
        system_prompt=(
            "You are a neurology clinical decision-support assistant for physicians. "
            "Provide differential diagnosis from patient history and symptom chronology. "
            "Highlight red flags, urgent exclusions, and recommended next diagnostics. "
            "Include a disclaimer that this is decision support and not a final diagnosis."
        ),
    )
