from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable


BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BENCHMARK_ROOT.parent
DEFAULT_REGISTRY_PATH = BENCHMARK_ROOT / "embedding_benchmark" / "config" / "model_registry.json"
DEFAULT_QUERY_SET_PATH = BENCHMARK_ROOT / "embedding_benchmark" / "query_sets" / "pubmed_chunks_benchmark_v1.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "benchmarks" / "embedding_benchmark"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_CHUNK_SHARDS = [
    PROJECT_ROOT / "data" / "processed" / "chunks_shard_0.parquet",
    PROJECT_ROOT / "data" / "processed" / "chunks_shard_1.parquet",
]


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    config: dict[str, Any]

    @property
    def display_name(self) -> str:
        return str(self.config.get("display_name") or self.slug)

    @property
    def family(self) -> str:
        return str(self.config["family"])

    @property
    def dim(self) -> int:
        return int(self.config["dim"])

    @property
    def batch_size(self) -> int:
        return int(self.config.get("batch_size") or 32)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def monotonic_seconds() -> float:
    return time.perf_counter()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry = load_json(path)
    if "models" not in registry or not isinstance(registry["models"], dict):
        raise RuntimeError(f"Model registry {path} must contain a 'models' object.")
    return registry


def get_model_spec(registry: dict[str, Any], model_slug: str) -> ModelSpec:
    models = registry["models"]
    if model_slug not in models:
        available = ", ".join(sorted(models))
        raise RuntimeError(f"Unknown model '{model_slug}'. Available: {available}")
    return ModelSpec(slug=model_slug, config=dict(models[model_slug]))


def model_slugs(registry: dict[str, Any], selected: Iterable[str] | None = None) -> list[str]:
    available = list(registry["models"].keys())
    if selected is None:
        return available
    requested = [item.strip() for item in selected if item.strip()]
    missing = sorted(set(requested) - set(available))
    if missing:
        raise RuntimeError(f"Unknown model(s): {', '.join(missing)}")
    return requested


def sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug.lower() or "model"


def collection_name_for(model: ModelSpec, *, prefix: str = "pubmed_v1", dtype: str = "f16") -> str:
    return f"{sanitize_slug(prefix)}_{model.slug}_dim{model.dim}_{dtype}"


def model_output_dir(output_root: Path, model_slug: str, precision: str = "fp16") -> Path:
    return output_root / model_slug / precision


def setup_logging(log_path: Path | None = None, *, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("embedding_benchmark")
    logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def file_is_nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def human_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


class ProgressTracker:
    def __init__(self, *, total: int, label: str, logger: logging.Logger, log_every_seconds: float = 30.0) -> None:
        self.total = max(int(total), 0)
        self.label = label
        self.logger = logger
        self.log_every_seconds = log_every_seconds
        self.started_at = monotonic_seconds()
        self.last_log_at = 0.0

    def update(self, done: int, *, force: bool = False, extra: str = "") -> None:
        now = monotonic_seconds()
        if not force and now - self.last_log_at < self.log_every_seconds and done < self.total:
            return
        self.last_log_at = now
        elapsed = now - self.started_at
        rate = done / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - done, 0)
        eta = remaining / rate if rate > 0 else 0.0
        percent = (done / self.total * 100.0) if self.total else 0.0
        bar = self._bar(percent)
        suffix = f" {extra}" if extra else ""
        self.logger.info(
            "%s %s %.2f%% %d/%d rate=%.2f/s elapsed=%s eta=%s%s",
            self.label,
            bar,
            percent,
            done,
            self.total,
            rate,
            human_seconds(elapsed),
            human_seconds(eta),
            suffix,
        )

    def _bar(self, percent: float) -> str:
        width = 24
        filled = int(round(width * max(min(percent, 100.0), 0.0) / 100.0))
        return "[" + "#" * filled + "-" * (width - filled) + "]"


def gpu_memory_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        devices = []
        for index in range(torch.cuda.device_count()):
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "allocated_bytes": int(torch.cuda.memory_allocated(index)),
                    "reserved_bytes": int(torch.cuda.memory_reserved(index)),
                    "max_allocated_bytes": int(torch.cuda.max_memory_allocated(index)),
                    "max_reserved_bytes": int(torch.cuda.max_memory_reserved(index)),
                }
            )
        return {"cuda_available": True, "devices": devices}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"cuda_available": None, "error": str(exc)}
