from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any


class TelemetryLogger:
    def __init__(self, *, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def _append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

    def log_chat_event(self, payload: dict[str, Any]) -> None:
        self._append(
            {
                "event_type": "chat",
                "logged_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )

    def log_feedback_event(self, payload: dict[str, Any]) -> None:
        self._append(
            {
                "event_type": "feedback",
                "logged_at": datetime.now(timezone.utc).isoformat(),
                **payload,
            }
        )

