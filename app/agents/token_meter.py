"""Token/call accounting for debate backends (cost-matched architecture comparison).

`InferenceBackend.complete` returns only text, so nothing downstream can see how
much a debate actually cost. `CountingBackend` wraps any backend, forwards the
call unchanged, and records what it cost. It satisfies the same Protocol, so it
drops in wherever a backend is expected without touching agents or orchestrators.

Exact counts arrive through the `LAST_USAGE` context variable that the Ollama
backend sets. Backends that cannot report usage (the mock) fall back to a crude
characters/4 estimate, which keeps the cost columns populated for mock runs
without pretending to be exact - `is_estimated` says which of the two you are
looking at.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.agents.backends import LAST_USAGE, InferenceBackend
from app.schemas import ChatMessage

CHARS_PER_TOKEN_ESTIMATE = 4


@dataclass(frozen=True)
class UsageSnapshot:
    """Cost of everything measured since the last `reset()`."""

    calls: int
    prompt_tokens: int
    completion_tokens: int
    is_estimated: bool

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "tokens_estimated": self.is_estimated,
        }


class CountingBackend:
    """Pass-through `InferenceBackend` that accumulates call and token counts."""

    def __init__(self, inner: InferenceBackend) -> None:
        self._inner = inner
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._estimated_any = False

    @property
    def inner(self) -> InferenceBackend:
        return self._inner

    async def complete(self, messages: list[ChatMessage], *, temperature: float = 0.3) -> str:
        LAST_USAGE.set(None)
        reply = await self._inner.complete(messages, temperature=temperature)
        self.calls += 1
        usage = LAST_USAGE.get()
        if usage:
            prompt_tokens, completion_tokens = usage
        else:
            prompt_tokens = _estimate_tokens("".join(message.content for message in messages))
            completion_tokens = _estimate_tokens(reply)
            self._estimated_any = True
        self.prompt_tokens += int(prompt_tokens)
        self.completion_tokens += int(completion_tokens)
        return reply

    def reset(self) -> None:
        """Zero the counters, typically once per benchmark case."""
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._estimated_any = False

    def snapshot(self) -> UsageSnapshot:
        return UsageSnapshot(
            calls=self.calls,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            is_estimated=self._estimated_any,
        )


def _estimate_tokens(text: str) -> int:
    return max(len(text) // CHARS_PER_TOKEN_ESTIMATE, 0)


def total_usage(*meters: CountingBackend | None) -> UsageSnapshot:
    """Sum several meters, e.g. the agent panel plus a separate supervisor backend."""
    live = [meter for meter in meters if meter is not None]
    return UsageSnapshot(
        calls=sum(meter.calls for meter in live),
        prompt_tokens=sum(meter.prompt_tokens for meter in live),
        completion_tokens=sum(meter.completion_tokens for meter in live),
        is_estimated=any(meter.snapshot().is_estimated for meter in live),
    )
