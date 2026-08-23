"""Per-case LLM usage accounting (calls + tokens) for compute-matched experiments.

The paper's main claim is "debate vs self-consistency **at matched compute**", so
every arm has to report how much inference it actually spent. Latency is not a
substitute: it depends on how many GPUs were free and on case concurrency.

Providers call :func:`record_llm_usage` after each request; the benchmark scripts
open a scope per case and read the totals. Outside a scope recording is a no-op,
so the API path pays nothing.

Scoping uses a :class:`~contextvars.ContextVar`: a scope opened inside a coroutine
is visible to every task spawned from it (agents in a debate round), but not to
sibling cases running concurrently, because each ``asyncio.Task`` gets its own
copy of the context.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class UsageTotals:
    """Mutable accumulator for one scope (typically one benchmark case).

    ``llm_calls`` counts requests that came back with a usable response — the
    number to compute-match on. ``failed_llm_calls`` counts requests that errored
    (timeout, connection refused, HTTP error); they still burned GPU time but
    their token counts are unknown, so they are kept separate instead of being
    silently folded into the totals.
    """

    llm_calls: int = 0
    failed_llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_by_model: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record(
        self,
        *,
        model: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        failed: bool = False,
    ) -> None:
        if failed:
            self.failed_llm_calls += 1
        else:
            self.llm_calls += 1
            self.prompt_tokens += max(0, int(prompt_tokens))
            self.completion_tokens += max(0, int(completion_tokens))
        key = model or "unknown"
        self.calls_by_model[key] = self.calls_by_model.get(key, 0) + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "llm_calls": self.llm_calls,
            "failed_llm_calls": self.failed_llm_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "calls_by_model": dict(self.calls_by_model),
        }


_ACTIVE_USAGE: ContextVar[UsageTotals | None] = ContextVar("llm_usage_totals", default=None)


def start_usage_scope() -> UsageTotals:
    """Begin accounting in the current context and return the accumulator.

    Preferred inside a per-case coroutine: the scope ends naturally when the task
    ends (concurrent cases) or when the next case starts its own scope
    (sequential cases). Use :func:`usage_scope` when the scope must be closed
    explicitly.
    """
    totals = UsageTotals()
    _ACTIVE_USAGE.set(totals)
    return totals


@contextmanager
def usage_scope() -> Iterator[UsageTotals]:
    """Account for LLM usage inside the ``with`` block, then restore the previous scope."""
    totals = UsageTotals()
    token = _ACTIVE_USAGE.set(totals)
    try:
        yield totals
    finally:
        _ACTIVE_USAGE.reset(token)


def current_usage() -> UsageTotals | None:
    """Active accumulator, or ``None`` when no scope is open."""
    return _ACTIVE_USAGE.get()


def record_llm_usage(
    *,
    model: str = "",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    failed: bool = False,
) -> None:
    """Record one inference request; no-op outside a scope."""
    totals = _ACTIVE_USAGE.get()
    if totals is None:
        return
    totals.record(
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        failed=failed,
    )
