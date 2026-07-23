"""Round-robin, supervisor-free debate orchestrator (MAC / MedAgent style rounds).

This is the *baseline* multi-agent architecture: a fixed panel of persona
agents debates a case with no supervisor/moderator mediating between them.
It is intentionally the simplest architecture in this package so future
variants (e.g. a supervisor that mediates turns, synthesizes a final answer,
or dynamically routes to specialists) have a stable interface to implement:
any orchestrator that exposes `async def run(patient_case: str) -> DebateResult`
can be swapped in wherever `DebateOrchestrator` is used today (CLI demo, eval
harness, and eventually the API).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import opinion_label
from app.agents.models import AgentRoundOpinion, DebateResult

DEFAULT_PERSONAS: tuple[tuple[str, str], ...] = (
    ("generalist", "generalist"),
    ("evidence_skeptic", "evidence_skeptic"),
    ("differential_expander", "differential_expander"),
    ("safety_officer", "safety_officer"),
)

EarlyExitFn = Callable[[int, list[AgentRoundOpinion]], bool]


class DebateOrchestrator:
    """
    Runs 2-3 rounds of peer debate among ClinicalAgent instances.

    Round 1 ("independent opinion"): every agent answers in parallel with no
    peer context, so nobody anchors on somebody else's first take.

    Round 2+ ("round-robin"): agents speak one at a time in a fixed order
    (`agents` order). Each agent sees the previous round's final opinions
    *and* the opinions already given by peers earlier in the current round -
    exactly like a real round-robin discussion, where the last speaker has
    heard strictly more of the conversation than the first. This is why
    those rounds run sequentially rather than concurrently: turn N depends on
    turn N-1's output.

    No supervisor: after the final round, returns full history and each
    agent's last opinion. Optional early-exit skips later rounds when the
    panel already agrees.
    """

    ARCHITECTURE: str = "round_robin_no_supervisor"

    def __init__(
        self,
        agents: list[ClinicalAgent],
        *,
        rounds: int = 3,
        early_exit: EarlyExitFn | None = None,
        agent_concurrency: int = 4,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("DebateOrchestrator requires at least 2 agents.")
        if rounds < 2 or rounds > 3:
            raise ValueError("rounds must be 2 or 3.")
        if agent_concurrency < 1:
            raise ValueError("agent_concurrency must be >= 1.")
        self.agents = agents
        self.rounds = rounds
        self.early_exit = early_exit
        self.agent_concurrency = agent_concurrency
        self.early_exits = 0

    async def run(self, patient_case: str) -> DebateResult:
        history: list[list[AgentRoundOpinion]] = []

        for round_number in range(1, self.rounds + 1):
            if round_number == 1:
                round_opinions = await self._run_independent_round(patient_case)
            else:
                round_opinions = await self._run_round_robin_round(
                    patient_case,
                    round_number=round_number,
                    previous_round=history[-1],
                )
            history.append(round_opinions)
            if (
                round_number < self.rounds
                and self.early_exit is not None
                and self.early_exit(round_number, round_opinions)
            ):
                self.early_exits += 1
                break

        return DebateResult(
            patient_case=patient_case,
            rounds=history,
            final_opinions=history[-1],
        )

    async def _run_independent_round(self, patient_case: str) -> list[AgentRoundOpinion]:
        """Round 1: every agent answers concurrently with no peer context."""
        semaphore = asyncio.Semaphore(self.agent_concurrency)

        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            async with semaphore:
                return await self._speak(agent, patient_case, round_number=1, context=None)

        return list(await asyncio.gather(*[_one(agent) for agent in self.agents]))

    async def _run_round_robin_round(
        self,
        patient_case: str,
        *,
        round_number: int,
        previous_round: list[AgentRoundOpinion],
    ) -> list[AgentRoundOpinion]:
        """Round 2+: agents take turns in order, each seeing all turns so far."""
        spoken_so_far: list[AgentRoundOpinion] = []
        for agent in self.agents:
            context = _round_robin_context(agent.agent_id, previous_round, spoken_so_far)
            entry = await self._speak(agent, patient_case, round_number=round_number, context=context)
            spoken_so_far.append(entry)
        return spoken_so_far

    async def _speak(
        self,
        agent: ClinicalAgent,
        patient_case: str,
        *,
        round_number: int,
        context: list[AgentRoundOpinion] | None,
    ) -> AgentRoundOpinion:
        opinion = await agent.generate_opinion(patient_case, context=context)
        return AgentRoundOpinion(
            agent_id=agent.agent_id,
            persona=agent.persona,
            round=round_number,
            opinion=opinion,
        )


def labels_unanimous(round_opinions: list[AgentRoundOpinion]) -> bool:
    labels = [opinion_label(entry.opinion) for entry in round_opinions]
    labels = [label for label in labels if label is not None]
    return bool(labels) and len(set(labels)) == 1


def _round_robin_context(
    agent_id: str,
    previous_round: list[AgentRoundOpinion],
    spoken_so_far: list[AgentRoundOpinion],
) -> list[AgentRoundOpinion]:
    """Previous round's peers (excluding self) followed by this round's turns so far."""
    previous_peers = [entry for entry in previous_round if entry.agent_id != agent_id]
    current_peers = [entry for entry in spoken_so_far if entry.agent_id != agent_id]
    return previous_peers + current_peers
