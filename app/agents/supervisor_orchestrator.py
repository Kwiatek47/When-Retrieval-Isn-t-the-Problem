"""Supervisor-mediated debate, with optional anonymization and information asymmetry.

Drop-in alternative to `DebateOrchestrator`: same `async def run(patient_case) ->
DebateResult` contract, same rectangular `rounds` shape, same stable agent ids, so
the eval harness, uncertainty signals and per-agent metrics all keep working.

Three mechanisms sit behind flags so each can be measured on its own rather than
as one undifferentiated change:

* **Supervisor closure** (always on here). The final label comes from a rigor-check
  call, not from counting votes. Vote counting is what lets a panel be confidently
  wrong together.
* **`anonymize`**. Peers' identities are stripped from the transcript, so an agent
  can only react to what an argument says, not to who said it.
* **`partitions`**. Each agent sees one segment of the evidence and must ask peers
  for the rest (InfoNav), instead of every agent independently reading the whole
  case and independently making the same mistake.

Turning all three off would just be the baseline, which is why the class always
runs a supervisor; use `DebateOrchestrator` for the no-supervisor arm.
"""

from __future__ import annotations

import asyncio
import zlib
from collections.abc import Callable

from app.agents.agent import ClinicalAgent
from app.agents.aggregation import majority_vote
from app.agents.models import AgentRoundOpinion, DebateResult
from app.agents.partitioning import ContextPartition, partition_patient_case
from app.agents.supervisor import SupervisorAgent, SupervisorVerdict, eligibility_gate

EarlyExitFn = Callable[[int, list[AgentRoundOpinion]], bool]

ARCHITECTURE_SHARED = "supervisor_shared_context"
ARCHITECTURE_ANONYMIZED = "supervisor_anonymized"
ARCHITECTURE_ASYMMETRIC = "supervisor_asymmetric_infonav"


class SupervisorOrchestrator:
    """Runs a supervisor-closed debate over a panel of agents."""

    def __init__(
        self,
        agents: list[ClinicalAgent],
        supervisor: SupervisorAgent,
        *,
        rounds: int = 2,
        anonymize: bool = False,
        partitions: int = 0,
        partition_overlap: int = 0,
        max_info_rounds: int = 1,
        agent_concurrency: int = 1,
        early_exit: EarlyExitFn | None = None,
        seed: int = 47,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("SupervisorOrchestrator requires at least 2 agents.")
        if agent_concurrency < 1:
            raise ValueError("agent_concurrency must be >= 1.")
        if partitions and partitions < 2:
            raise ValueError("partitions must be 0 (disabled) or >= 2.")
        if partitions and max_info_rounds < 1:
            raise ValueError("max_info_rounds must be >= 1 when partitioning is enabled.")

        self.agents = agents
        self.supervisor = supervisor
        self.anonymize = anonymize
        self.partitions = partitions
        self.partition_overlap = partition_overlap
        self.max_info_rounds = max_info_rounds
        self.agent_concurrency = agent_concurrency
        self.early_exit = early_exit
        self.seed = seed
        self.early_exits = 0

        # Under asymmetry the schedule is fixed: one local round, then a bounded
        # number of information-exchange rounds. `mas_general.md` warns that an
        # unbounded question loop is this architecture's failure mode.
        if partitions:
            self.rounds = 1 + max_info_rounds
        else:
            if rounds < 1 or rounds > 4:
                raise ValueError("rounds must be between 1 and 4.")
            self.rounds = rounds

        # Populated per run() for the eval harness to record.
        self.last_verdict: SupervisorVerdict | None = None
        self.last_exchange: list[dict] = []
        self.last_partitions: list[ContextPartition] = []

    @property
    def ARCHITECTURE(self) -> str:  # noqa: N802 - matches DebateOrchestrator's class attribute
        if self.partitions:
            return ARCHITECTURE_ASYMMETRIC
        if self.anonymize:
            return ARCHITECTURE_ANONYMIZED
        return ARCHITECTURE_SHARED

    async def run(self, patient_case: str) -> DebateResult:
        self.last_verdict = None
        self.last_exchange = []
        self.last_partitions = []

        if self.partitions:
            history = await self._run_asymmetric(patient_case)
        else:
            history = await self._run_shared_context(patient_case)

        self.last_verdict = await self._close(patient_case, history[-1])
        return DebateResult(
            patient_case=patient_case,
            rounds=history,
            final_opinions=history[-1],
        )

    # -- shared-context schedule (rungs B and C) ---------------------------------

    async def _run_shared_context(self, patient_case: str) -> list[list[AgentRoundOpinion]]:
        history: list[list[AgentRoundOpinion]] = []
        views = {agent.agent_id: patient_case for agent in self.agents}

        for round_number in range(1, self.rounds + 1):
            if round_number == 1:
                round_opinions = await self._independent_round(views, round_number=1)
            else:
                round_opinions = await self._round_robin_round(
                    views,
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
        return history

    # -- asymmetric schedule (rung D) --------------------------------------------

    async def _run_asymmetric(self, patient_case: str) -> list[list[AgentRoundOpinion]]:
        partitions = partition_patient_case(
            patient_case,
            k=self.partitions,
            overlap=self.partition_overlap,
        )
        self.last_partitions = partitions

        # Fewer segments than agents (very short evidence): agents wrap around, so
        # some share a segment rather than some getting nothing to read.
        assigned = {
            agent.agent_id: partitions[index % len(partitions)]
            for index, agent in enumerate(self.agents)
        }
        views = {agent_id: part.patient_case for agent_id, part in assigned.items()}
        segments = {agent_id: part.evidence_text for agent_id, part in assigned.items()}

        history: list[list[AgentRoundOpinion]] = [
            await self._independent_round(
                views,
                round_number=1,
                partial_evidence=True,
                segment_ids={aid: part.segment_id for aid, part in assigned.items()},
            )
        ]

        for offset in range(self.max_info_rounds):
            round_number = 2 + offset
            routed = await self._route_requests(
                patient_case=patient_case,
                previous_round=history[-1],
                segments=segments,
            )
            history.append(
                await self._round_robin_round(
                    views,
                    round_number=round_number,
                    previous_round=history[-1],
                    partial_evidence=True,
                    info_requests=routed,
                    segment_ids={aid: part.segment_id for aid, part in assigned.items()},
                )
            )
        return history

    async def _route_requests(
        self,
        *,
        patient_case: str,
        previous_round: list[AgentRoundOpinion],
        segments: dict[str, str],
    ) -> dict[str, list[str]]:
        """Two-layer routing: deterministic eligibility gate, then supervisor choice."""
        requests: dict[str, str] = {}
        askers: dict[str, str] = {}
        for entry in previous_round:
            for position, question in enumerate(entry.opinion.information_requests):
                text = question.strip()
                if not text:
                    continue
                request_id = f"{entry.agent_id}:{position}"
                requests[request_id] = text
                askers[request_id] = entry.agent_id

        if not requests:
            return {}

        eligible = {
            request_id: [
                agent_id
                for agent_id in eligibility_gate(
                    text,
                    segments,
                )
                # An agent answering its own question learns nothing it did not
                # already have.
                if agent_id != askers[request_id]
            ]
            for request_id, text in requests.items()
        }
        question, _ = _split_question(patient_case)
        routed = await self.supervisor.route(
            question=question,
            requests=requests,
            eligible=eligible,
        )

        inbox: dict[str, list[str]] = {agent.agent_id: [] for agent in self.agents}
        for request_id, targets in routed.items():
            for agent_id in targets:
                if agent_id in inbox:
                    inbox[agent_id].append(requests[request_id])
            self.last_exchange.append(
                {
                    "request_id": request_id,
                    "asked_by": askers[request_id],
                    "question": requests[request_id],
                    "eligible": eligible.get(request_id, []),
                    "routed_to": list(targets),
                }
            )
        return inbox

    # -- rounds ------------------------------------------------------------------

    async def _independent_round(
        self,
        views: dict[str, str],
        *,
        round_number: int,
        partial_evidence: bool = False,
        segment_ids: dict[str, str] | None = None,
    ) -> list[AgentRoundOpinion]:
        semaphore = asyncio.Semaphore(self.agent_concurrency)

        async def _one(agent: ClinicalAgent) -> AgentRoundOpinion:
            async with semaphore:
                return await self._speak(
                    agent,
                    views[agent.agent_id],
                    round_number=round_number,
                    context=None,
                    partial_evidence=partial_evidence,
                    segment_id=(segment_ids or {}).get(agent.agent_id, ""),
                )

        return list(await asyncio.gather(*[_one(agent) for agent in self.agents]))

    async def _round_robin_round(
        self,
        views: dict[str, str],
        *,
        round_number: int,
        previous_round: list[AgentRoundOpinion],
        partial_evidence: bool = False,
        info_requests: dict[str, list[str]] | None = None,
        segment_ids: dict[str, str] | None = None,
    ) -> list[AgentRoundOpinion]:
        spoken_so_far: list[AgentRoundOpinion] = []
        for agent in self.agents:
            context = [
                entry
                for entry in list(previous_round) + spoken_so_far
                if entry.agent_id != agent.agent_id
            ]
            entry = await self._speak(
                agent,
                views[agent.agent_id],
                round_number=round_number,
                context=context,
                partial_evidence=partial_evidence,
                info_requests=(info_requests or {}).get(agent.agent_id) or None,
                segment_id=(segment_ids or {}).get(agent.agent_id, ""),
            )
            spoken_so_far.append(entry)
        return spoken_so_far

    async def _speak(
        self,
        agent: ClinicalAgent,
        patient_case: str,
        *,
        round_number: int,
        context: list[AgentRoundOpinion] | None,
        partial_evidence: bool = False,
        info_requests: list[str] | None = None,
        segment_id: str = "",
    ) -> AgentRoundOpinion:
        opinion = await agent.generate_opinion(
            patient_case,
            context=context,
            anonymize=self.anonymize,
            info_requests=info_requests,
            partial_evidence=partial_evidence,
            # Seed per (agent, round) so the shuffle is reproducible but not the
            # same permutation for everyone, which would recreate a fixed order.
            # crc32, not hash(): Python randomizes string hashing per process, so
            # hash() would silently reshuffle between runs of the same experiment.
            shuffle_seed=zlib.crc32(
                f"{self.seed}:{agent.agent_id}:{round_number}".encode()
            ),
        )
        return AgentRoundOpinion(
            agent_id=agent.agent_id,
            persona=agent.persona,
            round=round_number,
            opinion=opinion,
            segment_id=segment_id,
        )

    # -- closure -----------------------------------------------------------------

    async def _close(
        self,
        patient_case: str,
        final_round: list[AgentRoundOpinion],
    ) -> SupervisorVerdict:
        """Rigor-check closure, degrading to a vote if the supervisor cannot answer."""
        question, _ = _split_question(patient_case)
        # Always unattributed: the point of the supervisor is to judge arguments,
        # and it has no reason to know which agent produced which.
        arguments = [
            {
                "argument_id": f"A{position + 1}",
                "label": entry.opinion.top_1_diagnosis,
                "evidence_conclusiveness": entry.opinion.evidence_conclusiveness,
                "confidence": entry.opinion.confidence_level,
                "supporting": entry.opinion.pros[:2],
                "against": entry.opinion.cons[:2],
            }
            for position, entry in enumerate(final_round)
        ]
        verdict = await self.supervisor.moderate(question=question, arguments=arguments)
        if verdict.ok:
            return verdict

        fallback_label, _ = majority_vote([entry.opinion for entry in final_round])
        return SupervisorVerdict(
            label=fallback_label,
            confidence=0.0,
            rule="supervisor_fallback_majority",
            rationale="Supervisor produced no usable verdict; fell back to panel majority.",
            error=verdict.error or "Supervisor verdict unusable.",
        )

    def verdict_share(self, verdict: SupervisorVerdict) -> dict[str, float]:
        """One-hot 'share' so supervisor runs fill the same report column as votes."""
        return {label: (1.0 if label == verdict.label else 0.0) for label in ("yes", "no", "maybe")}


def _split_question(patient_case: str) -> tuple[str, str]:
    """Best-effort (question, evidence) split; falls back to the whole text."""
    if "RESEARCH QUESTION:" not in patient_case:
        return patient_case.strip(), patient_case.strip()
    tail = patient_case.split("RESEARCH QUESTION:", 1)[1]
    if "EVIDENCE:" in tail:
        question, evidence = tail.split("EVIDENCE:", 1)
        return question.strip(), evidence.strip()
    return tail.strip(), ""
