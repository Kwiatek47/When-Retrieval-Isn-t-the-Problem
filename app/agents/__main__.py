"""Demo: run a 3-round clinical debate without a supervisor.

Usage:
  .venv/bin/python -m app.agents              # mock backend (offline)
  .venv/bin/python -m app.agents --ollama     # Ollama via project settings
"""

from __future__ import annotations

import argparse
import asyncio
import json

from app.agents import DebateOrchestrator, MockInferenceBackend, build_default_agents
from app.agents.backends import OllamaInferenceBackend


SAMPLE_CASE = """
65-year-old man with 3 days of fever (38.5 C), productive cough, and dyspnea on exertion.
History: hypertension, former smoker (20 pack-years). No chest pain. SpO2 unknown.
Meds: amlodipine. No recent travel. Exam not yet available.
""".strip()


async def _run(*, use_ollama: bool) -> None:
    if use_ollama:
        from app.core.config import get_settings
        from app.providers.ollama import OllamaProvider

        settings = get_settings()
        provider = OllamaProvider(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
            keep_alive=settings.ollama_keep_alive,
            num_predict=max(settings.ollama_num_predict, 800),
            num_ctx=settings.ollama_num_ctx,
        )
        backend = OllamaInferenceBackend(provider, model=settings.default_model, temperature=0.3)
        print(f"Using Ollama model={settings.default_model} at {settings.ollama_base_url}")
    else:
        backend = MockInferenceBackend()
        print("Using MockInferenceBackend (offline demo)")

    agents = build_default_agents(backend)
    orchestrator = DebateOrchestrator(agents, rounds=3)
    result = await orchestrator.run(SAMPLE_CASE)

    print(f"\nRounds completed: {len(result.rounds)} (no supervisor)")
    for round_opinions in result.rounds:
        round_no = round_opinions[0].round
        print(f"\n=== Round {round_no} ===")
        for entry in round_opinions:
            print(
                f"- {entry.agent_id}: {entry.opinion.top_1_diagnosis} "
                f"(confidence={entry.opinion.confidence_level:.2f})"
            )

    print("\n=== Final opinions (JSON) ===")
    print(
        json.dumps(
            [entry.model_dump() for entry in result.final_opinions],
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent clinical debate demo")
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama LLM backend instead of the offline mock",
    )
    args = parser.parse_args()
    asyncio.run(_run(use_ollama=args.ollama))


if __name__ == "__main__":
    main()
