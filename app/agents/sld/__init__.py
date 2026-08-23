"""Supervised Ledger Debate (MAS-Supervisor v2).

Keeps the Supervisor + persona panel + two debate rounds of the legacy
`app.agents.orchestrator` pipeline, but replaces the lossy natural-language
round memory with a typed, citation-verified `EvidenceLedger`. Lives
alongside the legacy debate path, which stays untouched.
"""
