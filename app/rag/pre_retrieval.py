import re

import httpx

from app.rag.models import PreRetrievalResult
from app.schemas import ChatMessage


class PreRetriever:
    """Prepare a clinical user query for retrieval."""

    _NON_RETRIEVAL_PATTERNS = (
        re.compile(r"^\s*(hi|hello|hey|czesc|cześć|dzień dobry|dzien dobry)\s*[!.]?\s*$", re.IGNORECASE),
        re.compile(r"^\s*(thanks|thank you|dzieki|dzięki|ok|okay)\s*[!.]?\s*$", re.IGNORECASE),
    )
    _QUERY_REWRITE_SYSTEM_PROMPT = (
        "Przetłumacz skrótowe zapytanie medyczne użytkownika na profesjonalne, "
        "rozbudowane zapytanie optymalne dla semantycznej wyszukiwarki bazy danych. "
        "Rozwiń skróty i dodaj synonimy. "
        "ZWRÓĆ TYLKO POPRAWIONE ZAPYTANIE. BEZ WSTĘPU I ZAKOŃCZENIA."
    )

    def __init__(self, *, ollama_base_url: str, rewrite_model: str, rewrite_timeout: float) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.rewrite_model = rewrite_model
        self.rewrite_timeout = rewrite_timeout

    async def prepare(self, messages: list[ChatMessage]) -> PreRetrievalResult:
        query = self._latest_user_message(messages)
        normalized_query = self._normalize(query)
        requires_retrieval = self._requires_retrieval(normalized_query)

        search_queries = [normalized_query]
        notes = []

        if requires_retrieval:
            rewritten_query = await self._rewrite_query(normalized_query)
            if rewritten_query and rewritten_query.lower() != normalized_query.lower():
                search_queries.append(rewritten_query)
                notes.append(f"Query rewritten for semantic retrieval with {self.rewrite_model}.")
            elif not rewritten_query:
                notes.append("Query rewriting unavailable; using normalized query.")

        if not requires_retrieval:
            notes.append("Retrieval skipped for simple conversational input.")

        return PreRetrievalResult(
            original_query=query,
            normalized_query=normalized_query,
            search_queries=search_queries,
            requires_retrieval=requires_retrieval,
            filters=self._extract_filters(normalized_query),
            notes=notes,
        )

    def _latest_user_message(self, messages: list[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return messages[-1].content

    def _normalize(self, query: str) -> str:
        return re.sub(r"\s+", " ", query).strip()

    async def _rewrite_query(self, query: str) -> str | None:
        payload = {
            "model": self.rewrite_model,
            "messages": [
                {"role": "system", "content": self._QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": "leki na PAD"},
                {
                    "role": "assistant",
                    "content": "leczenie symptomatic peripheral artery disease choroba tętnic obwodowych",
                },
                {"role": "user", "content": "powikłania T2DM"},
                {
                    "role": "assistant",
                    "content": "powikłania type 2 diabetes mellitus cukrzyca typu 2",
                },
                {"role": "user", "content": query},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64},
        }

        try:
            async with httpx.AsyncClient(base_url=self.ollama_base_url, timeout=self.rewrite_timeout) as client:
                response = await client.post("/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        content = str((data.get("message") or {}).get("content") or "")
        return self._sanitize_rewritten_query(content)

    def _sanitize_rewritten_query(self, query: str) -> str | None:
        sanitized = self._normalize(query).strip("\"'` ")
        if not sanitized:
            return None
        prefixes = ("Output:", "Assistant:", "Odpowiedz:", "Wynik:")
        for prefix in prefixes:
            if sanitized.lower().startswith(prefix.lower()):
                sanitized = sanitized[len(prefix) :].strip()
        return sanitized or None

    def _requires_retrieval(self, query: str) -> bool:
        if not query:
            return False
        return not any(pattern.match(query) for pattern in self._NON_RETRIEVAL_PATTERNS)

    def _extract_filters(self, query: str) -> dict[str, str]:
        filters = {}
        icd_codes = re.findall(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", query.upper())
        if icd_codes:
            filters["icd_code"] = ",".join(sorted(set(icd_codes)))
        return filters
