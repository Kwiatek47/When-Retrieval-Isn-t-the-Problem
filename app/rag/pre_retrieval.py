import re
from datetime import datetime

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

    _INTENT_PATTERNS = {
        "treatment": (
            "leczenie",
            "leczyć",
            "terapia",
            "terapeuty",
            "lek",
            "leki",
            "dawk",
            "treatment",
            "therapy",
            "drug",
            "medication",
            "antibiotic",
            "chemotherapy",
            "regimen",
            "dose",
            "guideline",
            "wytyczne",
        ),
        "diagnosis": (
            "diagno",
            "rozpozn",
            "screening",
            "test",
            "badanie",
            "diagnosis",
            "diagnostic",
        ),
        "adverse_effects": (
            "bezpieczeń",
            "bezpieczen",
            "działania niepożądane",
            "dzialania niepozadane",
            "powikł",
            "powikl",
            "ryzyk",
            "przeciwwsk",
            "adverse",
            "side effect",
            "contraindication",
            "safety",
        ),
        "prognosis": (
            "rokowanie",
            "śmiertel",
            "smiertel",
            "mortality",
            "survival",
            "outcome",
            "prognosis",
        ),
        "mechanism": (
            "mechanizm",
            "patofizj",
            "pathophysiology",
            "mechanism",
            "biomarker",
        ),
    }
    _RECENT_EVIDENCE_TERMS = (
        "aktual",
        "najnows",
        "wytyczne",
        "guideline",
        "standard",
        "dawk",
        "bezpieczeń",
        "bezpieczen",
        "safety",
        "contraindication",
    )
    _PUBLICATION_TYPE_POLICY = {
        "treatment": [
            "Practice Guideline",
            "Guideline",
            "Systematic Review",
            "Meta-Analysis",
            "Review",
            "Randomized Controlled Trial",
            "Clinical Trial",
            "Clinical Overview",
        ],
        "diagnosis": [
            "Practice Guideline",
            "Guideline",
            "Systematic Review",
            "Meta-Analysis",
            "Review",
            "Clinical Trial",
            "Clinical Overview",
        ],
        "adverse_effects": [
            "Practice Guideline",
            "Guideline",
            "Systematic Review",
            "Meta-Analysis",
            "Review",
            "Clinical Trial",
            "Clinical Overview",
        ],
        "prognosis": [
            "Systematic Review",
            "Meta-Analysis",
            "Review",
            "Clinical Trial",
            "Observational Study",
        ],
        "mechanism": [
            "Review",
            "Systematic Review",
            "Meta-Analysis",
        ],
        "general": [
            "Systematic Review",
            "Meta-Analysis",
            "Review",
            "Practice Guideline",
            "Guideline",
            "Clinical Overview",
        ],
    }
    _ACRONYM_EXPANSIONS = {
        "PAD": "peripheral artery disease choroba tetnic obwodowych",
        "T2DM": "type 2 diabetes mellitus cukrzyca typu 2",
        "CKD": "chronic kidney disease przewlekla choroba nerek",
        "AF": "atrial fibrillation migotanie przedsionkow",
        "DOAC": "direct oral anticoagulants doustne antykoagulanty niebędące antagonistami witaminy K",
        "EGFR": "estimated glomerular filtration rate kidney function renal function",
        "SGLT2": "sodium-glucose cotransporter 2 inhibitors inhibitory SGLT2",
        "ICS": "inhaled corticosteroids wziewne kortykosteroidy",
    }
    _MAX_QUERY_VARIANTS = 4

    def __init__(
        self,
        *,
        ollama_base_url: str,
        rewrite_model: str,
        rewrite_timeout: float,
        active_corpus_version: str | None = None,
    ) -> None:
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.rewrite_model = rewrite_model
        self.rewrite_timeout = rewrite_timeout
        self.active_corpus_version = (active_corpus_version or "").strip()

    async def prepare(self, messages: list[ChatMessage], *, allow_rewrite: bool = True) -> PreRetrievalResult:
        query = self._latest_user_message(messages)
        normalized_query = self._normalize(query)
        requires_retrieval = self._requires_retrieval(normalized_query)
        intent = self._classify_intent(normalized_query)
        requires_recent_evidence = self._requires_recent_evidence(normalized_query, intent)

        search_queries = self._unique_queries([query, normalized_query])
        notes = []

        if requires_retrieval:
            deterministic_query, expanded_acronyms = self._deterministic_query_expansion(normalized_query)
            if deterministic_query and deterministic_query.lower() != normalized_query.lower():
                search_queries = self._unique_queries([*search_queries, deterministic_query])
                notes.append(
                    "Query expanded deterministically for acronyms: "
                    f"{', '.join(expanded_acronyms)}."
                )
            if allow_rewrite:
                rewritten_query = await self._rewrite_query(normalized_query)
                if rewritten_query and rewritten_query.lower() != normalized_query.lower():
                    search_queries = self._unique_queries([*search_queries, rewritten_query])
                    notes.append(f"Query rewritten for semantic retrieval with {self.rewrite_model}.")
                elif not rewritten_query:
                    notes.append("Query rewriting unavailable; using normalized query.")
            else:
                notes.append("Query rewriting skipped for retrieval-only flow.")
            search_queries = search_queries[: self._MAX_QUERY_VARIANTS]

        if not requires_retrieval:
            notes.append("Retrieval skipped for simple conversational input.")

        return PreRetrievalResult(
            original_query=query,
            normalized_query=normalized_query,
            search_queries=search_queries,
            requires_retrieval=requires_retrieval,
            filters=self._extract_filters(normalized_query),
            intent=intent,
            preferred_publication_types=self._preferred_publication_types(intent),
            min_year=self._min_year(requires_recent_evidence),
            requires_recent_evidence=requires_recent_evidence,
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
        if not self.rewrite_model.strip():
            return None

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

    def _deterministic_query_expansion(self, query: str) -> tuple[str | None, list[str]]:
        matched = []
        upper_query = query.upper()
        for acronym, expansion in self._ACRONYM_EXPANSIONS.items():
            if re.search(rf"\b{re.escape(acronym)}\b", upper_query):
                matched.append((acronym, expansion))
        if not matched:
            return None, []

        expansion_terms = " ".join(expansion for _, expansion in matched)
        expanded_query = self._normalize(f"{query} {expansion_terms}")
        return expanded_query, [acronym for acronym, _ in matched]

    def _requires_retrieval(self, query: str) -> bool:
        if not query:
            return False
        return not any(pattern.match(query) for pattern in self._NON_RETRIEVAL_PATTERNS)

    def _extract_filters(self, query: str) -> dict[str, str]:
        filters = {}
        if self.active_corpus_version:
            filters["corpusVersion"] = self.active_corpus_version
        lower_query = query.lower()
        if re.search(r"\bnice\b", lower_query):
            filters["source"] = "nice"
        elif re.search(r"\bpubmed\b", lower_query):
            filters["source"] = "pubmed"
        nice_ids = re.findall(r"\b(?:NG|CG|TA|HTG|HST|AMR|MPG|PH|CSG|SG|SC)\s*\d+\b", query.upper())
        if nice_ids:
            filters["externalId"] = ",".join(sorted({value.replace(" ", "") for value in nice_ids}))
        icd_codes = re.findall(r"\b[A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?\b", query.upper())
        if icd_codes:
            filters["icd_code"] = ",".join(sorted(set(icd_codes)))
        return filters

    def _unique_queries(self, queries: list[str]) -> list[str]:
        unique = []
        seen = set()
        for query in queries:
            normalized = self._normalize(query)
            key = normalized.lower()
            if normalized and key not in seen:
                unique.append(normalized)
                seen.add(key)
        return unique

    def _classify_intent(self, query: str) -> str:
        lower_query = query.lower()
        for intent, patterns in self._INTENT_PATTERNS.items():
            if any(pattern in lower_query for pattern in patterns):
                return intent
        return "general"

    def _preferred_publication_types(self, intent: str) -> list[str]:
        return list(self._PUBLICATION_TYPE_POLICY.get(intent, self._PUBLICATION_TYPE_POLICY["general"]))

    def _requires_recent_evidence(self, query: str, intent: str) -> bool:
        lower_query = query.lower()
        return any(term in lower_query for term in self._RECENT_EVIDENCE_TERMS)

    def _min_year(self, requires_recent_evidence: bool) -> int | None:
        if not requires_recent_evidence:
            return None
        return datetime.now().year - 10
