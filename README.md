# Architektura-multiagentowego-systemu-diagnostycznego
https://moja.pg.edu.pl/auth/appjs/research-projects/project-topics?tab=contributor&amp;page=1&amp;pageSize=10&amp;sort=id&amp;sortDirection=asc

## MedChat

Minimalistyczny fundament chatbota medycznego oparty o FastAPI, asynchroniczny provider pattern i lokalne modele Ollama.

### Struktura

```text
.
├── app/
│   ├── api/              # Routery i dependency injection
│   ├── core/             # Konfiguracja aplikacji
│   ├── providers/        # Protokol LLMProvider i implementacje providerow
│   ├── rag/              # Sekwencyjny pipeline Pre-retrieval -> Retrieval -> Post-retrieval
│   ├── services/         # Logika domenowa gotowa pod RAG
│   ├── main.py           # Fabryka aplikacji FastAPI
│   └── schemas.py        # Modele request/response
├── main.py               # Entry point dla uvicorn main:app
└── static/
    ├── index.html        # Widok aplikacji
    └── app.js            # Logika UI
```

### Uruchomienie

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Aplikacja będzie dostępna pod adresem `http://127.0.0.1:8000/`.

Modele medyczne w Ollamie można pobrać przykładowo:

```bash
ollama pull medgemma
ollama pull meditron
ollama pull medllama2
```

### RAG pipeline

Endpoint `POST /api/chat` przechodzi przez trzy kroki przed wywolaniem modelu:

1. `PreRetriever` (`app/rag/pre_retrieval.py`) normalizuje ostatnie pytanie uzytkownika, rozwija podstawowe skroty medyczne, wykrywa proste przypadki bez potrzeby retrieval i wyciaga wstepne filtry, np. kody ICD.
2. `MedicalKnowledgeRetriever` (`app/rag/retrieval.py`) jest kontraktem pod baze wektorowa. Aktualnie podpiety jest `EmptyMedicalKnowledgeRetriever`, ktory zwraca brak dokumentow do czasu integracji VectorDB.
3. `PostRetriever` (`app/rag/post_retrieval.py`) sortuje i deduplikuje dokumenty, buduje blok kontekstu `MEDICAL_KNOWLEDGE_BASE`, dokleja instrukcje cytowania i zwraca metadane `citations` oraz `retrieval`.

Osoba implementujaca VectorDB powinna podmienic `get_medical_knowledge_retriever()` w `app/api/dependencies.py` na klase implementujaca:

```python
async def retrieve(query: PreRetrievalResult, *, limit: int) -> RetrievalResult:
    ...
```

Zwrocone dokumenty powinny miec typ `RetrievedDocument` z polami `id`, `title`, `content`, `source`, `score` i opcjonalnym `metadata`.

Konfiguracja srodowiskowa:

```bash
RAG_TOP_K=5
RAG_MAX_CONTEXT_CHARS=8000
```
