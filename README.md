# Architektura-multiagentowego-systemu-diagnostycznego
https://moja.pg.edu.pl/auth/appjs/research-projects/project-topics?tab=contributor&amp;page=1&amp;pageSize=10&amp;sort=id&amp;sortDirection=asc

## MedChat

Minimalistyczny fundament chatbota medycznego oparty o FastAPI, asynchroniczny provider pattern i lokalne modele Ollama.
Aktualny target: wsparcie lekarza w neurologicznym roznicowaniu diagnoz na podstawie wywiadu medycznego.

### Struktura

```text
.
├── app/
│   ├── api/              # Routery i dependency injection
│   ├── core/             # Konfiguracja aplikacji
│   ├── providers/        # Protokol LLMProvider i implementacje providerow
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

### Prompt versioning

Aplikacja wspiera wersje promptu systemowego (`v1`, `v2`, `v3`):

- aktywna wersja: zmienna `PROMPT_VERSION` (domyslnie `v1`)
- opcjonalny override per request: pole `prompt_version` w `POST /api/chat`

### Telemetria i feedback

- Wszystkie wywolania `POST /api/chat` i `POST /api/feedback` zapisywane sa do JSONL.
- Domyslna sciezka: `data/telemetry/events.jsonl`
- Mozesz zmienic sciezke przez `TELEMETRY_PATH`.

### Offline eval (CPU-only)

Benchmark i rubryka sa w katalogu `eval/`:

- dataset: `eval/dataset.jsonl`
- rubric: `eval/rubric.md`

Uruchomienie porownania promptow:

```bash
python scripts/run_eval.py --candidate v2
```

Raporty trafiaja do `eval/reports/` (`latest.md`, `latest.json` oraz wersje timestampowane).

### Profil specjalistyczny: neurologia

- Prompty `v1-v3` sa ukierunkowane na: roznicowanie neurologiczne, lokalizacje, czerwone flagi i kolejnosc badan.
- Dataset benchmarkowy zawiera przypadki neurologiczne (stroke, napad, neuroinfekcja, neuropatie, otepienia, zespoly rdzeniowe).
- Rubryka premiuje: jakosc roznicowania, lokalizacje neuroanatomiczna, plan diagnostyczny i bezpieczenstwo triage.
