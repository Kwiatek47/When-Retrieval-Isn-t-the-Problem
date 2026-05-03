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
