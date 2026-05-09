# Architektura-multiagentowego-systemu-diagnostycznego

MedChat to MVP medycznego chatbota RAG. Obecny stan projektu to lokalna aplikacja FastAPI z UI, osobnym `embedding-service`, baza wektorowa Qdrant i lokalnym modelem Ollama. Nazwa repo odnosi sie do kierunku rozwoju, ale aktualnie zaimplementowany jest sekwencyjny pipeline RAG, a nie pelny system multiagentowy.

## Co jest zaimplementowane

- UI czatu w `static/`
- API `POST /api/chat` w FastAPI
- lokalny provider LLM przez Ollama
- `PreRetriever -> Retrieval -> PostRetriever`
- embeddings biomedyczne MedCPT
- Qdrant jako Vector DB
- sample ingest do Qdranta z `data/pubmed_sample.json`
- opcjonalny reranking przez cross-encoder

## Architektura

```text
Uzytkownik
  -> UI / static
  -> FastAPI /api/chat
  -> PreRetriever
  -> Retriever
     -> embedding-service
     -> Qdrant
  -> PostRetriever
  -> Ollama
  -> odpowiedz z cytowaniami
```

Najwazniejsze katalogi:

```text
.
├── app/
│   ├── api/                    # endpointy FastAPI i dependency injection
│   ├── core/                   # konfiguracja
│   ├── providers/              # provider LLM, obecnie Ollama
│   ├── rag/                    # pipeline RAG i retrievery
│   ├── services/               # logika aplikacyjna
│   ├── main.py                 # create_app()
│   └── schemas.py              # request/response models
├── services/embedding-service/ # MedCPT query/document encoder
├── qdrant/                     # inicjalizacja kolekcji Qdrant
├── scripts/                    # skrypty pomocnicze, m.in. ingest sample
├── data/                       # sample corpus i statystyki BM25
└── static/                     # frontend
```

## Jak to dziala

Endpoint `POST /api/chat` wykonuje trzy etapy przed wywolaniem glownego modelu:

1. `PreRetriever` normalizuje ostatnie pytanie uzytkownika, opcjonalnie je przepisuje przez mniejszy model i wyciaga proste filtry, np. kody ICD.
2. `MedicalKnowledgeRetriever` pobiera dokumenty z bazy wiedzy. Domyslnie aplikacja uderza do `embedding-service`, ktory liczy embedding MedCPT i odpytuje Qdrant.
3. `PostRetriever` deduplikuje wyniki, buduje blok `MEDICAL_KNOWLEDGE_BASE`, dolacza instrukcje cytowania i zwraca metadane `citations` oraz `retrieval`.

Na koncu Ollama dostaje rozmowe z wstrzyknietym kontekstem i generuje odpowiedz.

## Ograniczenia obecnego MVP

- to nie jest jeszcze pelny system multiagentowy
- domyslny endpoint `/embed/hybrid/query` wykonuje dense retrieval, mimo nazwy `hybrid`
- korpus danych jest demonstracyjny i bardzo maly
- nie ma jeszcze pelnej ewaluacji retrievalu ani testow end-to-end

## Wymagania

- Python 3.10+
- Docker i Docker Compose
- Ollama
- GPU NVIDIA dla `embedding-service` w trybie CUDA

Jesli chcesz uzywac GPU w Dockerze, host musi miec:

- dzialajace `nvidia-smi`
- NVIDIA Container Toolkit
- Docker skonfigurowany do pracy z `--gpus all`

## Szybki start

### 1. Uruchom infrastrukture RAG

Z katalogu glownego projektu:

```bash
docker compose up --build qdrant qdrant-init embedding-service
```

To stawia:

- `qdrant` na `http://localhost:6333`
- `embedding-service` na `http://localhost:8081`

Sprawdz healthcheck:

```bash
curl http://localhost:8081/health
```

Jesli wszystko jest dobrze skonfigurowane, odpowiedz powinna zawierac:

```json
{
  "status": "ok",
  "device": "cuda"
}
```

Szczegoly konfiguracji GPU sa opisane w [services/embedding-service/README.md](./services/embedding-service/README.md).

### 2. Zaludnij baze wiedzy

Projekt ma gotowy sample ingest:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
EMBEDDING_SERVICE_URL=http://localhost:8081 QDRANT_URL=http://localhost:6333 python3 scripts/ingest_pubmed_sample.py
```

Skrypt:

- czyta `data/pubmed_sample.json`
- liczy embeddingi przez `embedding-service`
- zapisuje punkty do kolekcji `MedicalChunk`
- buduje `data/bm25_stats.json`

### 3. Uruchom Ollame

Przykladowe modele:

```bash
ollama serve
ollama pull medgemma:latest
ollama pull llama3.2:3b
```

`medgemma:latest` moze sluzyc jako model czatu, a `llama3.2:3b` jako model do query rewriting.

### 4. Uruchom aplikacje FastAPI

W nowym terminalu:

```bash
source .venv/bin/activate
EMBEDDING_SERVICE_URL=http://localhost:8081 uvicorn main:app --reload
```

Aplikacja bedzie dostepna pod:

```text
http://127.0.0.1:8000
```

## Jak uzywac

### UI

Otworz w przegladarce:

```text
http://127.0.0.1:8000
```

Mozesz:

- wybrac model Ollama z listy
- zadac pytanie medyczne
- zobaczyc odpowiedz oraz liste cytowanych zrodel

### API

Przykladowe zapytanie:

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "medgemma:latest",
    "messages": [
      {
        "role": "user",
        "content": "Jakie sa korzysci i ryzyka intensywnej kontroli cisnienia?"
      }
    ],
    "temperature": 0.2
  }'
```

Przykladowa odpowiedz zawiera:

- `message`
- `citations`
- `retrieval`

## Konfiguracja

Najwazniejsze zmienne srodowiskowe aplikacji:

```bash
OLLAMA_MODEL=medgemma
OLLAMA_BASE_URL=http://localhost:11434
RAG_RETRIEVER=embedding_service
RAG_TOP_K=5
RAG_MAX_CONTEXT_CHARS=8000
QUERY_REWRITE_MODEL=llama3.2:3b
QUERY_REWRITE_TIMEOUT=15
EMBEDDING_SERVICE_URL=http://localhost:8081
EMBEDDING_TIMEOUT=30
EMBEDDING_DIMENSION=768
QDRANT_HOST=localhost
QDRANT_PORT=6333
BM25_STATS_PATH=data/bm25_stats.json
```

Najwazniejsze zmienne `embedding-service`:

```bash
EMBEDDING_DEVICE=cuda
MEDCPT_QUERY_MODEL=ncbi/MedCPT-Query-Encoder
MEDCPT_DOCUMENT_MODEL=ncbi/MedCPT-Article-Encoder
EMBEDDING_BATCH_SIZE=16
```

## Tryby retrievalu

W kodzie sa dwie glowne sciezki retrievalu:

- `EmbeddingServiceHybridRetriever`
  obecnie domyslna sciezka aplikacji; aplikacja pyta `embedding-service`, a ten robi dense retrieval w Qdrancie
- `QdrantHybridKnowledgeRetriever`
  alternatywa po stronie aplikacji; laczy dense i sparse retrieval przez RRF, jesli ustawisz `RAG_RETRIEVER=qdrant_hybrid`

To oznacza, ze aktualny system ma podstawy pod hybryde, ale domyslnie nie wykonuje jeszcze pelnego hybrid search w endpointzie `embedding-service`.

## Typowy workflow developerski

```bash
docker compose up --build qdrant qdrant-init embedding-service
curl http://localhost:8081/health
source .venv/bin/activate
EMBEDDING_SERVICE_URL=http://localhost:8081 QDRANT_URL=http://localhost:6333 python3 scripts/ingest_pubmed_sample.py
ollama serve
EMBEDDING_SERVICE_URL=http://localhost:8081 uvicorn main:app --reload
```

## Co warto zrobic dalej

- wlaczyc prawdziwy hybrid retrieval jako domyslny
- dodac lepszy chunking i wiekszy korpus
- dodac testy integracyjne
- dodac walidacje cytowan i ewaluacje retrievalu
- dopiero potem budowac warstwe multiagentowa
