# Embedding Service

Osobny serwis HTTP odpowiedzialny wyłącznie za liczenie embeddingów dla pipeline RAG. Inne usługi nie powinny importować kodu tego serwisu bezpośrednio, tylko komunikować się z nim przez API.

Aktualna implementacja używa MedCPT:

- `ncbi/MedCPT-Article-Encoder` dla dokumentów i chunków,
- `ncbi/MedCPT-Query-Encoder` dla zapytań użytkownika.

Oba encodery pracują w tej samej przestrzeni retrievalowej i zwracają embeddingi o wymiarze `768`.

## Uruchomienie na GPU

Domyslny obraz Dockera instaluje PyTorch z CUDA i uruchamia MedCPT na GPU. Host musi miec:

- sterownik NVIDIA widoczny przez `nvidia-smi`,
- Docker z NVIDIA Container Toolkit,
- Compose obslugujacy `gpus: all`.

Z katalogu głównego projektu:

```bash
docker compose up --build embedding-service
```

Pierwszy start pobiera modele z Hugging Face. Cache modeli jest trzymany w wolumenie Docker Compose:

```text
huggingface-cache
```

Serwis jest wystawiony lokalnie jako:

```text
http://localhost:8081
```

W sieci Docker Compose inne kontenery powinny używać:

```text
http://embedding-service:8080
```

## Zmienne środowiskowe

```text
EMBEDDING_MODEL_NAME=medcpt-ncbi-v1
MEDCPT_QUERY_MODEL=ncbi/MedCPT-Query-Encoder
MEDCPT_DOCUMENT_MODEL=ncbi/MedCPT-Article-Encoder
MEDCPT_QUERY_MAX_LENGTH=64
MEDCPT_DOCUMENT_MAX_LENGTH=512
EMBEDDING_BATCH_SIZE=16
EMBEDDING_DEVICE=cuda
HF_HOME=/models/huggingface
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_TIMEOUT=10
QDRANT_COLLECTION=MedicalChunk
QDRANT_VECTOR_NAME=medcpt_dense
```

`EMBEDDING_MODEL_NAME` jest stabilnym identyfikatorem pary encoderów i powinien trafiać do metadanych Qdrant jako `embeddingModel`.

Wymiar embeddingu jest stały:

```text
768
```

Nie należy mieszać embeddingów z różnych modeli albo różnych wymiarów w jednej kolekcji Qdrant.

## Healthcheck

```bash
curl http://localhost:8081/health
```

Przykładowa odpowiedź:

```json
{
  "status": "ok",
  "model": "medcpt-ncbi-v1",
  "query_model": "ncbi/MedCPT-Query-Encoder",
  "document_model": "ncbi/MedCPT-Article-Encoder",
  "dimension": 768,
  "device": "cuda"
}
```

## Embedding dokumentów

Endpoint dla `ingestion-worker`.

```http
POST /embed/documents
Content-Type: application/json
```

Preferowany request:

```json
{
  "documents": [
    {
      "title": "Aspirin and fever reduction",
      "text": "Aspirin is an analgesic and antipyretic medication."
    },
    {
      "title": "Blood pressure monitoring in hypertension",
      "text": "Hypertension management requires repeated blood pressure measurement."
    }
  ]
}
```

Response:

```json
{
  "model": "medcpt-ncbi-v1",
  "encoder": "document",
  "encoder_model": "ncbi/MedCPT-Article-Encoder",
  "dimension": 768,
  "embeddings": [
    [0.01, -0.02, 0.03],
    [0.04, 0.05, -0.01]
  ]
}
```

Wektory w przykładzie są skrócone. Realnie każdy embedding ma `768` elementów.

Endpoint nadal przyjmuje prostszy format testowy:

```json
{
  "texts": [
    "Aspirin is an analgesic and antipyretic medication."
  ]
}
```

W tym wariancie tytuł dokumentu jest pusty. Dla danych PubMed/PMC lepiej używać formatu `documents`, bo MedCPT Article Encoder był projektowany dla par typu tytuł + abstrakt/tekst.

## Embedding zapytania

Niskopoziomowy endpoint do samego policzenia embeddingu zapytania.

```http
POST /embed/query
Content-Type: application/json
```

Request:

```json
{
  "text": "What reduces fever?"
}
```

Response:

```json
{
  "model": "medcpt-ncbi-v1",
  "encoder": "query",
  "encoder_model": "ncbi/MedCPT-Query-Encoder",
  "dimension": 768,
  "embeddings": [
    [0.02, -0.01, 0.05]
  ]
}
```

## Hybrydowe zapytanie do Qdrant

Endpoint dla `rag-api`. Liczy embedding zapytania przez MedCPT Query Encoder i od razu odpytuje Qdrant po named vectorze `medcpt_dense`.

```http
POST /embed/hybrid/query
Content-Type: application/json
```

Request:

```json
{
  "text": "What reduces fever?",
  "limit": 5
}
```

Response:

```json
{
  "model": "medcpt-ncbi-v1",
  "encoder": "query",
  "encoder_model": "ncbi/MedCPT-Query-Encoder",
  "dimension": 768,
  "collection": "MedicalChunk",
  "vector_name": "medcpt_dense",
  "documents": [
    {
      "id": "sample-pubmed-1:0",
      "title": "Aspirin and fever reduction",
      "content": "Aspirin is an analgesic and antipyretic medication.",
      "source": "sample-json: PMID 10000001",
      "score": 0.91,
      "metadata": {
        "pmid": "10000001",
        "year": 2024,
        "documentId": "sample-pubmed-1",
        "embeddingModel": "medcpt-ncbi-v1"
      }
    }
  ]
}
```

Nazwa endpointu zostaje `hybrid`, ale aktualny MVP wykonuje dense retrieval. Hybrydowe laczenie wynikow mozna rozszerzyc wewnatrz tego endpointu bez zmiany kontraktu `rag-api`.

## Kontrakt dla innych serwisów

`ingestion-worker` powinien:

1. sparsować dokument,
2. podzielić tekst na chunki,
3. wysłać chunki do `POST /embed/documents`,
4. zapisać do Qdrant tekst chunku, metadane i zwrócony vector,
5. zapisać wartość `model` z odpowiedzi jako `embeddingModel`.

`rag-api` powinno:

1. przyjąć pytanie użytkownika,
2. wysłać pytanie do `POST /embed/hybrid/query`,
3. opcjonalnie przekazać kandydatów do rerankera,
4. zbudować odpowiedź na podstawie znalezionych chunków.

## Uwagi produkcyjne

Mozna szybko sprawdzic, czy Docker widzi GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Jesli trzeba tymczasowo wrocic na CPU, zmien w `docker-compose.yml`:

```yaml
EMBEDDING_DEVICE: cpu
```

oraz zbuduj obraz z CPU-only PyTorch:

```bash
docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu embedding-service
```

Po zmianie encoderów trzeba utworzyć nową kolekcję Qdrant albo przeprowadzić pełną reindeksację korpusu. Nie dopisujemy embeddingów z nowej przestrzeni do starej kolekcji.
