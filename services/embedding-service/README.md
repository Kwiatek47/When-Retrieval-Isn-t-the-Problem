# Embedding Service

Osobny serwis HTTP odpowiedzialny wyłącznie za liczenie embeddingów dla pipeline RAG. Inne usługi nie powinny importować kodu tego serwisu bezpośrednio, tylko komunikować się z nim przez API.

Aktualna implementacja używa MedCPT:

- `ncbi/MedCPT-Article-Encoder` dla dokumentów i chunków,
- `ncbi/MedCPT-Query-Encoder` dla zapytań użytkownika.

Oba encodery pracują w tej samej przestrzeni retrievalowej i zwracają embeddingi o wymiarze `768`.

## Uruchomienie

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
EMBEDDING_DEVICE=cpu
HF_HOME=/models/huggingface
```

`EMBEDDING_MODEL_NAME` jest stabilnym identyfikatorem pary encoderów i powinien trafiać do metadanych Weaviate jako `embeddingModel`.

Wymiar embeddingu jest stały:

```text
768
```

Nie należy mieszać embeddingów z różnych modeli albo różnych wymiarów w jednej kolekcji Weaviate.

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
  "device": "cpu"
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

Endpoint dla `rag-api`.

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

## Kontrakt dla innych serwisów

`ingestion-worker` powinien:

1. sparsować dokument,
2. podzielić tekst na chunki,
3. wysłać chunki do `POST /embed/documents`,
4. zapisać do Weaviate tekst chunku, metadane i zwrócony vector,
5. zapisać wartość `model` z odpowiedzi jako `embeddingModel`.

`rag-api` powinno:

1. przyjąć pytanie użytkownika,
2. wysłać pytanie do `POST /embed/query`,
3. wysłać otrzymany vector do Weaviate jako zapytanie near-vector,
4. opcjonalnie przekazać kandydatów do rerankera,
5. zbudować odpowiedź na podstawie znalezionych chunków.

## Uwagi produkcyjne

Aktualnie serwis działa CPU-only. To jest poprawne dla MVP, ale przy większym ingestowaniu PubMed/PMC będzie wolne. Przejście na GPU powinno wymagać głównie zmiany obrazu Dockera, instalacji odpowiedniego PyTorch i ustawienia:

```text
EMBEDDING_DEVICE=cuda
```

Po zmianie encoderów trzeba utworzyć nową kolekcję Weaviate albo przeprowadzić pełną reindeksację korpusu. Nie dopisujemy embeddingów z nowej przestrzeni do starej kolekcji.
