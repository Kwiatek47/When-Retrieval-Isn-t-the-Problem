# Weaviate Vector Store

Osobna baza wektorowa dla pipeline RAG. Weaviate nie liczy embeddingów. Działa w trybie bring your own vectors, więc inne serwisy muszą dostarczać gotowe wektory.

## Uruchomienie

Z katalogu głównego projektu:

```bash
docker compose up --build weaviate weaviate-schema
```

Lokalne adresy:

```text
HTTP: http://localhost:8080
gRPC: localhost:50051
```

W sieci Docker Compose inne kontenery powinny używać:

```text
HTTP host: weaviate
HTTP port: 8080
gRPC host: weaviate
gRPC port: 50051
```

## Tryb pracy

W `docker-compose.yml` ustawione jest:

```text
DEFAULT_VECTORIZER_MODULE=none
ENABLE_MODULES=
```

To oznacza, że Weaviate nie wykonuje automatycznej wektoryzacji. Każdy zapis do kolekcji `MedicalChunk` musi zawierać gotowy vector wyliczony poza Weaviate, np. przez `embedding-service`.

## Inicjalizacja kolekcji

Kolekcję inicjalizuje jednorazowy serwis:

```text
weaviate-schema
```

Kod definicji znajduje się w:

```text
weaviate/schema.py
```

Uruchomienie:

```bash
docker compose up --build weaviate-schema
```

Jeśli kolekcja `MedicalChunk` już istnieje, skrypt niczego nie nadpisuje.

## Kolekcja MedicalChunk

Kolekcja przechowuje chunki dokumentów medycznych i metadane potrzebne do retrievalu.

Pola:

```text
text: text
pmid: text
title: text
journal: text
year: int
authors: text[]
meshTerms: text[]
section: text
source: text
chunkIndex: int
documentId: text
embeddingModel: text
corpusVersion: text
```

Vector jest self-provided, czyli przekazywany przy zapisie obiektu. Dystans w indeksie HNSW jest ustawiony na `cosine`.

Aktualny serwis embeddingów zwraca MedCPT vectors o wymiarze `768`. W polu `embeddingModel` zapisujemy wartość `model` zwróconą przez `embedding-service`, obecnie:

```text
medcpt-ncbi-v1
```

## Kontrakt zapisu dla ingestion-worker

`ingestion-worker` powinien zapisywać jeden obiekt na jeden chunk.

Minimalny przepływ:

1. sparsuj dokument PubMed/PMC,
2. zbuduj chunki,
3. wyślij teksty chunków do `embedding-service` przez `/embed/documents`,
4. dla każdego chunku zapisz do Weaviate:
   - `properties` z tekstem i metadanymi,
   - `vector` zwrócony przez embedding-service.

Przykładowy obiekt logiczny:

```json
{
  "properties": {
    "text": "Aspirin is an analgesic and antipyretic medication.",
    "pmid": "10000001",
    "title": "Aspirin and fever reduction",
    "journal": "Sample Medical Journal",
    "year": 2024,
    "authors": ["A. Kowalski", "B. Nowak"],
    "meshTerms": ["Aspirin", "Fever", "Analgesics"],
    "section": "abstract",
    "source": "sample-json",
    "chunkIndex": 0,
    "documentId": "sample-pubmed-1",
    "embeddingModel": "medcpt-ncbi-v1",
    "corpusVersion": "sample-v1"
  },
  "vector": [0.0, 0.12, -0.04]
}
```

Wektor w przykładzie jest skrócony.

## Przykład klienta Python

Kod dla usług `rag-api` i `ingestion-worker` powinien używać klienta Weaviate v4.

```python
import weaviate


client = weaviate.connect_to_custom(
    http_host="weaviate",
    http_port=8080,
    http_secure=False,
    grpc_host="weaviate",
    grpc_port=50051,
    grpc_secure=False,
)

try:
    chunks = client.collections.get("MedicalChunk")

    chunks.data.insert(
        properties={
            "text": "Aspirin is an analgesic and antipyretic medication.",
            "pmid": "10000001",
            "title": "Aspirin and fever reduction",
            "journal": "Sample Medical Journal",
            "year": 2024,
            "authors": ["A. Kowalski", "B. Nowak"],
            "meshTerms": ["Aspirin", "Fever", "Analgesics"],
            "section": "abstract",
            "source": "sample-json",
            "chunkIndex": 0,
            "documentId": "sample-pubmed-1",
            "embeddingModel": "medcpt-ncbi-v1",
            "corpusVersion": "sample-v1",
        },
        vector=[0.0] * 768,
    )
finally:
    client.close()
```

## Kontrakt odczytu dla rag-api

`rag-api` powinno:

1. wysłać pytanie do `embedding-service` przez `/embed/query`,
2. pobrać embedding zapytania,
3. wykonać wyszukiwanie near-vector w `MedicalChunk`,
4. zwrócić znalezione chunki wraz ze źródłami.

Przykład:

```python
from weaviate.classes.query import MetadataQuery


chunks = client.collections.get("MedicalChunk")
result = chunks.query.near_vector(
    near_vector=query_vector,
    limit=10,
    return_metadata=MetadataQuery(distance=True),
)

for item in result.objects:
    print(item.properties["title"], item.properties["pmid"], item.metadata.distance)
```

## Zasady produkcyjne

Nie mieszać różnych modeli embeddingowych w jednej kolekcji. Jeśli zmieniasz model, utwórz nową kolekcję albo nową wersję korpusu i przeprowadź reindeksację.

Oryginalne dokumenty mogą być poza Weaviate, np. filesystem, S3 albo MinIO. Weaviate powinien trzymać tylko chunki, wektory i metadane potrzebne do retrievalu oraz cytowań.
