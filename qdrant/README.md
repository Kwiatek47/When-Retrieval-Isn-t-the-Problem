# Qdrant Vector Store

Osobna baza wektorowa dla pipeline RAG. Qdrant nie liczy embeddingów MedCPT. Działa w trybie bring your own vectors, więc inne serwisy muszą dostarczać gotowe wektory.

Aktualny MVP używa tylko dense retrievalu:

```text
collection: MedicalChunk
vector name: medcpt_dense
vector size: 768
distance: cosine
datatype: float16
vector storage: on disk
HNSW index: on disk
```

`float16` i on-disk storage zmniejszają narzut pamięci względem trzymania pełnych `float32` w RAM. Kosztem może być niższa precyzja i większa latencja.

## Uruchomienie

Z katalogu głównego projektu:

```bash
docker compose up --build qdrant qdrant-init
```

Lokalne adresy:

```text
HTTP/REST: http://localhost:6333
gRPC: localhost:6334
```

W sieci Docker Compose inne kontenery powinny używać:

```text
QDRANT_HOST=qdrant
QDRANT_PORT=6333
```

## Inicjalizacja kolekcji

Kolekcję inicjalizuje jednorazowy serwis:

```text
qdrant-init
```

Kod definicji znajduje się w:

```text
qdrant/schema.py
```

Jeśli kolekcja `MedicalChunk` już istnieje, skrypt jej nie usuwa i nie nadpisuje. Dodatkowo tworzy indeksy payload dla pól metadanych.

## Payload MedicalChunk

Jeden punkt Qdrant odpowiada jednemu chunkowi tekstu.

Payload:

```text
text: string
pmid: keyword
title: text
journal: keyword
year: integer
authors: keyword[]
meshTerms: keyword[]
section: keyword
source: keyword
chunkIndex: integer
documentId: keyword
embeddingModel: keyword
corpusVersion: keyword
```

W polu `embeddingModel` zapisujemy wartość `model` zwróconą przez `embedding-service`, obecnie:

```text
medcpt-ncbi-v1
```

## Kontrakt zapisu dla ingestion-worker

`ingestion-worker` powinien zapisywać jeden punkt na jeden chunk.

Minimalny przepływ:

1. sparsuj dokument PubMed/PMC,
2. zbuduj chunki,
3. wyślij chunki do `embedding-service` przez `/embed/documents`,
4. zapisz do Qdrant payload chunku i named vector `medcpt_dense`.

Przykład:

```python
from qdrant_client import QdrantClient, models


client = QdrantClient(host="qdrant", port=6333)

client.upsert(
    collection_name="MedicalChunk",
    points=[
        models.PointStruct(
            id="sample-pubmed-1:0",
            vector={
                "medcpt_dense": dense_vector,
            },
            payload={
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
        )
    ],
)
```

## Kontrakt odczytu dla rag-api

`rag-api` powinno:

1. przyjąć pytanie użytkownika,
2. wysłać pytanie do `embedding-service` przez `/embed/query`,
3. wysłać otrzymany vector do Qdrant jako query po named vectorze `medcpt_dense`,
4. opcjonalnie przekazać kandydatów do rerankera,
5. zwrócić znalezione chunki wraz ze źródłami.

Przykład:

```python
from qdrant_client import QdrantClient


client = QdrantClient(host="qdrant", port=6333)

result = client.query_points(
    collection_name="MedicalChunk",
    query=query_vector,
    using="medcpt_dense",
    limit=10,
    with_payload=True,
)

for point in result.points:
    print(point.payload["title"], point.payload["pmid"], point.score)
```

## Hybrydowe wyszukiwanie

Qdrant nie wygeneruje za nas ani embeddingu MedCPT, ani klasycznego BM25. Ma natomiast mechanizmy, które pozwalają zbudować hybrydę:

- dense vector search,
- sparse vectors,
- payload full-text indexes,
- Query API z fuzją wyników, np. RRF albo DBSF.

Są dwa sensowne warianty.

### Wariant 1: hybryda wewnątrz Qdrant

Ten wariant trzyma dense i sparse vector w tym samym punkcie Qdrant.

Kolekcja musi mieć dwa wektory:

```python
from qdrant_client import models


client.create_collection(
    collection_name="MedicalChunk",
    vectors_config={
        "medcpt_dense": models.VectorParams(
            size=768,
            distance=models.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "bm25_sparse": models.SparseVectorParams(),
    },
)
```

`ingestion-worker` musi wtedy policzyć dwie reprezentacje:

- dense: `embedding-service` / MedCPT Article Encoder,
- sparse: osobny lexical encoder, np. BM25/SPLADE/fastembed sparse.

Zapis punktu:

```python
models.PointStruct(
    id="sample-pubmed-1:0",
    vector={
        "medcpt_dense": dense_vector,
        "bm25_sparse": models.SparseVector(
            indices=sparse_indices,
            values=sparse_values,
        ),
    },
    payload={...},
)
```

`rag-api` liczy dense embedding pytania oraz sparse vector pytania, a potem odpytuje Qdrant przez Query API z fuzją:

```python
result = client.query_points(
    collection_name="MedicalChunk",
    prefetch=[
        models.Prefetch(
            query=query_dense_vector,
            using="medcpt_dense",
            limit=100,
        ),
        models.Prefetch(
            query=models.SparseVector(
                indices=query_sparse_indices,
                values=query_sparse_values,
            ),
            using="bm25_sparse",
            limit=100,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=20,
    with_payload=True,
)
```

To jest najprostszy wariant operacyjnie, bo fuzję wyników robi Qdrant. Nadal trzeba samemu policzyć sparse vectors.

### Wariant 2: lexical search poza Qdrant

Ten wariant jest bardziej klasyczny dla dużych korpusów.

Architektura:

```text
rag-api
  -> embedding-service -> MedCPT dense query vector
  -> qdrant -> dense top 100
  -> lexical-service/OpenSearch/Tantivy/Postgres FTS -> BM25 top 100
  -> RRF/weighted fusion w rag-api
  -> opcjonalnie reranker-service
```

Prosty RRF:

```python
def rrf(result_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, point_id in enumerate(results, start=1):
            scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (k + rank)
    return scores
```

Ten wariant daje większą kontrolę nad BM25, analizatorami językowymi, synonymami MeSH i rankingiem lexical, ale wymaga utrzymania drugiego indeksu.

## Zasady produkcyjne

Nie mieszać różnych modeli embeddingowych w jednej kolekcji. Jeśli zmieniasz model, utwórz nową kolekcję albo przeprowadź pełną reindeksację.

Oryginalne dokumenty mogą być poza Qdrant, np. filesystem, S3 albo MinIO. Qdrant powinien trzymać chunki, wektory i metadane potrzebne do retrievalu oraz cytowań.
