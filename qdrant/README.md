# Qdrant Vector Store

Qdrant jest baza wektorowa dla pipeline RAG. Dziala w trybie bring-your-own-vectors: embeddingi MedCPT i sparse BM25 liczymy poza Qdrantem, a Qdrant przechowuje punkty i wykonuje hybrid search.

Aktualny MVP uzywa:

```text
collection: MedicalChunk
dense vector name: medcpt_dense
dense vector size: 768
dense distance: cosine
dense datatype: float16
dense vector storage: on disk
dense HNSW index: on disk
sparse vector name: bm25_sparse
sparse encoder: corpus-aware BM25
fusion: RRF
```

`float16` i on-disk storage zmniejszaja narzut RAM. Kosztem moze byc nizsza precyzja i wieksza latencja.

## Uruchomienie

GPU jest priorytetowym trybem dla `embedding-service`, ale sam Qdrant uruchamiasz tak samo:

```bash
docker compose up --build qdrant qdrant-init
```

Pelny wariant GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build qdrant qdrant-init embedding-service
```

Fallback CPU:

```bash
docker compose up --build qdrant qdrant-init embedding-service
```

Adresy:

```text
HTTP/REST: http://localhost:6333
gRPC: localhost:6334
```

## Inicjalizacja Kolekcji

Kolekcje inicjalizuje jednorazowy serwis:

```text
qdrant-init
```

Kod definicji jest w:

```text
qdrant/schema.py
```

Jesli kolekcja `MedicalChunk` juz istnieje bez sparse vectora `bm25_sparse`, ustaw `QDRANT_RECREATE_COLLECTION=true` albo uzyj `--recreate` w `scripts/rag/01_build_index.py`.

## Payload MedicalChunk

Jeden punkt Qdrant odpowiada jednemu chunkowi tekstu. Payload pozostaje w camelCase:

```text
text: string
chunkId: keyword
documentId: keyword
pmid: keyword
doi: keyword
title: text
journal: keyword
year: integer
authors: keyword[]
meshTerms: keyword[]
section: keyword
source: keyword
url: keyword
chunkIndex: integer
publicationDate: keyword
publicationTypes: keyword[]
isReview: bool
isSystematicReview: bool
wordCount: integer
embeddingModel: keyword
corpusVersion: keyword
textHash: keyword
```

`chunks.parquet` i `embeddings.parquet` uzywaja snake_case zgodnie z kontraktem zespolu. Mapowanie do camelCase dzieje sie podczas budowy indeksu.

## Budowanie Indeksu

Preferowany workflow:

```bash
python3 scripts/embeddings/01_embed_chunks.py \
  --chunks data/processed/chunks.parquet \
  --embedding-service-url http://localhost:8081 \
  --out data/embeddings/embeddings.parquet

python3 scripts/embeddings/02_validate_embeddings.py \
  --chunks data/processed/chunks.parquet \
  --embeddings data/embeddings/embeddings.parquet

python3 scripts/rag/01_build_index.py \
  --chunks data/processed/chunks.parquet \
  --embeddings data/embeddings/embeddings.parquet \
  --collection MedicalChunk \
  --qdrant-url http://localhost:6333 \
  --recreate
```

`01_build_index.py` moze tez policzyc embeddingi w locie, jesli nie podasz `--embeddings`, ale kontrakt zespolowy preferuje osobny plik `data/embeddings/embeddings.parquet`.

Skrypt zapisuje:

```text
data/bm25_stats.json
data/indexes/qdrant/index_manifest.json
```

Po przebudowie `data/bm25_stats.json` zrestartuj `embedding-service`, bo BM25 encoder jest cache'owany w procesie.

## Kontrakt Zapisu

Punkt Qdrant powinien zawierac dense i sparse vector:

```python
from qdrant_client import QdrantClient, models


client = QdrantClient(host="qdrant", port=6333)

client.upsert(
    collection_name="MedicalChunk",
    points=[
        models.PointStruct(
            id="stable-uuid-from-chunk-id",
            vector={
                "medcpt_dense": dense_vector,
                "bm25_sparse": models.SparseVector(
                    indices=sparse_indices,
                    values=sparse_values,
                ),
            },
            payload={
                "text": "Title: ... Abstract: ...",
                "chunkId": "pubmed:10000001:abstract:v1",
                "documentId": "pubmed:10000001",
                "pmid": "10000001",
                "doi": "10.example/sample",
                "title": "Aspirin and fever reduction",
                "journal": "Sample Medical Journal",
                "year": 2024,
                "section": "abstract",
                "source": "pubmed",
                "url": "https://pubmed.ncbi.nlm.nih.gov/10000001/",
                "chunkIndex": 0,
                "publicationDate": "2024-01-01",
                "publicationTypes": ["Review"],
                "isReview": True,
                "isSystematicReview": False,
                "wordCount": 120,
                "embeddingModel": "medcpt-ncbi-v1",
                "corpusVersion": "pubmed-rag-v1",
                "textHash": "source-text-hash",
            },
        )
    ],
)
```

## Kontrakt Odczytu

RAG API powinno zwracac znalezione chunki wraz ze zrodlami:

```text
chunk_id
score
pmid
title
text
doi
year
source
url
metadata
```

Publiczne endpointy:

```text
GET /search?q=...&top_k=10
GET /api/search?q=...&top_k=10
```

Smoke test:

```bash
python3 scripts/rag/02_search.py \
  --query "hypertension treatment" \
  --top-k 5 \
  --api-url http://127.0.0.1:8000 \
  --require-results
```

## Hybrid Search

Qdrant wykonuje RRF po dense i sparse prefetch:

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

Dense embedding zapytania liczy MedCPT Query Encoder. Sparse query vector powstaje z `data/bm25_stats.json`.

## Zasady Produkcyjne

Nie mieszaj roznych modeli embeddingowych w jednej kolekcji. Jesli zmieniasz model, utworz nowa kolekcje albo przeprowadz pelna reindeksacje.

Oryginalne dokumenty moga byc poza Qdrant, np. filesystem, S3 albo MinIO. Qdrant powinien trzymac chunki, wektory i metadane potrzebne do retrievalu oraz cytowan.
