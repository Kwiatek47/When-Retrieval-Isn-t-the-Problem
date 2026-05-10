from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.sparse import BM25SparseEncoder


DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_BM25_STATS_PATH = PROJECT_ROOT / "data" / "bm25_stats.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "indexes" / "qdrant" / "index_manifest.json"

REQUIRED_CHUNK_COLUMNS = {"chunk_id", "text"}
REQUIRED_EMBEDDING_COLUMNS = {"chunk_id", "embedding"}

PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "chunkId": models.PayloadSchemaType.KEYWORD,
    "documentId": models.PayloadSchemaType.KEYWORD,
    "pmid": models.PayloadSchemaType.KEYWORD,
    "doi": models.PayloadSchemaType.KEYWORD,
    "title": models.PayloadSchemaType.TEXT,
    "journal": models.PayloadSchemaType.KEYWORD,
    "year": models.PayloadSchemaType.INTEGER,
    "source": models.PayloadSchemaType.KEYWORD,
    "url": models.PayloadSchemaType.KEYWORD,
    "section": models.PayloadSchemaType.KEYWORD,
    "publicationTypes": models.PayloadSchemaType.KEYWORD,
    "embeddingModel": models.PayloadSchemaType.KEYWORD,
    "corpusVersion": models.PayloadSchemaType.KEYWORD,
    "textHash": models.PayloadSchemaType.KEYWORD,
}


def main() -> None:
    args = _parse_args()

    chunks = _load_chunks(args.chunks, limit=args.limit)
    if not chunks:
        raise RuntimeError(f"No chunks found in {args.chunks}.")

    if args.embeddings:
        embeddings, embedding_model = _load_embeddings(args.embeddings, chunks, args.embedding_dimension)
    else:
        embeddings, embedding_model = _embed_chunks(
            chunks=chunks,
            embedding_service_url=args.embedding_service_url,
            embedding_dimension=args.embedding_dimension,
            batch_size=args.embedding_batch_size,
        )

    bm25_encoder = BM25SparseEncoder.from_corpus(_sparse_text(chunk) for chunk in chunks)
    bm25_encoder.save(args.bm25_stats_out)

    client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
    try:
        _ensure_collection(client, args)
        _upsert_chunks(
            client=client,
            chunks=chunks,
            embeddings=embeddings,
            bm25_encoder=bm25_encoder,
            embedding_model=embedding_model,
            args=args,
        )
    finally:
        client.close()

    _write_manifest(
        path=args.manifest_out,
        chunks=chunks,
        embedding_model=embedding_model,
        args=args,
    )
    print(
        f"Indexed {len(chunks)} chunks into Qdrant collection {args.collection}. "
        f"BM25 stats: {args.bm25_stats_out}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Qdrant RAG index from data/processed/chunks.parquet and optional "
            "data/embeddings/embeddings.parquet."
        )
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "MedicalChunk"))
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-timeout", type=float, default=float(os.getenv("QDRANT_TIMEOUT", "30")))
    parser.add_argument("--dense-vector-name", default=os.getenv("QDRANT_VECTOR_NAME", "medcpt_dense"))
    parser.add_argument("--sparse-vector-name", default=os.getenv("QDRANT_SPARSE_VECTOR_NAME", "bm25_sparse"))
    parser.add_argument("--embedding-dimension", type=int, default=int(os.getenv("EMBEDDING_DIMENSION", "768")))
    parser.add_argument(
        "--embedding-service-url",
        default=os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8081"),
    )
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--bm25-stats-out", type=Path, default=DEFAULT_BM25_STATS_PATH)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--corpus-version", default=os.getenv("CORPUS_VERSION", "pubmed-rag-v1"))
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the Qdrant collection.")
    parser.add_argument("--limit", type=int, help="Index only the first N chunks for a pilot run.")
    return parser.parse_args()


def _load_chunks(path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Chunks parquet file does not exist: {path}")

    table = pq.read_table(path)
    _require_columns(table.column_names, REQUIRED_CHUNK_COLUMNS, path)
    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]

    chunks: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for index, row in enumerate(rows):
        chunk_id = _clean_str(row.get("chunk_id"))
        text = _clean_str(row.get("text"))
        if not chunk_id:
            raise RuntimeError(f"Chunk at row {index} has empty chunk_id.")
        if chunk_id in seen_chunk_ids:
            raise RuntimeError(f"Duplicate chunk_id in chunks parquet: {chunk_id}")
        if not text:
            raise RuntimeError(f"Chunk {chunk_id} has empty text.")

        seen_chunk_ids.add(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": _clean_str(row.get("doc_id")) or chunk_id,
                "pmid": _clean_str(row.get("pmid")),
                "title": _clean_str(row.get("title")) or "Untitled medical chunk",
                "text": text,
                "source": _clean_str(row.get("source")) or "pubmed",
                "url": _clean_str(row.get("url")),
                "doi": _clean_str(row.get("doi")),
                "year": _to_int(row.get("year")),
                "publication_date": _clean_str(row.get("publication_date")),
                "publication_types": _as_str_list(row.get("publication_types")),
                "journal": _clean_str(row.get("journal")),
                "section": _clean_str(row.get("section")) or "abstract",
                "is_review": _to_bool(row.get("is_review")),
                "is_systematic_review": _to_bool(row.get("is_systematic_review")),
                "word_count": _to_int(row.get("word_count")),
                "text_hash": _clean_str(row.get("text_hash")),
                "chunk_index": _to_int(row.get("chunk_index")) or index,
            }
        )

    return chunks


def _load_embeddings(
    path: Path,
    chunks: list[dict[str, Any]],
    expected_dimension: int,
) -> tuple[dict[str, list[float]], str]:
    if not path.exists():
        raise RuntimeError(f"Embeddings parquet file does not exist: {path}")

    table = pq.read_table(path)
    _require_columns(table.column_names, REQUIRED_EMBEDDING_COLUMNS, path)

    embeddings: dict[str, list[float]] = {}
    embedding_models: set[str] = set()
    dimensions: set[int] = set()

    for index, row in enumerate(table.to_pylist()):
        chunk_id = _clean_str(row.get("chunk_id"))
        if not chunk_id:
            raise RuntimeError(f"Embedding at row {index} has empty chunk_id.")
        if chunk_id in embeddings:
            raise RuntimeError(f"Duplicate embedding for chunk_id: {chunk_id}")

        vector = _float_vector(row.get("embedding"), label=f"embedding row {index}")
        dimensions.add(len(vector))
        declared_dimension = _to_int(row.get("embedding_dim"))
        if declared_dimension is not None and declared_dimension != len(vector):
            raise RuntimeError(
                f"Embedding row {index} declares dimension {declared_dimension}, "
                f"but vector has dimension {len(vector)}."
            )
        embeddings[chunk_id] = vector

        embedding_model = _clean_str(row.get("embedding_model"))
        if embedding_model:
            embedding_models.add(embedding_model)

    if dimensions != {expected_dimension}:
        raise RuntimeError(
            f"Expected embedding dimension {expected_dimension}, got {sorted(dimensions)} from {path}."
        )

    missing = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in embeddings]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Embeddings parquet is missing {len(missing)} chunks. Examples: {preview}")

    if len(embedding_models) > 1:
        raise RuntimeError(f"Embeddings parquet contains multiple embedding_model values: {sorted(embedding_models)}")

    embedding_model = next(iter(embedding_models), "external-embeddings")
    return embeddings, embedding_model


def _embed_chunks(
    *,
    chunks: list[dict[str, Any]],
    embedding_service_url: str,
    embedding_dimension: int,
    batch_size: int,
) -> tuple[dict[str, list[float]], str]:
    embeddings: dict[str, list[float]] = {}
    embedding_model = ""
    service_url = embedding_service_url.rstrip("/")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        payload = {
            "documents": [
                {
                    "title": chunk["title"],
                    "text": chunk["text"],
                }
                for chunk in batch
            ]
        }
        response = _request_json("POST", f"{service_url}/embed/documents", payload)
        dimension = int(response.get("dimension") or 0)
        if dimension != embedding_dimension:
            raise RuntimeError(f"Expected embedding dimension {embedding_dimension}, got {dimension}.")

        response_embeddings = response.get("embeddings") or []
        if len(response_embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings from embedding-service, got {len(response_embeddings)}."
            )

        embedding_model = _clean_str(response.get("model")) or embedding_model
        for chunk, vector in zip(batch, response_embeddings, strict=True):
            embeddings[chunk["chunk_id"]] = _float_vector(vector, label=f"chunk {chunk['chunk_id']}")

        print(f"Embedded {min(start + len(batch), len(chunks))}/{len(chunks)} chunks.")

    return embeddings, embedding_model or "embedding-service"


def _ensure_collection(client: QdrantClient, args: argparse.Namespace) -> None:
    if args.recreate and client.collection_exists(collection_name=args.collection):
        client.delete_collection(collection_name=args.collection)

    if not client.collection_exists(collection_name=args.collection):
        client.create_collection(
            collection_name=args.collection,
            vectors_config={
                args.dense_vector_name: models.VectorParams(
                    size=args.embedding_dimension,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                args.sparse_vector_name: models.SparseVectorParams(),
            },
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )
    else:
        collection = client.get_collection(collection_name=args.collection)
        vectors = getattr(collection.config.params, "vectors", {}) or {}
        sparse_vectors = getattr(collection.config.params, "sparse_vectors", {}) or {}
        if args.dense_vector_name not in vectors:
            raise RuntimeError(
                f"Collection {args.collection} exists without dense vector {args.dense_vector_name}. "
                "Use --recreate or choose another collection."
            )
        if args.sparse_vector_name not in sparse_vectors:
            raise RuntimeError(
                f"Collection {args.collection} exists without sparse vector {args.sparse_vector_name}. "
                "Use --recreate or choose another collection."
            )

    for field_name, field_schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(
                collection_name=args.collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise


def _upsert_chunks(
    *,
    client: QdrantClient,
    chunks: list[dict[str, Any]],
    embeddings: dict[str, list[float]],
    bm25_encoder: BM25SparseEncoder,
    embedding_model: str,
    args: argparse.Namespace,
) -> None:
    for start in range(0, len(chunks), args.upsert_batch_size):
        batch = chunks[start : start + args.upsert_batch_size]
        points = [
            _build_point(
                chunk=chunk,
                embedding=embeddings[chunk["chunk_id"]],
                bm25_encoder=bm25_encoder,
                embedding_model=embedding_model,
                args=args,
            )
            for chunk in batch
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        print(f"Upserted {min(start + len(batch), len(chunks))}/{len(chunks)} chunks.")


def _build_point(
    *,
    chunk: dict[str, Any],
    embedding: list[float],
    bm25_encoder: BM25SparseEncoder,
    embedding_model: str,
    args: argparse.Namespace,
) -> models.PointStruct:
    sparse_payload = bm25_encoder.encode_document(_sparse_text(chunk))
    sparse_vector = models.SparseVector(
        indices=sparse_payload["indices"],
        values=sparse_payload["values"],
    )
    payload = {
        "text": chunk["text"],
        "chunkId": chunk["chunk_id"],
        "documentId": chunk["doc_id"],
        "pmid": chunk["pmid"],
        "title": chunk["title"],
        "journal": chunk["journal"],
        "year": chunk["year"],
        "doi": chunk["doi"],
        "source": chunk["source"],
        "url": _source_url(chunk),
        "section": chunk["section"],
        "chunkIndex": chunk["chunk_index"],
        "publicationDate": chunk["publication_date"],
        "publicationTypes": chunk["publication_types"],
        "isReview": chunk["is_review"],
        "isSystematicReview": chunk["is_systematic_review"],
        "wordCount": chunk["word_count"],
        "textHash": chunk["text_hash"],
        "embeddingModel": embedding_model,
        "corpusVersion": args.corpus_version,
    }
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),
        vector={
            args.dense_vector_name: embedding,
            args.sparse_vector_name: sparse_vector,
        },
        payload={key: value for key, value in payload.items() if _has_payload_value(value)},
    )


def _write_manifest(
    *,
    path: Path,
    chunks: list[dict[str, Any]],
    embedding_model: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "chunk_count": len(chunks),
        "chunks_path": str(args.chunks),
        "embeddings_path": str(args.embeddings) if args.embeddings else None,
        "embedding_model": embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "dense_vector_name": args.dense_vector_name,
        "sparse_vector_name": args.sparse_vector_name,
        "bm25_stats_path": str(args.bm25_stats_out),
        "corpus_version": args.corpus_version,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _sparse_text(chunk: dict[str, Any]) -> str:
    return " ".join(
        [
            chunk["title"],
            chunk["text"],
            chunk["journal"],
            chunk["pmid"],
            chunk["doi"],
            " ".join(chunk["publication_types"]),
        ]
    )


def _source_url(chunk: dict[str, Any]) -> str:
    if chunk["url"]:
        return chunk["url"]
    if chunk["pmid"]:
        return f"https://pubmed.ncbi.nlm.nih.gov/{chunk['pmid']}/"
    return chunk["source"]


def _has_payload_value(value: Any) -> bool:
    if value is None:
        return False
    if value == "":
        return False
    if isinstance(value, list) and not value:
        return False
    return True


def _require_columns(column_names: list[str], required: set[str], path: Path) -> None:
    missing = sorted(required - set(column_names))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")


def _request_json(method: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urlopen(request, timeout=240) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _float_vector(value: Any, *, label: str) -> list[float]:
    if value is None:
        raise RuntimeError(f"{label} is empty.")
    vector = [float(item) for item in value]
    if not vector:
        raise RuntimeError(f"{label} is empty.")
    if any(math.isnan(item) or math.isinf(item) for item in vector):
        raise RuntimeError(f"{label} contains NaN or infinity.")
    if not any(item != 0.0 for item in vector):
        raise RuntimeError(f"{label} is a zero vector.")
    return vector


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(";") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


if __name__ == "__main__":
    main()
