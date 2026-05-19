from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.rag.sparse import BM25SparseEncoder, BM25Stats


DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_BM25_STATS_PATH = PROJECT_ROOT / "data" / "bm25_stats.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "indexes" / "qdrant"
DEFAULT_MANIFEST_PATH = DEFAULT_INDEX_DIR / "index_manifest.json"
DEFAULT_CHECKPOINT_PATH = DEFAULT_INDEX_DIR / "index_checkpoint.json"
DEFAULT_CHUNK_STORE_PATH = DEFAULT_INDEX_DIR / "chunk_store.sqlite"

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
    "publicationDate": models.PayloadSchemaType.KEYWORD,
    "publicationTypes": models.PayloadSchemaType.KEYWORD,
    "isReview": models.PayloadSchemaType.BOOL,
    "isSystematicReview": models.PayloadSchemaType.BOOL,
    "wordCount": models.PayloadSchemaType.INTEGER,
    "embeddingModel": models.PayloadSchemaType.KEYWORD,
    "corpusVersion": models.PayloadSchemaType.KEYWORD,
    "textHash": models.PayloadSchemaType.KEYWORD,
}


def main() -> None:
    args = _parse_args()
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    args.chunk_store.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(args.chunk_store)
    try:
        chunk_count, bm25_encoder = _prepare_chunk_store(conn, args)
        bm25_encoder.save(args.bm25_stats_out)

        client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
        try:
            _ensure_collection(client, args)
            if args.embeddings:
                embedding_model, upserted_count, failed_count = _upsert_embedding_files(
                    conn=conn,
                    client=client,
                    bm25_encoder=bm25_encoder,
                    args=args,
                )
            else:
                embedding_model, upserted_count, failed_count = _embed_and_upsert_chunks(
                    conn=conn,
                    client=client,
                    bm25_encoder=bm25_encoder,
                    args=args,
                )
        finally:
            client.close()

        if upserted_count < chunk_count and not args.allow_missing_embeddings:
            raise RuntimeError(
                f"Indexed {upserted_count}/{chunk_count} chunks. "
                "Pass --allow-missing-embeddings only for intentional partial builds."
            )

        _write_manifest(
            path=args.manifest_out,
            chunk_count=chunk_count,
            upserted_count=upserted_count,
            failed_count=failed_count,
            embedding_model=embedding_model,
            args=args,
        )
        _write_checkpoint(
            args,
            stage="complete",
            upserted_count=upserted_count,
            failed_count=failed_count,
            chunk_count=chunk_count,
        )
        print(
            f"Indexed {upserted_count}/{chunk_count} chunks into Qdrant collection {args.collection}. "
            f"BM25 stats: {args.bm25_stats_out}"
        )
    finally:
        conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a streaming Qdrant RAG index from chunks.parquet and one or more "
            "embedding shard parquet files."
        )
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument(
        "--embeddings",
        type=Path,
        nargs="+",
        action="append",
        default=[],
        help=(
            "Embedding parquet shard. Pass multiple paths after one flag or repeat "
            "the flag for multiple shards."
        ),
    )
    parser.add_argument(
        "--collection",
        default=os.getenv("QDRANT_COLLECTION", "MedicalChunk_pubmed_reviews_v1_medcpt_20260518"),
    )
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
    parser.add_argument("--read-batch-size", type=int, default=2048)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--bm25-stats-out", type=Path, default=DEFAULT_BM25_STATS_PATH)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--checkpoint-out", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--chunk-store", type=Path, default=DEFAULT_CHUNK_STORE_PATH)
    parser.add_argument("--corpus-version", default=os.getenv("CORPUS_VERSION", "pubmed-reviews-v1"))
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the Qdrant collection.")
    parser.add_argument("--resume", action="store_true", help="Reuse the chunk store and skip already indexed chunks.")
    parser.add_argument("--allow-missing-embeddings", action="store_true")
    parser.add_argument("--limit", type=int, help="Index only the first N chunks for a pilot run.")
    args = parser.parse_args()
    args.embeddings = [path for group in args.embeddings for path in group]
    return args


def _prepare_chunk_store(
    conn: sqlite3.Connection,
    args: argparse.Namespace,
) -> tuple[int, BM25SparseEncoder]:
    if args.resume and args.chunk_store.exists() and args.bm25_stats_out.exists():
        _initialize_store(conn, reset=False)
        chunk_count = _chunk_store_count(conn)
        if chunk_count > 0:
            print(f"Reusing chunk store with {chunk_count} chunks: {args.chunk_store}")
            return chunk_count, BM25SparseEncoder.from_file(args.bm25_stats_out)

    _initialize_store(conn, reset=True)
    document_frequency: Counter[str] = Counter()
    total_document_length = 0
    chunk_count = 0

    for row_group_index, batch_index, rows in _iter_parquet_rows(
        args.chunks,
        required_columns=REQUIRED_CHUNK_COLUMNS,
        batch_size=args.read_batch_size,
    ):
        for row in rows:
            if args.limit is not None and chunk_count >= args.limit:
                break
            chunk = _chunk_from_row(row, index=chunk_count)
            sparse_text = _sparse_text(chunk)
            tokens = _tokenize(sparse_text)
            total_document_length += len(tokens)
            document_frequency.update(set(tokens))
            try:
                conn.execute(
                    "INSERT INTO chunks(chunk_id, payload_json, sparse_text) VALUES (?, ?, ?)",
                    (chunk["chunk_id"], json.dumps(chunk, ensure_ascii=False), sparse_text),
                )
            except sqlite3.IntegrityError as exc:
                raise RuntimeError(f"Duplicate chunk_id in chunks parquet: {chunk['chunk_id']}") from exc
            chunk_count += 1
        conn.commit()
        _write_checkpoint(
            args,
            stage="build_chunk_store",
            last_row_group=row_group_index,
            last_batch=batch_index,
            chunk_count=chunk_count,
        )
        print(f"Prepared {chunk_count} chunks for indexing.")
        if args.limit is not None and chunk_count >= args.limit:
            break

    if chunk_count <= 0:
        raise RuntimeError(f"No chunks found in {args.chunks}.")
    if total_document_length <= 0:
        raise RuntimeError("Cannot build BM25 stats because all chunks tokenize to zero length.")

    vocabulary = {term: index for index, term in enumerate(sorted(document_frequency))}
    bm25_stats = BM25Stats(
        document_count=chunk_count,
        average_document_length=total_document_length / chunk_count,
        document_frequency=dict(document_frequency),
        vocabulary=vocabulary,
    )
    return chunk_count, BM25SparseEncoder(bm25_stats)


def _initialize_store(conn: sqlite3.Connection, *, reset: bool) -> None:
    if reset:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute("DROP TABLE IF EXISTS indexed_embeddings")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            sparse_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_embeddings (
            chunk_id TEXT PRIMARY KEY,
            embedding_path TEXT NOT NULL,
            embedding_row INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()


def _chunk_store_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    return int(row[0] or 0)


def _upsert_embedding_files(
    *,
    conn: sqlite3.Connection,
    client: QdrantClient,
    bm25_encoder: BM25SparseEncoder,
    args: argparse.Namespace,
) -> tuple[str, int, int]:
    embedding_models: set[str] = set()
    upserted_count = _indexed_count(conn) if args.resume else 0
    failed_count = 0

    for file_index, path in enumerate(args.embeddings):
        if not path.exists():
            raise RuntimeError(f"Embeddings parquet file does not exist: {path}")

        file_row_offset = 0
        for row_group_index, batch_index, rows in _iter_parquet_rows(
            path,
            required_columns=REQUIRED_EMBEDDING_COLUMNS,
            selected_columns=_embedding_columns(path),
            batch_size=args.read_batch_size,
        ):
            pending: list[tuple[str, models.PointStruct, int]] = []
            seen_in_batch: set[str] = set()
            for row_index, row in enumerate(rows, start=file_row_offset):
                chunk_id = _clean_str(row.get("chunk_id"))
                if not chunk_id:
                    raise RuntimeError(f"Embedding at row {row_index} in {path} has empty chunk_id.")
                if chunk_id in seen_in_batch:
                    raise RuntimeError(f"Duplicate embedding for chunk_id in one batch: {chunk_id}")
                seen_in_batch.add(chunk_id)
                if _is_indexed(conn, chunk_id):
                    if args.resume:
                        continue
                    raise RuntimeError(f"Duplicate embedding for already indexed chunk_id: {chunk_id}")

                stored_chunk = _chunk_from_store(conn, chunk_id)
                if stored_chunk is None:
                    if args.limit is not None:
                        continue
                    failed_count += 1
                    raise RuntimeError(f"Embedding shard contains chunk_id missing from chunks parquet: {chunk_id}")

                vector = _float_vector(row.get("embedding"), label=f"{path} row {row_index}")
                if len(vector) != args.embedding_dimension:
                    raise RuntimeError(
                        f"Expected embedding dimension {args.embedding_dimension}, "
                        f"got {len(vector)} for {chunk_id}."
                    )
                declared_dimension = _to_int(row.get("embedding_dim"))
                if declared_dimension is not None and declared_dimension != len(vector):
                    raise RuntimeError(
                        f"Embedding row {row_index} declares dimension {declared_dimension}, "
                        f"but vector has dimension {len(vector)}."
                    )

                embedding_model = _clean_str(row.get("embedding_model"))
                if embedding_model:
                    embedding_models.add(embedding_model)
                point = _build_point(
                    chunk=stored_chunk["chunk"],
                    sparse_text=stored_chunk["sparse_text"],
                    embedding=vector,
                    bm25_encoder=bm25_encoder,
                    embedding_model=embedding_model or "external-embeddings",
                    args=args,
                )
                pending.append((chunk_id, point, row_index))

            upserted_count += _upsert_pending_points(
                conn=conn,
                client=client,
                pending=pending,
                embedding_path=path,
                args=args,
            )
            file_row_offset += len(rows)
            _write_checkpoint(
                args,
                stage="embedding_shard",
                embedding_path=str(path),
                embedding_file_index=file_index,
                last_row_group=row_group_index,
                last_batch=batch_index,
                upserted_count=upserted_count,
                failed_count=failed_count,
            )
            print(f"Indexed {upserted_count} chunks after {path.name} row group {row_group_index}.")

    if len(embedding_models) > 1:
        raise RuntimeError(f"Embedding shards contain multiple embedding_model values: {sorted(embedding_models)}")
    return next(iter(embedding_models), "external-embeddings"), upserted_count, failed_count


def _embed_and_upsert_chunks(
    *,
    conn: sqlite3.Connection,
    client: QdrantClient,
    bm25_encoder: BM25SparseEncoder,
    args: argparse.Namespace,
) -> tuple[str, int, int]:
    upserted_count = _indexed_count(conn) if args.resume else 0
    failed_count = 0
    embedding_model = ""
    service_url = args.embedding_service_url.rstrip("/")

    for batch_index, rows in enumerate(_chunk_store_batches(conn, args.embedding_batch_size)):
        chunks = [json.loads(row["payload_json"]) for row in rows]
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if args.resume:
            chunks = [chunk for chunk in chunks if not _is_indexed(conn, chunk["chunk_id"])]
        if not chunks:
            continue

        payload = {
            "documents": [
                {
                    "title": chunk["title"],
                    "text": chunk["text"],
                }
                for chunk in chunks
            ]
        }
        response = _request_json("POST", f"{service_url}/embed/documents", payload)
        dimension = int(response.get("dimension") or 0)
        if dimension != args.embedding_dimension:
            raise RuntimeError(f"Expected embedding dimension {args.embedding_dimension}, got {dimension}.")
        response_embeddings = response.get("embeddings") or []
        if len(response_embeddings) != len(chunks):
            raise RuntimeError(
                f"Expected {len(chunks)} embeddings from embedding-service, got {len(response_embeddings)}."
            )

        embedding_model = _clean_str(response.get("model")) or embedding_model
        pending = []
        for row_index, (chunk, vector) in enumerate(zip(chunks, response_embeddings)):
            stored_chunk = _chunk_from_store(conn, chunk["chunk_id"])
            if stored_chunk is None:
                failed_count += 1
                continue
            point = _build_point(
                chunk=chunk,
                sparse_text=stored_chunk["sparse_text"],
                embedding=_float_vector(vector, label=f"chunk {chunk['chunk_id']}"),
                bm25_encoder=bm25_encoder,
                embedding_model=embedding_model or "embedding-service",
                args=args,
            )
            pending.append((chunk["chunk_id"], point, row_index))

        upserted_count += _upsert_pending_points(
            conn=conn,
            client=client,
            pending=pending,
            embedding_path=args.embedding_service_url,
            args=args,
        )
        _write_checkpoint(
            args,
            stage="embed_and_upsert",
            last_batch=batch_index,
            upserted_count=upserted_count,
            failed_count=failed_count,
            last_chunk_ids=chunk_ids[:5],
        )
        print(f"Embedded and indexed {upserted_count} chunks.")

    return embedding_model or "embedding-service", upserted_count, failed_count


def _upsert_pending_points(
    *,
    conn: sqlite3.Connection,
    client: QdrantClient,
    pending: list[tuple[str, models.PointStruct, int]],
    embedding_path: Path | str,
    args: argparse.Namespace,
) -> int:
    indexed = 0
    for start in range(0, len(pending), args.upsert_batch_size):
        batch = pending[start : start + args.upsert_batch_size]
        points = [point for _, point, _ in batch]
        if not points:
            continue
        client.upsert(collection_name=args.collection, points=points, wait=True)
        indexed_at = datetime.now(timezone.utc).isoformat()
        for chunk_id, _, embedding_row in batch:
            conn.execute(
                """
                INSERT INTO indexed_embeddings(chunk_id, embedding_path, embedding_row, indexed_at)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, str(embedding_path), embedding_row, indexed_at),
            )
        conn.commit()
        indexed += len(batch)
    return indexed


def _indexed_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM indexed_embeddings").fetchone()
    return int(row[0] or 0)


def _is_indexed(conn: sqlite3.Connection, chunk_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM indexed_embeddings WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return row is not None


def _chunk_from_store(conn: sqlite3.Connection, chunk_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT payload_json, sparse_text FROM chunks WHERE chunk_id = ?",
        (chunk_id,),
    ).fetchone()
    if row is None:
        return None
    return {"chunk": json.loads(row[0]), "sparse_text": row[1]}


def _chunk_store_batches(conn: sqlite3.Connection, batch_size: int) -> Iterator[list[sqlite3.Row]]:
    conn.row_factory = sqlite3.Row
    offset = 0
    while True:
        rows = conn.execute(
            "SELECT payload_json, sparse_text FROM chunks ORDER BY rowid LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break
        yield rows
        offset += len(rows)


def _iter_parquet_rows(
    path: Path,
    *,
    required_columns: set[str],
    selected_columns: list[str] | None = None,
    batch_size: int,
) -> Iterator[tuple[int, int, list[dict[str, Any]]]]:
    if not path.exists():
        raise RuntimeError(f"Parquet file does not exist: {path}")
    parquet_file = pq.ParquetFile(path)
    column_names = list(parquet_file.schema_arrow.names)
    _require_columns(column_names, required_columns, path)
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        batches = parquet_file.iter_batches(
            batch_size=batch_size,
            row_groups=[row_group_index],
            columns=selected_columns,
        )
        for batch_index, batch in enumerate(batches):
            try:
                rows = batch.to_pylist()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed reading {path} row_group={row_group_index} batch={batch_index}."
                ) from exc
            yield row_group_index, batch_index, rows


def _embedding_columns(path: Path) -> list[str]:
    column_names = set(pq.ParquetFile(path).schema_arrow.names)
    return [
        column
        for column in ("chunk_id", "embedding", "embedding_dim", "embedding_model")
        if column in column_names
    ]


def _chunk_from_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    chunk_id = _clean_str(row.get("chunk_id"))
    text = _clean_str(row.get("text"))
    if not chunk_id:
        raise RuntimeError(f"Chunk at row {index} has empty chunk_id.")
    if not text:
        raise RuntimeError(f"Chunk {chunk_id} has empty text.")

    return {
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


def _build_point(
    *,
    chunk: dict[str, Any],
    sparse_text: str,
    embedding: list[float],
    bm25_encoder: BM25SparseEncoder,
    embedding_model: str,
    args: argparse.Namespace,
) -> models.PointStruct:
    sparse_payload = bm25_encoder.encode_document(sparse_text)
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
    chunk_count: int,
    upserted_count: int,
    failed_count: int,
    embedding_model: str,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "build_command": sys.argv,
        "collection": args.collection,
        "qdrant_url": args.qdrant_url,
        "chunk_count": chunk_count,
        "upserted_count": upserted_count,
        "failed_count": failed_count,
        "chunks_path": str(args.chunks),
        "chunks_sha256": _sha256(args.chunks),
        "embeddings": [
            {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in args.embeddings
        ],
        "embedding_model": embedding_model,
        "embedding_dimension": args.embedding_dimension,
        "dense_vector_name": args.dense_vector_name,
        "sparse_vector_name": args.sparse_vector_name,
        "dense_distance": "cosine",
        "dense_datatype": "float16",
        "dense_on_disk": True,
        "sparse_encoder": "bm25",
        "fusion": "rrf",
        "bm25_stats_path": str(args.bm25_stats_out),
        "bm25_stats_sha256": _sha256(args.bm25_stats_out),
        "corpus_version": args.corpus_version,
        "checkpoint_path": str(args.checkpoint_out),
        "chunk_store_path": str(args.chunk_store),
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _write_checkpoint(args: argparse.Namespace, **fields: Any) -> None:
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "collection": args.collection,
        "chunks_path": str(args.chunks),
        "chunk_store_path": str(args.chunk_store),
        **fields,
    }
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    with args.checkpoint_out.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, sort_keys=True)
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


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in re.split(r"[;,]", value) if item.strip()]
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
