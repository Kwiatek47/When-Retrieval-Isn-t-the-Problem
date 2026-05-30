from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any, Iterator
from uuid import NAMESPACE_URL, uuid5

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from embedding_benchmark.common import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    append_jsonl,
    collection_name_for,
    get_model_spec,
    human_seconds,
    load_registry,
    model_output_dir,
    monotonic_seconds,
    now_iso,
    ProgressTracker,
    setup_logging,
    sha256_file,
    write_json_atomic,
)


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
    registry = load_registry(args.registry)
    spec = get_model_spec(registry, args.model)
    output_dir = model_output_dir(args.output_root, spec.slug, args.precision)
    index_dir = output_dir / "qdrant"
    log = setup_logging(output_dir / "logs" / "build_qdrant_index.log")
    collection = args.collection or collection_name_for(spec, prefix=args.collection_prefix, dtype="f16")
    embeddings = args.embeddings or [
        output_dir / "embeddings_shard_0.parquet",
        output_dir / "embeddings_shard_1.parquet",
    ]
    checkpoint_path = index_dir / "index_checkpoint.json"
    chunk_store_path = index_dir / "chunk_store.sqlite"
    bm25_stats_path = index_dir / "bm25_stats.json"
    metrics_path = output_dir / "metrics.jsonl"

    if args.force:
        _delete_file(checkpoint_path)
        _delete_file(chunk_store_path)
        _delete_file(bm25_stats_path)

    index_dir.mkdir(parents=True, exist_ok=True)
    started_at = monotonic_seconds()
    conn = sqlite3.connect(chunk_store_path)
    try:
        chunk_count, bm25_encoder = _prepare_chunk_store(
            conn=conn,
            chunks_path=args.chunks,
            bm25_stats_path=bm25_stats_path,
            read_batch_size=args.read_batch_size,
            limit=args.limit,
            resume=args.resume,
            log=log,
        )
        client = QdrantClient(url=args.qdrant_url, timeout=args.qdrant_timeout)
        try:
            _ensure_collection(
                client=client,
                collection=collection,
                vector_name=args.dense_vector_name,
                sparse_vector_name=args.sparse_vector_name,
                dim=spec.dim,
                recreate=args.recreate,
                defer_payload_indexes=args.defer_payload_indexes,
                log=log,
            )
            if args.recreate:
                _clear_indexed_embeddings(conn, checkpoint_path=checkpoint_path, log=log)
            indexed_count = _upsert_embeddings(
                conn=conn,
                client=client,
                collection=collection,
                vector_name=args.dense_vector_name,
                sparse_vector_name=args.sparse_vector_name,
                embeddings=embeddings,
                embedding_dim=spec.dim,
                model_slug=spec.slug,
                bm25_encoder=bm25_encoder,
                read_batch_size=args.read_batch_size,
                upsert_batch_size=args.upsert_batch_size,
                resume=args.resume,
                checkpoint_path=checkpoint_path,
                metrics_path=metrics_path,
                log=log,
            )
            if args.defer_payload_indexes:
                _create_payload_indexes(client=client, collection=collection, log=log)
        finally:
            client.close()
    finally:
        conn.close()

    elapsed = monotonic_seconds() - started_at
    manifest = {
        "created_at": now_iso(),
        "model": spec.slug,
        "display_name": spec.display_name,
        "precision": args.precision,
        "collection": collection,
        "qdrant_url": args.qdrant_url,
        "dense_vector_name": args.dense_vector_name,
        "sparse_vector_name": args.sparse_vector_name,
        "qdrant_vector_datatype": "float16",
        "qdrant_on_disk": True,
        "embedding_dim": spec.dim,
        "chunks_path": str(args.chunks),
        "chunks_sha256": sha256_file(args.chunks),
        "embeddings": [{"path": str(path), "sha256": sha256_file(path)} for path in embeddings if path.exists()],
        "chunk_count": chunk_count,
        "indexed_count": indexed_count,
        "bm25_stats_path": str(bm25_stats_path),
        "bm25_stats_sha256": sha256_file(bm25_stats_path),
        "chunk_store_path": str(chunk_store_path),
        "defer_payload_indexes": args.defer_payload_indexes,
        "upsert_batch_size": args.upsert_batch_size,
        "elapsed_seconds": elapsed,
        "elapsed_human": human_seconds(elapsed),
    }
    write_json_atomic(index_dir / "qdrant_index_manifest.json", manifest)
    append_jsonl(
        metrics_path,
        {
            "event": "qdrant_index_complete",
            "created_at": now_iso(),
            "model": spec.slug,
            "collection": collection,
            "chunk_count": chunk_count,
            "indexed_count": indexed_count,
            "elapsed_seconds": elapsed,
            "rows_per_second": indexed_count / elapsed if elapsed > 0 else None,
        },
    )
    log.info("Qdrant index complete collection=%s indexed=%d/%d elapsed=%s", collection, indexed_count, chunk_count, human_seconds(elapsed))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Qdrant FLOAT16 dense+BM25 index for one benchmark model.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embeddings", type=Path, nargs="*")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--qdrant-timeout", type=float, default=120.0)
    parser.add_argument("--collection")
    parser.add_argument("--collection-prefix", default="pubmed_v1")
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="bm25")
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--defer-payload-indexes", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _prepare_chunk_store(
    *,
    conn: sqlite3.Connection,
    chunks_path: Path,
    bm25_stats_path: Path,
    read_batch_size: int,
    limit: int | None,
    resume: bool,
    log: Any,
) -> tuple[int, "BM25SparseEncoder"]:
    if resume and _chunk_store_count(conn) > 0 and bm25_stats_path.exists():
        _ensure_indexed_embeddings_table(conn)
        log.info("Reusing existing chunk store: %s", _chunk_store_count(conn))
        return _chunk_store_count(conn), BM25SparseEncoder.from_file(bm25_stats_path)

    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("DROP TABLE IF EXISTS indexed_embeddings")
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            sparse_text TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE indexed_embeddings (
            chunk_id TEXT PRIMARY KEY,
            embedding_path TEXT NOT NULL,
            embedding_row INTEGER NOT NULL,
            indexed_at TEXT NOT NULL
        )
        """
    )
    conn.commit()

    document_frequency: Counter[str] = Counter()
    total_document_length = 0
    chunk_count = 0
    parquet = pq.ParquetFile(chunks_path)
    expected_total = min(int(parquet.metadata.num_rows), limit) if limit is not None else int(parquet.metadata.num_rows)
    progress = ProgressTracker(total=expected_total, label="prepare_chunk_store", logger=log)
    for _, _, rows in _iter_parquet_rows(chunks_path, batch_size=read_batch_size, columns=None):
        for row in rows:
            if limit is not None and chunk_count >= limit:
                break
            chunk = _chunk_from_row(row, chunk_count)
            sparse_text = _sparse_text(chunk)
            tokens = _tokenize(sparse_text)
            document_frequency.update(set(tokens))
            total_document_length += len(tokens)
            conn.execute(
                "INSERT INTO chunks(chunk_id, payload_json, sparse_text) VALUES (?, ?, ?)",
                (chunk["chunk_id"], json.dumps(chunk, ensure_ascii=False), sparse_text),
            )
            chunk_count += 1
        conn.commit()
        log.info("Prepared chunk store rows=%d", chunk_count)
        progress.update(chunk_count)
        if limit is not None and chunk_count >= limit:
            break
    if chunk_count <= 0:
        raise RuntimeError(f"No chunks prepared from {chunks_path}.")
    vocabulary = {term: index for index, term in enumerate(sorted(document_frequency))}
    bm25 = BM25SparseEncoder(
        BM25Stats(
            document_count=chunk_count,
            average_document_length=total_document_length / max(chunk_count, 1),
            document_frequency=dict(document_frequency),
            vocabulary=vocabulary,
        )
    )
    bm25.save(bm25_stats_path)
    progress.update(chunk_count, force=True, extra="complete")
    return chunk_count, bm25


def _ensure_indexed_embeddings_table(conn: sqlite3.Connection) -> None:
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


def _clear_indexed_embeddings(conn: sqlite3.Connection, *, checkpoint_path: Path, log: Any) -> None:
    _ensure_indexed_embeddings_table(conn)
    conn.execute("DELETE FROM indexed_embeddings")
    conn.commit()
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    log.info("Cleared local indexed_embeddings checkpoint because Qdrant collection was recreated.")


def _ensure_collection(
    *,
    client: QdrantClient,
    collection: str,
    vector_name: str,
    sparse_vector_name: str,
    dim: int,
    recreate: bool,
    defer_payload_indexes: bool,
    log: Any,
) -> None:
    if recreate and client.collection_exists(collection_name=collection):
        log.info("Deleting existing collection: %s", collection)
        client.delete_collection(collection_name=collection)
    if not client.collection_exists(collection_name=collection):
        log.info("Creating collection=%s dim=%d datatype=float16 on_disk=true", collection, dim)
        client.create_collection(
            collection_name=collection,
            vectors_config={
                vector_name: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    datatype=models.Datatype.FLOAT16,
                    on_disk=True,
                )
            },
            sparse_vectors_config={sparse_vector_name: models.SparseVectorParams()},
            hnsw_config=models.HnswConfigDiff(on_disk=True),
        )
    if not defer_payload_indexes:
        _create_payload_indexes(client=client, collection=collection, log=log)


def _create_payload_indexes(*, client: QdrantClient, collection: str, log: Any) -> None:
    for field_name, field_schema in PAYLOAD_INDEXES.items():
        try:
            client.create_payload_index(collection_name=collection, field_name=field_name, field_schema=field_schema)
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise
    log.info("Payload indexes are ready for collection=%s", collection)


def _upsert_embeddings(
    *,
    conn: sqlite3.Connection,
    client: QdrantClient,
    collection: str,
    vector_name: str,
    sparse_vector_name: str,
    embeddings: list[Path],
    embedding_dim: int,
    model_slug: str,
    bm25_encoder: "BM25SparseEncoder",
    read_batch_size: int,
    upsert_batch_size: int,
    resume: bool,
    checkpoint_path: Path,
    metrics_path: Path,
    log: Any,
) -> int:
    indexed = _indexed_count(conn) if resume else 0
    total_rows = 0
    for embedding_path in embeddings:
        total_rows += pq.ParquetFile(embedding_path).metadata.num_rows
    progress = ProgressTracker(total=int(total_rows), label=f"qdrant_upsert:{model_slug}", logger=log)
    progress.update(indexed, extra="resume_state" if indexed else "")
    for embedding_path in embeddings:
        if not embedding_path.exists():
            raise RuntimeError(f"Embedding file does not exist: {embedding_path}")
        row_offset = 0
        for row_group, batch_index, rows in _iter_parquet_rows(
            embedding_path,
            batch_size=read_batch_size,
            columns=["chunk_id", "embedding", "embedding_dim", "embedding_model"],
        ):
            pending = []
            for row_index, row in enumerate(rows, start=row_offset):
                chunk_id = str(row.get("chunk_id") or "")
                if not chunk_id:
                    raise RuntimeError(f"Missing chunk_id in {embedding_path} row {row_index}")
                if resume and _is_indexed(conn, chunk_id):
                    continue
                stored = _chunk_from_store(conn, chunk_id)
                if stored is None:
                    raise RuntimeError(f"Embedding chunk_id not found in chunk store: {chunk_id}")
                vector = _float_vector(row.get("embedding"), expected_dim=embedding_dim, label=f"{embedding_path}:{row_index}")
                point = _build_point(
                    chunk=stored["chunk"],
                    sparse_text=stored["sparse_text"],
                    dense_vector=vector,
                    vector_name=vector_name,
                    sparse_vector_name=sparse_vector_name,
                    bm25_encoder=bm25_encoder,
                    model_slug=model_slug,
                )
                pending.append((chunk_id, row_index, point))

            started = monotonic_seconds()
            written = _upsert_pending(
                conn=conn,
                client=client,
                collection=collection,
                embedding_path=embedding_path,
                pending=pending,
                upsert_batch_size=upsert_batch_size,
            )
            elapsed = monotonic_seconds() - started
            indexed += written
            append_jsonl(
                metrics_path,
                {
                    "event": "qdrant_upsert_batch",
                    "created_at": now_iso(),
                    "model": model_slug,
                    "collection": collection,
                    "embedding_path": str(embedding_path),
                    "row_group": row_group,
                    "batch": batch_index,
                    "rows_upserted": written,
                    "elapsed_seconds": elapsed,
                    "rows_per_second": written / elapsed if elapsed > 0 else None,
                    "indexed_total": indexed,
                },
            )
            write_json_atomic(
                checkpoint_path,
                {
                    "updated_at": now_iso(),
                    "stage": "qdrant_upsert",
                    "model": model_slug,
                    "collection": collection,
                    "embedding_path": str(embedding_path),
                    "last_row_group": row_group,
                    "last_batch": batch_index,
                    "indexed_total": indexed,
                },
            )
            log.info(
                "Qdrant upsert path=%s row_group=%d batch=%d rows=%d indexed_total=%d",
                embedding_path.name,
                row_group,
                batch_index,
                written,
                indexed,
            )
            progress.update(indexed, extra=f"file={embedding_path.name}")
            row_offset += len(rows)
    final_indexed = _indexed_count(conn)
    progress.update(final_indexed, force=True, extra="complete")
    return final_indexed


def _upsert_pending(
    *,
    conn: sqlite3.Connection,
    client: QdrantClient,
    collection: str,
    embedding_path: Path,
    pending: list[tuple[str, int, models.PointStruct]],
    upsert_batch_size: int,
) -> int:
    written = 0
    for start in range(0, len(pending), upsert_batch_size):
        batch = pending[start : start + upsert_batch_size]
        if not batch:
            continue
        client.upsert(collection_name=collection, points=[point for _, _, point in batch], wait=True)
        indexed_at = now_iso()
        for chunk_id, row_index, _ in batch:
            conn.execute(
                """
                INSERT OR IGNORE INTO indexed_embeddings(chunk_id, embedding_path, embedding_row, indexed_at)
                VALUES (?, ?, ?, ?)
                """,
                (chunk_id, str(embedding_path), row_index, indexed_at),
            )
        conn.commit()
        written += len(batch)
    return written


def _iter_parquet_rows(path: Path, *, batch_size: int, columns: list[str] | None) -> Iterator[tuple[int, int, list[dict[str, Any]]]]:
    parquet = pq.ParquetFile(path)
    for row_group_index in range(parquet.metadata.num_row_groups):
        for batch_index, batch in enumerate(
            parquet.iter_batches(row_groups=[row_group_index], batch_size=batch_size, columns=columns)
        ):
            yield row_group_index, batch_index, batch.to_pylist()


def _chunk_from_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    chunk_id = _clean_str(row.get("chunk_id"))
    text = _clean_str(row.get("text"))
    if not chunk_id or not text:
        raise RuntimeError(f"Invalid chunk row at index {index}: missing chunk_id/text.")
    return {
        "chunk_id": chunk_id,
        "doc_id": _clean_str(row.get("doc_id")) or chunk_id,
        "pmid": _clean_str(row.get("pmid")),
        "title": _clean_str(row.get("title")) or "Untitled medical chunk",
        "text": text,
        "source": _clean_str(row.get("source")) or "pubmed",
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


def _build_point(
    *,
    chunk: dict[str, Any],
    sparse_text: str,
    dense_vector: list[float],
    vector_name: str,
    sparse_vector_name: str,
    bm25_encoder: "BM25SparseEncoder",
    model_slug: str,
) -> models.PointStruct:
    sparse_payload = bm25_encoder.encode_document(sparse_text)
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
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{chunk['pmid']}/" if chunk["pmid"] else chunk["source"],
        "section": chunk["section"],
        "chunkIndex": chunk["chunk_index"],
        "publicationDate": chunk["publication_date"],
        "publicationTypes": chunk["publication_types"],
        "isReview": chunk["is_review"],
        "isSystematicReview": chunk["is_systematic_review"],
        "wordCount": chunk["word_count"],
        "textHash": chunk["text_hash"],
        "embeddingModel": model_slug,
        "corpusVersion": "pubmed-chunks-v1",
    }
    return models.PointStruct(
        id=str(uuid5(NAMESPACE_URL, chunk["chunk_id"])),
        vector={
            vector_name: dense_vector,
            sparse_vector_name: models.SparseVector(
                indices=sparse_payload["indices"],
                values=sparse_payload["values"],
            ),
        },
        payload={key: value for key, value in payload.items() if _has_value(value)},
    )


class BM25Stats:
    def __init__(self, *, document_count: int, average_document_length: float, document_frequency: dict[str, int], vocabulary: dict[str, int]) -> None:
        self.document_count = document_count
        self.average_document_length = average_document_length
        self.document_frequency = document_frequency
        self.vocabulary = vocabulary


class BM25SparseEncoder:
    def __init__(self, stats: BM25Stats, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.stats = stats
        self.k1 = k1
        self.b = b

    def encode_document(self, text: str) -> dict[str, list[int] | list[float]]:
        tokens = _tokenize(text)
        if not tokens:
            return {"indices": [], "values": []}
        counts = Counter(tokens)
        doc_len = len(tokens)
        indices = []
        values = []
        for token, tf in counts.items():
            index = self.stats.vocabulary.get(token)
            if index is None:
                continue
            df = self.stats.document_frequency.get(token, 0)
            idf = math.log(1.0 + (self.stats.document_count - df + 0.5) / (df + 0.5))
            denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.stats.average_document_length, 1e-6))
            score = idf * (tf * (self.k1 + 1.0)) / denom
            indices.append(index)
            values.append(float(score))
        return {"indices": indices, "values": values}

    def save(self, path: Path) -> None:
        write_json_atomic(
            path,
            {
                "document_count": self.stats.document_count,
                "average_document_length": self.stats.average_document_length,
                "document_frequency": self.stats.document_frequency,
                "vocabulary": self.stats.vocabulary,
                "k1": self.k1,
                "b": self.b,
            },
        )

    @classmethod
    def from_file(cls, path: Path) -> "BM25SparseEncoder":
        with path.open(encoding="utf-8") as file:
            payload = json.load(file)
        return cls(
            BM25Stats(
                document_count=int(payload["document_count"]),
                average_document_length=float(payload["average_document_length"]),
                document_frequency={str(k): int(v) for k, v in payload["document_frequency"].items()},
                vocabulary={str(k): int(v) for k, v in payload["vocabulary"].items()},
            ),
            k1=float(payload.get("k1", 1.5)),
            b=float(payload.get("b", 0.75)),
        )


def _chunk_store_count(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0] or 0)


def _indexed_count(conn: sqlite3.Connection) -> int:
    try:
        row = conn.execute("SELECT COUNT(*) FROM indexed_embeddings").fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0] or 0)


def _is_indexed(conn: sqlite3.Connection, chunk_id: str) -> bool:
    row = conn.execute("SELECT 1 FROM indexed_embeddings WHERE chunk_id = ?", (chunk_id,)).fetchone()
    return row is not None


def _chunk_from_store(conn: sqlite3.Connection, chunk_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT payload_json, sparse_text FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
    if row is None:
        return None
    return {"chunk": json.loads(row[0]), "sparse_text": row[1]}


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


def _float_vector(value: Any, *, expected_dim: int, label: str) -> list[float]:
    if value is None:
        raise RuntimeError(f"{label}: empty vector")
    vector = [float(item) for item in value]
    if len(vector) != expected_dim:
        raise RuntimeError(f"{label}: expected dim={expected_dim}, got {len(vector)}")
    if any(math.isnan(item) or math.isinf(item) for item in vector):
        raise RuntimeError(f"{label}: vector contains NaN/inf")
    return vector


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


def _clean_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


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


def _has_value(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, list) and not value:
        return False
    return True


def _delete_file(path: Path) -> None:
    if path.exists():
        path.unlink()


if __name__ == "__main__":
    main()
