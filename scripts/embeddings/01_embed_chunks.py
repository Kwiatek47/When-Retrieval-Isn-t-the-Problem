from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.parquet"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "embeddings" / "embedding_manifest.json"


def main() -> None:
    args = _parse_args()
    chunks = _load_chunks(
        path=args.chunks,
        id_column=args.id_column,
        text_column=args.text_column,
        title_column=args.title_column,
        limit=args.limit,
    )
    if not chunks:
        raise RuntimeError(f"No chunks found in {args.chunks}.")

    created_at = datetime.now(timezone.utc)
    embeddings, response_model, encoder_model, embedding_dimension = _embed_chunks(
        chunks=chunks,
        embedding_service_url=args.embedding_service_url,
        batch_size=args.batch_size,
        expected_model=args.model,
        expected_dimension=args.embedding_dimension,
    )
    service_health = _service_health(args.embedding_service_url)

    _write_embeddings(
        path=args.out,
        chunks=chunks,
        embeddings=embeddings,
        embedding_model=response_model,
        embedding_dimension=embedding_dimension,
        created_at=created_at,
    )
    _write_manifest(
        path=args.manifest_out,
        chunks=chunks,
        chunks_path=args.chunks,
        embeddings_path=args.out,
        embedding_service_url=args.embedding_service_url,
        embedding_model=response_model,
        encoder_model=encoder_model or _clean_str(service_health.get("document_model")),
        query_model=_clean_str(service_health.get("query_model")),
        device=_clean_str(service_health.get("device")),
        embedding_dimension=embedding_dimension,
        batch_size=args.batch_size,
        created_at=created_at,
    )
    print(f"Wrote {len(chunks)} embeddings to {args.out}")
    print(f"Wrote embedding manifest to {args.manifest_out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunks.parquet into data/embeddings/embeddings.parquet.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="chunk_id")
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--model", help="Expected embedding model name returned by embedding-service.")
    parser.add_argument("--embedding-dimension", type=int, default=768)
    parser.add_argument("--embedding-service-url", default="http://localhost:8081")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--limit", type=int, help="Embed only the first N chunks for a pilot run.")
    return parser.parse_args()


def _load_chunks(
    *,
    path: Path,
    id_column: str,
    text_column: str,
    title_column: str,
    limit: int | None,
) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"Chunks parquet file does not exist: {path}")
    table = pq.read_table(path)
    required = {id_column, text_column}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")

    rows = table.to_pylist()
    if limit is not None:
        rows = rows[:limit]

    chunks: list[dict[str, str]] = []
    seen_chunk_ids: set[str] = set()
    for index, row in enumerate(rows):
        chunk_id = _clean_str(row.get(id_column))
        text = _clean_str(row.get(text_column))
        if not chunk_id:
            raise RuntimeError(f"Chunk at row {index} has empty {id_column}.")
        if chunk_id in seen_chunk_ids:
            raise RuntimeError(f"Duplicate {id_column}: {chunk_id}")
        if not text:
            raise RuntimeError(f"Chunk {chunk_id} has empty {text_column}.")
        seen_chunk_ids.add(chunk_id)
        chunks.append(
            {
                "chunk_id": chunk_id,
                "title": _clean_str(row.get(title_column)) if title_column in table.column_names else "",
                "text": text,
            }
        )
    return chunks


def _embed_chunks(
    *,
    chunks: list[dict[str, str]],
    embedding_service_url: str,
    batch_size: int,
    expected_model: str | None,
    expected_dimension: int,
) -> tuple[dict[str, list[float]], str, str, int]:
    embeddings: dict[str, list[float]] = {}
    response_model = ""
    encoder_model = ""
    service_url = embedding_service_url.rstrip("/")

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = _request_json(
            "POST",
            f"{service_url}/embed/documents",
            {
                "documents": [
                    {
                        "title": chunk["title"],
                        "text": chunk["text"],
                    }
                    for chunk in batch
                ]
            },
        )
        model = _clean_str(response.get("model"))
        if expected_model and model != expected_model:
            raise RuntimeError(f"Expected embedding model {expected_model}, got {model}.")
        dimension = int(response.get("dimension") or 0)
        if dimension != expected_dimension:
            raise RuntimeError(f"Expected embedding dimension {expected_dimension}, got {dimension}.")
        response_embeddings = response.get("embeddings") or []
        if len(response_embeddings) != len(batch):
            raise RuntimeError(
                f"Expected {len(batch)} embeddings from embedding-service, got {len(response_embeddings)}."
            )
        response_model = model or response_model
        encoder_model = _clean_str(response.get("encoder_model")) or encoder_model

        for chunk, vector in zip(batch, response_embeddings, strict=True):
            embeddings[chunk["chunk_id"]] = _float_vector(vector, label=f"chunk {chunk['chunk_id']}")

        print(f"Embedded {min(start + len(batch), len(chunks))}/{len(chunks)} chunks.")

    return embeddings, response_model or "embedding-service", encoder_model, expected_dimension


def _write_embeddings(
    *,
    path: Path,
    chunks: list[dict[str, str]],
    embeddings: dict[str, list[float]],
    embedding_model: str,
    embedding_dimension: int,
    created_at: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        rows.append(
            {
                "vector_id": str(uuid5(NAMESPACE_URL, f"{embedding_model}:{chunk_id}")),
                "chunk_id": chunk_id,
                "embedding": embeddings[chunk_id],
                "embedding_model": embedding_model,
                "embedding_dim": embedding_dimension,
                "created_at": created_at,
            }
        )

    table = pa.Table.from_pydict(
        {
            "vector_id": pa.array([row["vector_id"] for row in rows], type=pa.string()),
            "chunk_id": pa.array([row["chunk_id"] for row in rows], type=pa.string()),
            "embedding": pa.array([row["embedding"] for row in rows], type=pa.list_(pa.float32())),
            "embedding_model": pa.array([row["embedding_model"] for row in rows], type=pa.string()),
            "embedding_dim": pa.array([row["embedding_dim"] for row in rows], type=pa.int32()),
            "created_at": pa.array([row["created_at"] for row in rows], type=pa.timestamp("us", tz="UTC")),
        }
    )
    pq.write_table(table, path)


def _write_manifest(
    *,
    path: Path,
    chunks: list[dict[str, str]],
    chunks_path: Path,
    embeddings_path: Path,
    embedding_service_url: str,
    embedding_model: str,
    encoder_model: str,
    query_model: str,
    device: str,
    embedding_dimension: int,
    batch_size: int,
    created_at: datetime,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": created_at.isoformat(),
        "chunks_path": str(chunks_path),
        "chunks_sha256": _sha256(chunks_path),
        "embeddings_path": str(embeddings_path),
        "chunk_count": len(chunks),
        "embedding_service_url": embedding_service_url,
        "embedding_model": embedding_model,
        "document_model": encoder_model,
        "query_model": query_model or None,
        "embedding_dim": embedding_dimension,
        "batch_size": batch_size,
        "device": device or None,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _request_json(method: str, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
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


def _service_health(embedding_service_url: str) -> dict[str, Any]:
    url = f"{embedding_service_url.rstrip('/')}/health"
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, ValueError):
        return {}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
