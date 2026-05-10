from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.parquet"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "data" / "embeddings" / "embedding_quality_report.md"

REQUIRED_CHUNK_COLUMNS = {"chunk_id"}
REQUIRED_EMBEDDING_COLUMNS = {
    "vector_id",
    "chunk_id",
    "embedding",
    "embedding_model",
    "embedding_dim",
    "created_at",
}


def main() -> None:
    args = _parse_args()
    chunk_ids = _load_chunk_ids(args.chunks)
    embedding_rows = _load_embedding_rows(args.embeddings)
    report, passed = _validate(
        chunk_ids=chunk_ids,
        embedding_rows=embedding_rows,
        chunks_path=args.chunks,
        embeddings_path=args.embeddings,
    )
    _write_report(args.out, report)
    print(report)
    if not passed:
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate embeddings.parquet against chunks.parquet.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def _load_chunk_ids(path: Path) -> list[str]:
    if not path.exists():
        raise RuntimeError(f"Chunks parquet file does not exist: {path}")
    table = pq.read_table(path)
    missing = sorted(REQUIRED_CHUNK_COLUMNS - set(table.column_names))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")
    return [_clean_str(row.get("chunk_id")) for row in table.to_pylist()]


def _load_embedding_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"Embeddings parquet file does not exist: {path}")
    table = pq.read_table(path)
    missing = sorted(REQUIRED_EMBEDDING_COLUMNS - set(table.column_names))
    if missing:
        raise RuntimeError(f"{path} is missing required columns: {', '.join(missing)}")
    return table.to_pylist()


def _validate(
    *,
    chunk_ids: list[str],
    embedding_rows: list[dict[str, Any]],
    chunks_path: Path,
    embeddings_path: Path,
) -> tuple[str, bool]:
    issues: list[str] = []

    chunk_id_set = set(chunk_ids)
    duplicate_chunk_ids = len(chunk_ids) - len(chunk_id_set)
    if duplicate_chunk_ids:
        issues.append(f"`chunks.parquet` has {duplicate_chunk_ids} duplicate `chunk_id` values.")
    if any(not chunk_id for chunk_id in chunk_ids):
        issues.append("`chunks.parquet` contains empty `chunk_id` values.")

    embedding_chunk_ids: list[str] = []
    vector_ids: list[str] = []
    dimensions: set[int] = set()
    declared_dimensions: set[int] = set()
    models: set[str] = set()
    nan_count = 0
    zero_vector_count = 0
    bad_declared_dimension_count = 0
    empty_vector_count = 0

    for index, row in enumerate(embedding_rows):
        chunk_id = _clean_str(row.get("chunk_id"))
        vector_id = _clean_str(row.get("vector_id"))
        model = _clean_str(row.get("embedding_model"))
        vector = _vector(row.get("embedding"))
        declared_dimension = _to_int(row.get("embedding_dim"))

        embedding_chunk_ids.append(chunk_id)
        vector_ids.append(vector_id)
        if model:
            models.add(model)
        if declared_dimension is not None:
            declared_dimensions.add(declared_dimension)
        if not vector:
            empty_vector_count += 1
            continue
        dimensions.add(len(vector))
        if declared_dimension is not None and declared_dimension != len(vector):
            bad_declared_dimension_count += 1
        if any(math.isnan(value) or math.isinf(value) for value in vector):
            nan_count += 1
        if not any(value != 0.0 for value in vector):
            zero_vector_count += 1
        if not chunk_id:
            issues.append(f"Embedding row {index} has empty `chunk_id`.")
        if not vector_id:
            issues.append(f"Embedding row {index} has empty `vector_id`.")

    duplicate_embedding_chunk_ids = len(embedding_chunk_ids) - len(set(embedding_chunk_ids))
    duplicate_vector_ids = len(vector_ids) - len(set(vector_ids))
    missing_embeddings = sorted(chunk_id_set - set(embedding_chunk_ids))
    extra_embeddings = sorted(set(embedding_chunk_ids) - chunk_id_set)

    if duplicate_embedding_chunk_ids:
        issues.append(f"`embeddings.parquet` has {duplicate_embedding_chunk_ids} duplicate `chunk_id` values.")
    if duplicate_vector_ids:
        issues.append(f"`embeddings.parquet` has {duplicate_vector_ids} duplicate `vector_id` values.")
    if missing_embeddings:
        issues.append(f"`embeddings.parquet` is missing {len(missing_embeddings)} chunks.")
    if extra_embeddings:
        issues.append(f"`embeddings.parquet` has {len(extra_embeddings)} chunk ids not present in chunks.")
    if len(dimensions) != 1:
        issues.append(f"Embedding vector dimensions are not constant: {sorted(dimensions)}.")
    if len(declared_dimensions) > 1:
        issues.append(f"Declared `embedding_dim` values are not constant: {sorted(declared_dimensions)}.")
    if bad_declared_dimension_count:
        issues.append(f"{bad_declared_dimension_count} rows declare a dimension different from vector length.")
    if len(models) != 1:
        issues.append(f"`embedding_model` should contain exactly one model, got: {sorted(models)}.")
    if nan_count:
        issues.append(f"{nan_count} embedding rows contain NaN or infinity.")
    if zero_vector_count:
        issues.append(f"{zero_vector_count} embedding rows are zero vectors.")
    if empty_vector_count:
        issues.append(f"{empty_vector_count} embedding rows have empty vectors.")

    passed = not issues
    lines = [
        "# Embedding Quality Report",
        "",
        f"- Chunks file: `{chunks_path}`",
        f"- Embeddings file: `{embeddings_path}`",
        f"- Chunks: {len(chunk_ids)}",
        f"- Embeddings: {len(embedding_rows)}",
        f"- Embedding models: {', '.join(sorted(models)) if models else '-'}",
        f"- Vector dimensions: {', '.join(str(value) for value in sorted(dimensions)) if dimensions else '-'}",
        f"- Missing embeddings: {len(missing_embeddings)}",
        f"- Extra embeddings: {len(extra_embeddings)}",
        f"- NaN/Inf rows: {nan_count}",
        f"- Zero-vector rows: {zero_vector_count}",
        "",
        f"Status: **{'PASS' if passed else 'FAIL'}**",
        "",
    ]
    if issues:
        lines.extend(["## Issues", ""])
        lines.extend(f"- {issue}" for issue in issues)
        lines.append("")
        if missing_embeddings:
            lines.extend(["## Missing Examples", ""])
            lines.extend(f"- `{chunk_id}`" for chunk_id in missing_embeddings[:20])
            lines.append("")
        if extra_embeddings:
            lines.extend(["## Extra Examples", ""])
            lines.extend(f"- `{chunk_id}`" for chunk_id in extra_embeddings[:20])
            lines.append("")

    return "\n".join(lines), passed


def _write_report(path: Path, report: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def _vector(value: Any) -> list[float]:
    if value is None:
        return []
    return [float(item) for item in value]


def _to_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


if __name__ == "__main__":
    main()
