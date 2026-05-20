from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "chunks_child.parquet"


def main() -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    args = _parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for rows in _iter_child_rows(
            args.input,
            read_batch_size=args.read_batch_size,
            max_words=args.max_words,
            split_threshold_words=args.split_threshold_words,
        ):
            if not rows:
                continue
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema)
            table = table.cast(writer.schema)
            writer.write_table(table)
            total_rows += table.num_rows
            print(f"Wrote {total_rows} child-aware chunks.")
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise RuntimeError(f"No chunks written from {args.input}.")
    print(f"Wrote {total_rows} rows to {args.output}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split long RAG chunks into parent-child child chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--split-threshold-words", type=int, default=450)
    parser.add_argument("--max-words", type=int, default=450)
    parser.add_argument("--read-batch-size", type=int, default=2048)
    return parser.parse_args()


def _iter_child_rows(
    path: Path,
    *,
    read_batch_size: int,
    max_words: int,
    split_threshold_words: int,
) -> Iterator[list[dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    for row_group_index in range(parquet_file.metadata.num_row_groups):
        for batch in parquet_file.iter_batches(batch_size=read_batch_size, row_groups=[row_group_index]):
            yield build_child_rows(
                batch.to_pylist(),
                max_words=max_words,
                split_threshold_words=split_threshold_words,
            )


def build_child_rows(
    rows: list[dict[str, Any]],
    *,
    max_words: int = 450,
    split_threshold_words: int = 450,
) -> list[dict[str, Any]]:
    child_rows = []
    for row in rows:
        child_rows.extend(
            build_child_rows_for_record(
                row,
                max_words=max_words,
                split_threshold_words=split_threshold_words,
            )
        )
    return child_rows


def build_child_rows_for_record(
    row: dict[str, Any],
    *,
    max_words: int = 450,
    split_threshold_words: int = 450,
) -> list[dict[str, Any]]:
    text = str(row.get("text") or "").strip()
    parent_word_count = _word_count(text)
    declared_word_count = _optional_int(row.get("word_count")) or parent_word_count
    if declared_word_count <= split_threshold_words:
        return [_with_parent_fields(row, parent_chunk_id="", parent_word_count=declared_word_count)]

    parent_chunk_id = str(row.get("chunk_id") or "").strip()
    if not parent_chunk_id:
        raise RuntimeError("Cannot split a chunk without chunk_id.")

    child_texts = _split_text(text, max_words=max_words)
    if len(child_texts) <= 1:
        return [_with_parent_fields(row, parent_chunk_id="", parent_word_count=declared_word_count)]

    child_rows = []
    for index, child_text in enumerate(child_texts):
        child_row = dict(row)
        child_row["chunk_id"] = f"{parent_chunk_id}:part:{index}"
        child_row["text"] = child_text
        child_row["word_count"] = _word_count(child_text)
        child_row["chunk_index"] = index
        child_rows.append(
            _with_parent_fields(
                child_row,
                parent_chunk_id=parent_chunk_id,
                parent_word_count=declared_word_count,
            )
        )
    return child_rows


def _with_parent_fields(
    row: dict[str, Any],
    *,
    parent_chunk_id: str,
    parent_word_count: int,
) -> dict[str, Any]:
    enriched = dict(row)
    enriched["parent_chunk_id"] = parent_chunk_id
    enriched["parent_word_count"] = parent_word_count
    return enriched


def _split_text(text: str, *, max_words: int) -> list[str]:
    sentences = _sentences(text)
    if not sentences:
        return [text]

    chunks = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = _word_count(sentence)
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current).strip())
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(" ".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"[\w]+", text))


def _optional_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    main()
