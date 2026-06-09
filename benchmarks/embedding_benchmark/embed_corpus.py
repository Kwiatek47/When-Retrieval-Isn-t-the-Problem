from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from embedding_benchmark.common import (
    DEFAULT_CHUNK_SHARDS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    append_jsonl,
    file_is_nonempty,
    get_model_spec,
    gpu_memory_snapshot,
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
from embedding_benchmark.model_adapters import TextRecord, create_adapter


def main() -> None:
    args = _parse_args()
    registry = load_registry(args.registry)
    spec = get_model_spec(registry, args.model)
    output_dir = model_output_dir(args.output_root, spec.slug, args.precision)
    log = setup_logging(output_dir / "logs" / f"embed_shard_{args.shard_index}.log")

    shard_path = _resolve_shard(args)
    final_path = output_dir / f"embeddings_shard_{args.shard_index}.parquet"
    checkpoint_path = output_dir / f"embed_checkpoint_shard_{args.shard_index}.json"
    metrics_path = output_dir / "metrics.jsonl"
    batch_dir = output_dir / "batches" / f"shard_{args.shard_index}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and not args.force:
        log.info("Final shard already exists, skipping: %s", final_path)
        return
    if args.force:
        _remove_final_and_batches(final_path, batch_dir)

    log.info("Starting embedding model=%s shard=%s chunks=%s out=%s", spec.slug, args.shard_index, shard_path, final_path)
    started_at = monotonic_seconds()
    adapter = create_adapter(spec, device=args.device, precision=args.precision)
    try:
        total_rows = _embed_shard(
            adapter=adapter,
            chunks_path=shard_path,
            batch_dir=batch_dir,
            checkpoint_path=checkpoint_path,
            metrics_path=metrics_path,
            model_slug=spec.slug,
            embedding_dim=spec.dim,
            batch_size=args.batch_size or spec.batch_size,
            read_batch_size=args.read_batch_size,
            log=log,
        )
    finally:
        adapter.close()
        gc.collect()
        _empty_cuda_cache()

    _merge_batches(
        batch_dir=batch_dir,
        final_path=final_path,
        expected_rows=total_rows,
        keep_batches=args.keep_batches,
        log=log,
    )
    elapsed = monotonic_seconds() - started_at
    manifest = {
        "created_at": now_iso(),
        "model": spec.slug,
        "display_name": spec.display_name,
        "precision": args.precision,
        "device": args.device,
        "chunk_shard_path": str(shard_path),
        "chunk_shard_sha256": sha256_file(shard_path),
        "embedding_path": str(final_path),
        "embedding_sha256": sha256_file(final_path),
        "embedding_dim": spec.dim,
        "rows": total_rows,
        "elapsed_seconds": elapsed,
        "elapsed_human": human_seconds(elapsed),
        "gpu_memory": gpu_memory_snapshot(),
        "batch_size": args.batch_size or spec.batch_size,
        "read_batch_size": args.read_batch_size,
    }
    write_json_atomic(output_dir / f"embedding_manifest_shard_{args.shard_index}.json", manifest)
    append_jsonl(
        metrics_path,
        {
            "event": "embed_shard_complete",
            "created_at": now_iso(),
            "model": spec.slug,
            "shard_index": args.shard_index,
            "rows": total_rows,
            "elapsed_seconds": elapsed,
            "rows_per_second": total_rows / elapsed if elapsed > 0 else None,
            "embedding_path": str(final_path),
            "gpu_memory": gpu_memory_snapshot(),
        },
    )
    log.info("Embedding complete rows=%d elapsed=%s final=%s", total_rows, human_seconds(elapsed), final_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed one PubMed chunk shard for one model with resumable batches.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--chunks-shard", type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--read-batch-size", type=int, default=4096)
    parser.add_argument("--keep-batches", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _resolve_shard(args: argparse.Namespace) -> Path:
    if args.chunks_shard is not None:
        return args.chunks_shard
    if args.shard_index < 0 or args.shard_index >= len(DEFAULT_CHUNK_SHARDS):
        raise RuntimeError(f"Default shard index must be 0..{len(DEFAULT_CHUNK_SHARDS) - 1}.")
    return DEFAULT_CHUNK_SHARDS[args.shard_index]


def _embed_shard(
    *,
    adapter: Any,
    chunks_path: Path,
    batch_dir: Path,
    checkpoint_path: Path,
    metrics_path: Path,
    model_slug: str,
    embedding_dim: int,
    batch_size: int,
    read_batch_size: int,
    log: Any,
) -> int:
    parquet = pq.ParquetFile(chunks_path)
    required = {"chunk_id", "text", "title"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise RuntimeError(f"{chunks_path} missing columns: {', '.join(sorted(missing))}")

    expected_total = int(parquet.metadata.num_rows)
    progress = ProgressTracker(total=expected_total, label=f"embed:{model_slug}:{chunks_path.name}", logger=log)
    total_rows = 0
    completed_rows = 0
    for row_group_index in range(parquet.metadata.num_row_groups):
        batches = parquet.iter_batches(
            row_groups=[row_group_index],
            batch_size=read_batch_size,
            columns=["chunk_id", "title", "text"],
        )
        for batch_index, batch in enumerate(batches):
            batch_path = batch_dir / f"rg{row_group_index:05d}_batch{batch_index:06d}.parquet"
            row_count = batch.num_rows
            total_rows += row_count
            if file_is_nonempty(batch_path):
                log.info("Skipping existing embedding batch %s rows=%d", batch_path.name, row_count)
                completed_rows += row_count
                progress.update(completed_rows, extra="resumed_existing_batch")
                continue

            rows = batch.to_pylist()
            records = [
                TextRecord(
                    chunk_id=str(row["chunk_id"]),
                    title=str(row.get("title") or ""),
                    text=str(row.get("text") or ""),
                )
                for row in rows
            ]
            started = monotonic_seconds()
            embeddings = adapter.encode_documents(records, batch_size=batch_size)
            elapsed = monotonic_seconds() - started
            _write_embedding_batch(
                path=batch_path,
                records=records,
                embeddings=embeddings,
                embedding_model=model_slug,
                embedding_dim=embedding_dim,
            )
            event = {
                "event": "embed_batch",
                "created_at": now_iso(),
                "model": model_slug,
                "row_group": row_group_index,
                "batch": batch_index,
                "rows": row_count,
                "elapsed_seconds": elapsed,
                "rows_per_second": row_count / elapsed if elapsed > 0 else None,
                "batch_path": str(batch_path),
                "gpu_memory": gpu_memory_snapshot(),
            }
            append_jsonl(metrics_path, event)
            write_json_atomic(
                checkpoint_path,
                {
                    "updated_at": now_iso(),
                    "stage": "embedding_batches",
                    "model": model_slug,
                    "chunks_path": str(chunks_path),
                    "last_row_group": row_group_index,
                    "last_batch": batch_index,
                    "rows_seen": total_rows,
                    "last_batch_path": str(batch_path),
                },
            )
            log.info(
                "Embedded batch row_group=%d batch=%d rows=%d speed=%.2f rows/s",
                row_group_index,
                batch_index,
                row_count,
                row_count / elapsed if elapsed > 0 else 0.0,
            )
            completed_rows += row_count
            progress.update(completed_rows, extra=f"last_batch={batch_path.name}")
    progress.update(completed_rows, force=True, extra="complete")
    return total_rows


def _write_embedding_batch(
    *,
    path: Path,
    records: list[TextRecord],
    embeddings: np.ndarray,
    embedding_model: str,
    embedding_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "chunk_id": pa.array([record.chunk_id for record in records], type=pa.string()),
            "embedding": pa.array([row.tolist() for row in embeddings], type=pa.list_(pa.float32())),
            "embedding_model": pa.array([embedding_model] * len(records), type=pa.string()),
            "embedding_dim": pa.array([embedding_dim] * len(records), type=pa.int32()),
            "created_at": pa.array([datetime.now(timezone.utc)] * len(records), type=pa.timestamp("us", tz="UTC")),
        }
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp_path, compression="zstd")
    tmp_path.replace(path)


def _merge_batches(
    *,
    batch_dir: Path,
    final_path: Path,
    expected_rows: int,
    keep_batches: bool,
    log: Any,
) -> None:
    batch_paths = sorted(batch_dir.glob("*.parquet"))
    if not batch_paths:
        raise RuntimeError(f"No embedding batches found in {batch_dir}.")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    writer: pq.ParquetWriter | None = None
    merged_rows = 0
    progress = ProgressTracker(total=len(batch_paths), label=f"merge:{final_path.name}", logger=log, log_every_seconds=10.0)
    try:
        for index, batch_path in enumerate(batch_paths, start=1):
            table = pq.read_table(batch_path)
            merged_rows += table.num_rows
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table)
            progress.update(index, extra=f"rows={merged_rows}")
        if writer is not None:
            writer.close()
            writer = None
        if merged_rows != expected_rows:
            raise RuntimeError(f"Merged {merged_rows} rows, expected {expected_rows}.")
        tmp_path.replace(final_path)
        progress.update(len(batch_paths), force=True, extra=f"rows={merged_rows}")
        log.info("Merged %d batch files into %s rows=%d", len(batch_paths), final_path, merged_rows)
    finally:
        if writer is not None:
            writer.close()
        if tmp_path.exists():
            tmp_path.unlink()

    if not keep_batches:
        for batch_path in batch_paths:
            batch_path.unlink()
        try:
            batch_dir.rmdir()
        except OSError:
            pass
        log.info("Removed %d intermediate embedding batch files from %s", len(batch_paths), batch_dir)


def _remove_final_and_batches(final_path: Path, batch_dir: Path) -> None:
    if final_path.exists():
        final_path.unlink()
    if batch_dir.exists():
        for path in batch_dir.glob("*.parquet"):
            path.unlink()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


if __name__ == "__main__":
    main()
