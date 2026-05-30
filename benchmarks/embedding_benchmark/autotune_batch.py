from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from embedding_benchmark.common import (
    DEFAULT_CHUNK_SHARDS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    ProgressTracker,
    get_model_spec,
    gpu_memory_snapshot,
    human_seconds,
    load_registry,
    model_output_dir,
    monotonic_seconds,
    now_iso,
    setup_logging,
    write_json_atomic,
)
from embedding_benchmark.model_adapters import TextRecord, create_adapter


def main() -> None:
    args = _parse_args()
    registry = load_registry(args.registry)
    spec = get_model_spec(registry, args.model)
    output_dir = model_output_dir(args.output_root, spec.slug, args.precision)
    output_path = output_dir / f"autotune_{_device_slug(args.device)}.json"
    log = setup_logging(output_dir / "logs" / f"autotune_{_device_slug(args.device)}.log")

    if output_path.exists() and not args.force:
        log.info("Autotune result already exists, skipping: %s", output_path)
        return

    sample_records = _load_sample(args.chunks_shard, sample_size=args.sample_size)
    adapter = create_adapter(spec, device=args.device, precision=args.precision)
    started_at = monotonic_seconds()
    progress = ProgressTracker(total=args.max_batch, label=f"autotune:{spec.slug}:{args.device}", logger=log, log_every_seconds=5.0)
    successful: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    try:
        candidate = args.min_batch
        while candidate <= args.max_batch:
            result = _try_batch(adapter=adapter, records=sample_records, batch_size=candidate)
            result["candidate_batch_size"] = candidate
            result["gpu_memory"] = gpu_memory_snapshot()
            if result["ok"]:
                successful.append(result)
                log.info(
                    "Autotune OK model=%s device=%s batch=%d elapsed=%s rows_per_second=%.2f",
                    spec.slug,
                    args.device,
                    candidate,
                    human_seconds(float(result["elapsed_seconds"])),
                    float(result["rows_per_second"]),
                )
                progress.update(candidate, extra="ok")
                candidate *= 2
                continue
            failed.append(result)
            log.info("Autotune failed model=%s device=%s batch=%d error=%s", spec.slug, args.device, candidate, result["error"])
            break

        low = successful[-1]["candidate_batch_size"] if successful else 0
        high = failed[0]["candidate_batch_size"] if failed else min(args.max_batch * 2, max(args.max_batch, low * 2))
        while high - low > args.step:
            candidate = low + ((high - low) // 2)
            candidate = max(args.min_batch, (candidate // args.step) * args.step)
            if candidate <= low:
                break
            result = _try_batch(adapter=adapter, records=sample_records, batch_size=candidate)
            result["candidate_batch_size"] = candidate
            result["gpu_memory"] = gpu_memory_snapshot()
            if result["ok"]:
                successful.append(result)
                low = candidate
                log.info("Autotune OK model=%s device=%s batch=%d", spec.slug, args.device, candidate)
            else:
                failed.append(result)
                high = candidate
                log.info("Autotune failed model=%s device=%s batch=%d error=%s", spec.slug, args.device, candidate, result["error"])
            progress.update(min(candidate, args.max_batch), extra="binary_search")
    finally:
        adapter.close()
        gc.collect()
        _empty_cuda_cache()

    if not successful:
        raise RuntimeError(f"No working batch size found for {spec.slug} on {args.device}.")
    best = max(successful, key=lambda item: int(item["candidate_batch_size"]))
    recommended = max(args.min_batch, int(best["candidate_batch_size"] * args.safety_factor))
    recommended = max(args.min_batch, (recommended // args.step) * args.step)
    elapsed = monotonic_seconds() - started_at
    payload = {
        "created_at": now_iso(),
        "model": spec.slug,
        "device": args.device,
        "precision": args.precision,
        "sample_size": len(sample_records),
        "min_batch": args.min_batch,
        "max_batch": args.max_batch,
        "safety_factor": args.safety_factor,
        "largest_successful_batch_size": int(best["candidate_batch_size"]),
        "recommended_batch_size": int(recommended),
        "successful": successful,
        "failed": failed,
        "elapsed_seconds": elapsed,
        "elapsed_human": human_seconds(elapsed),
    }
    write_json_atomic(output_path, payload)
    progress.update(args.max_batch, force=True, extra=f"recommended={recommended}")
    log.info("Autotune complete model=%s device=%s recommended_batch_size=%d", spec.slug, args.device, recommended)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autotune max safe embedding batch size for one model and one GPU.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--chunks-shard", type=Path, default=DEFAULT_CHUNK_SHARDS[0])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-size", type=int, default=512)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=512)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--safety-factor", type=float, default=0.85)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _load_sample(path: Path, *, sample_size: int) -> list[TextRecord]:
    parquet = pq.ParquetFile(path)
    records: list[TextRecord] = []
    for row_group_index in range(parquet.metadata.num_row_groups):
        for batch in parquet.iter_batches(
            row_groups=[row_group_index],
            batch_size=sample_size,
            columns=["chunk_id", "title", "text"],
        ):
            for row in batch.to_pylist():
                records.append(
                    TextRecord(
                        chunk_id=str(row["chunk_id"]),
                        title=str(row.get("title") or ""),
                        text=str(row.get("text") or ""),
                    )
                )
                if len(records) >= sample_size:
                    return records
    return records


def _try_batch(*, adapter: Any, records: list[TextRecord], batch_size: int) -> dict[str, Any]:
    try:
        started_at = monotonic_seconds()
        sample = records[: min(len(records), batch_size)]
        adapter.encode_documents(sample, batch_size=batch_size)
        elapsed = monotonic_seconds() - started_at
        return {
            "ok": True,
            "elapsed_seconds": elapsed,
            "rows": len(sample),
            "rows_per_second": len(sample) / elapsed if elapsed > 0 else None,
        }
    except RuntimeError as exc:
        if _looks_like_oom(exc):
            _empty_cuda_cache()
            return {"ok": False, "error": str(exc), "oom": True}
        raise


def _looks_like_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error" in text and "memory" in text


def _device_slug(device: str) -> str:
    return device.replace(":", "_").replace("/", "_")


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


if __name__ == "__main__":
    main()
