from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import json

from embedding_benchmark.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    ProgressTracker,
    append_jsonl,
    get_model_spec,
    load_registry,
    model_output_dir,
    model_slugs,
    now_iso,
    setup_logging,
    write_json_atomic,
)


HEAVY_MODELS = {"qwen3_4b", "qwen3_8b"}


def main() -> None:
    args = _parse_args()
    registry = load_registry(args.registry)
    selected_models = model_slugs(registry, args.models)
    pipeline_log = setup_logging(args.output_root / "pipeline.log")
    state_path = args.output_root / "pipeline_state.json"
    args.output_root.mkdir(parents=True, exist_ok=True)

    pipeline_log.info("Starting pipeline models=%s", selected_models)
    stages_per_model = int(not args.skip_embedding) + int(not args.skip_index) + int(not args.skip_eval)
    stage_total = len(selected_models) * stages_per_model + int(not args.skip_aggregate)
    stage_done = 0
    core_models = [model for model in selected_models if model not in HEAVY_MODELS]
    core_aggregate_written = False
    pipeline_progress = ProgressTracker(total=stage_total, label="pipeline", logger=pipeline_log, log_every_seconds=5.0)
    for model_index, model_slug in enumerate(selected_models, start=1):
        spec = get_model_spec(registry, model_slug)
        output_dir = model_output_dir(args.output_root, model_slug, args.precision)
        output_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(
            args.output_root / "pipeline_events.jsonl",
            {
                "event": "model_started",
                "created_at": now_iso(),
                "model": model_slug,
                "output_dir": str(output_dir),
            },
        )
        write_json_atomic(
            state_path,
            {
                "updated_at": now_iso(),
                "stage": "model_started",
                "model": model_slug,
                "models": selected_models,
            },
        )
        try:
            if not args.skip_embedding:
                batch_size = None
                if not args.skip_autotune:
                    batch_size = _run_autotune_stage(args, model_slug, pipeline_log)
                _run_embedding_stage(args, model_slug, pipeline_log, batch_size=batch_size)
                stage_done += 1
                pipeline_progress.update(
                    stage_done,
                    force=True,
                    extra=f"model={model_slug} stage=embedding model_index={model_index}/{len(selected_models)}",
                )
            if not args.skip_index:
                _run_index_stage(args, model_slug, pipeline_log)
                stage_done += 1
                pipeline_progress.update(
                    stage_done,
                    force=True,
                    extra=f"model={model_slug} stage=index model_index={model_index}/{len(selected_models)}",
                )
            if not args.skip_eval:
                _run_evaluation_stage(args, model_slug, pipeline_log)
                stage_done += 1
                pipeline_progress.update(
                    stage_done,
                    force=True,
                    extra=f"model={model_slug} stage=evaluation model_index={model_index}/{len(selected_models)}",
                )

            append_jsonl(
                args.output_root / "pipeline_events.jsonl",
                {"event": "model_completed", "created_at": now_iso(), "model": model_slug},
            )
            write_json_atomic(
                state_path,
                {
                    "updated_at": now_iso(),
                    "stage": "model_completed",
                    "model": model_slug,
                    "models": selected_models,
                },
            )
            pipeline_log.info("Completed model=%s display_name=%s", model_slug, spec.display_name)
            if (
                not args.skip_aggregate
                and not core_aggregate_written
                and core_models
                and _all_model_summaries_exist(args, core_models)
            ):
                _run_aggregate_stage(
                    args,
                    core_models,
                    pipeline_log,
                    out_dir=args.output_root / "reports" / "core_without_qwen4b_8b",
                )
                core_aggregate_written = True
        except Exception as exc:
            failure_payload = {
                "created_at": now_iso(),
                "model": model_slug,
                "display_name": spec.display_name,
                "error": repr(exc),
                "stage": "model_failed",
            }
            write_json_atomic(output_dir / "failure.json", failure_payload)
            append_jsonl(args.output_root / "pipeline_events.jsonl", {"event": "model_failed", **failure_payload})
            pipeline_log.exception("Model failed model=%s; continuing=%s", model_slug, not args.stop_on_error)
            if args.stop_on_error:
                raise
            continue

    if not args.skip_aggregate:
        _run_aggregate_stage(
            args,
            selected_models,
            pipeline_log,
            out_dir=args.output_root / "reports" / "full_with_qwen4b_8b",
        )
        stage_done += 1
        pipeline_progress.update(stage_done, force=True, extra="stage=aggregate")
    write_json_atomic(state_path, {"updated_at": now_iso(), "stage": "complete", "models": selected_models})
    pipeline_log.info("Pipeline complete")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding, Qdrant indexing and evaluation model-by-model.")
    parser.add_argument("--models", nargs="*", help="Subset of model slugs. Defaults to all registry models.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--query-set", type=Path, default=Path("benchmarks/embedding_benchmark/query_sets/pubmed_chunks_benchmark_v1.jsonl"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--shard-0", type=Path, default=Path("data/processed/chunks_shard_0.parquet"))
    parser.add_argument("--shard-1", type=Path, default=Path("data/processed/chunks_shard_1.parquet"))
    parser.add_argument("--device-0", default="cuda:0")
    parser.add_argument("--device-1", default="cuda:1")
    parser.add_argument("--eval-device", default="cuda:0")
    parser.add_argument("--embed-read-batch-size", type=int, default=4096)
    parser.add_argument("--index-read-batch-size", type=int, default=4096)
    parser.add_argument("--upsert-batch-size", type=int, default=256)
    parser.add_argument("--qdrant-fast-bulk", action="store_true")
    parser.add_argument("--qdrant-fast-upsert-batch-size", type=int, default=1024)
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument("--collection-prefix", default="pubmed_v1")
    parser.add_argument("--dense-vector-name", default="dense")
    parser.add_argument("--sparse-vector-name", default="bm25")
    parser.add_argument("--eval-top-k", type=int, default=20)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-autotune", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--keep-batches", action="store_true")
    parser.add_argument("--recreate-index", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--sequential-shards", action="store_true")
    parser.add_argument("--autotune-sample-size", type=int, default=512)
    parser.add_argument("--autotune-max-batch", type=int, default=512)
    parser.add_argument("--autotune-min-batch", type=int, default=1)
    parser.add_argument("--autotune-step", type=int, default=1)
    parser.add_argument("--autotune-safety-factor", type=float, default=0.85)
    parser.add_argument("--force-autotune", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _run_autotune_stage(args: argparse.Namespace, model_slug: str, log: Any) -> int:
    commands = [
        (
            args.device_0,
            args.shard_0,
            _base_python_args(args)
            + [
                "-m",
                "embedding_benchmark.autotune_batch",
                "--model",
                model_slug,
                "--registry",
                str(args.registry),
                "--chunks-shard",
                str(args.shard_0),
                "--output-root",
                str(args.output_root),
                "--precision",
                args.precision,
                "--device",
                args.device_0,
                "--sample-size",
                str(args.autotune_sample_size),
                "--min-batch",
                str(args.autotune_min_batch),
                "--max-batch",
                str(args.autotune_max_batch),
                "--step",
                str(args.autotune_step),
                "--safety-factor",
                str(args.autotune_safety_factor),
            ]
            + (["--force"] if args.force_autotune else []),
        ),
        (
            args.device_1,
            args.shard_1,
            _base_python_args(args)
            + [
                "-m",
                "embedding_benchmark.autotune_batch",
                "--model",
                model_slug,
                "--registry",
                str(args.registry),
                "--chunks-shard",
                str(args.shard_1),
                "--output-root",
                str(args.output_root),
                "--precision",
                args.precision,
                "--device",
                args.device_1,
                "--sample-size",
                str(args.autotune_sample_size),
                "--min-batch",
                str(args.autotune_min_batch),
                "--max-batch",
                str(args.autotune_max_batch),
                "--step",
                str(args.autotune_step),
                "--safety-factor",
                str(args.autotune_safety_factor),
            ]
            + (["--force"] if args.force_autotune else []),
        ),
    ]
    if args.sequential_shards:
        for _, _, command in commands:
            _run_command(command, cwd=args.project_root, env=_env(args), log=log)
    else:
        log.info("Starting autotune in parallel for model=%s", model_slug)
        processes = [subprocess.Popen(command, cwd=args.project_root, env=_env(args)) for _, _, command in commands]
        exit_codes = [process.wait() for process in processes]
        if any(code != 0 for code in exit_codes):
            raise RuntimeError(f"Autotune failed for {model_slug}: exit_codes={exit_codes}")

    recommended = []
    for device, _, _ in commands:
        path = model_output_dir(args.output_root, model_slug, args.precision) / f"autotune_{_device_slug(device)}.json"
        with path.open(encoding="utf-8") as file:
            recommended.append(int(json.load(file)["recommended_batch_size"]))
    chosen = min(recommended)
    log.info("Autotune selected model=%s batch_size=%d from device_recommendations=%s", model_slug, chosen, recommended)
    return chosen


def _run_embedding_stage(args: argparse.Namespace, model_slug: str, log: Any, *, batch_size: int | None = None) -> None:
    batch_attempts = _batch_attempts(batch_size, min_batch=args.autotune_min_batch)
    last_error: Exception | None = None
    for attempt_index, attempt_batch_size in enumerate(batch_attempts, start=1):
        try:
            _run_embedding_stage_once(
                args,
                model_slug,
                log,
                batch_size=attempt_batch_size,
                attempt_index=attempt_index,
                attempt_count=len(batch_attempts),
            )
            return
        except Exception as exc:
            last_error = exc
            log.exception(
                "Embedding attempt failed model=%s attempt=%d/%d batch_size=%s",
                model_slug,
                attempt_index,
                len(batch_attempts),
                attempt_batch_size,
            )
            if attempt_index < len(batch_attempts):
                _remove_partial_embedding_outputs(args, model_slug, log=log)
                log.info("Retrying model=%s with smaller batch_size=%s", model_slug, batch_attempts[attempt_index])
    if last_error is not None:
        raise last_error


def _run_embedding_stage_once(
    args: argparse.Namespace,
    model_slug: str,
    log: Any,
    *,
    batch_size: int | None,
    attempt_index: int,
    attempt_count: int,
) -> None:
    commands = [
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.embed_corpus",
            "--model",
            model_slug,
            "--registry",
            str(args.registry),
            "--chunks-shard",
            str(args.shard_0),
            "--shard-index",
            "0",
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--device",
            args.device_0,
            "--read-batch-size",
            str(args.embed_read_batch_size),
        ]
        + (["--batch-size", str(batch_size)] if batch_size is not None else [])
        + (["--keep-batches"] if args.keep_batches else []),
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.embed_corpus",
            "--model",
            model_slug,
            "--registry",
            str(args.registry),
            "--chunks-shard",
            str(args.shard_1),
            "--shard-index",
            "1",
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--device",
            args.device_1,
            "--read-batch-size",
            str(args.embed_read_batch_size),
        ]
        + (["--batch-size", str(batch_size)] if batch_size is not None else [])
        + (["--keep-batches"] if args.keep_batches else []),
    ]
    if args.sequential_shards:
        log.info(
            "Starting sequential shard embedding model=%s batch_size=%s attempt=%d/%d",
            model_slug,
            batch_size,
            attempt_index,
            attempt_count,
        )
        for command in commands:
            _run_command(command, cwd=args.project_root, env=_env(args), log=log)
        return

    log.info(
        "Starting shard embedding in parallel model=%s batch_size=%s attempt=%d/%d",
        model_slug,
        batch_size,
        attempt_index,
        attempt_count,
    )
    processes = [
        subprocess.Popen(command, cwd=args.project_root, env=_env(args))
        for command in commands
    ]
    exit_codes = [process.wait() for process in processes]
    if any(code != 0 for code in exit_codes):
        raise RuntimeError(f"Embedding shard process failed for {model_slug}: exit_codes={exit_codes}")
    log.info("Shard embedding complete for model=%s", model_slug)


def _batch_attempts(batch_size: int | None, *, min_batch: int) -> list[int | None]:
    if batch_size is None:
        return [None]
    attempts: list[int | None] = []
    current = max(int(batch_size), min_batch)
    while current >= min_batch:
        attempts.append(current)
        if current == min_batch:
            break
        current = max(min_batch, current // 2)
        current = max(min_batch, (current // min_batch) * min_batch)
    return attempts


def _remove_partial_embedding_outputs(args: argparse.Namespace, model_slug: str, *, log: Any) -> None:
    output_dir = model_output_dir(args.output_root, model_slug, args.precision)
    paths = [
        output_dir / "embeddings_shard_0.parquet",
        output_dir / "embeddings_shard_1.parquet",
        output_dir / "embedding_manifest_shard_0.json",
        output_dir / "embedding_manifest_shard_1.json",
    ]
    for path in paths:
        if path.exists():
            path.unlink()
            log.info("Removed partial output before retry: %s", path)
    batch_root = output_dir / "batches"
    if batch_root.exists():
        for path in batch_root.rglob("*.parquet"):
            path.unlink()
        log.info("Removed partial batch files before retry: %s", batch_root)


def _run_index_stage(args: argparse.Namespace, model_slug: str, log: Any) -> None:
    upsert_batch_size = args.qdrant_fast_upsert_batch_size if args.qdrant_fast_bulk else args.upsert_batch_size
    command = (
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.build_qdrant_index",
            "--model",
            model_slug,
            "--registry",
            str(args.registry),
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--qdrant-url",
            args.qdrant_url,
            "--collection-prefix",
            args.collection_prefix,
            "--dense-vector-name",
            args.dense_vector_name,
            "--sparse-vector-name",
            args.sparse_vector_name,
            "--read-batch-size",
            str(args.index_read_batch_size),
            "--upsert-batch-size",
            str(upsert_batch_size),
        ]
        + (["--recreate"] if args.recreate_index else [])
        + (["--no-resume"] if args.no_resume else [])
        + (["--defer-payload-indexes"] if args.qdrant_fast_bulk else [])
    )
    _run_command(
        command,
        cwd=args.project_root,
        env=_env(args),
        log=log,
    )


def _run_evaluation_stage(args: argparse.Namespace, model_slug: str, log: Any) -> None:
    _run_command(
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.evaluate_qdrant",
            "--model",
            model_slug,
            "--registry",
            str(args.registry),
            "--query-set",
            str(args.query_set),
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--device",
            args.eval_device,
            "--qdrant-url",
            args.qdrant_url,
            "--collection-prefix",
            args.collection_prefix,
            "--dense-vector-name",
            args.dense_vector_name,
            "--top-k",
            str(args.eval_top_k),
        ]
        + (["--force"] if args.force_eval else []),
        cwd=args.project_root,
        env=_env(args),
        log=log,
    )


def _run_aggregate_stage(args: argparse.Namespace, models: list[str], log: Any, *, out_dir: Path) -> None:
    _run_command(
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.aggregate_results",
            "--registry",
            str(args.registry),
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--out-dir",
            str(out_dir),
            "--models",
            *models,
        ],
        cwd=args.project_root,
        env=_env(args),
        log=log,
    )


def _all_model_summaries_exist(args: argparse.Namespace, models: list[str]) -> bool:
    for model_slug in models:
        summary_path = model_output_dir(args.output_root, model_slug, args.precision) / "evaluation" / "summary.json"
        if not summary_path.exists():
            return False
    return True


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], log: Any) -> None:
    log.info("Running command: %s", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")


def _base_python_args(args: argparse.Namespace) -> list[str]:
    return [args.python]


def _env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    benchmark_path = str((args.project_root / "benchmarks").resolve())
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = benchmark_path if not existing else benchmark_path + os.pathsep + existing
    return env


def _device_slug(device: str) -> str:
    return device.replace(":", "_").replace("/", "_")


if __name__ == "__main__":
    main()
