from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import json

from embedding_benchmark.common import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_REGISTRY_PATH,
    ProgressTracker,
    append_jsonl,
    collection_name_for,
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

    pipeline_log.info(
        "Starting pipeline models=%s shards=%s eval_device=%s",
        selected_models,
        [str(args.shard_0), str(args.shard_1)],
        args.eval_device,
    )
    run_pubmedqa_pipeline_eval = not args.skip_pubmedqa_pipeline_eval
    stages_per_model = (
        int(not args.skip_embedding)
        + int(not args.skip_index)
        + int(not args.skip_eval)
        + int(run_pubmedqa_pipeline_eval)
    )
    stage_total = len(selected_models) * stages_per_model + int(not args.skip_aggregate) + int(not args.skip_aggregate and run_pubmedqa_pipeline_eval)
    stage_done = 0
    core_models = [model for model in selected_models if model not in HEAVY_MODELS]
    core_aggregate_written = False
    pipeline_progress = ProgressTracker(total=stage_total, label="pipeline", logger=pipeline_log, log_every_seconds=5.0)
    resume_completed_models = _resume_completed_models_enabled(args)
    for model_index, model_slug in enumerate(selected_models, start=1):
        spec = get_model_spec(registry, model_slug)
        output_dir = model_output_dir(args.output_root, model_slug, args.precision)
        output_dir.mkdir(parents=True, exist_ok=True)
        if resume_completed_models and _model_outputs_complete(args, model_slug, run_pubmedqa_pipeline_eval):
            stage_done += stages_per_model
            _write_model_complete_marker(args, model_slug, spec, run_pubmedqa_pipeline_eval)
            append_jsonl(
                args.output_root / "pipeline_events.jsonl",
                {
                    "event": "model_skipped_completed",
                    "created_at": now_iso(),
                    "model": model_slug,
                    "output_dir": str(output_dir),
                },
            )
            write_json_atomic(
                state_path,
                {
                    "updated_at": now_iso(),
                    "stage": "model_skipped_completed",
                    "model": model_slug,
                    "models": selected_models,
                },
            )
            pipeline_log.info("Skipping completed model=%s output_dir=%s", model_slug, output_dir)
            if args.cleanup_model_data and not (output_dir / "cleanup_manifest.json").exists():
                _cleanup_model_data(args, model_slug, spec, pipeline_log)
            pipeline_progress.update(
                stage_done,
                force=True,
                extra=f"model={model_slug} stage=skipped_completed model_index={model_index}/{len(selected_models)}",
            )
            continue
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
            if run_pubmedqa_pipeline_eval:
                _run_pubmedqa_pipeline_stage(args, model_slug, pipeline_log)
                stage_done += 1
                pipeline_progress.update(
                    stage_done,
                    force=True,
                    extra=f"model={model_slug} stage=pubmedqa_pipeline model_index={model_index}/{len(selected_models)}",
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
            _write_model_complete_marker(args, model_slug, spec, run_pubmedqa_pipeline_eval)
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
            if args.cleanup_model_data:
                _cleanup_model_data(args, model_slug, spec, pipeline_log)
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
        if run_pubmedqa_pipeline_eval:
            _run_pubmedqa_pipeline_aggregate_stage(args, selected_models, pipeline_log)
            stage_done += 1
            pipeline_progress.update(stage_done, force=True, extra="stage=pubmedqa_pipeline_aggregate")
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
    parser.add_argument("--cleanup-model-data", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-completed-models", action=argparse.BooleanOptionalAction, default=True)
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
    parser.add_argument("--skip-pubmedqa-pipeline-eval", action="store_true")
    parser.add_argument("--pubmedqa-dataset", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/eval.json"))
    parser.add_argument("--pubmedqa-corpus", type=Path, default=Path("data/benchmarks/pubmedqa/official_pqal_test/corpus.json"))
    parser.add_argument("--pubmedqa-candidate-k", type=int, default=20)
    parser.add_argument("--pubmedqa-top-k", type=int, default=3)
    parser.add_argument("--pubmedqa-llm-model", default=os.getenv("PUBMEDQA_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b")))
    parser.add_argument("--pubmedqa-ollama-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    parser.add_argument("--pubmedqa-temperature", type=float, default=float(os.getenv("PUBMEDQA_EVAL_TEMPERATURE", "0.0")))
    parser.add_argument("--pubmedqa-cross-encoder-model", default="ncbi/MedCPT-Cross-Encoder")
    parser.add_argument("--pubmedqa-cross-encoder-device")
    parser.add_argument("--pubmedqa-force", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _resume_completed_models_enabled(args: argparse.Namespace) -> bool:
    if args.no_resume or not args.resume_completed_models:
        return False
    return not any(
        (
            args.force_autotune and not args.skip_autotune,
            args.force_eval and not args.skip_eval,
            args.pubmedqa_force and not args.skip_pubmedqa_pipeline_eval,
            args.recreate_index and not args.skip_index,
        )
    )


def _model_outputs_complete(args: argparse.Namespace, model_slug: str, run_pubmedqa_pipeline_eval: bool) -> bool:
    output_dir = model_output_dir(args.output_root, model_slug, args.precision)
    complete_marker = output_dir / "model_complete.json"
    cleanup_marker = output_dir / "cleanup_manifest.json"

    checks: list[Path] = []
    if not args.skip_eval:
        checks.extend(
            [
                output_dir / "evaluation" / "summary.json",
                output_dir / "evaluation" / "qdrant_results.jsonl",
            ]
        )
    if run_pubmedqa_pipeline_eval:
        checks.extend(
            [
                output_dir / "pubmedqa_pipeline" / "summary.json",
                output_dir / "pubmedqa_pipeline" / "results.jsonl",
            ]
        )

    if checks:
        return all(_path_is_ready(path) for path in checks)
    return complete_marker.exists() or cleanup_marker.exists()


def _write_model_complete_marker(
    args: argparse.Namespace,
    model_slug: str,
    spec: Any,
    run_pubmedqa_pipeline_eval: bool,
) -> None:
    output_dir = model_output_dir(args.output_root, model_slug, args.precision)
    payload = {
        "created_at": now_iso(),
        "model": model_slug,
        "display_name": spec.display_name,
        "precision": args.precision,
        "requested_stages": {
            "embedding": not args.skip_embedding,
            "index": not args.skip_index,
            "eval": not args.skip_eval,
            "pubmedqa_pipeline_eval": run_pubmedqa_pipeline_eval,
        },
        "outputs": {
            "evaluation_summary": str(output_dir / "evaluation" / "summary.json"),
            "pubmedqa_pipeline_summary": str(output_dir / "pubmedqa_pipeline" / "summary.json"),
            "qdrant_index_manifest": str(output_dir / "qdrant" / "qdrant_index_manifest.json"),
        },
    }
    write_json_atomic(output_dir / "model_complete.json", payload)


def _path_is_ready(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _run_autotune_stage(args: argparse.Namespace, model_slug: str, log: Any) -> int:
    commands = [
        (
            device,
            shard,
            _base_python_args(args)
            + [
                "-m",
                "embedding_benchmark.autotune_batch",
                "--model",
                model_slug,
                "--registry",
                str(args.registry),
                "--chunks-shard",
                str(shard),
                "--output-root",
                str(args.output_root),
                "--precision",
                args.precision,
                "--device",
                device,
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
        )
        for device, shard in ((args.device_0, args.shard_0), (args.device_1, args.shard_1))
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
            str(shard),
            "--shard-index",
            str(shard_index),
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--device",
            device,
            "--read-batch-size",
            str(args.embed_read_batch_size),
        ]
        + (["--batch-size", str(batch_size)] if batch_size is not None else [])
        + (["--keep-batches"] if args.keep_batches else [])
        for shard_index, (shard, device) in enumerate(((args.shard_0, args.device_0), (args.shard_1, args.device_1)))
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
    completed_shards = set()
    for shard_index in range(2):
        final_path = output_dir / f"embeddings_shard_{shard_index}.parquet"
        manifest_path = output_dir / f"embedding_manifest_shard_{shard_index}.json"
        if final_path.exists():
            completed_shards.add(shard_index)
            log.info("Preserving completed shard before retry: %s", final_path)
            continue
        if manifest_path.exists():
            manifest_path.unlink()
            log.info("Removed stale manifest before retry: %s", manifest_path)

    batch_root = output_dir / "batches"
    if batch_root.exists():
        for shard_index in range(2):
            if shard_index in completed_shards:
                continue
            shard_batch_dir = batch_root / f"shard_{shard_index}"
            if not shard_batch_dir.exists():
                continue
            for path in shard_batch_dir.rglob("*.parquet"):
                path.unlink()
            log.info("Removed partial batch files before retry: %s", shard_batch_dir)


def _run_index_stage(args: argparse.Namespace, model_slug: str, log: Any) -> None:
    upsert_batch_size = args.qdrant_fast_upsert_batch_size if args.qdrant_fast_bulk else args.upsert_batch_size
    output_dir = model_output_dir(args.output_root, model_slug, args.precision)
    embedding_paths = [output_dir / "embeddings_shard_0.parquet", output_dir / "embeddings_shard_1.parquet"]
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
            "--embeddings",
            *[str(path) for path in embedding_paths],
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


def _run_pubmedqa_pipeline_stage(args: argparse.Namespace, model_slug: str, log: Any) -> None:
    _run_command(
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.evaluate_pubmedqa_pipeline",
            "--model",
            model_slug,
            "--registry",
            str(args.registry),
            "--dataset",
            str(args.pubmedqa_dataset),
            "--pubmedqa-corpus",
            str(args.pubmedqa_corpus),
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
            "--candidate-k",
            str(args.pubmedqa_candidate_k),
            "--top-k",
            str(args.pubmedqa_top_k),
            "--llm-model",
            args.pubmedqa_llm_model,
            "--ollama-url",
            args.pubmedqa_ollama_url,
            "--temperature",
            str(args.pubmedqa_temperature),
            "--cross-encoder-model",
            args.pubmedqa_cross_encoder_model,
        ]
        + (["--cross-encoder-device", args.pubmedqa_cross_encoder_device] if args.pubmedqa_cross_encoder_device else [])
        + (["--force"] if args.pubmedqa_force else []),
        cwd=args.project_root,
        env=_env(args),
        log=log,
    )


def _run_pubmedqa_pipeline_aggregate_stage(args: argparse.Namespace, models: list[str], log: Any) -> None:
    _run_command(
        _base_python_args(args)
        + [
            "-m",
            "embedding_benchmark.aggregate_pubmedqa_pipeline",
            "--registry",
            str(args.registry),
            "--output-root",
            str(args.output_root),
            "--precision",
            args.precision,
            "--out-dir",
            str(args.output_root / "reports" / "pubmedqa_pipeline"),
            "--models",
            *models,
        ],
        cwd=args.project_root,
        env=_env(args),
        log=log,
    )
def _cleanup_model_data(args: argparse.Namespace, model_slug: str, spec: Any, log: Any) -> None:
    output_dir = model_output_dir(args.output_root, model_slug, args.precision)
    removed: list[str] = []
    missing: list[str] = []

    collection = collection_name_for(spec, prefix=args.collection_prefix, dtype="f16")
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=args.qdrant_url, timeout=120.0)
        try:
            if _collection_exists(client, collection):
                client.delete_collection(collection_name=collection)
                removed.append(f"qdrant_collection:{collection}")
                log.info("Deleted Qdrant collection after successful model eval: %s", collection)
            else:
                missing.append(f"qdrant_collection:{collection}")
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                close()
    except Exception as exc:
        log.warning("Could not delete Qdrant collection during cleanup collection=%s error=%r", collection, exc)

    for pattern in (
        "embeddings_shard_*.parquet",
        "embeddings_shard_*.parquet.tmp",
        "embedding_manifest_shard_*.json",
        "embed_checkpoint_shard_*.json",
    ):
        for path in output_dir.glob(pattern):
            _remove_path(path, removed=removed, missing=missing, log=log)

    for path in (
        output_dir / "batches",
        output_dir / "qdrant" / "chunk_store.sqlite",
        output_dir / "qdrant" / "bm25_stats.json",
        output_dir / "qdrant" / "index_checkpoint.json",
    ):
        _remove_path(path, removed=removed, missing=missing, log=log)

    cleanup_payload = {
        "created_at": now_iso(),
        "model": model_slug,
        "collection": collection,
        "removed": removed,
        "missing": missing,
        "retained": [
            str(output_dir / "logs"),
            str(output_dir / "metrics.jsonl"),
            str(output_dir / "evaluation"),
            str(output_dir / "pubmedqa_pipeline"),
            str(output_dir / "qdrant" / "qdrant_index_manifest.json"),
        ],
    }
    write_json_atomic(output_dir / "cleanup_manifest.json", cleanup_payload)
    append_jsonl(args.output_root / "pipeline_events.jsonl", {"event": "model_data_cleaned", **cleanup_payload})
    log.info("Cleanup complete model=%s removed_count=%d", model_slug, len(removed))


def _collection_exists(client: Any, collection: str) -> bool:
    collection_exists = getattr(client, "collection_exists", None)
    if collection_exists is not None:
        try:
            return bool(collection_exists(collection_name=collection))
        except Exception as exc:
            if not _is_collection_exists_route_error(exc):
                raise
    try:
        client.get_collection(collection_name=collection)
        return True
    except Exception as exc:
        if _is_missing_collection_error(exc):
            return False
        raise


def _is_collection_exists_route_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "/exists" in text or "collection `exists`" in text


def _is_missing_collection_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "404" in text or "not found" in text or "doesn't exist" in text or "does not exist" in text


def _remove_path(path: Path, *, removed: list[str], missing: list[str], log: Any) -> None:
    if not path.exists():
        missing.append(str(path))
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(str(path))
    log.info("Removed model data path=%s", path)


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
