from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


def main() -> None:
    args = _parse_args()
    manifest = _read_optional_json(args.index_manifest)
    gate = _read_optional_json(args.gate_result)
    report = _read_optional_json(args.report)

    lock = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "PubMedQA official PQA-L 500",
        "dataset": _file_entry(args.dataset),
        "chunks": _file_entry(args.chunks),
        "bm25_stats": {
            **_file_entry(args.bm25_stats),
            "version": args.bm25_stats_version,
        },
        "index": {
            "manifest_path": str(args.index_manifest),
            "manifest_sha256": _sha256_optional(args.index_manifest),
            "manifest": manifest,
        },
        "runtime": {
            "python": _python_version(),
            "dependencies": _pip_freeze(),
        },
        "model": {
            "chat_model": args.model,
            "embedding_model": manifest.get("embedding_model", args.embedding_model),
        },
        "qdrant": {
            "url": args.qdrant_url,
            "collection": args.collection,
            "dense_vector_name": args.dense_vector_name,
            "sparse_vector_name": args.sparse_vector_name,
        },
        "eval_config": {
            "api_url": args.api_url,
            "candidate_k": args.candidate_k,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "mode": args.mode,
            "metric_path": args.metric_path,
            "allowed_drop": args.allowed_drop,
            "flags": _selected_env(
                [
                    "RAG_RETRIEVER",
                    "RAG_CORPUS_VERSION",
                    "BM25_STATS_PATH",
                    "RAG_EVIDENCE_FILTER_ENABLED",
                    "RAG_EVIDENCE_JUDGE_METHOD",
                    "RAG_EVIDENCE_CLASSIFIER_ENABLED",
                    "RAG_EVIDENCE_CLASSIFIER_MODEL_PATH",
                    "RAG_EVIDENCE_CLASSIFIER_FAST_THRESHOLD",
                    "RAG_EVIDENCE_CLASSIFIER_HINT_THRESHOLD",
                    "RAG_ANSWER_QUALITY_GATE_ENABLED",
                    "RAG_TOP_K",
                    "RAG_CANDIDATE_K",
                    "RAG_MAX_EXCERPT_CHARS",
                    "OLLAMA_BASE_URL",
                    "OLLAMA_MODEL",
                    "EMBEDDING_SERVICE_URL",
                    "QDRANT_COLLECTION",
                    "QDRANT_VECTOR_NAME",
                    "QDRANT_SPARSE_VECTOR_NAME",
                ]
            ),
        },
        "report": {
            "path": str(args.report),
            "sha256": _sha256_optional(args.report),
            "summary": report.get("summary"),
        },
        "regression_gate": gate,
    }
    _write_json(args.out, lock)
    print(f"Wrote lockfile: {args.out}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a reproducibility lockfile for the official PQA-L 500 eval.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--bm25-stats", type=Path, required=True)
    parser.add_argument("--index-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--gate-result", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", default="medcpt-ncbi-v1")
    parser.add_argument("--collection", required=True)
    parser.add_argument("--qdrant-url", required=True)
    parser.add_argument("--dense-vector-name", required=True)
    parser.add_argument("--sparse-vector-name", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--candidate-k", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--mode", default="benchmark_pqal")
    parser.add_argument("--metric-path", required=True)
    parser.add_argument("--allowed-drop", type=float, required=True)
    parser.add_argument("--bm25-stats-version", default="bm25-v1")
    return parser.parse_args()


def _file_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": _sha256_optional(path),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def _sha256_optional(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    return data if isinstance(data, dict) else {}


def _python_version() -> str:
    import sys

    return sys.version.replace("\n", " ")


def _pip_freeze() -> list[str]:
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []
    return [line for line in result.stdout.splitlines() if line.strip()]


def _selected_env(keys: list[str]) -> dict[str, str]:
    return {key: os.getenv(key, "") for key in keys}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
