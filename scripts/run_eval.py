#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any
from urllib import error, request


DISCLAIMER_RE = re.compile(
    r"(not medical advice|clinical decision support|not a final diagnosis|does not replace physician judgment|nie.*porad[ay] lekarsk)",
    re.IGNORECASE,
)


@dataclass
class CaseResult:
    case_id: str
    prompt_version: str
    total_score: float
    disclaimer_score: float
    keyword_score: float
    forbidden_penalty: float
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline regression benchmark for prompt versions.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/api/chat")
    parser.add_argument("--dataset", default="eval/dataset.jsonl")
    parser.add_argument("--model", default="medgemma")
    parser.add_argument("--baseline", default="v1")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", default="eval/reports")
    parser.add_argument("--timeout", type=float, default=90.0)
    return parser.parse_args()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            cases.append(json.loads(stripped))
    if not cases:
        raise ValueError(f"Dataset at {path} is empty.")
    return cases


def call_chat(
    *,
    api_url: str,
    model: str,
    prompt_version: str,
    question: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "temperature": 0.0,
        "prompt_version": prompt_version,
        "messages": [{"role": "user", "content": question}],
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        api_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.URLError as exc:
        raise RuntimeError(f"API request failed: {exc}") from exc
    return json.loads(raw)


def compute_case_score(case: dict[str, Any], answer: str) -> tuple[float, float, float, float]:
    answer_lower = answer.lower()
    expected_keywords: list[str] = case.get("expected_keywords", [])
    forbidden_keywords: list[str] = case.get("forbidden_keywords", [])
    require_disclaimer = bool(case.get("require_disclaimer", True))

    if expected_keywords:
        matched = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
        keyword_score = matched / len(expected_keywords)
    else:
        keyword_score = 1.0

    has_forbidden = any(keyword.lower() in answer_lower for keyword in forbidden_keywords)
    forbidden_penalty = 1.0 if has_forbidden else 0.0

    if require_disclaimer:
        disclaimer_score = 1.0 if DISCLAIMER_RE.search(answer) else 0.0
    else:
        disclaimer_score = 1.0

    total_score = (0.5 * keyword_score) + (0.4 * disclaimer_score) + (0.1 * (1.0 - forbidden_penalty))
    return total_score, disclaimer_score, keyword_score, forbidden_penalty


def run_version(
    *,
    cases: list[dict[str, Any]],
    api_url: str,
    model: str,
    prompt_version: str,
    timeout: float,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        output = call_chat(
            api_url=api_url,
            model=model,
            prompt_version=prompt_version,
            question=case["question"],
            timeout=timeout,
        )
        response_text = str((output.get("message") or {}).get("content") or "").strip()
        total, disclaimer, keywords, forbidden = compute_case_score(case, response_text)
        results.append(
            CaseResult(
                case_id=case["id"],
                prompt_version=prompt_version,
                total_score=total,
                disclaimer_score=disclaimer,
                keyword_score=keywords,
                forbidden_penalty=forbidden,
                response=response_text,
            )
        )
    return results


def summarize(results: list[CaseResult]) -> dict[str, float]:
    return {
        "total_score_avg": mean(result.total_score for result in results),
        "disclaimer_score_avg": mean(result.disclaimer_score for result in results),
        "keyword_score_avg": mean(result.keyword_score for result in results),
        "forbidden_penalty_avg": mean(result.forbidden_penalty for result in results),
    }


def write_report(
    *,
    output_dir: Path,
    baseline_name: str,
    candidate_name: str,
    baseline_results: list[CaseResult],
    candidate_results: list[CaseResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_json = output_dir / f"report_{timestamp}.json"
    report_md = output_dir / f"report_{timestamp}.md"
    latest_json = output_dir / "latest.json"
    latest_md = output_dir / "latest.md"

    baseline_summary = summarize(baseline_results)
    candidate_summary = summarize(candidate_results)
    delta_total = candidate_summary["total_score_avg"] - baseline_summary["total_score_avg"]

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {"name": baseline_name, "summary": baseline_summary},
        "candidate": {"name": candidate_name, "summary": candidate_summary},
        "delta": {"total_score_avg": delta_total},
        "cases": [
            {
                "case_id": candidate.case_id,
                "baseline_total": baseline.total_score,
                "candidate_total": candidate.total_score,
                "delta_total": candidate.total_score - baseline.total_score,
                "candidate_response": candidate.response,
            }
            for baseline, candidate in zip(baseline_results, candidate_results, strict=True)
        ],
    }
    report_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    lines = [
        "# Prompt Regression Report",
        "",
        f"- Baseline: `{baseline_name}`",
        f"- Candidate: `{candidate_name}`",
        f"- Cases: `{len(candidate_results)}`",
        f"- Delta total score: `{delta_total:+.4f}`",
        "",
        "## Aggregate metrics",
        "",
        f"- Baseline total avg: `{baseline_summary['total_score_avg']:.4f}`",
        f"- Candidate total avg: `{candidate_summary['total_score_avg']:.4f}`",
        f"- Baseline disclaimer avg: `{baseline_summary['disclaimer_score_avg']:.4f}`",
        f"- Candidate disclaimer avg: `{candidate_summary['disclaimer_score_avg']:.4f}`",
        "",
        "## Per-case deltas",
        "",
    ]
    for baseline, candidate in zip(baseline_results, candidate_results, strict=True):
        lines.append(
            f"- `{candidate.case_id}`: baseline `{baseline.total_score:.3f}` -> candidate `{candidate.total_score:.3f}` "
            f"(delta `{candidate.total_score - baseline.total_score:+.3f}`)"
        )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    latest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    output_dir = Path(args.output_dir)

    cases = load_dataset(dataset_path)
    baseline_results = run_version(
        cases=cases,
        api_url=args.api_url,
        model=args.model,
        prompt_version=args.baseline,
        timeout=args.timeout,
    )
    candidate_results = run_version(
        cases=cases,
        api_url=args.api_url,
        model=args.model,
        prompt_version=args.candidate,
        timeout=args.timeout,
    )
    write_report(
        output_dir=output_dir,
        baseline_name=args.baseline,
        candidate_name=args.candidate,
        baseline_results=baseline_results,
        candidate_results=candidate_results,
    )
    print(f"Report saved to {output_dir}")


if __name__ == "__main__":
    main()
