#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from statistics import mean
from typing import Any

from tqdm import tqdm
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
    parser.add_argument(
        "--reuse-baseline-report",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "JSON report from a previous run (e.g. eval/reports/latest.json). "
            "Skips baseline API calls; baseline metrics are taken from the file. "
            "Must match --baseline against the report's baseline name."
        ),
    )
    parser.add_argument("--output-dir", default="eval/reports")
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
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
    max_retries: int,
    retry_backoff: float,
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

    def format_http_error(exc: error.HTTPError) -> str:
        details = [f"HTTP {exc.code}"]
        try:
            body = exc.read().decode("utf-8").strip()
        except Exception:
            body = ""
        if body:
            try:
                parsed = json.loads(body)
                detail = parsed.get("detail")
                if detail:
                    details.append(str(detail))
                else:
                    details.append(body)
            except json.JSONDecodeError:
                details.append(body)
        return ": ".join(details)
    attempt = 0
    while True:
        try:
            with request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except (error.HTTPError, error.URLError) as exc:
            attempt += 1
            is_http_error = isinstance(exc, error.HTTPError)
            status = getattr(exc, "code", None) if is_http_error else None
            retryable_statuses = {429, 500, 502, 503, 504}
            retryable = (status in retryable_statuses) if is_http_error else True
            if attempt > max_retries or not retryable:
                message = format_http_error(exc) if is_http_error else str(exc)
                raise RuntimeError(f"API request failed after {attempt} attempt(s): {message}") from exc
            delay = retry_backoff ** attempt
            time.sleep(delay)


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
    max_retries: int,
    retry_backoff: float,
    progress: tqdm | None = None,
) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in cases:
        output = call_chat(
            api_url=api_url,
            model=model,
            prompt_version=prompt_version,
            question=case["question"],
            timeout=timeout,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
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
        if progress is not None:
            progress.update(1)
            progress.set_postfix(case=case.get("id", "?"))
    return results


def summarize(results: list[CaseResult]) -> dict[str, float]:
    return {
        "total_score_avg": mean(result.total_score for result in results),
        "disclaimer_score_avg": mean(result.disclaimer_score for result in results),
        "keyword_score_avg": mean(result.keyword_score for result in results),
        "forbidden_penalty_avg": mean(result.forbidden_penalty for result in results),
    }


def load_baseline_from_report(
    path: Path,
    *,
    expected_baseline_name: str,
    cases: list[dict[str, Any]],
) -> list[CaseResult]:
    """Rebuild baseline CaseResult list from a prior report JSON (skips re-querying the API)."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    stored_name = str((raw.get("baseline") or {}).get("name") or "").strip()
    if not stored_name:
        raise ValueError(f"No baseline.name in report: {path}")
    if stored_name != expected_baseline_name:
        raise ValueError(
            f"Report baseline is {stored_name!r} but --baseline is {expected_baseline_name!r}. "
            "Use matching names or omit --reuse-baseline-report."
        )

    rows = raw.get("cases") or []
    by_id: dict[str, dict[str, Any]] = {str(row["case_id"]): row for row in rows}
    results: list[CaseResult] = []

    for case in cases:
        cid = str(case["id"])
        if cid not in by_id:
            raise ValueError(f"Case id {cid!r} missing in baseline report {path}")
        row = by_id[cid]

        answer_text = str(row.get("baseline_response") or "").strip()
        if answer_text:
            total, disclaimer, keywords, forbidden = compute_case_score(case, answer_text)
            results.append(
                CaseResult(
                    case_id=cid,
                    prompt_version=stored_name,
                    total_score=total,
                    disclaimer_score=disclaimer,
                    keyword_score=keywords,
                    forbidden_penalty=forbidden,
                    response=answer_text,
                )
            )
            continue

        if "baseline_total" not in row:
            raise ValueError(f"Case {cid!r} in {path} has no baseline_response and no baseline_total")
        total_score = float(row["baseline_total"])
        results.append(
            CaseResult(
                case_id=cid,
                prompt_version=stored_name,
                total_score=total_score,
                disclaimer_score=float(row.get("baseline_disclaimer_score", 0.0)),
                keyword_score=float(row.get("baseline_keyword_score", 0.0)),
                forbidden_penalty=float(row.get("baseline_forbidden_penalty", 0.0)),
                response="",
            )
        )

    if rows and not any(str(r.get("baseline_response") or "").strip() for r in rows):
        print(
            "Note: reuse report has no baseline_response text; baseline disclaimer/keyword averages "
            "may be zeroed. Run a full eval once without --reuse-baseline-report to refresh metrics.",
            file=sys.stderr,
        )

    return results


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
                "baseline_disclaimer_score": baseline.disclaimer_score,
                "baseline_keyword_score": baseline.keyword_score,
                "baseline_forbidden_penalty": baseline.forbidden_penalty,
                "baseline_response": baseline.response,
                "candidate_total": candidate.total_score,
                "candidate_disclaimer_score": candidate.disclaimer_score,
                "candidate_keyword_score": candidate.keyword_score,
                "candidate_forbidden_penalty": candidate.forbidden_penalty,
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
    n = len(cases)

    if args.reuse_baseline_report is not None:
        baseline_results = load_baseline_from_report(
            args.reuse_baseline_report,
            expected_baseline_name=args.baseline,
            cases=cases,
        )
        print(f"Baseline loaded from {args.reuse_baseline_report} ({args.baseline}); skipping API baseline run.")
    else:
        with tqdm(
            total=n,
            desc=f"Baseline ({args.baseline})",
            unit="case",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ) as pbar_base:
            baseline_results = run_version(
                cases=cases,
                api_url=args.api_url,
                model=args.model,
                prompt_version=args.baseline,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_backoff=args.retry_backoff,
                progress=pbar_base,
            )
    with tqdm(
        total=n,
        desc=f"Candidate ({args.candidate})",
        unit="case",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ) as pbar_cand:
        candidate_results = run_version(
            cases=cases,
            api_url=args.api_url,
            model=args.model,
            prompt_version=args.candidate,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
            progress=pbar_cand,
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
