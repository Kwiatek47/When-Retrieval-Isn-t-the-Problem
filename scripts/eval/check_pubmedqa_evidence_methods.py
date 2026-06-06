from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


def main() -> None:
    args = _parse_args()
    report = _load_json(args.report)
    methods = (((report.get("summary") or {}).get("evidence_decision") or {}).get("methods") or {})
    case_count = int((report.get("summary") or {}).get("case_count") or 0)
    only_forbidden = len(methods) == 1 and next(iter(methods), "") in set(args.forbid_only_method)
    required_prefix_count = sum(
        count
        for method, count in methods.items()
        if any(str(method).startswith(prefix) for prefix in args.require_method_prefix)
    )
    passed = not only_forbidden and required_prefix_count >= args.min_required_count
    payload = {
        "report": str(args.report),
        "case_count": case_count,
        "methods": methods,
        "forbid_only_method": args.forbid_only_method,
        "require_method_prefix": args.require_method_prefix,
        "min_required_count": args.min_required_count,
        "required_prefix_count": required_prefix_count,
        "passed": passed,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not passed:
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail PubMedQA evals that silently used the wrong evidence method.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--forbid-only-method", action="append", default=["rules"])
    parser.add_argument("--require-method-prefix", action="append", default=["deberta_classifier"])
    parser.add_argument("--min-required-count", type=int, default=1)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
