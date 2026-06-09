from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from typing import Any


def main() -> None:
    args = _parse_args()
    report: dict[str, Any] = {
        "cuda_required": args.require_cuda,
        "h100_required": args.require_h100,
        "bf16_required": args.require_bf16,
    }
    issues: list[str] = []

    try:
        import torch
    except Exception as exc:
        if not args.require_cuda and not args.require_h100 and not args.require_bf16:
            report["torch_importable"] = False
            report["torch_error"] = str(exc)
            print(json.dumps(report, indent=2, sort_keys=True))
            return
        raise RuntimeError("PyTorch is not importable. In Colab, select a GPU runtime before setup.") from exc

    report["torch_importable"] = True
    report["torch_version"] = str(torch.__version__)
    report["torch_cuda_version"] = str(getattr(torch.version, "cuda", "") or "")
    cuda_available = bool(torch.cuda.is_available())
    report["cuda_available"] = cuda_available

    if not cuda_available:
        if args.require_cuda:
            issues.append("torch.cuda.is_available() is false.")
    else:
        device_index = int(torch.cuda.current_device())
        device_name = torch.cuda.get_device_name(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        report.update(
            {
                "device_index": device_index,
                "device_name": device_name,
                "compute_capability": ".".join(str(part) for part in capability),
                "bf16_supported": bf16_supported,
                "device_count": int(torch.cuda.device_count()),
            }
        )
        if args.require_h100 and "H100" not in device_name.upper() and capability != (9, 0):
            issues.append(f"Expected an H100-class GPU, got {device_name} capability={capability}.")
        if args.require_bf16 and not bf16_supported:
            issues.append(f"bf16 is required for the H100 runner, but {device_name} reports no bf16 support.")

    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
        report["nvidia_smi"] = result.stdout.strip()
    else:
        report["nvidia_smi"] = ""

    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Google Colab H100/PyTorch runtime before embedding benchmarks.")
    parser.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-h100", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
