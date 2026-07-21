"""Read-only dependency and frozen-release checks for the smoke experiment."""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


THIS_FILE = Path(__file__).resolve()
EXPERIMENT_ROOT = THIS_FILE.parents[1]
REPO_ROOT = THIS_FILE.parents[3]
WORKSPACE_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline_v4 import release  # noqa: E402


def command_output(command: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip()


def artifact_check(manifest: dict, role: str, *, hash_file: bool) -> dict:
    entry = release.artifact(manifest, role)
    path = REPO_ROOT / entry["path"]
    release.verify_artifact(
        manifest,
        str(REPO_ROOT),
        role,
        str(path),
        verify_sha256=hash_file,
    )
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "expected_sha256": entry["sha256"],
        "sha256_checked": hash_file,
    }


def main() -> None:
    config = json.loads((EXPERIMENT_ROOT / "config" / "smoke.json").read_text(encoding="utf-8"))
    manifest = release.load_manifest(str(REPO_ROOT))
    if config["v41"]["release"] != release.RELEASE_VERSION:
        raise RuntimeError("smoke config does not target the checked-out v4.1 release")

    report = {
        "status": "ready_for_fixture",
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "v41": {
            "release": manifest["release_version"],
            "manifest": str(REPO_ROOT / release.MANIFEST_NAME),
            "artifacts": {
                "l3_checkpoint": artifact_check(manifest, "l3_checkpoint", hash_file=True),
                "l3_dictionary": artifact_check(manifest, "l3_dictionary", hash_file=False),
                "l2_checkpoint": artifact_check(manifest, "l2_checkpoint", hash_file=True),
            },
        },
        "external": {
            "jacobian_lens_source": (WORKSPACE_ROOT / "external" / "jacobian-lens").is_dir(),
            "causcale_source": (WORKSPACE_ROOT / "external" / "CauScale").is_dir(),
            "jlens_importable_here": importlib.util.find_spec("jlens") is not None,
        },
        "model_cache": {
            "qwen3_5_4b": any(
                path.is_dir()
                for base in [WORKSPACE_ROOT / ".hf-jlens", WORKSPACE_ROOT / ".hf_cache"]
                for path in base.glob("**/models--Qwen--Qwen3.5-4B")
            ),
            "prefit_lens": any(
                path.is_file()
                for base in [WORKSPACE_ROOT / ".hf-jlens", WORKSPACE_ROOT / ".hf_cache"]
                for path in base.glob("**/*Qwen3.5-4B_jacobian_lens_n1000.pt")
            ),
            "causcale_synthetic": any(
                path.is_file()
                for base in [WORKSPACE_ROOT / ".hf-causcale", WORKSPACE_ROOT / ".hf_cache"]
                for path in base.glob("**/*auprc=0.905_migrated.ckpt")
            ),
        },
    }

    if not report["cuda_available"]:
        report["status"] = "blocked_no_cuda"
    output = EXPERIMENT_ROOT / "runs" / "preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
