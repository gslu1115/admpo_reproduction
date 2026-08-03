from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def environment_manifest(root: Path) -> dict:
    packages = {}
    for package in (
        "torch", "gym", "gymnasium", "D4RL", "mujoco", "mujoco-py",
        "numpy", "h5py", "matplotlib", "PyYAML", "Cython", "tensorboard",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda = torch.version.cuda
    except Exception:
        gpu, cuda = None, None
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
    except Exception:
        driver = None
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": packages,
        "gpu": gpu,
        "driver": driver,
        "cuda": cuda,
        "repository_commit": _git(root, "rev-parse", "HEAD"),
        "vendor_admpo_commit": _git(root / "vendor" / "ADMPO", "rev-parse", "HEAD"),
    }


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
