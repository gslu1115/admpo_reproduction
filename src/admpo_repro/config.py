from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def load_config(experiment: str, phase: str, seeds: list[int] | None = None) -> dict[str, Any]:
    path = ROOT / "configs" / f"{experiment}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    override = config.pop("phase_overrides", {}).get(phase, {})
    config = _merge(config, override)
    config["phase"] = phase
    if seeds is not None:
        config["seeds"] = seeds
    config["root"] = str(ROOT)
    return config
