from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "gsm8k_qwen35.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_config_path"] = str(config_path)
    return config


def experiment_dir(config: dict[str, Any]) -> Path:
    root = Path(config.get("output_dir", "outputs"))
    return root / config["experiment_name"]


def ensure_output_dirs(config: dict[str, Any]) -> dict[str, Path]:
    base = experiment_dir(config)
    dirs = {
        "base": base,
        "traces": base / "traces",
        "features": base / "features",
        "stats": base / "stats",
        "figs": base / "figs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__").replace(".", "_").replace("-", "_")

