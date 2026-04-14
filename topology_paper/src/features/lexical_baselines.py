from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_logits_npz(path: Path, logits_summary: dict[str, np.ndarray] | None, full_logits: np.ndarray | None = None) -> str | None:
    if logits_summary is None and full_logits is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if logits_summary:
        payload.update(logits_summary)
    if full_logits is not None:
        payload["full_logits"] = full_logits
    np.savez_compressed(path, **payload)
    return str(path)


def lexical_summary_features(logits_path: str | None) -> dict[str, float]:
    if not logits_path:
        return {}
    path = Path(logits_path)
    if not path.exists():
        return {}
    data = np.load(path)
    features: dict[str, float] = {}
    for key in ["entropy", "max_prob", "chosen_logprob", "top2_margin"]:
        if key not in data.files:
            continue
        values = np.asarray(data[key], dtype=np.float32)
        if len(values) == 0:
            continue
        features[f"logit_{key}_mean"] = float(np.mean(values))
        features[f"logit_{key}_std"] = float(np.std(values))
        features[f"logit_{key}_max"] = float(np.max(values))
        features[f"logit_{key}_min"] = float(np.min(values))
    return features

