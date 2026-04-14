from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from topology_paper.src.features.topology import compute_tda


@dataclass
class TrajectoryCloud:
    mode: str
    t: int
    window_size: int | None
    cloud: np.ndarray


def iterate_trajectory_clouds(
    H: np.ndarray,
    mode: str,
    window_sizes: list[int],
    prefix_min_tokens: int,
    stride: int,
) -> Iterator[TrajectoryCloud]:
    T = len(H)
    if mode == "prefix":
        for t in range(prefix_min_tokens, T + 1, stride):
            yield TrajectoryCloud(mode="prefix", t=t, window_size=None, cloud=H[:t])
        if T >= prefix_min_tokens and (T - prefix_min_tokens) % stride != 0:
            yield TrajectoryCloud(mode="prefix", t=T, window_size=None, cloud=H[:T])
        return

    if mode == "window":
        for window in window_sizes:
            if T < window:
                continue
            for t in range(window, T + 1, stride):
                yield TrajectoryCloud(mode="window", t=t, window_size=window, cloud=H[t - window : t])
            if (T - window) % stride != 0:
                yield TrajectoryCloud(mode="window", t=T, window_size=window, cloud=H[T - window : T])
        return

    raise ValueError(f"Unknown trajectory mode: {mode}")


def compute_dynamic_features(
    H: np.ndarray,
    sample_id: str,
    model_id: str,
    topology_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    window_sizes = list(topology_config.get("window_sizes", [16, 32, 64]))
    prefix_min_tokens = int(topology_config.get("prefix_min_tokens", 8))
    stride = int(topology_config.get("stride", 4))

    for mode in ["prefix", "window"]:
        for item in iterate_trajectory_clouds(H, mode, window_sizes, prefix_min_tokens, stride):
            features = compute_tda(item.cloud, topology_config)
            rows.append(
                {
                    "sample_id": sample_id,
                    "model_id": model_id,
                    "t": item.t,
                    "mode": item.mode,
                    "window_size": item.window_size,
                    **features,
                }
            )

    if not rows:
        return rows

    df = pd.DataFrame(rows).sort_values(["mode", "window_size", "t"], na_position="first")
    grouped = df.groupby(["mode", "window_size"], dropna=False, sort=False)
    df["delta_h0"] = grouped["h0_entropy"].diff().fillna(0.0)
    df["rolling_mean_h0"] = grouped["h0_entropy"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    df["rolling_std_h0"] = grouped["h0_entropy"].transform(lambda s: s.rolling(3, min_periods=1).std().fillna(0.0))
    return df.to_dict(orient="records")


def aggregate_dynamic_features(
    trace_meta: dict[str, Any],
    static_features: dict[str, Any],
    dynamic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": trace_meta["sample_id"],
        "model_id": trace_meta["model_id"],
        "correct": bool(trace_meta.get("correct", False)),
        "n_generated_tokens": int(trace_meta.get("n_generated_tokens", static_features.get("n_points", 0))),
        "h0_entropy_final": static_features.get("h0_entropy", 0.0),
        "h0_total_persistence_final": static_features.get("h0_total_persistence", 0.0),
        "h0_max_persistence_final": static_features.get("h0_max_persistence", 0.0),
        "dust_score_final": static_features.get("dust_score", 0.0),
        "h1_max_persistence_final": static_features.get("h1_max_persistence", 0.0),
    }
    if not dynamic_rows:
        return row

    df = pd.DataFrame(dynamic_rows)
    for (mode, window_size), part in df.groupby(["mode", "window_size"], dropna=False):
        suffix = mode if mode == "prefix" else f"window_{int(window_size)}"
        metrics = _aggregate_one(part)
        for key, value in metrics.items():
            row[f"{suffix}_{key}"] = value

        if suffix == "prefix":
            for key, value in metrics.items():
                row[key] = value

    return row


def _aggregate_one(part: pd.DataFrame) -> dict[str, Any]:
    part = part.sort_values("t")
    h0 = part["h0_entropy"].astype(float).to_numpy()
    t = part["t"].astype(float).to_numpy()
    if len(h0) == 0:
        return {}
    threshold = float(h0.mean() + h0.std())
    above = h0 > threshold
    recovery_count = 0
    for i in range(1, len(above)):
        if above[i - 1] and not above[i]:
            recovery_count += 1
    peak_idx = int(np.argmax(h0))
    post_peak = np.where((np.arange(len(h0)) > peak_idx) & (h0 <= h0[0]))[0]
    time_to_recovery = float(t[post_peak[0]] - t[peak_idx]) if len(post_peak) else np.nan
    late_start = max(0, int(np.floor(len(h0) * 0.8)) - 1)
    terminal_slope = _terminal_slope(t, h0)
    return {
        "peak_h0_entropy": float(h0.max()),
        "mean_h0_entropy": float(h0.mean()),
        "auc_h0_entropy": float(np.trapezoid(h0, t) / max(t[-1], 1.0)),
        "time_to_peak_h0": float(t[peak_idx]),
        "time_to_recovery": time_to_recovery,
        "num_fragmentation_spikes": int(np.sum((part["delta_h0"].to_numpy() > h0.std()) & (part["delta_h0"].to_numpy() > 0))),
        "max_positive_delta_h0": float(max(0.0, part["delta_h0"].max())),
        "terminal_h0_slope": terminal_slope,
        "late_stage_mean_h0": float(h0[late_start:].mean()),
        "recovery_count": int(recovery_count),
        "peak_h0_entropy_len_norm": float(h0.max() / np.log(len(h0) + 2)),
    }


def _terminal_slope(t: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    k = min(5, len(y))
    x = t[-k:]
    z = y[-k:]
    if np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, z, 1)[0])
