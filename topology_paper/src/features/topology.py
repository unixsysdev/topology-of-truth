from __future__ import annotations

import math
from typing import Any

import numpy as np
from ripser import ripser
from sklearn.metrics import pairwise_distances

from topology_paper.src.features.point_cloud import preprocess_point_cloud


def persistent_entropy(diagram: np.ndarray) -> float:
    if diagram is None or len(diagram) == 0:
        return 0.0
    finite = diagram[np.isfinite(diagram[:, 1])]
    if len(finite) == 0:
        return 0.0
    lifetimes = finite[:, 1] - finite[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes) == 0:
        return 0.0
    probs = lifetimes / lifetimes.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def persistence_stats(diagram: np.ndarray) -> dict[str, float | int]:
    if diagram is None or len(diagram) == 0:
        return {"num": 0, "total_persistence": 0.0, "max_persistence": 0.0, "entropy": 0.0}
    finite = diagram[np.isfinite(diagram[:, 1])]
    lifetimes = finite[:, 1] - finite[:, 0] if len(finite) else np.array([], dtype=np.float32)
    lifetimes = lifetimes[lifetimes > 0]
    return {
        "num": int(len(diagram)),
        "total_persistence": float(lifetimes.sum()) if len(lifetimes) else 0.0,
        "max_persistence": float(lifetimes.max()) if len(lifetimes) else 0.0,
        "entropy": persistent_entropy(diagram),
    }


def compute_tda(point_cloud: np.ndarray, topology_config: dict[str, Any]) -> dict[str, float | int]:
    if len(point_cloud) < 2:
        return empty_features(n_points=len(point_cloud))

    H = preprocess_point_cloud(point_cloud, topology_config)
    maxdim = int(topology_config.get("max_homology_dim", 1))
    distance = topology_config.get("distance", "euclidean")
    thresh = topology_config.get("max_edge_length")

    kwargs = {"maxdim": maxdim}
    if thresh is not None:
        kwargs["thresh"] = float(thresh)

    if distance == "cosine":
        distances = pairwise_distances(H, metric="cosine")
        result = ripser(distances, distance_matrix=True, **kwargs)
    elif distance == "euclidean":
        result = ripser(H, **kwargs)
    else:
        distances = pairwise_distances(H, metric=distance)
        result = ripser(distances, distance_matrix=True, **kwargs)

    diagrams = result["dgms"]
    h0 = persistence_stats(diagrams[0] if diagrams else np.empty((0, 2)))
    h1 = persistence_stats(diagrams[1] if maxdim >= 1 and len(diagrams) > 1 else np.empty((0, 2)))
    n_points = int(len(point_cloud))
    dust_score = h0["entropy"] / (math.log(n_points + 1) + 1e-12)

    return {
        "n_points": n_points,
        "h0_num_components": h0["num"],
        "h0_entropy": h0["entropy"],
        "h0_total_persistence": h0["total_persistence"],
        "h0_max_persistence": h0["max_persistence"],
        "dust_score": float(dust_score),
        "h1_num_features": h1["num"],
        "h1_entropy": h1["entropy"],
        "h1_total_persistence": h1["total_persistence"],
        "h1_max_persistence": h1["max_persistence"],
    }


def empty_features(n_points: int = 0) -> dict[str, float | int]:
    return {
        "n_points": int(n_points),
        "h0_num_components": int(n_points),
        "h0_entropy": 0.0,
        "h0_total_persistence": 0.0,
        "h0_max_persistence": 0.0,
        "dust_score": 0.0,
        "h1_num_features": 0,
        "h1_entropy": 0.0,
        "h1_total_persistence": 0.0,
        "h1_max_persistence": 0.0,
    }

