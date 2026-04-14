from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def normalize_point_cloud(point_cloud: np.ndarray, standardize: bool = True) -> np.ndarray:
    H = np.asarray(point_cloud, dtype=np.float32)
    if H.ndim != 2:
        raise ValueError(f"Expected [tokens, hidden_dim], got shape {H.shape}")
    if len(H) == 0:
        raise ValueError("Cannot compute topology on an empty point cloud")
    H = H - H.mean(axis=0, keepdims=True)
    if standardize:
        H = H / (H.std(axis=0, keepdims=True) + 1e-8)
    return H.astype(np.float32)


def reduce_point_cloud(
    point_cloud: np.ndarray,
    reduce: str = "pca",
    pca_dims: int = 64,
    pca_explained_variance: float | None = None,
    pca_whiten: bool = True,
) -> np.ndarray:
    if reduce in {None, "none", "off"}:
        return point_cloud
    if reduce != "pca":
        raise ValueError(f"Unsupported reduction policy: {reduce}")

    n_points, n_dims = point_cloud.shape
    if n_points < 3 or n_dims < 2:
        return point_cloud

    max_components = min(n_points - 1, n_dims, pca_dims)
    if pca_explained_variance is not None:
        n_components: int | float = float(pca_explained_variance)
    else:
        n_components = max_components

    pca = PCA(n_components=n_components, whiten=pca_whiten, svd_solver="full")
    reduced = pca.fit_transform(point_cloud)
    if reduced.shape[1] > max_components:
        reduced = reduced[:, :max_components]
    return reduced.astype(np.float32)


def preprocess_point_cloud(point_cloud: np.ndarray, topology_config: dict) -> np.ndarray:
    H = normalize_point_cloud(point_cloud, standardize=bool(topology_config.get("standardize", True)))
    return reduce_point_cloud(
        H,
        reduce=topology_config.get("reduce", "pca"),
        pca_dims=int(topology_config.get("pca_dims", 64)),
        pca_explained_variance=topology_config.get("pca_explained_variance"),
        pca_whiten=bool(topology_config.get("pca_whiten", True)),
    )

