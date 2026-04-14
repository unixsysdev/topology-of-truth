from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STATIC_FEATURES = [
    "h0_entropy_final",
    "h0_total_persistence_final",
    "h0_max_persistence_final",
    "dust_score_final",
    "h1_max_persistence_final",
]
DYNAMIC_FEATURES = [
    "peak_h0_entropy",
    "mean_h0_entropy",
    "auc_h0_entropy",
    "time_to_peak_h0",
    "time_to_recovery",
    "num_fragmentation_spikes",
    "max_positive_delta_h0",
    "terminal_h0_slope",
    "late_stage_mean_h0",
    "recovery_count",
]
LENGTH_FEATURES = ["n_generated_tokens"]
LEXICAL_FEATURES = [
    "logit_entropy_mean",
    "logit_entropy_std",
    "logit_max_prob_mean",
    "logit_chosen_logprob_mean",
    "logit_top2_margin_mean",
]


def run_statistics(sample_features_path: Path, output_dir: Path, stats_config: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_parquet(sample_features_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if df["correct"].nunique() < 2:
        result = {"error": "Need both correct and incorrect samples for predictive statistics."}
        _write_json(output_dir / "stats_summary.json", result)
        return result

    random_seed = int(stats_config.get("random_seed", 42))
    bootstrap_samples = int(stats_config.get("bootstrap_samples", 2000))
    permutation_samples = int(stats_config.get("permutation_samples", 500))
    feature_sets = {
        "length_baseline": LENGTH_FEATURES,
        "static_h0": [c for c in STATIC_FEATURES if c != "h1_max_persistence_final"],
        "dynamic_h0": DYNAMIC_FEATURES,
        "static_plus_dynamic": STATIC_FEATURES + DYNAMIC_FEATURES,
        "dynamic_h0_plus_length": DYNAMIC_FEATURES + LENGTH_FEATURES,
        "dynamic_h0_plus_lexical": DYNAMIC_FEATURES + LENGTH_FEATURES + LEXICAL_FEATURES,
    }

    results = {}
    for name, columns in feature_sets.items():
        available = [c for c in columns if c in df.columns]
        if not available:
            continue
        results[name] = evaluate_feature_set(df, available, random_seed, bootstrap_samples)

    results["permutation_dynamic_h0"] = permutation_test_auc(
        df,
        [c for c in DYNAMIC_FEATURES if c in df.columns],
        random_seed=random_seed,
        n_permutations=permutation_samples,
        n_bootstrap=bootstrap_samples,
    )
    _write_json(output_dir / "stats_summary.json", results)
    return results


def evaluate_feature_set(
    df: pd.DataFrame,
    columns: list[str],
    random_seed: int,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    y = df["correct"].astype(int).to_numpy()
    X = df[columns + (["model_id"] if "model_id" in df.columns else [])]
    categorical = ["model_id"] if "model_id" in X.columns else []
    numeric = [c for c in X.columns if c not in categorical]
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        return {
            "error": "Need at least 2 samples in each class for cross-validated AUC.",
            "features": columns,
        }
    cv_splits = max(2, min(5, min_class))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ],
        remainder="drop",
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_seed)),
        ]
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_seed)
    scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, scores)
    return {
        "features": columns,
        "roc_auc": float(auc),
        "bootstrap_ci": bootstrap_auc_ci(y, scores, n_bootstrap=n_bootstrap, random_seed=random_seed),
    }


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, n_bootstrap: int = 2000, random_seed: int = 42) -> dict[str, float]:
    if n_bootstrap <= 0:
        return {"low": float("nan"), "high": float("nan")}
    rng = np.random.default_rng(random_seed)
    aucs = []
    n = len(y)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], scores[idx]))
    if not aucs:
        return {"low": float("nan"), "high": float("nan")}
    return {"low": float(np.percentile(aucs, 2.5)), "high": float(np.percentile(aucs, 97.5))}


def permutation_test_auc(
    df: pd.DataFrame,
    columns: list[str],
    random_seed: int,
    n_permutations: int = 500,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:
    if not columns or df["correct"].nunique() < 2:
        return {"error": "No dynamic columns or only one class."}
    observed = evaluate_feature_set(df, columns, random_seed, n_bootstrap)
    if "roc_auc" not in observed:
        return observed
    rng = np.random.default_rng(random_seed)
    aucs = []
    shuffled = df.copy()
    for _ in range(n_permutations):
        shuffled["correct"] = rng.permutation(shuffled["correct"].to_numpy())
        if shuffled["correct"].nunique() < 2:
            continue
        permuted = evaluate_feature_set(shuffled, columns, random_seed, n_bootstrap=0)
        if "roc_auc" in permuted:
            aucs.append(permuted["roc_auc"])
    p_value = float((np.sum(np.asarray(aucs) >= observed["roc_auc"]) + 1) / (len(aucs) + 1)) if aucs else float("nan")
    return {"observed_auc": observed["roc_auc"], "p_value": p_value, "n_permutations": len(aucs)}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
