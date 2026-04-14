from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import ensure_dir


EARLY_LATE_WINDOW = 32


def main() -> None:
    parser = argparse.ArgumentParser(description="Directional hidden-state pilot on existing 2B traces.")
    parser.add_argument("--artifact-dir", default="artifacts/paired_qwen35_2b_10/traces")
    parser.add_argument("--out-dir", default="artifacts/qwen35_2b_directional_pilot")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    records = load_records(Path(args.artifact_dir))
    summary_df = build_summary(records)
    summary_df.to_csv(out_dir / "sample_directional_summary.csv", index=False)

    classifier_df = run_classifiers(summary_df)
    classifier_df.to_csv(out_dir / "classifier_auc.csv", index=False)

    corr_df = feature_correlation_table(summary_df)
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_directional(summary_df, out_dir / "directional_vs_length.png")


def load_records(artifact_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_dir in sorted(artifact_dir.glob("sample_*")):
        sample_path = sample_dir / "sample.json"
        hidden_path = sample_dir / "hidden.npy"
        if not sample_path.exists() or not hidden_path.exists():
            continue
        with sample_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        record["hidden_path"] = str(hidden_path)
        record.setdefault("answer_length", len(str(record.get("generated_text", "")).split()))
        record["extractor_confident"] = bool(record.get("pred_answer"))
        records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        hidden = np.load(rec["hidden_path"]).astype(np.float32)
        directional = directional_features(hidden)
        rows.append(
            {
                "sample_id": rec["sample_id"],
                "correct": bool(rec["correct"]),
                "token_count": int(rec["token_count"]),
                "answer_length": int(rec["answer_length"]),
                "stop_reason": rec["stop_reason"],
                "pred_answer": rec["pred_answer"],
                "gold_answer": rec["gold_answer"],
                "extractor_confident": bool(rec["extractor_confident"]),
                **directional,
            }
        )
    return pd.DataFrame(rows)


def directional_features(hidden: np.ndarray) -> dict[str, float]:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    if centered.shape[0] < 4:
        return zero_directional_features()

    state_cos = consecutive_cosines(centered[:-1], centered[1:])
    steps = np.diff(centered, axis=0)
    unit_steps = normalize_rows(steps)
    step_dir_cos = consecutive_cosines(unit_steps[:-1], unit_steps[1:]) if len(unit_steps) >= 2 else np.array([0.0], dtype=np.float32)
    turning_angles = np.arccos(np.clip(step_dir_cos, -1.0, 1.0))

    late_states = late_slice_rows(centered)
    late_steps = late_slice_rows(unit_steps)
    late_state_cos = late_slice_values(state_cos)
    late_step_dir_cos = late_slice_values(step_dir_cos)
    late_turning_angles = late_slice_values(turning_angles)

    late_direction_stability = resultant_norm(late_steps)
    full_direction_stability = resultant_norm(unit_steps)
    early_basis = fit_basis(early_window(centered))
    late_basis = fit_basis(late_fixed_window(centered))
    subspace_alignment_mean, subspace_alignment_min = principal_alignment(early_basis, late_basis)

    late_top1_var_frac, late_top3_var_frac, late_participation_ratio = spectrum_features(late_fixed_window(centered))

    return {
        "state_cosine_mean": float(state_cos.mean()),
        "state_cosine_std": float(state_cos.std()),
        "state_cosine_min": float(state_cos.min()),
        "late_state_cosine_mean": float(late_state_cos.mean()),
        "late_state_cosine_std": float(late_state_cos.std()),
        "step_dir_cosine_mean": float(step_dir_cos.mean()),
        "step_dir_cosine_std": float(step_dir_cos.std()),
        "step_dir_cosine_min": float(step_dir_cos.min()),
        "late_step_dir_cosine_mean": float(late_step_dir_cos.mean()),
        "late_step_dir_cosine_std": float(late_step_dir_cos.std()),
        "turning_angle_mean": float(turning_angles.mean()),
        "turning_angle_std": float(turning_angles.std()),
        "turning_angle_max": float(turning_angles.max()),
        "late_turning_angle_mean": float(late_turning_angles.mean()),
        "late_turning_angle_std": float(late_turning_angles.std()),
        "direction_stability_full": float(full_direction_stability),
        "direction_stability_late": float(late_direction_stability),
        "subspace_alignment_mean": float(subspace_alignment_mean),
        "subspace_alignment_min": float(subspace_alignment_min),
        "late_top1_var_frac": float(late_top1_var_frac),
        "late_top3_var_frac": float(late_top3_var_frac),
        "late_participation_ratio": float(late_participation_ratio),
    }


def zero_directional_features() -> dict[str, float]:
    keys = [
        "state_cosine_mean",
        "state_cosine_std",
        "state_cosine_min",
        "late_state_cosine_mean",
        "late_state_cosine_std",
        "step_dir_cosine_mean",
        "step_dir_cosine_std",
        "step_dir_cosine_min",
        "late_step_dir_cosine_mean",
        "late_step_dir_cosine_std",
        "turning_angle_mean",
        "turning_angle_std",
        "turning_angle_max",
        "late_turning_angle_mean",
        "late_turning_angle_std",
        "direction_stability_full",
        "direction_stability_late",
        "subspace_alignment_mean",
        "subspace_alignment_min",
        "late_top1_var_frac",
        "late_top3_var_frac",
        "late_participation_ratio",
    ]
    return {key: 0.0 for key in keys}


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    if len(mat) == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-8, None)


def consecutive_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.array([0.0], dtype=np.float32)
    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = dot / np.clip(denom, 1e-8, None)
    return np.clip(cos.astype(np.float32), -1.0, 1.0)


def late_slice_values(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([0.0], dtype=np.float32)
    n = max(1, int(np.ceil(len(values) * 0.2)))
    return values[-n:]


def late_slice_rows(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return values
    n = max(3, int(np.ceil(len(values) * 0.2)))
    return values[-n:]


def early_window(hidden: np.ndarray) -> np.ndarray:
    n = min(EARLY_LATE_WINDOW, hidden.shape[0])
    return hidden[:n]


def late_fixed_window(hidden: np.ndarray) -> np.ndarray:
    n = min(EARLY_LATE_WINDOW, hidden.shape[0])
    return hidden[-n:]


def fit_basis(hidden: np.ndarray, max_components: int = 4) -> np.ndarray:
    if hidden.shape[0] < 2:
        return np.zeros((hidden.shape[1], 1), dtype=np.float32)
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    comps = min(max_components, centered.shape[0] - 1, centered.shape[1])
    if comps < 1:
        return np.zeros((centered.shape[1], 1), dtype=np.float32)
    pca = PCA(n_components=comps, whiten=False, svd_solver="full")
    pca.fit(centered)
    return pca.components_.T.astype(np.float32)


def principal_alignment(basis_a: np.ndarray, basis_b: np.ndarray) -> tuple[float, float]:
    if basis_a.size == 0 or basis_b.size == 0:
        return 0.0, 0.0
    k = min(basis_a.shape[1], basis_b.shape[1])
    if k < 1:
        return 0.0, 0.0
    overlap = basis_a[:, :k].T @ basis_b[:, :k]
    singular_values = np.linalg.svd(overlap, compute_uv=False)
    singular_values = np.clip(singular_values, 0.0, 1.0)
    return float(singular_values.mean()), float(singular_values.min())


def spectrum_features(hidden: np.ndarray) -> tuple[float, float, float]:
    if hidden.shape[0] < 3:
        return 0.0, 0.0, 0.0
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.sort(np.clip(eigvals, 0.0, None))[::-1]
    total = float(eigvals.sum())
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    probs = eigvals / total
    top1 = float(eigvals[:1].sum() / total)
    top3 = float(eigvals[:3].sum() / total)
    participation = float(1.0 / np.sum(np.square(probs) + 1e-12))
    return top1, top3, participation


def resultant_norm(unit_vectors: np.ndarray) -> float:
    if len(unit_vectors) == 0:
        return 0.0
    return float(np.linalg.norm(unit_vectors.sum(axis=0)) / max(len(unit_vectors), 1))


def run_classifiers(summary_df: pd.DataFrame) -> pd.DataFrame:
    primary = summary_df[(summary_df["stop_reason"] != "max_new_tokens") & (summary_df["extractor_confident"])].copy()
    controls = ["token_count", "answer_length"]
    directional = [
        "state_cosine_mean",
        "state_cosine_std",
        "state_cosine_min",
        "late_state_cosine_mean",
        "late_state_cosine_std",
        "step_dir_cosine_mean",
        "step_dir_cosine_std",
        "step_dir_cosine_min",
        "late_step_dir_cosine_mean",
        "late_step_dir_cosine_std",
        "turning_angle_mean",
        "turning_angle_std",
        "turning_angle_max",
        "late_turning_angle_mean",
        "late_turning_angle_std",
        "direction_stability_full",
        "direction_stability_late",
        "subspace_alignment_mean",
        "subspace_alignment_min",
        "late_top1_var_frac",
        "late_top3_var_frac",
        "late_participation_ratio",
    ]
    rows = [
        evaluate_feature_set(primary, controls, "controls_only"),
        evaluate_feature_set(primary, controls + directional, "controls_plus_directional"),
    ]
    return pd.DataFrame(rows)


def evaluate_feature_set(df: pd.DataFrame, columns: list[str], label: str) -> dict[str, Any]:
    valid_columns = [c for c in columns if c in df.columns and not df[c].isna().all()]
    if not valid_columns:
        return {"model": label, "roc_auc": float("nan"), "status": "unavailable", "features": ""}
    y = df["correct"].astype(int).to_numpy()
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        return {"model": label, "roc_auc": float("nan"), "status": "insufficient_classes", "features": ",".join(valid_columns)}
    X = df[valid_columns]
    cv = StratifiedKFold(n_splits=max(2, min(4, min_class)), shuffle=True, random_state=42)
    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), valid_columns)]
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]
    )
    scores = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    return {"model": label, "roc_auc": float(roc_auc_score(y, scores)), "status": "ok", "features": ",".join(valid_columns)}


def feature_correlation_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    primary = summary_df[(summary_df["stop_reason"] != "max_new_tokens") & (summary_df["extractor_confident"])].copy()
    numeric_cols = [
        c
        for c in primary.columns
        if c not in {"sample_id", "stop_reason", "pred_answer", "gold_answer"}
        and pd.api.types.is_numeric_dtype(primary[c])
    ]
    corr = primary[numeric_cols].corr(numeric_only=True)
    rows = []
    for col in numeric_cols:
        if col in {"correct", "token_count", "answer_length", "extractor_confident"}:
            continue
        rows.append(
            {
                "feature": col,
                "corr_with_correct": float(corr.loc[col, "correct"]),
                "corr_with_token_count": float(corr.loc[col, "token_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_with_correct", key=lambda s: np.abs(s), ascending=False)


def plot_directional(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=summary_df,
        x="token_count",
        y="direction_stability_late",
        hue="correct",
        style="stop_reason",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
