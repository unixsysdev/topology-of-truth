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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import ensure_dir
from topology_paper.src.features.lexical_baselines import lexical_summary_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Local-geometry pilot on existing 2B traces.")
    parser.add_argument("--artifact-dir", default="artifacts/paired_qwen35_2b_10/traces")
    parser.add_argument("--out-dir", default="artifacts/qwen35_2b_geometry_pilot")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    records = load_records(Path(args.artifact_dir))
    summary_df = build_summary(records)
    summary_df.to_csv(out_dir / "sample_geometry_summary.csv", index=False)

    classifier_df = run_classifiers(summary_df)
    classifier_df.to_csv(out_dir / "classifier_auc.csv", index=False)

    corr_df = feature_correlation_table(summary_df)
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_geometry(summary_df, out_dir / "geometry_vs_length.png")


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
        logits_path = sample_dir / "logits_summary.npz"
        if logits_path.exists():
            record["logits_path"] = str(logits_path)
        record.setdefault("answer_length", len(str(record.get("generated_text", "")).split()))
        record["extractor_confident"] = bool(record.get("pred_answer"))
        records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        hidden = np.load(rec["hidden_path"]).astype(np.float32)
        geometry = local_geometry_features(hidden)
        uncertainty = uncertainty_features(rec.get("logits_path"))
        row = {
            "sample_id": rec["sample_id"],
            "correct": bool(rec["correct"]),
            "token_count": int(rec["token_count"]),
            "answer_length": int(rec["answer_length"]),
            "stop_reason": rec["stop_reason"],
            "pred_answer": rec["pred_answer"],
            "gold_answer": rec["gold_answer"],
            "extractor_confident": bool(rec["extractor_confident"]),
            "has_uncertainty": bool(uncertainty),
            **geometry,
            **uncertainty,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def local_geometry_features(hidden: np.ndarray) -> dict[str, float]:
    centered = hidden - hidden.mean(axis=0, keepdims=True)
    n_tokens = centered.shape[0]
    if n_tokens < 3:
        return {
            "path_length_total": 0.0,
            "path_length_per_token": 0.0,
            "net_displacement": 0.0,
            "displacement_ratio": 0.0,
            "step_norm_mean": 0.0,
            "step_norm_max": 0.0,
            "step_norm_std": 0.0,
            "late_step_norm_mean": 0.0,
            "late_step_norm_std": 0.0,
            "state_cosine_mean": 0.0,
            "state_cosine_min": 0.0,
            "state_cosine_std": 0.0,
            "late_state_cosine_mean": 0.0,
            "step_dir_cosine_mean": 0.0,
            "step_dir_cosine_min": 0.0,
            "step_dir_cosine_std": 0.0,
            "late_step_dir_cosine_mean": 0.0,
            "curvature_mean": 0.0,
            "curvature_max": 0.0,
            "late_curvature_mean": 0.0,
            "late_top1_var_frac": 0.0,
            "late_top3_var_frac": 0.0,
            "late_participation_ratio": 0.0,
        }

    diffs = np.diff(centered, axis=0)
    step_norms = np.linalg.norm(diffs, axis=1)
    path_length_total = float(step_norms.sum())
    path_length_per_token = float(path_length_total / max(len(step_norms), 1))
    net_displacement = float(np.linalg.norm(centered[-1] - centered[0]))
    displacement_ratio = float(net_displacement / max(path_length_total, 1e-8))

    state_cos = consecutive_cosines(centered[:-1], centered[1:])
    step_dir_cos = consecutive_cosines(diffs[:-1], diffs[1:]) if len(diffs) >= 2 else np.array([0.0], dtype=np.float32)
    curvature = 1.0 - step_dir_cos

    late_step_norms = late_slice(step_norms)
    late_state_cos = late_slice(state_cos)
    late_step_dir_cos = late_slice(step_dir_cos)
    late_curvature = late_slice(curvature)
    late_trace = late_trace_slice(centered)

    top1_var_frac, top3_var_frac, participation_ratio = late_spectrum_features(late_trace)

    return {
        "path_length_total": path_length_total,
        "path_length_per_token": path_length_per_token,
        "net_displacement": net_displacement,
        "displacement_ratio": displacement_ratio,
        "step_norm_mean": float(step_norms.mean()),
        "step_norm_max": float(step_norms.max()),
        "step_norm_std": float(step_norms.std()),
        "late_step_norm_mean": float(late_step_norms.mean()),
        "late_step_norm_std": float(late_step_norms.std()),
        "state_cosine_mean": float(state_cos.mean()),
        "state_cosine_min": float(state_cos.min()),
        "state_cosine_std": float(state_cos.std()),
        "late_state_cosine_mean": float(late_state_cos.mean()),
        "step_dir_cosine_mean": float(step_dir_cos.mean()),
        "step_dir_cosine_min": float(step_dir_cos.min()),
        "step_dir_cosine_std": float(step_dir_cos.std()),
        "late_step_dir_cosine_mean": float(late_step_dir_cos.mean()),
        "curvature_mean": float(curvature.mean()),
        "curvature_max": float(curvature.max()),
        "late_curvature_mean": float(late_curvature.mean()),
        "late_top1_var_frac": top1_var_frac,
        "late_top3_var_frac": top3_var_frac,
        "late_participation_ratio": participation_ratio,
    }


def consecutive_cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.array([0.0], dtype=np.float32)
    dot = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = dot / np.clip(denom, 1e-8, None)
    return np.clip(cos.astype(np.float32), -1.0, 1.0)


def late_slice(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.array([0.0], dtype=np.float32)
    n = max(1, int(np.ceil(len(values) * 0.2)))
    return values[-n:]


def late_trace_slice(hidden: np.ndarray) -> np.ndarray:
    if len(hidden) == 0:
        return hidden
    n = max(3, int(np.ceil(len(hidden) * 0.2)))
    return hidden[-n:]


def late_spectrum_features(hidden: np.ndarray) -> tuple[float, float, float]:
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


def uncertainty_features(logits_path: str | None) -> dict[str, float]:
    if not logits_path:
        return {}
    features = lexical_summary_features(logits_path)
    path = Path(logits_path)
    if not path.exists():
        return features
    data = np.load(path)
    for key in ["entropy", "top2_margin", "max_prob", "chosen_logprob"]:
        if key not in data.files:
            continue
        values = np.asarray(data[key], dtype=np.float32)
        if len(values) == 0:
            continue
        late_vals = late_slice(values)
        features[f"logit_{key}_late_mean"] = float(late_vals.mean())
        features[f"logit_{key}_late_min"] = float(late_vals.min())
        features[f"logit_{key}_late_max"] = float(late_vals.max())
    return features


def run_classifiers(summary_df: pd.DataFrame) -> pd.DataFrame:
    primary = summary_df[(summary_df["stop_reason"] != "max_new_tokens") & (summary_df["extractor_confident"])].copy()
    controls = ["token_count", "answer_length"]
    uncertainty = sorted([c for c in primary.columns if c.startswith("logit_")])
    geometry = [
        "path_length_total",
        "path_length_per_token",
        "net_displacement",
        "displacement_ratio",
        "step_norm_mean",
        "step_norm_max",
        "step_norm_std",
        "late_step_norm_mean",
        "late_step_norm_std",
        "state_cosine_mean",
        "state_cosine_min",
        "state_cosine_std",
        "late_state_cosine_mean",
        "step_dir_cosine_mean",
        "step_dir_cosine_min",
        "step_dir_cosine_std",
        "late_step_dir_cosine_mean",
        "curvature_mean",
        "curvature_max",
        "late_curvature_mean",
        "late_top1_var_frac",
        "late_top3_var_frac",
        "late_participation_ratio",
    ]
    rows = [
        evaluate_feature_set(primary, controls, "controls_only"),
        evaluate_feature_set(primary, controls + uncertainty, "controls_plus_uncertainty"),
        evaluate_feature_set(primary, controls + geometry, "controls_plus_local_geometry"),
    ]
    if uncertainty:
        rows.append(
            evaluate_feature_set(primary, controls + uncertainty + geometry, "controls_plus_uncertainty_plus_local_geometry")
        )
    else:
        rows.append(
            {
                "model": "controls_plus_uncertainty_plus_local_geometry",
                "roc_auc": float("nan"),
                "status": "unavailable",
                "features": "",
            }
        )
    return pd.DataFrame(rows)


def evaluate_feature_set(df: pd.DataFrame, columns: list[str], label: str) -> dict[str, Any]:
    valid_columns = [c for c in columns if c in df.columns and not df[c].isna().all()]
    if label.endswith("uncertainty") and not any(c.startswith("logit_") for c in valid_columns):
        return {"model": label, "roc_auc": float("nan"), "status": "unavailable", "features": ""}
    if "uncertainty" in label and not any(c.startswith("logit_") for c in valid_columns):
        return {"model": label, "roc_auc": float("nan"), "status": "unavailable", "features": ""}
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
        if c
        not in {"sample_id", "stop_reason", "pred_answer", "gold_answer", "has_uncertainty"}
        and pd.api.types.is_numeric_dtype(primary[c])
    ]
    corr = primary[numeric_cols].corr(numeric_only=True)
    rows = []
    for col in numeric_cols:
        if col in {"correct", "token_count", "answer_length"}:
            continue
        rows.append(
            {
                "feature": col,
                "corr_with_correct": float(corr.loc[col, "correct"]) if "correct" in corr.index else float("nan"),
                "corr_with_token_count": float(corr.loc[col, "token_count"]) if "token_count" in corr.index else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_with_correct", key=lambda s: np.abs(s), ascending=False)


def plot_geometry(summary_df: pd.DataFrame, out_path: Path) -> None:
    df = summary_df.copy()
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="token_count", y="path_length_per_token", hue="correct", style="stop_reason")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
