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


def main() -> None:
    parser = argparse.ArgumentParser(description="Uncertainty baseline on 2B traces with saved logits.")
    parser.add_argument("--artifact-dir", default="artifacts/qwen35_2b_uncertainty_15/traces")
    parser.add_argument("--out-dir", default="artifacts/qwen35_2b_uncertainty_15_analysis")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    records = load_records(Path(args.artifact_dir))
    summary_df = build_summary(records)
    summary_df.to_csv(out_dir / "sample_uncertainty_summary.csv", index=False)

    classifier_df = run_classifiers(summary_df)
    classifier_df.to_csv(out_dir / "classifier_auc.csv", index=False)

    corr_df = feature_correlation_table(summary_df)
    corr_df.to_csv(out_dir / "feature_correlations.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_uncertainty(summary_df, out_dir / "uncertainty_vs_length.png")


def load_records(artifact_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample_dir in sorted(artifact_dir.glob("sample_*")):
        sample_path = sample_dir / "sample.json"
        hidden_path = sample_dir / "hidden.npy"
        logits_path = sample_dir / "logits_summary.npz"
        if not sample_path.exists() or not hidden_path.exists() or not logits_path.exists():
            continue
        with sample_path.open("r", encoding="utf-8") as f:
            record = json.load(f)
        record["hidden_path"] = str(hidden_path)
        record["logits_path"] = str(logits_path)
        record.setdefault("answer_length", len(str(record.get("generated_text", "")).split()))
        record["extractor_confident"] = bool(record.get("pred_answer"))
        records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        row = {
            "sample_id": rec["sample_id"],
            "correct": bool(rec["correct"]),
            "token_count": int(rec["token_count"]),
            "answer_length": int(rec["answer_length"]),
            "stop_reason": rec["stop_reason"],
            "pred_answer": rec["pred_answer"],
            "gold_answer": rec["gold_answer"],
            "extractor_confident": bool(rec["extractor_confident"]),
            **uncertainty_features(rec["logits_path"]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def uncertainty_features(logits_path: str) -> dict[str, float]:
    data = np.load(logits_path)
    entropy = np.asarray(data["entropy"], dtype=np.float32)
    margin = np.asarray(data["top2_margin"], dtype=np.float32)

    late_entropy = late_slice(entropy)
    late_margin = late_slice(margin)

    return {
        "mean_token_entropy": float(entropy.mean()),
        "max_token_entropy": float(entropy.max()),
        "late_mean_entropy": float(late_entropy.mean()),
        "final_token_entropy": float(entropy[-1]),
        "mean_top2_margin": float(margin.mean()),
        "min_top2_margin": float(margin.min()),
        "late_mean_top2_margin": float(late_margin.mean()),
    }


def late_slice(values: np.ndarray) -> np.ndarray:
    n = max(1, int(np.ceil(len(values) * 0.2)))
    return values[-n:]


def run_classifiers(summary_df: pd.DataFrame) -> pd.DataFrame:
    primary = summary_df[(summary_df["stop_reason"] != "max_new_tokens") & (summary_df["extractor_confident"])].copy()
    controls = ["token_count", "answer_length"]
    uncertainty = [
        "mean_token_entropy",
        "max_token_entropy",
        "late_mean_entropy",
        "final_token_entropy",
        "mean_top2_margin",
        "min_top2_margin",
        "late_mean_top2_margin",
    ]
    rows = [
        evaluate_feature_set(primary, controls, "controls_only"),
        evaluate_feature_set(primary, controls + uncertainty, "controls_plus_uncertainty"),
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


def plot_uncertainty(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        data=summary_df,
        x="token_count",
        y="late_mean_entropy",
        hue="correct",
        style="stop_reason",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
