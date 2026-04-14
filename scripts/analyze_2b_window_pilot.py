from __future__ import annotations

import argparse
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
from topology_paper.src.config import load_config
from topology_paper.src.features.dynamic_topology import compute_dynamic_features
from topology_paper.src.features.lexical_baselines import lexical_summary_features


WINDOWS = [16, 32, 64]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window H0 pilot analysis for Qwen3.5-2B.")
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    parser.add_argument("--artifact-dir", default="artifacts/paired_qwen35_2b_10/traces")
    parser.add_argument("--out-dir", default="artifacts/qwen35_2b_window_pilot")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(args.out_dir)
    topology_config = dict(config["topology"])
    topology_config["max_homology_dim"] = 0

    records = load_records(Path(args.artifact_dir))
    dynamic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for rec in records:
        hidden = np.load(rec["hidden_path"])
        dyn = compute_dynamic_features(hidden, rec["sample_id"], rec["model_id"], topology_config)
        window_rows = [row for row in dyn if row["mode"] == "window"]
        for row in window_rows:
            row["correct"] = rec["correct"]
            row["token_count"] = rec["token_count"]
            row["answer_length"] = rec["answer_length"]
            row["stop_reason"] = rec["stop_reason"]
        dynamic_rows.extend(window_rows)

        summary = {
            "sample_id": rec["sample_id"],
            "correct": rec["correct"],
            "token_count": rec["token_count"],
            "answer_length": rec["answer_length"],
            "stop_reason": rec["stop_reason"],
            "pred_answer": rec["pred_answer"],
            "gold_answer": rec["gold_answer"],
            "extractor_confident": bool(rec["pred_answer"]),
        }
        summary.update(lexical_summary_features(rec.get("logits_path")))
        summary.update(aggregate_windows(pd.DataFrame(window_rows)))
        summary_rows.append(summary)

    dynamic_df = pd.DataFrame(dynamic_rows)
    summary_df = pd.DataFrame(summary_rows)
    dynamic_df.to_parquet(out_dir / "window_dynamic_features.parquet", index=False)
    summary_df.to_csv(out_dir / "sample_window_summary.csv", index=False)

    primary = summary_df[(summary_df["stop_reason"] != "max_new_tokens") & (summary_df["extractor_confident"])].copy()
    controls = ["token_count", "answer_length", "logit_entropy_mean", "logit_entropy_max"]
    topology = sorted([c for c in primary.columns if any(c.startswith(prefix) for prefix in ["mean_window_h0_", "max_window_h0_", "var_window_h0_", "late_window_mean_h0_", "max_positive_delta_window_h0_", "fraction_high_h0_windows_"])])
    auc_rows = [
        evaluate_feature_set(primary, [c for c in controls if c in primary.columns], "controls_only"),
        evaluate_feature_set(primary, topology, "topology_only"),
        evaluate_feature_set(primary, [c for c in controls if c in primary.columns] + topology, "controls_plus_topology"),
    ]
    pd.DataFrame(auc_rows).to_csv(out_dir / "classifier_auc.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_window_progress(dynamic_df, out_dir / "mean_window_h0_progress_2b.png")
    plot_length_vs_h0(summary_df, out_dir / "length_vs_window_h0_2b.png")


def load_records(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for sample_dir in sorted(artifact_dir.glob("sample_*")):
        sample_path = sample_dir / "sample.json"
        hidden_path = sample_dir / "hidden.npy"
        if not sample_path.exists() or not hidden_path.exists():
            continue
        record = pd.read_json(sample_path).iloc[0].to_dict()
        record["hidden_path"] = str(hidden_path)
        record["model_id"] = "Qwen/Qwen3.5-2B"
        record.setdefault("answer_length", len(str(record.get("generated_text", "")).split()))
        rows.append(record)
    return rows


def aggregate_windows(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if df.empty:
        return out
    for window in WINDOWS:
        part = df[df["window_size"] == window].sort_values("t")
        if part.empty:
            continue
        h0 = part["h0_entropy"].to_numpy(dtype=float)
        threshold = robust_threshold(h0)
        out[f"mean_window_h0_{window}"] = float(np.mean(h0))
        out[f"max_window_h0_{window}"] = float(np.max(h0))
        out[f"var_window_h0_{window}"] = float(np.var(h0))
        late_n = max(1, int(np.ceil(len(h0) * 0.2)))
        out[f"late_window_mean_h0_{window}"] = float(np.mean(h0[-late_n:]))
        out[f"max_positive_delta_window_h0_{window}"] = float(np.maximum(part["delta_h0"].to_numpy(dtype=float), 0.0).max())
        out[f"fraction_high_h0_windows_{window}"] = float(np.mean(h0 > threshold))
    return out


def robust_threshold(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + 1.5 * mad


def evaluate_feature_set(df: pd.DataFrame, columns: list[str], label: str) -> dict[str, Any]:
    valid_columns = [c for c in columns if c in df.columns and not df[c].isna().all()]
    if not valid_columns:
        return {"model": label, "roc_auc": float("nan"), "features": ""}
    y = df["correct"].astype(int).to_numpy()
    min_class = int(np.bincount(y).min())
    if min_class < 2:
        return {"model": label, "roc_auc": float("nan"), "features": ",".join(valid_columns)}
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
    return {"model": label, "roc_auc": float(roc_auc_score(y, scores)), "features": ",".join(valid_columns)}


def plot_window_progress(dynamic_df: pd.DataFrame, out_path: Path) -> None:
    df = dynamic_df[dynamic_df["stop_reason"] != "max_new_tokens"].copy()
    df["progress"] = df["t"] / df["token_count"]
    df["progress_bin"] = np.clip(np.ceil(df["progress"] * 20) / 20.0, 0.05, 1.0)
    agg = df[df["window_size"].isin([32, 64])].groupby(["correct", "window_size", "progress_bin"], as_index=False)["h0_entropy"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=agg, x="progress_bin", y="h0_entropy", hue="correct", style="window_size")
    plt.xlabel("Normalized generation progress")
    plt.ylabel("Mean sliding-window H0 entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_length_vs_h0(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary_df, x="token_count", y="mean_window_h0_32", hue="correct", style="stop_reason")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
