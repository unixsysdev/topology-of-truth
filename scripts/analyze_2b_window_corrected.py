from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ripser import ripser
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _common import ensure_dir
from topology_paper.src.features.lexical_baselines import lexical_summary_features


WINDOWS = [16, 32, 64]
STRIDE = 4


@dataclass
class VariantSpec:
    name: str
    distance: str
    use_fixed_trace_pca: bool
    pca_dims: int = 32


VARIANTS = [
    VariantSpec(name="variant_a_center_only", distance="euclidean", use_fixed_trace_pca=False),
    VariantSpec(name="variant_b_fixed_trace_pca", distance="euclidean", use_fixed_trace_pca=True, pca_dims=32),
    VariantSpec(name="variant_c_center_only_cosine", distance="cosine", use_fixed_trace_pca=False),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Corrected 2B window H0 analysis on existing traces.")
    parser.add_argument("--artifact-dir", default="artifacts/paired_qwen35_2b_10/traces")
    parser.add_argument("--out-dir", default="artifacts/qwen35_2b_window_corrected")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    records = load_records(Path(args.artifact_dir))

    all_variant_summaries = []
    chosen_dynamic = None
    chosen_summary = None
    chosen_name = None

    for variant in VARIANTS:
        dynamic_df, summary_df, diagnostics_df = run_variant(records, variant)
        dynamic_df.to_parquet(out_dir / f"{variant.name}_dynamic_features.parquet", index=False)
        summary_df.to_csv(out_dir / f"{variant.name}_sample_summary.csv", index=False)
        diagnostics_df.to_csv(out_dir / f"{variant.name}_diagnostics.csv", index=False)
        all_variant_summaries.append(diagnostics_df.assign(variant=variant.name))

    diagnostics_all = pd.concat(all_variant_summaries, ignore_index=True)
    diagnostics_all.to_csv(out_dir / "window_variant_diagnostics.csv", index=False)

    # Pick the first non-degenerate variant by average entropy std across windows.
    scores = (
        diagnostics_all.groupby("variant", as_index=False)["h0_entropy_std"].mean().sort_values("h0_entropy_std", ascending=False)
    )
    chosen_name = scores.iloc[0]["variant"]
    chosen_dynamic = pd.read_parquet(out_dir / f"{chosen_name}_dynamic_features.parquet")
    chosen_summary = pd.read_csv(out_dir / f"{chosen_name}_sample_summary.csv")

    chosen_summary.to_csv(out_dir / "chosen_sample_summary.csv", index=False)
    chosen_dynamic.to_parquet(out_dir / "chosen_dynamic_features.parquet", index=False)
    scores.to_csv(out_dir / "variant_ranking.csv", index=False)

    final_df = add_final_state_baseline(chosen_summary, records)
    final_df.to_csv(out_dir / "chosen_sample_summary_with_final_state.csv", index=False)
    classifier_df = run_classifiers(final_df)
    classifier_df.to_csv(out_dir / "classifier_auc.csv", index=False)

    sns.set_theme(style="whitegrid")
    plot_window_progress(chosen_dynamic, out_dir / "chosen_window_progress.png")
    plot_length_vs_feature(final_df, out_dir / "chosen_length_vs_window_h0.png")


def load_records(artifact_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for sample_dir in sorted(artifact_dir.glob("sample_*")):
        sample_path = sample_dir / "sample.json"
        hidden_path = sample_dir / "hidden.npy"
        if not sample_path.exists() or not hidden_path.exists():
            continue
        record = pd.read_json(sample_path).iloc[0].to_dict()
        record["hidden_path"] = str(hidden_path)
        record.setdefault("answer_length", len(str(record.get("generated_text", "")).split()))
        rows.append(record)
    return rows


def run_variant(records: list[dict[str, Any]], variant: VariantSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dynamic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []

    for rec in records:
        hidden = np.load(rec["hidden_path"]).astype(np.float32)
        centered_full = hidden - hidden.mean(axis=0, keepdims=True)
        projector = fit_trace_pca(centered_full, variant) if variant.use_fixed_trace_pca else None
        rows = []
        for window in WINDOWS:
            if len(hidden) < window:
                continue
            prev_h0 = None
            for t in range(window, len(hidden) + 1, STRIDE):
                sub = centered_full[t - window : t]
                features = compute_corrected_tda(sub, variant, projector)
                delta = 0.0 if prev_h0 is None else float(features["h0_entropy"] - prev_h0)
                prev_h0 = float(features["h0_entropy"])
                rows.append(
                    {
                        "sample_id": rec["sample_id"],
                        "window_size": window,
                        "t": t,
                        "progress": t / max(int(rec["token_count"]), 1),
                        "correct": bool(rec["correct"]),
                        "token_count": int(rec["token_count"]),
                        "answer_length": int(rec["answer_length"]),
                        "stop_reason": rec["stop_reason"],
                        "pred_answer": rec["pred_answer"],
                        "gold_answer": rec["gold_answer"],
                        "delta_h0": delta,
                        **features,
                    }
                )
            if rows and rows[-1]["t"] != len(hidden):
                sub = centered_full[-window:]
                features = compute_corrected_tda(sub, variant, projector)
                delta = 0.0 if prev_h0 is None else float(features["h0_entropy"] - prev_h0)
                rows.append(
                    {
                        "sample_id": rec["sample_id"],
                        "window_size": window,
                        "t": len(hidden),
                        "progress": 1.0,
                        "correct": bool(rec["correct"]),
                        "token_count": int(rec["token_count"]),
                        "answer_length": int(rec["answer_length"]),
                        "stop_reason": rec["stop_reason"],
                        "pred_answer": rec["pred_answer"],
                        "gold_answer": rec["gold_answer"],
                        "delta_h0": delta,
                        **features,
                    }
                )

        sample_dynamic = pd.DataFrame(rows)
        dynamic_rows.extend(rows)
        summary = {
            "sample_id": rec["sample_id"],
            "correct": bool(rec["correct"]),
            "token_count": int(rec["token_count"]),
            "answer_length": int(rec["answer_length"]),
            "stop_reason": rec["stop_reason"],
            "pred_answer": rec["pred_answer"],
            "gold_answer": rec["gold_answer"],
            "extractor_confident": bool(rec["pred_answer"]),
        }
        summary.update(lexical_summary_features(rec.get("logits_path")))
        summary.update(aggregate_windows(sample_dynamic))
        summary_rows.append(summary)

        for window in WINDOWS:
            part = sample_dynamic[sample_dynamic["window_size"] == window]
            if part.empty:
                continue
            diagnostics_rows.append(
                {
                    "sample_id": rec["sample_id"],
                    "window_size": window,
                    "h0_entropy_min": float(part["h0_entropy"].min()),
                    "h0_entropy_max": float(part["h0_entropy"].max()),
                    "h0_entropy_std": float(part["h0_entropy"].std()),
                    "h0_total_persistence_std": float(part["h0_total_persistence"].std()),
                    "dust_score_std": float(part["dust_score"].std()),
                }
            )

    return pd.DataFrame(dynamic_rows), pd.DataFrame(summary_rows), pd.DataFrame(diagnostics_rows)


def fit_trace_pca(centered_full: np.ndarray, variant: VariantSpec) -> PCA | None:
    n_points, n_dims = centered_full.shape
    comps = min(variant.pca_dims, n_points - 1, n_dims)
    if comps < 2:
        return None
    pca = PCA(n_components=comps, whiten=False, svd_solver="full")
    pca.fit(centered_full)
    return pca


def compute_corrected_tda(centered_window: np.ndarray, variant: VariantSpec, projector: PCA | None) -> dict[str, float]:
    H = centered_window
    if projector is not None:
        H = projector.transform(H)
    if variant.distance == "cosine":
        distances = pairwise_distances(H, metric="cosine")
        dgms = ripser(distances, distance_matrix=True, maxdim=0)["dgms"]
    else:
        dgms = ripser(H, maxdim=0)["dgms"]
    h0 = dgms[0]
    finite = h0[np.isfinite(h0[:, 1])]
    lifetimes = finite[:, 1] - finite[:, 0] if len(finite) else np.array([], dtype=np.float32)
    lifetimes = lifetimes[lifetimes > 0]
    if len(lifetimes):
        probs = lifetimes / lifetimes.sum()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        total_persistence = float(lifetimes.sum())
        max_persistence = float(lifetimes.max())
    else:
        entropy = 0.0
        total_persistence = 0.0
        max_persistence = 0.0
    dust_score = entropy / (np.log(len(centered_window) + 1) + 1e-12)
    return {
        "h0_entropy": entropy,
        "h0_total_persistence": total_persistence,
        "h0_max_persistence": max_persistence,
        "dust_score": float(dust_score),
    }


def aggregate_windows(sample_dynamic: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if sample_dynamic.empty:
        return out
    for window in WINDOWS:
        part = sample_dynamic[sample_dynamic["window_size"] == window].sort_values("t")
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


def add_final_state_baseline(summary_df: pd.DataFrame, records: list[dict[str, Any]]) -> pd.DataFrame:
    df = summary_df.copy()
    final_vecs = []
    index = []
    for rec in records:
        hidden = np.load(rec["hidden_path"]).astype(np.float32)
        final_state = hidden[-1]
        final_vecs.append(final_state)
        index.append(rec["sample_id"])
    mat = np.stack(final_vecs)
    df = df.set_index("sample_id")
    df["final_state_l2"] = np.linalg.norm(mat, axis=1)
    df["final_state_mean"] = mat.mean(axis=1)
    df["final_state_std"] = mat.std(axis=1)
    df["final_state_maxabs"] = np.abs(mat).max(axis=1)
    comps = min(4, mat.shape[0] - 1, mat.shape[1])
    if comps >= 1:
        pca = PCA(n_components=comps, whiten=False, svd_solver="full")
        pcs = pca.fit_transform(mat - mat.mean(axis=0, keepdims=True))
        for i in range(comps):
            df[f"final_state_pc{i+1}"] = pcs[:, i]
    df.reset_index(inplace=True)
    return df


def run_classifiers(summary_df: pd.DataFrame) -> pd.DataFrame:
    primary = summary_df[(summary_df["stop_reason"] != "max_new_tokens") & (summary_df["extractor_confident"])].copy()
    controls = [c for c in ["token_count", "answer_length", "logit_entropy_mean", "logit_entropy_max"] if c in primary.columns]
    topology = sorted([c for c in primary.columns if any(c.startswith(prefix) for prefix in ["mean_window_h0_", "max_window_h0_", "var_window_h0_", "late_window_mean_h0_", "max_positive_delta_window_h0_", "fraction_high_h0_windows_"])])
    final_state = sorted([c for c in primary.columns if c.startswith("final_state_")])
    rows = [
        evaluate_feature_set(primary, controls, "controls_only"),
        evaluate_feature_set(primary, topology, "corrected_topology_only"),
        evaluate_feature_set(primary, controls + topology, "controls_plus_corrected_topology"),
        evaluate_feature_set(primary, controls + final_state, "controls_plus_final_state"),
        evaluate_feature_set(primary, controls + topology + final_state, "controls_plus_topology_plus_final_state"),
    ]
    return pd.DataFrame(rows)


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
    df["progress_bin"] = np.clip(np.ceil(df["progress"] * 20) / 20.0, 0.05, 1.0)
    agg = df[df["window_size"].isin([32, 64])].groupby(["correct", "window_size", "progress_bin"], as_index=False)["h0_entropy"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=agg, x="progress_bin", y="h0_entropy", hue="correct", style="window_size")
    plt.xlabel("Normalized generation progress")
    plt.ylabel("Corrected sliding-window H0 entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_length_vs_feature(summary_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=summary_df, x="token_count", y="mean_window_h0_32", hue="correct", style="stop_reason")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
