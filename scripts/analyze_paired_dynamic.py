from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from _common import ensure_dir
from topology_paper.src.config import load_config
from topology_paper.src.features.dynamic_topology import compute_dynamic_features


def load_trace_dir(trace_dir: Path, model_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample_dir in sorted(trace_dir.glob("sample_*")):
        sample_json = sample_dir / "sample.json"
        hidden_npy = sample_dir / "hidden.npy"
        if not sample_json.exists() or not hidden_npy.exists():
            continue
        sample = pd.read_json(sample_json)
        record = sample.iloc[0].to_dict()
        record["hidden_path"] = str(hidden_npy)
        record["model_id"] = model_id
        rows.append(record)
    return rows


def mad_threshold(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + 1.5 * mad


def aggregate_dynamic(part: pd.DataFrame, token_count: int) -> dict[str, Any]:
    part = part.sort_values("t").copy()
    h0 = part["h0_entropy"].to_numpy(dtype=float)
    t = part["t"].to_numpy(dtype=float)
    if len(h0) == 0:
        return {}
    threshold = mad_threshold(h0)
    above = h0 > threshold
    spike_count = int(np.sum((above[1:] & ~above[:-1]).astype(int))) if len(above) > 1 else 0
    late_n = max(1, int(math.ceil(len(h0) * 0.2)))
    late_h0 = h0[-late_n:]
    late_t = t[-late_n:]
    late_slope = slope(late_t, late_h0)
    peak_idx = int(np.argmax(h0))
    return {
        "peak_h0_entropy": float(np.max(h0)),
        "mean_h0_entropy": float(np.mean(h0)),
        "auc_h0_entropy": float(np.trapezoid(h0, t)),
        "auc_h0_entropy_per_token": float(np.trapezoid(h0, t) / max(token_count, 1)),
        "late_stage_mean_h0": float(np.mean(late_h0)),
        "late_stage_slope_h0": late_slope,
        "max_delta_h0": float(part["delta_h0"].max()),
        "num_h0_spikes": spike_count,
        "time_to_peak_h0": float(t[peak_idx]),
        "spike_threshold_h0": threshold,
    }


def slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def single_feature_auc(values: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = float(ranks[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dynamic H0 analysis on paired 0.8B/2B artifact batch.")
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    parser.add_argument("--artifact-08b", default="artifacts/paired_qwen35_08b_10/traces")
    parser.add_argument("--artifact-2b", default="artifacts/paired_qwen35_2b_10/traces")
    parser.add_argument("--out-dir", default="artifacts/paired_dynamic_qwen35")
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir)
    config = load_config(args.config)
    topology_config = dict(config["topology"])
    topology_config["max_homology_dim"] = 0

    records = []
    records.extend(load_trace_dir(Path(args.artifact_08b), "Qwen/Qwen3.5-0.8B"))
    records.extend(load_trace_dir(Path(args.artifact_2b), "Qwen/Qwen3.5-2B"))

    dynamic_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for rec in records:
        hidden = np.load(rec["hidden_path"])
        dyn = compute_dynamic_features(hidden, rec["sample_id"], rec["model_id"], topology_config)
        for row in dyn:
            row["token_count"] = rec["token_count"]
            row["correct"] = bool(rec["correct"])
            row["stop_reason"] = rec["stop_reason"]
            row["pred_answer"] = rec["pred_answer"]
            row["gold_answer"] = rec["gold_answer"]
        dynamic_rows.extend(dyn)

        prefix = pd.DataFrame([r for r in dyn if r["mode"] == "prefix"])
        windows = pd.DataFrame([r for r in dyn if r["mode"] == "window"])
        summary = {
            "sample_id": rec["sample_id"],
            "model_id": rec["model_id"],
            "correct": bool(rec["correct"]),
            "token_count": int(rec["token_count"]),
            "stop_reason": rec["stop_reason"],
            "pred_answer": rec["pred_answer"],
            "gold_answer": rec["gold_answer"],
            "extractor_confident": bool(rec.get("pred_answer")),
        }
        summary.update(aggregate_dynamic(prefix, int(rec["token_count"])))
        for window in [16, 32, 64]:
            part = windows[windows["window_size"] == window]
            if len(part):
                summary[f"mean_window_h0_{window}"] = float(part["h0_entropy"].mean())
                summary[f"peak_window_h0_{window}"] = float(part["h0_entropy"].max())
        summary["h0_entropy_final"] = float(prefix["h0_entropy"].iloc[-1]) if len(prefix) else float("nan")
        summary_rows.append(summary)

    dynamic_df = pd.DataFrame(dynamic_rows)
    summary_df = pd.DataFrame(summary_rows)
    dynamic_df.to_parquet(out_dir / "dynamic_features.parquet", index=False)
    summary_df.to_csv(out_dir / "sample_dynamic_summary.csv", index=False)

    primary_df = summary_df[
        (summary_df["model_id"] == "Qwen/Qwen3.5-2B")
        & (summary_df["stop_reason"] != "max_new_tokens")
        & (summary_df["extractor_confident"])
    ].copy()

    separability_rows = []
    y = primary_df["correct"].astype(int).to_numpy()
    for feature in ["h0_entropy_final", "peak_h0_entropy", "auc_h0_entropy_per_token", "late_stage_mean_h0", "max_delta_h0"]:
        if feature not in primary_df.columns:
            continue
        values = primary_df[feature].to_numpy(dtype=float)
        auc = single_feature_auc(values, y)
        separability_rows.append(
            {
                "feature": feature,
                "roc_auc_raw": auc,
                "roc_auc_inverted": 1.0 - auc if np.isfinite(auc) else float("nan"),
                "correct_mean": float(primary_df.loc[primary_df["correct"], feature].mean()),
                "wrong_mean": float(primary_df.loc[~primary_df["correct"], feature].mean()),
            }
        )
    pd.DataFrame(separability_rows).to_csv(out_dir / "static_vs_dynamic_separability_2b.csv", index=False)

    sns.set_theme(style="whitegrid")
    make_prefix_plot(dynamic_df, out_dir / "prefix_h0_progress_2b.png")
    make_window_plot(dynamic_df, out_dir / "window_h0_progress_2b.png")
    make_qualitative_plot(dynamic_df, summary_df, out_dir / "qualitative_trajectories.png")


def make_prefix_plot(dynamic_df: pd.DataFrame, out_path: Path) -> None:
    df = dynamic_df[(dynamic_df["model_id"] == "Qwen/Qwen3.5-2B") & (dynamic_df["mode"] == "prefix") & (dynamic_df["stop_reason"] != "max_new_tokens")].copy()
    df["progress"] = df["t"] / df["token_count"]
    df["progress_bin"] = np.clip(np.ceil(df["progress"] * 20) / 20.0, 0.05, 1.0)
    agg = df.groupby(["correct", "progress_bin"], as_index=False)["h0_entropy"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=agg, x="progress_bin", y="h0_entropy", hue="correct")
    plt.xlabel("Normalized generation progress")
    plt.ylabel("Mean prefix H0 entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_window_plot(dynamic_df: pd.DataFrame, out_path: Path) -> None:
    df = dynamic_df[
        (dynamic_df["model_id"] == "Qwen/Qwen3.5-2B")
        & (dynamic_df["mode"] == "window")
        & (dynamic_df["window_size"].isin([32, 64]))
        & (dynamic_df["stop_reason"] != "max_new_tokens")
    ].copy()
    df["progress"] = df["t"] / df["token_count"]
    df["progress_bin"] = np.clip(np.ceil(df["progress"] * 20) / 20.0, 0.05, 1.0)
    agg = df.groupby(["correct", "window_size", "progress_bin"], as_index=False)["h0_entropy"].mean()
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=agg, x="progress_bin", y="h0_entropy", hue="correct", style="window_size")
    plt.xlabel("Normalized generation progress")
    plt.ylabel("Mean sliding-window H0 entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_qualitative_plot(dynamic_df: pd.DataFrame, summary_df: pd.DataFrame, out_path: Path) -> None:
    prefix = dynamic_df[dynamic_df["mode"] == "prefix"].copy()
    cases = []
    two_b_correct = summary_df[(summary_df["model_id"] == "Qwen/Qwen3.5-2B") & (summary_df["correct"])]
    two_b_wrong = summary_df[(summary_df["model_id"] == "Qwen/Qwen3.5-2B") & (~summary_df["correct"]) & (summary_df["stop_reason"] != "max_new_tokens")]
    zero_b_wrong_short = summary_df[(summary_df["model_id"] == "Qwen/Qwen3.5-0.8B") & (~summary_df["correct"]) & (summary_df["stop_reason"] != "max_new_tokens")].sort_values("token_count")
    zero_b_wrong_long = summary_df[(summary_df["model_id"] == "Qwen/Qwen3.5-0.8B") & (~summary_df["correct"])].sort_values("token_count", ascending=False)
    if len(two_b_correct):
        cases.append(("2B correct", two_b_correct.iloc[0]["sample_id"]))
    if len(two_b_wrong):
        cases.append(("2B wrong", two_b_wrong.iloc[0]["sample_id"]))
    if len(zero_b_wrong_short):
        cases.append(("0.8B wrong short", zero_b_wrong_short.iloc[0]["sample_id"]))
    if len(zero_b_wrong_long):
        cases.append(("0.8B wrong long/capped", zero_b_wrong_long.iloc[0]["sample_id"]))
    plt.figure(figsize=(10, 8))
    for idx, (title, sample_id) in enumerate(cases, start=1):
        ax = plt.subplot(2, 2, idx)
        part = prefix[prefix["sample_id"] == sample_id].sort_values("t")
        ax.plot(part["t"], part["h0_entropy"])
        ax.set_title(title)
        ax.set_xlabel("Token step")
        ax.set_ylabel("H0 entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
