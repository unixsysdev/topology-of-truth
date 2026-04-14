from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def make_plots(sample_features_path: Path, dynamic_features_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = pd.read_parquet(sample_features_path)
    dynamic_df = pd.read_parquet(dynamic_features_path)
    sns.set_theme(style="whitegrid")
    plot_final_h0(sample_df, output_dir / "02_final_h0_by_correctness.png")
    plot_dynamic_h0(dynamic_df, sample_df, output_dir / "03_dynamic_h0_trajectory.png")
    plot_representative_traces(dynamic_df, sample_df, output_dir / "04_representative_traces.png")


def plot_final_h0(df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="model_id", y="h0_entropy_final", hue="correct")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_dynamic_h0(dynamic_df: pd.DataFrame, sample_df: pd.DataFrame, path: Path) -> None:
    prefix = dynamic_df[dynamic_df["mode"] == "prefix"].merge(
        sample_df[["sample_id", "model_id", "correct"]], on=["sample_id", "model_id"], how="left"
    )
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=prefix, x="t", y="h0_entropy", hue="correct", style="model_id", errorbar=("ci", 95))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_representative_traces(dynamic_df: pd.DataFrame, sample_df: pd.DataFrame, path: Path) -> None:
    prefix = dynamic_df[dynamic_df["mode"] == "prefix"]
    reps = []
    for model_id in sample_df["model_id"].unique():
        for correct in [True, False]:
            candidates = sample_df[(sample_df["model_id"] == model_id) & (sample_df["correct"] == correct)]
            if len(candidates):
                reps.append(candidates.iloc[0][["sample_id", "model_id", "correct"]].to_dict())
    if not reps:
        return
    keys = pd.DataFrame(reps)
    data = prefix.merge(keys, on=["sample_id", "model_id"], how="inner")
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=data, x="t", y="h0_entropy", hue="model_id", style="correct", units="sample_id", estimator=None)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

