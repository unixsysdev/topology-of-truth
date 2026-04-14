#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _compute_for_run(run_name: str, run_dir: Path) -> pd.DataFrame:
    pair_path = run_dir / "within_world_pair_distances.csv"
    idx_path = run_dir / "verdict_state_index.parquet"

    pairs = pd.read_csv(pair_path)
    idx = pd.read_parquet(idx_path)[["example_id", "state_key", "proposition_id"]].drop_duplicates()

    a = idx.rename(columns={"example_id": "example_a", "proposition_id": "proposition_a"})
    b = idx.rename(columns={"example_id": "example_b", "proposition_id": "proposition_b"})

    merged = pairs.merge(
        a[["example_a", "state_key", "proposition_a"]],
        on=["example_a", "state_key"],
        how="left",
    ).merge(
        b[["example_b", "state_key", "proposition_b"]],
        on=["example_b", "state_key"],
        how="left",
    )

    cross = merged[merged["proposition_a"] != merged["proposition_b"]].copy()

    world = (
        cross.groupby(["world_id", "state_key"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "mean_same_crossprop": g.loc[g["same_label"], "cosine_distance"].mean(),
                    "mean_diff_crossprop": g.loc[~g["same_label"], "cosine_distance"].mean(),
                }
            )
        )
        .reset_index(drop=True)
    )
    world["gap_crossprop"] = world["mean_diff_crossprop"] - world["mean_same_crossprop"]

    summary = (
        world.groupby("state_key", as_index=False)
        .agg(
            n_valid_worlds=("gap_crossprop", "count"),
            worlds_positive_gap=("gap_crossprop", lambda s: int((s > 0).sum())),
            positive_gap_rate=("gap_crossprop", lambda s: float((s > 0).mean())),
            mean_diff_minus_same_crossprop=("gap_crossprop", "mean"),
            median_diff_minus_same_crossprop=("gap_crossprop", "median"),
        )
        .sort_values("state_key")
    )
    summary.insert(0, "run", run_name)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute proposition-exclusion geometry control summaries for micro-world runs."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("RUN_NAME", "RUN_DIR"),
        required=True,
        help="Run label and analysis directory containing within_world_pair_distances.csv and verdict_state_index.parquet.",
    )
    parser.add_argument("--out-csv", type=Path, required=True, help="Output CSV path.")
    args = parser.parse_args()

    frames = []
    for run_name, run_dir in args.run:
        frames.append(_compute_for_run(run_name, Path(run_dir)))

    out = pd.concat(frames, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
