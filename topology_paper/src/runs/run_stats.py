from __future__ import annotations

import argparse
from pathlib import Path

from topology_paper.src.config import ensure_output_dirs, load_config
from topology_paper.src.eval.plots import make_plots
from topology_paper.src.eval.statistics import run_statistics


def run_stats(config_path: str | Path | None = None, plots: bool = True) -> None:
    config = load_config(config_path)
    dirs = ensure_output_dirs(config)
    sample_features = dirs["features"] / "sample_features.parquet"
    dynamic_features = dirs["features"] / "dynamic_features.parquet"
    run_statistics(sample_features, dirs["stats"], config.get("stats", {}))
    if plots:
        make_plots(sample_features, dynamic_features, dirs["figs"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run predictive statistics and plots.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_stats(args.config, plots=not args.no_plots)


if __name__ == "__main__":
    main()

