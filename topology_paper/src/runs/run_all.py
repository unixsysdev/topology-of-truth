from __future__ import annotations

import argparse

from topology_paper.src.runs.run_generation import run_generation
from topology_paper.src.runs.run_stats import run_stats
from topology_paper.src.runs.run_topology import run_topology


def main() -> None:
    parser = argparse.ArgumentParser(description="Run generation, topology, and stats.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_generation(args.config, models=args.models, start=args.start, end=args.end)
    run_topology(args.config)
    run_stats(args.config, plots=not args.no_plots)


if __name__ == "__main__":
    main()

