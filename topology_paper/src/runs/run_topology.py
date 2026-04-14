from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from topology_paper.src.config import ensure_output_dirs, load_config
from topology_paper.src.features.dynamic_topology import aggregate_dynamic_features, compute_dynamic_features
from topology_paper.src.features.lexical_baselines import lexical_summary_features
from topology_paper.src.features.topology import compute_tda


def trace_hit_cap(trace: dict[str, Any], config: dict[str, Any]) -> bool:
    if "hit_max_new_tokens" in trace:
        return bool(trace["hit_max_new_tokens"])
    max_new_tokens = int(config["generation"].get("max_new_tokens", 512))
    return int(trace.get("n_generated_tokens", 0)) >= max_new_tokens


def run_topology(config_path: str | Path | None = None) -> None:
    config = load_config(config_path)
    dirs = ensure_output_dirs(config)
    trace_files = sorted(dirs["traces"].glob("**/trace.json"))
    if not trace_files:
        raise FileNotFoundError(f"No trace.json files found under {dirs['traces']}")

    dynamic_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    skipped_capped = 0
    for trace_file in tqdm(trace_files, desc="topology"):
        with trace_file.open("r", encoding="utf-8") as f:
            trace = json.load(f)
        if trace_hit_cap(trace, config):
            skipped_capped += 1
            continue
        H = np.load(trace["hidden_path"])
        static_features = compute_tda(H, config["topology"])
        dyn = compute_dynamic_features(H, trace["sample_id"], trace["model_id"], config["topology"])
        dynamic_rows.extend(dyn)
        sample = aggregate_dynamic_features(trace, static_features, dyn)
        sample.update(lexical_summary_features(trace.get("logits_path")))
        sample_rows.append(sample)

    pd.DataFrame(dynamic_rows).to_parquet(dirs["features"] / "dynamic_features.parquet", index=False)
    pd.DataFrame(sample_rows).to_parquet(dirs["features"] / "sample_features.parquet", index=False)
    print(f"Saved topology features for {len(sample_rows)} traces; skipped {skipped_capped} capped traces.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute static and dynamic topology features.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    run_topology(args.config)


if __name__ == "__main__":
    main()
