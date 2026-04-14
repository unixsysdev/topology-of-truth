from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import reopen_json, write_json
from topology_paper.src.config import load_config
from topology_paper.src.features.topology import compute_tda


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute H0 summary for one saved hidden-state artifact.")
    parser.add_argument("--artifact-dir", default="artifacts/one_question")
    parser.add_argument("--sample-name", default="sample_0001")
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    sample = reopen_json(artifact_dir / f"{args.sample_name}.json")
    hidden = np.load(artifact_dir / f"{args.sample_name}_hidden.npy")

    config = load_config(args.config)
    topology_config = dict(config["topology"])
    topology_config["max_homology_dim"] = 0
    features = compute_tda(hidden, topology_config)
    payload = {
        "sample_id": sample["sample_id"],
        "model_id": sample["model_id"],
        "token_count": sample["token_count"],
        "hidden_shape": sample["hidden_shape"],
        "h0_num_components": features["h0_num_components"],
        "h0_entropy": features["h0_entropy"],
        "h0_total_persistence": features["h0_total_persistence"],
        "h0_max_persistence": features["h0_max_persistence"],
        "dust_score": features["dust_score"],
    }
    out_path = artifact_dir / f"{args.sample_name}_topology.json"
    write_json(out_path, payload)
    reopen_json(out_path)


if __name__ == "__main__":
    main()
