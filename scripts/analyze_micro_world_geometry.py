from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STATE_KEYS = ["final_prompt", "prompt_tail_mean", "verdict_token", "verdict_span_mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze micro-world verdict-state geometry.")
    parser.add_argument("--manifest", default="artifacts/micro_world_v1/generations/Qwen__Qwen3_5_2B/manifest.csv")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out_dir) if args.out_dir else manifest_path.parent.parent.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["status"] == "ok"].copy()
    rows = load_state_rows(manifest)
    if not rows:
        raise RuntimeError(f"No state rows loaded from {manifest_path}")

    features = pd.DataFrame([{k: v for k, v in row.items() if k != "state"} for row in rows])
    features.to_parquet(out_dir / "verdict_state_index.parquet", index=False)

    summary_rows = []
    distance_rows = []
    for state_key in STATE_KEYS:
        state_rows = [row for row in rows if row["state_key"] == state_key]
        for world_id, group in group_by(state_rows, "world_id").items():
            if len(group) < 3:
                continue
            summary, distances = analyze_world_state(world_id, state_key, group)
            summary_rows.append(summary)
            distance_rows.extend(distances)

    pd.DataFrame(summary_rows).to_csv(out_dir / "within_world_geometry_summary.csv", index=False)
    pd.DataFrame(distance_rows).to_csv(out_dir / "within_world_pair_distances.csv", index=False)
    aggregate_summary(pd.DataFrame(summary_rows)).to_csv(out_dir / "aggregate_geometry_summary.csv", index=False)


def load_state_rows(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, rec in manifest.iterrows():
        states_path = Path(str(rec["states_path"]))
        if not states_path.exists():
            continue
        with np.load(states_path) as data:
            for state_key in STATE_KEYS:
                if state_key not in data.files:
                    continue
                rows.append(
                    {
                        "example_id": rec["example_id"],
                        "world_id": rec["world_id"],
                        "proposition_id": rec["proposition_id"],
                        "paraphrase_id": rec["paraphrase_id"],
                        "label": rec["label"],
                        "pred_label": rec["pred_label"] if isinstance(rec["pred_label"], str) else "",
                        "correct": bool(rec["correct"]),
                        "template_id": rec["template_id"],
                        "template_family": rec["template_family"],
                        "lf_type": rec["lf_type"],
                        "state_key": state_key,
                        "state": np.asarray(data[state_key], dtype=np.float32),
                    }
                )
    return rows


def analyze_world_state(world_id: str, state_key: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    states = np.stack([row["state"] for row in rows]).astype(np.float32)
    labels = [row["label"] for row in rows]
    example_ids = [row["example_id"] for row in rows]
    distances = pairwise_cosine_distances(states)

    pair_rows = []
    same = []
    different = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            same_label = labels[i] == labels[j]
            dist = float(distances[i, j])
            if same_label:
                same.append(dist)
            else:
                different.append(dist)
            pair_rows.append(
                {
                    "world_id": world_id,
                    "state_key": state_key,
                    "example_a": example_ids[i],
                    "example_b": example_ids[j],
                    "label_a": labels[i],
                    "label_b": labels[j],
                    "same_label": same_label,
                    "cosine_distance": dist,
                }
            )

    purity = nearest_neighbor_purity(distances, labels)
    summary = {
        "world_id": world_id,
        "state_key": state_key,
        "n_examples": len(rows),
        "n_true": labels.count("True"),
        "n_false": labels.count("False"),
        "n_unknown": labels.count("Unknown"),
        "same_label_mean_cosine": float(np.mean(same)) if same else float("nan"),
        "different_label_mean_cosine": float(np.mean(different)) if different else float("nan"),
        "same_minus_different_cosine": float(np.mean(same) - np.mean(different)) if same and different else float("nan"),
        "nearest_neighbor_purity": purity,
    }
    return summary, pair_rows


def pairwise_cosine_distances(states: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    normalized = states / np.clip(norms, 1e-8, None)
    similarity = normalized @ normalized.T
    return 1.0 - similarity


def nearest_neighbor_purity(distances: np.ndarray, labels: list[str]) -> float:
    if len(labels) < 2:
        return float("nan")
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = np.argmin(masked, axis=1)
    return float(np.mean([labels[i] == labels[int(nearest[i])] for i in range(len(labels))]))


def aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for state_key, group in summary.groupby("state_key"):
        rows.append(
            {
                "state_key": state_key,
                "n_worlds": len(group),
                "mean_same_label_cosine": group["same_label_mean_cosine"].mean(),
                "mean_different_label_cosine": group["different_label_mean_cosine"].mean(),
                "mean_same_minus_different_cosine": group["same_minus_different_cosine"].mean(),
                "median_same_minus_different_cosine": group["same_minus_different_cosine"].median(),
                "mean_nearest_neighbor_purity": group["nearest_neighbor_purity"].mean(),
                "worlds_same_closer": int((group["same_minus_different_cosine"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


if __name__ == "__main__":
    main()
