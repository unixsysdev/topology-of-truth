from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


STATE_KEYS = ["final_prompt", "prompt_tail_mean", "verdict_token", "verdict_span_mean"]
LABELS = ["True", "False", "Unknown"]
PRED_LABELS = LABELS + ["MISSING"]


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
    if manifest.empty:
        raise RuntimeError(f"No successful rows in {manifest_path}")

    manifest["pred_label_norm"] = manifest["pred_label"].map(normalize_pred_label)
    manifest["correct_bool"] = manifest["correct"].map(to_bool)

    rows = load_state_rows(manifest)
    if not rows:
        raise RuntimeError(f"No state rows loaded from {manifest_path}")

    features = pd.DataFrame([{k: v for k, v in row.items() if k != "state"} for row in rows])
    features.to_parquet(out_dir / "verdict_state_index.parquet", index=False)

    write_classification_outputs(manifest, out_dir)

    summary_rows = []
    distance_rows = []
    pair_summary_rows = []
    for state_key in STATE_KEYS:
        state_rows = [row for row in rows if row["state_key"] == state_key]
        for world_id, group in group_by(state_rows, "world_id").items():
            if len(group) < 3:
                continue
            summary, distances, pair_summary = analyze_world_state(world_id, state_key, group)
            summary_rows.append(summary)
            distance_rows.extend(distances)
            pair_summary_rows.extend(pair_summary)

    world_summary = pd.DataFrame(summary_rows)
    pair_distances = pd.DataFrame(distance_rows)
    label_pair_summary = pd.DataFrame(pair_summary_rows)

    world_summary.to_csv(out_dir / "within_world_geometry_summary.csv", index=False)
    pair_distances.to_csv(out_dir / "within_world_pair_distances.csv", index=False)
    label_pair_summary.to_csv(out_dir / "within_world_label_pair_summary.csv", index=False)

    aggregate_geometry = aggregate_summary(world_summary)
    aggregate_geometry.to_csv(out_dir / "aggregate_geometry_summary.csv", index=False)
    sign_test_summary(world_summary).to_csv(out_dir / "sign_test_summary.csv", index=False)
    aggregate_label_pairs(label_pair_summary).to_csv(out_dir / "aggregate_label_pair_summary.csv", index=False)


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
                        "pred_label": rec["pred_label_norm"],
                        "correct": rec["correct_bool"],
                        "template_id": rec["template_id"],
                        "template_family": rec["template_family"],
                        "lf_type": rec["lf_type"],
                        "state_key": state_key,
                        "state": np.asarray(data[state_key], dtype=np.float32),
                    }
                )
    return rows


def write_classification_outputs(manifest: pd.DataFrame, out_dir: Path) -> None:
    overall = classification_summary(manifest)
    overall.to_csv(out_dir / "classification_summary.csv", index=False)

    per_world = []
    for world_id, group in manifest.groupby("world_id"):
        world_summary = classification_summary(group)
        world_summary.insert(0, "world_id", world_id)
        per_world.append(world_summary)
    pd.concat(per_world, ignore_index=True).to_csv(out_dir / "per_world_classification_summary.csv", index=False)

    confusion = confusion_table(manifest)
    confusion.to_csv(out_dir / "confusion_matrix.csv", index=False)

    by_label = []
    precision, recall, f1, support = precision_recall_fscore_support(
        manifest["label"],
        manifest["pred_label_norm"],
        labels=LABELS,
        zero_division=0,
    )
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        by_label.append(
            {
                "label": label,
                "support": int(s),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "accuracy_within_label": float((manifest.loc[manifest["label"] == label, "pred_label_norm"] == label).mean()),
                "predicted_count": int((manifest["pred_label_norm"] == label).sum()),
            }
        )
    pd.DataFrame(by_label).to_csv(out_dir / "classification_by_label.csv", index=False)


def classification_summary(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "n_examples": int(len(manifest)),
            "accuracy": float((manifest["label"] == manifest["pred_label_norm"]).mean()),
            "macro_f1": float(f1_score(manifest["label"], manifest["pred_label_norm"], labels=LABELS, average="macro", zero_division=0)),
            "pred_missing_rate": float((manifest["pred_label_norm"] == "MISSING").mean()),
        }
    ]
    for label in LABELS:
        mask = manifest["label"] == label
        rows[0][f"support_{label.lower()}"] = int(mask.sum())
        rows[0][f"accuracy_{label.lower()}"] = float((manifest.loc[mask, "pred_label_norm"] == label).mean()) if mask.any() else float("nan")
        rows[0][f"pred_rate_{label.lower()}"] = float((manifest["pred_label_norm"] == label).mean())
    return pd.DataFrame(rows)


def confusion_table(manifest: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(
        pd.Categorical(manifest["label"], categories=LABELS, ordered=True),
        pd.Categorical(manifest["pred_label_norm"], categories=PRED_LABELS, ordered=True),
        dropna=False,
    )
    rows = []
    for true_label in LABELS:
        row = {"true_label": true_label}
        for pred_label in PRED_LABELS:
            row[f"pred_{pred_label.lower()}"] = int(matrix.loc[true_label, pred_label])
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_world_state(world_id: str, state_key: str, rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    states = np.stack([row["state"] for row in rows]).astype(np.float32)
    labels = [row["label"] for row in rows]
    example_ids = [row["example_id"] for row in rows]
    distances = pairwise_cosine_distances(states)

    pair_rows = []
    same = []
    different = []
    pair_buckets: dict[str, list[float]] = {}
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            pair_type = canonical_pair_type(labels[i], labels[j])
            same_label = labels[i] == labels[j]
            dist = float(distances[i, j])
            pair_buckets.setdefault(pair_type, []).append(dist)
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
                    "pair_type": pair_type,
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
        "different_minus_same_cosine": float(np.mean(different) - np.mean(same)) if same and different else float("nan"),
        "nearest_neighbor_purity": purity,
    }

    pair_summary = []
    for pair_type, values in sorted(pair_buckets.items()):
        pair_summary.append(
            {
                "world_id": world_id,
                "state_key": state_key,
                "pair_type": pair_type,
                "mean_cosine_distance": float(np.mean(values)),
                "median_cosine_distance": float(np.median(values)),
                "n_pairs": int(len(values)),
                "same_label_pair": "__" not in pair_type,
            }
        )
    return summary, pair_rows, pair_summary


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
                "mean_different_minus_same_cosine": group["different_minus_same_cosine"].mean(),
                "median_different_minus_same_cosine": group["different_minus_same_cosine"].median(),
                "mean_nearest_neighbor_purity": group["nearest_neighbor_purity"].mean(),
                "worlds_positive_gap": int((group["different_minus_same_cosine"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def sign_test_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for state_key, group in summary.groupby("state_key"):
        valid = group["different_minus_same_cosine"].dropna()
        rows.append(
            {
                "state_key": state_key,
                "n_worlds": int(len(group)),
                "n_valid_worlds": int(len(valid)),
                "worlds_positive_gap": int((valid > 0).sum()),
                "worlds_zero_gap": int((valid == 0).sum()),
                "worlds_negative_gap": int((valid < 0).sum()),
                "positive_gap_rate": float((valid > 0).mean()) if len(valid) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def aggregate_label_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (state_key, pair_type), group in summary.groupby(["state_key", "pair_type"]):
        rows.append(
            {
                "state_key": state_key,
                "pair_type": pair_type,
                "n_worlds": int(len(group)),
                "mean_cosine_distance": group["mean_cosine_distance"].mean(),
                "median_cosine_distance": group["median_cosine_distance"].median(),
                "mean_n_pairs": group["n_pairs"].mean(),
            }
        )
    return pd.DataFrame(rows)


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row[key]), []).append(row)
    return out


def normalize_pred_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if not isinstance(value, str) or not value.strip():
        return "MISSING"
    value = value.strip()
    if value in LABELS:
        return value
    return "MISSING"


def to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def canonical_pair_type(label_a: str, label_b: str) -> str:
    if label_a == label_b:
        return label_a
    left, right = sorted([label_a, label_b])
    return f"{left}__{right}"


if __name__ == "__main__":
    main()
