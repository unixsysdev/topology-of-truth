from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from _common import ensure_dir


LATE_WINDOWS = [16, 32]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze within-question final-state and late-window basins.")
    parser.add_argument("--artifact-dir", default="artifacts/qwen35_2b_within_question")
    parser.add_argument("--out-dir", default="")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    out_dir = ensure_dir(args.out_dir or artifact_dir / "analysis")

    pair_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    for question_dir in sorted(artifact_dir.glob("q_*")):
        samples = load_samples(question_dir)
        if not samples:
            continue
        pair_rows.extend(pairwise_rows(question_dir.name, samples))
        question_rows.append(question_summary(question_dir.name, samples))
        reference = load_reference(question_dir)
        if reference is not None:
            reference_rows.extend(reference_distance_rows(question_dir.name, samples, reference))

    pair_df = pd.DataFrame(pair_rows)
    question_df = pd.DataFrame(question_rows)
    reference_df = pd.DataFrame(reference_rows)

    pair_df.to_csv(out_dir / "pairwise_summary.csv", index=False)
    question_df.to_csv(out_dir / "question_summary.csv", index=False)
    if not reference_df.empty:
        reference_df.to_csv(out_dir / "reference_distance_summary.csv", index=False)

    if not pair_df.empty:
        aggregate_pairwise(pair_df).to_csv(out_dir / "aggregate_pairwise.csv", index=False)
        aggregate_pair_subtypes(pair_df).to_csv(out_dir / "aggregate_pair_subtypes.csv", index=False)
    if not reference_df.empty:
        aggregate_reference(reference_df).to_csv(out_dir / "aggregate_reference.csv", index=False)


def load_samples(question_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for sample_json in sorted(question_dir.glob("sample_*/sample.json")):
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        hidden = np.load(question_dir / payload["sample_id"] / "hidden.npy").astype(np.float32)
        rows.append(
            {
                **payload,
                "hidden": hidden,
                "final_state": hidden[-1] if len(hidden) else np.zeros((0,), dtype=np.float32),
                "answer_state": mean_answer_state(hidden, payload.get("answer_token_indices", [])),
            }
        )
    return rows


def load_reference(question_dir: Path) -> dict[str, Any] | None:
    ref_json = question_dir / "reference" / "reference.json"
    ref_hidden = question_dir / "reference" / "reference_hidden.npy"
    if not ref_json.exists() or not ref_hidden.exists():
        return None
    payload = json.loads(ref_json.read_text(encoding="utf-8"))
    hidden = np.load(ref_hidden).astype(np.float32)
    return {
        **payload,
        "hidden": hidden,
        "final_state": hidden[-1] if len(hidden) else np.zeros((0,), dtype=np.float32),
        "answer_state": mean_answer_state(hidden, payload.get("answer_token_indices", [])),
    }


def pairwise_rows(question_id: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    valid = [s for s in samples if s["stop_reason"] != "max_new_tokens" and s["extractor_confident"]]
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a = valid[i]
            b = valid[j]
            pair_type = classify_pair(a["correct"], b["correct"])
            final_euclid = vector_distance(a["final_state"], b["final_state"], metric="euclidean")
            final_cosine = vector_distance(a["final_state"], b["final_state"], metric="cosine")
            row = {
                "question_id": question_id,
                "sample_a": a["sample_id"],
                "sample_b": b["sample_id"],
                "pair_type": pair_type,
                "pair_subtype": classify_pair_subtype(a, b),
                "correct_a": bool(a["correct"]),
                "correct_b": bool(b["correct"]),
                "pred_answer_a": a.get("pred_answer", ""),
                "pred_answer_b": b.get("pred_answer", ""),
                "same_pred_answer": normalize_answer(a.get("pred_answer", "")) == normalize_answer(b.get("pred_answer", "")),
                "final_state_euclidean": final_euclid,
                "final_state_cosine": final_cosine,
            }
            for window in LATE_WINDOWS:
                row[f"late_alignment_cosine_{window}"] = late_alignment(a["hidden"], b["hidden"], window)
            row["answer_state_euclidean"] = vector_distance(a["answer_state"], b["answer_state"], metric="euclidean")
            row["answer_state_cosine"] = vector_distance(a["answer_state"], b["answer_state"], metric="cosine")
            rows.append(row)
    return rows


def question_summary(question_id: str, samples: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [s for s in samples if s["stop_reason"] != "max_new_tokens" and s["extractor_confident"]]
    correct = [s for s in valid if s["correct"]]
    wrong = [s for s in valid if not s["correct"]]
    row = {
        "question_id": question_id,
        "n_valid": len(valid),
        "n_correct": len(correct),
        "n_wrong": len(wrong),
        "mixed_labels": bool(correct and wrong),
    }
    if len(correct) >= 2:
        centroid = np.mean(np.stack([s["final_state"] for s in correct]), axis=0)
        row["mean_correct_to_correct_centroid"] = float(np.mean([vector_distance(s["final_state"], centroid, "euclidean") for s in correct]))
        if wrong:
            row["mean_wrong_to_correct_centroid"] = float(np.mean([vector_distance(s["final_state"], centroid, "euclidean") for s in wrong]))
    else:
        row["mean_correct_to_correct_centroid"] = float("nan")
        row["mean_wrong_to_correct_centroid"] = float("nan")
    return row


def reference_distance_rows(question_id: str, samples: list[dict[str, Any]], reference: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        if sample["stop_reason"] == "max_new_tokens" or not sample["extractor_confident"]:
            continue
        row = {
            "question_id": question_id,
            "sample_id": sample["sample_id"],
            "correct": bool(sample["correct"]),
            "reference_final_state_euclidean": vector_distance(sample["final_state"], reference["final_state"], "euclidean"),
            "reference_final_state_cosine": vector_distance(sample["final_state"], reference["final_state"], "cosine"),
            "reference_answer_state_euclidean": vector_distance(sample["answer_state"], reference["answer_state"], "euclidean"),
            "reference_answer_state_cosine": vector_distance(sample["answer_state"], reference["answer_state"], "cosine"),
        }
        for window in LATE_WINDOWS:
            row[f"reference_late_alignment_cosine_{window}"] = late_alignment(sample["hidden"], reference["hidden"], window)
        rows.append(row)
    return rows


def mean_answer_state(hidden: np.ndarray, indices: list[int]) -> np.ndarray | None:
    if hidden.size == 0 or not indices:
        return None
    valid = [idx for idx in indices if 0 <= idx < len(hidden)]
    if not valid:
        return None
    return hidden[valid].mean(axis=0)


def late_alignment(a_hidden: np.ndarray, b_hidden: np.ndarray, window: int) -> float:
    if len(a_hidden) == 0 or len(b_hidden) == 0:
        return float("nan")
    k = min(window, len(a_hidden), len(b_hidden))
    a = a_hidden[-k:]
    b = b_hidden[-k:]
    numer = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosine = numer / np.clip(denom, 1e-8, None)
    return float(np.mean(cosine))


def vector_distance(a: np.ndarray | None, b: np.ndarray | None, metric: str) -> float:
    if a is None or b is None:
        return float("nan")
    if metric == "euclidean":
        return float(euclidean_distances(a[None, :], b[None, :])[0, 0])
    if metric == "cosine":
        return float(cosine_distances(a[None, :], b[None, :])[0, 0])
    raise ValueError(metric)


def classify_pair(a_correct: bool, b_correct: bool) -> str:
    if a_correct and b_correct:
        return "correct_correct"
    if (not a_correct) and (not b_correct):
        return "wrong_wrong"
    return "correct_wrong"


def classify_pair_subtype(a: dict[str, Any], b: dict[str, Any]) -> str:
    if a["correct"] and b["correct"]:
        return "correct_correct"
    if a["correct"] != b["correct"]:
        return "correct_wrong"
    if normalize_answer(a.get("pred_answer", "")) == normalize_answer(b.get("pred_answer", "")):
        return "wrong_wrong_same_answer"
    return "wrong_wrong_different_answer"


def normalize_answer(answer: Any) -> str:
    return str(answer).strip().lower()


def aggregate_pairwise(pair_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_state_euclidean",
        "final_state_cosine",
        "late_alignment_cosine_16",
        "late_alignment_cosine_32",
        "answer_state_euclidean",
        "answer_state_cosine",
    ]
    rows = []
    for metric in metrics:
        agg = pair_df.groupby("pair_type")[metric].mean(numeric_only=True).to_dict()
        rows.append({"metric": metric, **agg})
    return pd.DataFrame(rows)


def aggregate_pair_subtypes(pair_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "final_state_euclidean",
        "final_state_cosine",
        "late_alignment_cosine_16",
        "late_alignment_cosine_32",
        "answer_state_euclidean",
        "answer_state_cosine",
    ]
    rows = []
    for metric in metrics:
        agg = pair_df.groupby("pair_subtype")[metric].mean(numeric_only=True).to_dict()
        row = {"metric": metric}
        row.update(agg)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_reference(reference_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "reference_final_state_euclidean",
        "reference_final_state_cosine",
        "reference_answer_state_euclidean",
        "reference_answer_state_cosine",
        "reference_late_alignment_cosine_16",
        "reference_late_alignment_cosine_32",
    ]
    rows = []
    for metric in metrics:
        agg = reference_df.groupby("correct")[metric].mean(numeric_only=True).to_dict()
        rows.append({"metric": metric, "wrong": agg.get(False, float("nan")), "correct": agg.get(True, float("nan"))})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
