from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["True", "False", "Unknown"]
STATE_KEYS_DEFAULT = ["final_prompt", "verdict_token", "verdict_span_mean"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate linear probes on micro-world verdict states.")
    parser.add_argument("--train-manifest", default="artifacts/micro_world_v1/generations/Qwen__Qwen3_5_2B_train20/manifest.csv")
    parser.add_argument("--test-manifest", default="artifacts/micro_world_v1/generations/Qwen__Qwen3_5_2B_eval_full/manifest_complete_worlds.csv")
    parser.add_argument("--state-keys", nargs="+", default=STATE_KEYS_DEFAULT)
    parser.add_argument("--out-dir", default="artifacts/micro_world_v1/probe_qwen35_2b_train20_eval20")
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = pd.read_csv(args.train_manifest)
    test_manifest = pd.read_csv(args.test_manifest)
    train_manifest = train_manifest[train_manifest["status"] == "ok"].copy()
    test_manifest = test_manifest[test_manifest["status"] == "ok"].copy()
    if train_manifest.empty or test_manifest.empty:
        raise RuntimeError("Train/test manifests must have at least one successful row.")

    all_summary_rows: list[dict[str, Any]] = []
    all_label_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    all_pred_rows: list[dict[str, Any]] = []

    for state_key in args.state_keys:
        train_x, train_y, train_meta = load_state_matrix(train_manifest, state_key)
        test_x, test_y, test_meta = load_state_matrix(test_manifest, state_key)
        if len(train_x) == 0 or len(test_x) == 0:
            continue

        model = Pipeline(
            [
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(args.max_iter),
                        C=float(args.c),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        pred_proba = model.predict_proba(test_x)

        all_summary_rows.append(summary_row(state_key, train_y, test_y, pred, train_meta, test_meta))
        all_label_rows.extend(by_label_rows(state_key, test_y, pred))
        all_confusion_rows.extend(confusion_rows(state_key, test_y, pred))
        all_pred_rows.extend(prediction_rows(state_key, test_meta, test_y, pred, pred_proba, model.classes_))

    if not all_summary_rows:
        raise RuntimeError("No probe outputs were produced. Check state keys and manifests.")

    pd.DataFrame(all_summary_rows).to_csv(out_dir / "probe_summary.csv", index=False)
    pd.DataFrame(all_label_rows).to_csv(out_dir / "probe_by_label.csv", index=False)
    pd.DataFrame(all_confusion_rows).to_csv(out_dir / "probe_confusion.csv", index=False)
    pd.DataFrame(all_pred_rows).to_parquet(out_dir / "probe_predictions.parquet", index=False)
    decoder_baseline(test_manifest).to_csv(out_dir / "decoder_baseline_eval.csv", index=False)


def load_state_matrix(manifest: pd.DataFrame, state_key: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    states: list[np.ndarray] = []
    labels: list[str] = []
    meta: list[dict[str, Any]] = []
    for _, rec in manifest.iterrows():
        state_path = Path(str(rec["states_path"]))
        if not state_path.exists():
            continue
        label = str(rec["label"])
        if label not in LABELS:
            continue
        with np.load(state_path) as data:
            if state_key not in data.files:
                continue
            states.append(np.asarray(data[state_key], dtype=np.float32))
        labels.append(label)
        meta.append(
            {
                "example_id": rec["example_id"],
                "world_id": rec["world_id"],
                "proposition_id": rec["proposition_id"],
                "paraphrase_id": rec["paraphrase_id"],
            }
        )
    if not states:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object), []
    return np.stack(states).astype(np.float32), np.asarray(labels, dtype=object), meta


def summary_row(
    state_key: str,
    train_y: np.ndarray,
    test_y: np.ndarray,
    pred_y: np.ndarray,
    train_meta: list[dict[str, Any]],
    test_meta: list[dict[str, Any]],
) -> dict[str, Any]:
    p, r, f, _ = precision_recall_fscore_support(test_y, pred_y, labels=LABELS, zero_division=0)
    metrics = {
        "state_key": state_key,
        "n_train": int(len(train_y)),
        "n_test": int(len(test_y)),
        "train_worlds": int(len({row["world_id"] for row in train_meta})),
        "test_worlds": int(len({row["world_id"] for row in test_meta})),
        "accuracy": float(np.mean(test_y == pred_y)),
        "macro_f1": float(f1_score(test_y, pred_y, labels=LABELS, average="macro", zero_division=0)),
    }
    for label, precision, recall, f1 in zip(LABELS, p, r, f):
        lower = label.lower()
        metrics[f"precision_{lower}"] = float(precision)
        metrics[f"recall_{lower}"] = float(recall)
        metrics[f"f1_{lower}"] = float(f1)
    return metrics


def by_label_rows(state_key: str, true_y: np.ndarray, pred_y: np.ndarray) -> list[dict[str, Any]]:
    p, r, f, s = precision_recall_fscore_support(true_y, pred_y, labels=LABELS, zero_division=0)
    rows = []
    for label, precision, recall, f1, support in zip(LABELS, p, r, f, s):
        rows.append(
            {
                "state_key": state_key,
                "label": label,
                "support": int(support),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_count": int((pred_y == label).sum()),
            }
        )
    return rows


def confusion_rows(state_key: str, true_y: np.ndarray, pred_y: np.ndarray) -> list[dict[str, Any]]:
    matrix = confusion_matrix(true_y, pred_y, labels=LABELS)
    rows: list[dict[str, Any]] = []
    for true_label, values in zip(LABELS, matrix):
        row = {"state_key": state_key, "true_label": true_label}
        for pred_label, value in zip(LABELS, values):
            row[f"pred_{pred_label.lower()}"] = int(value)
        rows.append(row)
    return rows


def prediction_rows(
    state_key: str,
    meta: list[dict[str, Any]],
    true_y: np.ndarray,
    pred_y: np.ndarray,
    pred_proba: np.ndarray,
    class_order: np.ndarray,
) -> list[dict[str, Any]]:
    class_index = {str(label): idx for idx, label in enumerate(class_order)}
    rows = []
    for idx, rec in enumerate(meta):
        row = {
            "state_key": state_key,
            "example_id": rec["example_id"],
            "world_id": rec["world_id"],
            "proposition_id": rec["proposition_id"],
            "paraphrase_id": rec["paraphrase_id"],
            "label": str(true_y[idx]),
            "pred_label": str(pred_y[idx]),
            "correct": bool(true_y[idx] == pred_y[idx]),
        }
        for label in LABELS:
            pos = class_index.get(label)
            row[f"prob_{label.lower()}"] = float(pred_proba[idx, pos]) if pos is not None else float("nan")
        rows.append(row)
    return rows


def decoder_baseline(test_manifest: pd.DataFrame) -> pd.DataFrame:
    labels = test_manifest["label"].astype(str).to_numpy()
    preds = np.asarray([normalize_pred_label(v) for v in test_manifest["pred_label"]], dtype=object)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=LABELS, zero_division=0)
    row: dict[str, Any] = {
        "n_test": int(len(test_manifest)),
        "accuracy": float(np.mean(labels == preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro", zero_division=0)),
    }
    for label, precision, recall, f1 in zip(LABELS, p, r, f):
        lower = label.lower()
        row[f"precision_{lower}"] = float(precision)
        row[f"recall_{lower}"] = float(recall)
        row[f"f1_{lower}"] = float(f1)
    return pd.DataFrame([row])


def normalize_pred_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value if value in LABELS else ""


if __name__ == "__main__":
    main()
