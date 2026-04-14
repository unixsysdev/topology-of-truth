#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LABELS = ["True", "False", "Unknown"]
DEFAULT_STATE_KEYS = ["final_prompt", "verdict_token", "verdict_span_mean"]


@dataclass(frozen=True)
class ProbeRun:
    model_label: str
    train_manifest: Path
    test_manifest: Path
    out_subdir: str


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


def normalize_pred_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if not isinstance(value, str):
        return ""
    value = value.strip()
    return value if value in LABELS else ""


def decoder_baseline(test_manifest: pd.DataFrame) -> pd.DataFrame:
    labels = test_manifest["label"].astype(str).to_numpy()
    preds = np.asarray([normalize_pred_label(v) for v in test_manifest["pred_label"]], dtype=object)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=LABELS, zero_division=0)
    row: dict[str, Any] = {
        "n_test": int(len(test_manifest)),
        "accuracy": float(np.mean(labels == preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro", zero_division=0)),
    }
    for label, precision, recall, f1v in zip(LABELS, p, r, f):
        low = label.lower()
        row[f"precision_{low}"] = float(precision)
        row[f"recall_{low}"] = float(recall)
        row[f"f1_{low}"] = float(f1v)
    return pd.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shallow MLP probe sensitivity on micro-world verdict states.")
    parser.add_argument("--out-dir", default="artifacts/micro_world_v1/probe_mlp_sensitivity")
    parser.add_argument("--state-keys", nargs="+", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gen = root / "artifacts" / "micro_world_v1" / "generations"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        ProbeRun(
            model_label="Qwen3.5-2B",
            train_manifest=gen / "Qwen__Qwen3_5_2B_train20" / "manifest.csv",
            test_manifest=gen / "Qwen__Qwen3_5_2B_eval_full" / "manifest_complete_worlds.csv",
            out_subdir="qwen35_2b_train20_eval20",
        ),
        ProbeRun(
            model_label="Qwen3.5-4B (no-think)",
            train_manifest=gen / "Qwen__Qwen3_5_4B_train20_nothink" / "manifest.csv",
            test_manifest=gen / "Qwen__Qwen3_5_4B_eval20_nothink" / "manifest.csv",
            out_subdir="qwen35_4b_train20_eval20_nothink",
        ),
        ProbeRun(
            model_label="Gemma-3-4B-it (no-think)",
            train_manifest=gen / "Gemma__gemma_3_4b_it_train20_nothink" / "manifest.csv",
            test_manifest=gen / "Gemma__gemma_3_4b_it_eval20_nothink" / "manifest.csv",
            out_subdir="gemma_3_4b_it_train20_eval20_nothink",
        ),
    ]

    aggregate_rows: list[dict[str, Any]] = []
    for run in runs:
        train_manifest = pd.read_csv(run.train_manifest)
        test_manifest = pd.read_csv(run.test_manifest)
        train_manifest = train_manifest[train_manifest["status"] == "ok"].copy()
        test_manifest = test_manifest[test_manifest["status"] == "ok"].copy()

        run_dir = out_dir / run.out_subdir
        run_dir.mkdir(parents=True, exist_ok=True)

        summary_rows: list[dict[str, Any]] = []
        pred_rows: list[dict[str, Any]] = []
        for state_key in args.state_keys:
            x_train, y_train, train_meta = load_state_matrix(train_manifest, state_key)
            x_test, y_test, test_meta = load_state_matrix(test_manifest, state_key)
            if len(x_train) == 0 or len(x_test) == 0:
                continue
            if set(np.unique(y_train)) != set(LABELS):
                continue

            label_to_int = {label: i for i, label in enumerate(LABELS)}
            int_to_label = {i: label for label, i in label_to_int.items()}
            y_train_int = np.asarray([label_to_int[str(v)] for v in y_train], dtype=np.int64)
            y_test_int = np.asarray([label_to_int[str(v)] for v in y_test], dtype=np.int64)

            probe = Pipeline(
                [
                    ("scale", StandardScaler(with_mean=True, with_std=True)),
                    (
                        "mlp",
                        MLPClassifier(
                            hidden_layer_sizes=(int(args.hidden),),
                            activation="relu",
                            alpha=float(args.alpha),
                            solver="adam",
                            learning_rate_init=1e-3,
                            max_iter=int(args.max_iter),
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=20,
                            random_state=int(args.seed),
                        ),
                    ),
                ]
            )
            probe.fit(x_train, y_train_int)
            pred_int = probe.predict(x_test)
            pred = np.asarray([int_to_label[int(v)] for v in pred_int], dtype=object)
            pred_proba = probe.predict_proba(x_test)
            cls = probe.named_steps["mlp"].classes_
            cls_idx = {int(c): i for i, c in enumerate(cls)}

            p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=LABELS, zero_division=0)
            row: dict[str, Any] = {
                "model": run.model_label,
                "state_key": state_key,
                "n_train": int(len(y_train)),
                "n_test": int(len(y_test)),
                "train_worlds": int(len({m["world_id"] for m in train_meta})),
                "test_worlds": int(len({m["world_id"] for m in test_meta})),
                "probe_kind": "mlp_shallow",
                "accuracy": float(np.mean(y_test == pred)),
                "macro_f1": float(f1_score(y_test, pred, labels=LABELS, average="macro", zero_division=0)),
                "worlds_with_nonzero_unknown_recall": int(
                    sum(
                        (
                            (np.asarray([y_test[i] for i, m in enumerate(test_meta) if m["world_id"] == w]) == "Unknown")
                            & (np.asarray([pred[i] for i, m in enumerate(test_meta) if m["world_id"] == w]) == "Unknown")
                        ).any()
                        for w in sorted({m["world_id"] for m in test_meta})
                    )
                ),
            }
            for label, precision, recall, f1v in zip(LABELS, p, r, f):
                low = label.lower()
                row[f"precision_{low}"] = float(precision)
                row[f"recall_{low}"] = float(recall)
                row[f"f1_{low}"] = float(f1v)
            summary_rows.append(row)

            for i, m in enumerate(test_meta):
                out = {
                    "model": run.model_label,
                    "state_key": state_key,
                    "example_id": m["example_id"],
                    "world_id": m["world_id"],
                    "proposition_id": m["proposition_id"],
                    "paraphrase_id": m["paraphrase_id"],
                    "label": str(y_test[i]),
                    "pred_label": str(pred[i]),
                    "correct": bool(y_test[i] == pred[i]),
                }
                for label in LABELS:
                    j = cls_idx.get(label, None)
                    li = label_to_int[label]
                    j = cls_idx.get(li, None)
                    out[f"prob_{label.lower()}"] = float(pred_proba[i, j]) if j is not None else float("nan")
                pred_rows.append(out)

        if not summary_rows:
            raise RuntimeError(f"{run.model_label}: no MLP probe outputs produced.")

        summary_df = pd.DataFrame(summary_rows).sort_values("state_key").reset_index(drop=True)
        summary_df.to_csv(run_dir / "probe_summary_mlp.csv", index=False)
        pd.DataFrame(pred_rows).to_parquet(run_dir / "probe_predictions_mlp.parquet", index=False)
        decoder_baseline(test_manifest).to_csv(run_dir / "decoder_baseline_eval.csv", index=False)
        aggregate_rows.extend(summary_rows)

    agg = pd.DataFrame(aggregate_rows).sort_values(["model", "state_key"]).reset_index(drop=True)
    agg.to_csv(out_dir / "comparison_probe_states_mlp.csv", index=False)

    # Join against existing linear comparison for direct sensitivity table.
    linear = pd.read_csv(root / "artifacts" / "micro_world_v1" / "comparison_probe_states_qwen_gemma.csv")
    linear = linear.rename(
        columns={
            "probe_accuracy": "linear_probe_accuracy",
            "probe_macro_f1": "linear_probe_macro_f1",
            "probe_unknown_recall": "linear_probe_unknown_recall",
            "worlds_with_nonzero_unknown_recall": "linear_worlds_with_nonzero_unknown_recall",
        }
    )
    keep_linear = [
        "model",
        "state_key",
        "linear_probe_accuracy",
        "linear_probe_macro_f1",
        "linear_probe_unknown_recall",
        "linear_worlds_with_nonzero_unknown_recall",
    ]
    merged = linear[keep_linear].merge(
        agg.rename(
            columns={
                "accuracy": "mlp_probe_accuracy",
                "macro_f1": "mlp_probe_macro_f1",
                "recall_unknown": "mlp_probe_unknown_recall",
                "worlds_with_nonzero_unknown_recall": "mlp_worlds_with_nonzero_unknown_recall",
            }
        ),
        on=["model", "state_key"],
        how="inner",
    )
    merged["delta_unknown_recall_mlp_minus_linear"] = (
        merged["mlp_probe_unknown_recall"] - merged["linear_probe_unknown_recall"]
    )
    merged.to_csv(out_dir / "comparison_probe_linear_vs_mlp.csv", index=False)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
