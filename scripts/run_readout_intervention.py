#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


LABELS = ["True", "False", "Unknown"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    logits_parquet: Path
    probe_parquet: Path


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, 1e-8, 1.0))


def _prepare_frame(cfg: RunConfig) -> pd.DataFrame:
    logits = pd.read_parquet(cfg.logits_parquet).copy()
    probe = pd.read_parquet(cfg.probe_parquet).copy()
    probe = probe[probe["state_key"] == "verdict_token"].copy()

    keep_logits = [
        "example_id",
        "world_id",
        "gold_label",
        "decoder_pred_label",
        "argmax_logit_label",
        "logp_true",
        "logp_false",
        "logp_unknown",
    ]
    keep_probe = ["example_id", "world_id", "label", "prob_true", "prob_false", "prob_unknown"]
    df = logits[keep_logits].merge(probe[keep_probe], on=["example_id", "world_id"], how="inner")

    # Guard against label mismatches between files.
    mismatch = (df["gold_label"] != df["label"]).sum()
    if mismatch:
        raise ValueError(f"{cfg.name}: found {mismatch} gold-label mismatches across logits/probe merges")

    probe_log_u = _safe_log(df["prob_unknown"].to_numpy(dtype=float))
    probe_log_best_non_u = np.maximum(
        _safe_log(df["prob_true"].to_numpy(dtype=float)),
        _safe_log(df["prob_false"].to_numpy(dtype=float)),
    )
    df["probe_unknown_margin"] = probe_log_u - probe_log_best_non_u
    return df


def _predict_adjusted(df: pd.DataFrame, alpha: float, beta: float) -> np.ndarray:
    lt = df["logp_true"].to_numpy(dtype=float)
    lf = df["logp_false"].to_numpy(dtype=float)
    lu = df["logp_unknown"].to_numpy(dtype=float) + beta + alpha * df["probe_unknown_margin"].to_numpy(dtype=float)
    stacked = np.stack([lt, lf, lu], axis=1)
    return np.array([LABELS[i] for i in np.argmax(stacked, axis=1)], dtype=object)


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y = y_true.astype(str).to_numpy()
    return {
        "accuracy": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "unknown_recall": float(recall_score(y, y_pred, labels=["Unknown"], average="macro", zero_division=0)),
        "true_precision": float(precision_score(y, y_pred, labels=["True"], average="macro", zero_division=0)),
        "false_precision": float(precision_score(y, y_pred, labels=["False"], average="macro", zero_division=0)),
        "pred_unknown_rate": float(np.mean(y_pred == "Unknown")),
    }


def _grid_tune(train_df: pd.DataFrame, alphas: list[float], betas: list[float]) -> tuple[float, float]:
    best = None
    best_key = None
    for alpha in alphas:
        for beta in betas:
            pred = _predict_adjusted(train_df, alpha=alpha, beta=beta)
            m = _metrics(train_df["gold_label"], pred)
            # prioritize macro-F1, then Unknown recall, then accuracy.
            key = (m["macro_f1"], m["unknown_recall"], m["accuracy"])
            if best is None or key > best_key:
                best = (alpha, beta)
                best_key = key
    assert best is not None
    return best


def _run_loow_cv(df: pd.DataFrame, alphas: list[float], betas: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = []
    tuned = []
    for world in sorted(df["world_id"].unique()):
        train = df[df["world_id"] != world]
        test = df[df["world_id"] == world]
        alpha, beta = _grid_tune(train, alphas=alphas, betas=betas)
        pred = _predict_adjusted(test, alpha=alpha, beta=beta)
        tmp = test[["example_id", "world_id", "gold_label"]].copy()
        tmp["pred_label"] = pred
        tmp["alpha"] = alpha
        tmp["beta"] = beta
        preds.append(tmp)
        tuned.append({"world_id": world, "alpha": alpha, "beta": beta})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(tuned)


def run_for_config(cfg: RunConfig, out_dir: Path) -> dict[str, object]:
    df = _prepare_frame(cfg)

    baseline_decoder = _metrics(df["gold_label"], df["decoder_pred_label"].astype(str).to_numpy())
    baseline_argmax = _metrics(df["gold_label"], df["argmax_logit_label"].astype(str).to_numpy())

    # Fixed tiny intervention (no tuning).
    fixed_alpha = 1.0
    fixed_beta = 0.25
    fixed_pred = _predict_adjusted(df, alpha=fixed_alpha, beta=fixed_beta)
    fixed_metrics = _metrics(df["gold_label"], fixed_pred)

    # Leave-one-world-out tuned intervention to avoid direct overfit on held-out world.
    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    betas = [-0.5, -0.25, 0.0, 0.25, 0.5]
    cv_pred_df, cv_tuned_df = _run_loow_cv(df, alphas=alphas, betas=betas)
    cv_metrics = _metrics(cv_pred_df["gold_label"], cv_pred_df["pred_label"].to_numpy())

    run_out = out_dir / cfg.name
    run_out.mkdir(parents=True, exist_ok=True)
    cv_pred_df.to_csv(run_out / "loow_predictions.csv", index=False)
    cv_tuned_df.to_csv(run_out / "loow_selected_params.csv", index=False)

    rows = []
    for variant, m in [
        ("decoder_emitted", baseline_decoder),
        ("argmax_label_logits", baseline_argmax),
        ("fixed_intervention", fixed_metrics),
        ("loow_intervention", cv_metrics),
    ]:
        rows.append(
            {
                "run": cfg.name,
                "variant": variant,
                **m,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(run_out / "intervention_summary.csv", index=False)

    # Confusion matrix for key variants.
    for variant, pred in [
        ("decoder_emitted", df["decoder_pred_label"].astype(str).to_numpy()),
        ("argmax_label_logits", df["argmax_logit_label"].astype(str).to_numpy()),
        ("fixed_intervention", fixed_pred),
        ("loow_intervention", cv_pred_df.sort_values("example_id")["pred_label"].to_numpy()),
    ]:
        # Align y_true to prediction ordering.
        if variant == "loow_intervention":
            aligned = cv_pred_df.sort_values("example_id")
            y_true = aligned["gold_label"].to_numpy()
        else:
            aligned = df.sort_values("example_id")
            y_true = aligned["gold_label"].to_numpy()
            pred = pd.Series(pred, index=df.index).loc[aligned.index].to_numpy()
        mat = pd.crosstab(
            pd.Series(y_true, name="gold"),
            pd.Series(pred, name="pred"),
            dropna=False,
        ).reindex(index=LABELS, columns=LABELS, fill_value=0)
        mat.to_csv(run_out / f"confusion_{variant}.csv")

    return {
        "run": cfg.name,
        "n_examples": int(len(df)),
        "decoder_unknown_recall": baseline_decoder["unknown_recall"],
        "loow_unknown_recall": cv_metrics["unknown_recall"],
        "decoder_accuracy": baseline_decoder["accuracy"],
        "loow_accuracy": cv_metrics["accuracy"],
        "decoder_macro_f1": baseline_decoder["macro_f1"],
        "loow_macro_f1": cv_metrics["macro_f1"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal readout intervention on micro-world verdict logits.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/micro_world_v1/readout_intervention"))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    mw = root / "artifacts" / "micro_world_v1"
    runs = [
        RunConfig(
            name="gemma_3_4b_it_raw",
            logits_parquet=mw / "label_logits_gemma_3_4b_it_eval20_raw" / "label_logits_detailed.parquet",
            probe_parquet=mw / "probe_gemma_3_4b_it_train20_eval20_raw" / "probe_predictions.parquet",
        ),
        RunConfig(
            name="gemma_3_4b_pt_basefmt",
            logits_parquet=mw / "label_logits_gemma_3_4b_pt_eval20_basefmt" / "label_logits_detailed.parquet",
            probe_parquet=mw / "probe_gemma_3_4b_pt_train20_eval20_basefmt" / "probe_predictions.parquet",
        ),
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = []
    for cfg in runs:
        aggregate.append(run_for_config(cfg, out_dir=args.out_dir))
    pd.DataFrame(aggregate).to_csv(args.out_dir / "aggregate_readout_intervention.csv", index=False)
    print(f"Wrote {args.out_dir}")


if __name__ == "__main__":
    main()
