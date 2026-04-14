#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _unknown_recall(df: pd.DataFrame) -> float:
    return float((df["pred_label"] == "Unknown").mean()) if len(df) else float("nan")


def _bootstrap_world_ci(
    dec_unknown: pd.DataFrame,
    probe_unknown: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    worlds = sorted(set(dec_unknown["world_id"]).intersection(set(probe_unknown["world_id"])))
    if not worlds:
        raise ValueError("No overlapping worlds between decoder and probe inputs.")

    dec_unknown = dec_unknown[dec_unknown["world_id"].isin(worlds)].copy()
    probe_unknown = probe_unknown[probe_unknown["world_id"].isin(worlds)].copy()

    dec_world = {w: g for w, g in dec_unknown.groupby("world_id")}
    probe_world = {w: g for w, g in probe_unknown.groupby("world_id")}

    rng = np.random.default_rng(seed)
    world_arr = np.array(worlds)
    n_worlds = len(worlds)
    dec_vals = np.empty(n_bootstrap, dtype=float)
    probe_vals = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        sample = world_arr[rng.integers(0, n_worlds, size=n_worlds)]
        dec_sample = pd.concat([dec_world[w] for w in sample], ignore_index=True)
        probe_sample = pd.concat([probe_world[w] for w in sample], ignore_index=True)
        dec_vals[i] = _unknown_recall(dec_sample)
        probe_vals[i] = _unknown_recall(probe_sample)

    return dec_vals, probe_vals


def _compute_row(
    model_name: str,
    decoder_manifest: Path,
    probe_predictions: Path,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    dec = pd.read_csv(decoder_manifest)
    dec = dec[(dec["status"] == "ok") & dec["label"].isin(["True", "False", "Unknown"])].copy()
    dec["pred_label"] = dec["pred_label"].astype(str)
    dec_u = dec[dec["label"] == "Unknown"].copy()

    probe = pd.read_parquet(probe_predictions)
    probe = probe[probe["state_key"] == "verdict_token"].copy()
    probe_u = probe[probe["label"] == "Unknown"].copy()

    dec_bs, probe_bs = _bootstrap_world_ci(dec_u, probe_u, n_bootstrap=n_bootstrap, seed=seed)

    point_dec = _unknown_recall(dec_u)
    point_probe = _unknown_recall(probe_u)
    gap_bs = probe_bs - dec_bs

    worlds = sorted(set(dec_u["world_id"]).intersection(set(probe_u["world_id"])))
    return {
        "model": model_name,
        "n_eval_worlds": len(worlds),
        "decoder_unknown_recall": point_dec,
        "decoder_unknown_recall_ci_low": float(np.quantile(dec_bs, 0.025)),
        "decoder_unknown_recall_ci_high": float(np.quantile(dec_bs, 0.975)),
        "probe_unknown_recall_verdict_token": point_probe,
        "probe_unknown_recall_verdict_token_ci_low": float(np.quantile(probe_bs, 0.025)),
        "probe_unknown_recall_verdict_token_ci_high": float(np.quantile(probe_bs, 0.975)),
        "gap_probe_minus_decoder": point_probe - point_dec,
        "gap_probe_minus_decoder_ci_low": float(np.quantile(gap_bs, 0.025)),
        "gap_probe_minus_decoder_ci_high": float(np.quantile(gap_bs, 0.975)),
        "bootstrap_resamples": n_bootstrap,
        "bootstrap_unit": "world",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap world-level CIs for Unknown recall comparisons.")
    parser.add_argument(
        "--model",
        action="append",
        nargs=3,
        metavar=("MODEL_NAME", "DECODER_MANIFEST_CSV", "PROBE_PREDICTIONS_PARQUET"),
        required=True,
        help="Model label, decoder manifest path, and probe predictions path.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for model_name, dec_csv, probe_parquet in args.model:
        rows.append(
            _compute_row(
                model_name=model_name,
                decoder_manifest=Path(dec_csv),
                probe_predictions=Path(probe_parquet),
                n_bootstrap=args.n_bootstrap,
                seed=args.seed,
            )
        )

    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
