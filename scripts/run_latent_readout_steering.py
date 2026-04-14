#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import json
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

LABELS = ["True", "False", "Unknown"]


@dataclass(frozen=True)
class SteeringRun:
    run: str
    model_id: str
    train_manifest: Path
    eval_manifest: Path


def resolve_snapshot_path(model_id: str) -> Path:
    snap = snapshot_download(
        repo_id=model_id,
        local_files_only=True,
        allow_patterns=[
            "model.safetensors.index.json",
            "model-*.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "config.json",
        ],
    )
    return Path(snap)


def normalize_pred_label(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if not isinstance(value, str):
        return ""
    value = value.strip()
    if value in LABELS:
        return value
    low = value.lower()
    if low == "true":
        return "True"
    if low == "false":
        return "False"
    if low == "unknown":
        return "Unknown"
    return ""


def build_label_first_token_candidates(tokenizer) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for label in LABELS:
        variants = [f" {label}", label]
        ids: list[int] = []
        seen: set[int] = set()
        for v in variants:
            tok_ids = tokenizer.encode(v, add_special_tokens=False)
            if not tok_ids:
                continue
            first = int(tok_ids[0])
            if first not in seen:
                seen.add(first)
                ids.append(first)
        if not ids:
            raise RuntimeError(f"No token candidates for label={label}")
        out[label] = ids
    return out


def load_candidate_embedding_rows(snapshot: Path, token_ids: list[int]) -> np.ndarray:
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.exists():
        raise RuntimeError(f"Missing index file: {index_path}")
    idx = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = idx["weight_map"]
    key = "language_model.model.embed_tokens.weight"
    shard_name = weight_map.get(key)
    if shard_name is None:
        raise RuntimeError(f"{key} missing from {index_path}")

    shard_path = snapshot / shard_name
    with safe_open(str(shard_path), framework="pt", device="cpu") as f:
        sl = f.get_slice(key)
        rows = []
        for tid in token_ids:
            rows.append(sl[int(tid) : int(tid) + 1, :].to(torch.float32))
        mat = torch.cat(rows, dim=0).cpu().numpy().astype(np.float32)
    return mat


def load_state_matrix(manifest: pd.DataFrame, state_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[str] = []
    worlds: list[str] = []
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
            xs.append(np.asarray(data[state_key], dtype=np.float32))
        ys.append(label)
        worlds.append(str(rec["world_id"]))
    if not xs:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=object), np.empty((0,), dtype=object)
    return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=object), np.asarray(worlds, dtype=object)


def compute_unknown_direction(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    u = train_x[train_y == "Unknown"]
    n = train_x[train_y != "Unknown"]
    if len(u) == 0 or len(n) == 0:
        raise RuntimeError("Train split missing Unknown or non-Unknown examples for direction estimation.")
    vec = u.mean(axis=0) - n.mean(axis=0)
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        raise RuntimeError("Unknown steering vector has zero norm.")
    return (vec / norm).astype(np.float32)


def compute_label_scores(
    hidden: np.ndarray,
    direction: np.ndarray,
    alpha: float,
    token_embed_rows: np.ndarray,
    candidate_ids: list[int],
    label_candidates: dict[str, list[int]],
) -> np.ndarray:
    h = hidden + alpha * direction[None, :]
    id_to_col = {tok: i for i, tok in enumerate(candidate_ids)}
    logits = h @ token_embed_rows.T
    scores = np.zeros((h.shape[0], len(LABELS)), dtype=np.float32)
    for li, label in enumerate(LABELS):
        cols = [id_to_col[t] for t in label_candidates[label]]
        scores[:, li] = np.max(logits[:, cols], axis=1)
    return scores


def scores_to_pred(scores: np.ndarray) -> np.ndarray:
    idx = np.argmax(scores, axis=1)
    return np.asarray([LABELS[i] for i in idx], dtype=object)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "unknown_recall": float(recall_score(y_true, y_pred, labels=["Unknown"], average="macro", zero_division=0)),
        "true_precision": float(precision_score(y_true, y_pred, labels=["True"], average="macro", zero_division=0)),
        "false_precision": float(precision_score(y_true, y_pred, labels=["False"], average="macro", zero_division=0)),
        "pred_unknown_rate": float(np.mean(y_pred == "Unknown")),
    }


def pick_alpha(train_y: np.ndarray, train_scores_by_alpha: dict[float, np.ndarray]) -> float:
    best = None
    best_key = None
    for alpha, scores in train_scores_by_alpha.items():
        pred = scores_to_pred(scores)
        m = metrics(train_y, pred)
        key = (m["macro_f1"], m["unknown_recall"], m["accuracy"])
        if best is None or key > best_key:
            best = alpha
            best_key = key
    assert best is not None
    return float(best)


def run_loow(
    eval_y: np.ndarray,
    eval_worlds: np.ndarray,
    eval_scores_by_alpha: dict[float, np.ndarray],
    alphas: list[float],
) -> tuple[np.ndarray, pd.DataFrame]:
    pred = np.empty_like(eval_y, dtype=object)
    rows = []
    unique_worlds = sorted(set(eval_worlds.tolist()))
    for w in unique_worlds:
        tr = eval_worlds != w
        te = eval_worlds == w
        train_scores = {a: s[tr] for a, s in eval_scores_by_alpha.items()}
        best_alpha = pick_alpha(eval_y[tr], train_scores)
        pred[te] = scores_to_pred(eval_scores_by_alpha[best_alpha][te])
        rows.append({"world_id": w, "alpha": best_alpha})
    return pred, pd.DataFrame(rows)


def run_one(cfg: SteeringRun, out_dir: Path, fixed_alpha: float, alpha_grid: list[float]) -> dict[str, Any]:
    train_manifest = pd.read_csv(cfg.train_manifest)
    eval_manifest = pd.read_csv(cfg.eval_manifest)
    train_manifest = train_manifest[train_manifest["status"] == "ok"].copy()
    eval_manifest = eval_manifest[eval_manifest["status"] == "ok"].copy()
    if train_manifest.empty or eval_manifest.empty:
        raise RuntimeError(f"{cfg.run}: empty train/eval manifest after status filter.")

    train_x, train_y, _ = load_state_matrix(train_manifest, "final_prompt")
    eval_x, eval_y, eval_worlds = load_state_matrix(eval_manifest, "final_prompt")
    if len(train_x) == 0 or len(eval_x) == 0:
        raise RuntimeError(f"{cfg.run}: missing state vectors.")

    # Baseline decoder emitted labels from manifest order aligned to load_state_matrix traversal via state file.
    eval_preds_decoder = []
    for _, rec in eval_manifest.iterrows():
        sp = Path(str(rec["states_path"]))
        if not sp.exists():
            continue
        if str(rec["label"]) not in LABELS:
            continue
        eval_preds_decoder.append(normalize_pred_label(rec.get("pred_label")))
    eval_preds_decoder = np.asarray(eval_preds_decoder, dtype=object)
    if len(eval_preds_decoder) != len(eval_y):
        raise RuntimeError(f"{cfg.run}: decoder/eval alignment mismatch: {len(eval_preds_decoder)} vs {len(eval_y)}")

    direction = compute_unknown_direction(train_x, train_y)
    snapshot = resolve_snapshot_path(cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=True)
    candidates = build_label_first_token_candidates(tokenizer)
    candidate_ids = sorted({tok for ids in candidates.values() for tok in ids})
    token_embed_rows = load_candidate_embedding_rows(snapshot, candidate_ids)

    alphas = sorted(set([float(a) for a in alpha_grid] + [float(fixed_alpha), 0.0]))
    eval_scores_by_alpha = {
        a: compute_label_scores(eval_x, direction, a, token_embed_rows, candidate_ids, candidates) for a in alphas
    }

    baseline_argmax = scores_to_pred(eval_scores_by_alpha[0.0])
    fixed_pred = scores_to_pred(eval_scores_by_alpha[float(fixed_alpha)])
    loow_pred, loow_params = run_loow(eval_y, eval_worlds, eval_scores_by_alpha, alphas)

    run_dir = out_dir / cfg.run
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(run_dir / "unknown_direction.npz", direction=direction)
    loow_params.to_csv(run_dir / "loow_selected_alphas.csv", index=False)

    eval_frame = pd.DataFrame(
        {
            "world_id": eval_worlds,
            "gold_label": eval_y,
            "decoder_pred": eval_preds_decoder,
            "argmax_pred": baseline_argmax,
            "fixed_pred": fixed_pred,
            "loow_pred": loow_pred,
        }
    )
    eval_frame.to_csv(run_dir / "steering_predictions.csv", index=False)

    rows = []
    for variant, pred in [
        ("decoder_emitted", eval_preds_decoder),
        ("argmax_prompt_logit", baseline_argmax),
        (f"latent_steering_fixed_alpha_{fixed_alpha:g}", fixed_pred),
        ("latent_steering_loow", loow_pred),
    ]:
        rows.append({"run": cfg.run, "variant": variant, **metrics(eval_y, pred)})
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "latent_steering_summary.csv", index=False)

    # Unknown margin diagnostics on gold-Unknown subset
    gold_u = eval_y == "Unknown"
    diag_rows = []
    for a, scores in eval_scores_by_alpha.items():
        su = scores[gold_u]
        m = su[:, 2] - np.maximum(su[:, 0], su[:, 1])
        diag_rows.append(
            {
                "run": cfg.run,
                "alpha": float(a),
                "gold_unknown_count": int(gold_u.sum()),
                "mean_unknown_margin_on_gold_unknown": float(np.mean(m)),
                "unknown_argmax_rate_on_gold_unknown": float(np.mean(np.argmax(su, axis=1) == 2)),
            }
        )
    pd.DataFrame(diag_rows).sort_values("alpha").to_csv(run_dir / "unknown_margin_by_alpha.csv", index=False)

    loow_m = metrics(eval_y, loow_pred)
    dec_m = metrics(eval_y, eval_preds_decoder)
    return {
        "run": cfg.run,
        "n_eval": int(len(eval_y)),
        "decoder_unknown_recall": dec_m["unknown_recall"],
        "latent_loow_unknown_recall": loow_m["unknown_recall"],
        "decoder_accuracy": dec_m["accuracy"],
        "latent_loow_accuracy": loow_m["accuracy"],
        "decoder_macro_f1": dec_m["macro_f1"],
        "latent_loow_macro_f1": loow_m["macro_f1"],
        "fixed_alpha": float(fixed_alpha),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Residual-stream latent steering at final prompt state.")
    parser.add_argument(
        "--out-dir",
        default="artifacts/micro_world_v1/latent_readout_steering",
        help="Output directory",
    )
    parser.add_argument("--fixed-alpha", type=float, default=1.0)
    parser.add_argument(
        "--alpha-grid",
        nargs="+",
        type=float,
        default=[-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    gen = root / "artifacts" / "micro_world_v1" / "generations"
    runs = [
        SteeringRun(
            run="gemma_3_4b_it_raw",
            model_id="google/gemma-3-4b-it",
            train_manifest=gen / "Gemma__gemma_3_4b_it_train20_raw" / "manifest.csv",
            eval_manifest=gen / "Gemma__gemma_3_4b_it_eval20_raw" / "manifest.csv",
        ),
        SteeringRun(
            run="gemma_3_4b_pt_basefmt",
            model_id="google/gemma-3-4b-pt",
            train_manifest=gen / "Gemma__gemma_3_4b_pt_train20_basefmt" / "manifest.csv",
            eval_manifest=gen / "Gemma__gemma_3_4b_pt_eval20_basefmt" / "manifest.csv",
        ),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cfg in runs:
        rows.append(run_one(cfg, out_dir=out_dir, fixed_alpha=float(args.fixed_alpha), alpha_grid=list(args.alpha_grid)))
    pd.DataFrame(rows).to_csv(out_dir / "aggregate_latent_steering.csv", index=False)
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
