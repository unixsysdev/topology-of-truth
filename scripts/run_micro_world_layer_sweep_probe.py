from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


LABELS = ["True", "False", "Unknown"]


@dataclass
class SplitTensors:
    labels: np.ndarray
    prompt_last: np.ndarray
    verdict_token: np.ndarray
    verdict_valid: np.ndarray


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer-sweep probe for micro-world label recoverability."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-iter", type=int, default=4000)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_manifest = load_manifest(Path(args.train_manifest), args.limit_train)
    test_manifest = load_manifest(Path(args.test_manifest), args.limit_test)
    if train_manifest.empty or test_manifest.empty:
        raise RuntimeError("Train/test manifests are empty after filtering.")

    model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())
    model.eval()

    train_split = extract_split_tensors(model, tokenizer, train_manifest)
    test_split = extract_split_tensors(model, tokenizer, test_manifest)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(
        run_probe_sweep(
            train_split,
            test_split,
            state_kind="prompt_last",
            max_iter=int(args.max_iter),
            c=float(args.c),
        )
    )
    summary_rows.extend(
        run_probe_sweep(
            train_split,
            test_split,
            state_kind="verdict_token",
            max_iter=int(args.max_iter),
            c=float(args.c),
        )
    )
    if not summary_rows:
        raise RuntimeError("No layer probe results were produced.")

    summary = pd.DataFrame(summary_rows).sort_values(["state_kind", "layer_idx"]).reset_index(drop=True)
    summary.to_csv(out_dir / "layer_sweep_summary.csv", index=False)

    best = (
        summary.sort_values(
            ["state_kind", "recall_unknown", "macro_f1", "accuracy"],
            ascending=[True, False, False, False],
        )
        .groupby("state_kind", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    best.to_csv(out_dir / "layer_sweep_best.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "model_id": args.model_id,
                "n_train": int(len(train_split.labels)),
                "n_test": int(len(test_split.labels)),
                "n_layers": int(train_split.prompt_last.shape[1]),
                "hidden_dim": int(train_split.prompt_last.shape[2]),
                "train_verdict_valid": int(train_split.verdict_valid.sum()),
                "test_verdict_valid": int(test_split.verdict_valid.sum()),
            }
        ]
    )
    metadata.to_csv(out_dir / "layer_sweep_metadata.csv", index=False)


def load_manifest(path: Path, limit: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df = df[df["label"].isin(LABELS)].copy()
    if limit > 0:
        df = df.head(int(limit)).copy()
    return df.reset_index(drop=True)


@torch.no_grad()
def extract_split_tensors(model, tokenizer, manifest: pd.DataFrame) -> SplitTensors:
    n = len(manifest)
    labels = manifest["label"].astype(str).to_numpy()

    prompt_last = None
    verdict_token = None
    verdict_valid = np.zeros(n, dtype=bool)

    for row_idx, rec in manifest.iterrows():
        sample_path = Path(str(rec["sample_path"]))
        if not sample_path.exists():
            continue
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        prompt = str(sample.get("prompt", ""))
        if not prompt:
            continue

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        input_ids = prompt_ids
        prompt_len = int(prompt_ids.shape[1])

        generated = sample.get("generated_token_ids", [])
        has_verdict = isinstance(generated, list) and len(generated) > 0
        if has_verdict:
            tok = torch.tensor([[int(generated[0])]], dtype=input_ids.dtype, device=input_ids.device)
            input_ids = torch.cat([input_ids, tok], dim=1)
            verdict_valid[row_idx] = True

        out = model(input_ids, output_hidden_states=True, use_cache=False)
        hidden_states = out.hidden_states
        if hidden_states is None:
            continue

        n_layers = len(hidden_states)
        hidden_dim = int(hidden_states[-1].shape[-1])
        if prompt_last is None:
            prompt_last = np.zeros((n, n_layers, hidden_dim), dtype=np.float16)
            verdict_token = np.zeros((n, n_layers, hidden_dim), dtype=np.float16)

        for layer_idx, hs in enumerate(hidden_states):
            prompt_vec = hs[0, prompt_len - 1].detach().cpu().to(torch.float16).numpy()
            prompt_last[row_idx, layer_idx, :] = prompt_vec
            if has_verdict:
                verdict_vec = hs[0, prompt_len].detach().cpu().to(torch.float16).numpy()
                verdict_token[row_idx, layer_idx, :] = verdict_vec

    if prompt_last is None or verdict_token is None:
        raise RuntimeError("No examples were successfully extracted for split tensors.")

    return SplitTensors(
        labels=labels,
        prompt_last=prompt_last,
        verdict_token=verdict_token,
        verdict_valid=verdict_valid,
    )


def run_probe_sweep(
    train_split: SplitTensors,
    test_split: SplitTensors,
    state_kind: str,
    max_iter: int,
    c: float,
) -> list[dict[str, Any]]:
    if state_kind == "prompt_last":
        x_train_all = train_split.prompt_last
        x_test_all = test_split.prompt_last
        y_train = train_split.labels
        y_test = test_split.labels
    elif state_kind == "verdict_token":
        tr_mask = train_split.verdict_valid
        te_mask = test_split.verdict_valid
        x_train_all = train_split.verdict_token[tr_mask]
        x_test_all = test_split.verdict_token[te_mask]
        y_train = train_split.labels[tr_mask]
        y_test = test_split.labels[te_mask]
    else:
        raise ValueError(f"Unknown state_kind={state_kind}")

    if len(x_train_all) == 0 or len(x_test_all) == 0:
        return []
    if set(np.unique(y_train)) != set(LABELS):
        return []

    n_layers = x_train_all.shape[1]
    rows: list[dict[str, Any]] = []
    for layer_idx in range(n_layers):
        x_train = x_train_all[:, layer_idx, :].astype(np.float32)
        x_test = x_test_all[:, layer_idx, :].astype(np.float32)
        # Some layers can produce non-finite activations on a few samples.
        # Keep the sweep robust by zeroing non-finite entries.
        x_train = np.nan_to_num(x_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        x_test = np.nan_to_num(x_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        probe = Pipeline(
            [
                ("scale", StandardScaler(with_mean=True, with_std=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=int(max_iter),
                        C=float(c),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        probe.fit(x_train, y_train)
        pred = probe.predict(x_test)

        p, r, f, s = precision_recall_fscore_support(
            y_test, pred, labels=LABELS, zero_division=0
        )
        row: dict[str, Any] = {
            "state_kind": state_kind,
            "layer_idx": int(layer_idx),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "accuracy": float(np.mean(pred == y_test)),
            "macro_f1": float(
                f1_score(y_test, pred, labels=LABELS, average="macro", zero_division=0)
            ),
        }
        for label, prec, rec, f1v, sup in zip(LABELS, p, r, f, s):
            low = label.lower()
            row[f"precision_{low}"] = float(prec)
            row[f"recall_{low}"] = float(rec)
            row[f"f1_{low}"] = float(f1v)
            row[f"support_{low}"] = int(sup)
        rows.append(row)
    return rows


if __name__ == "__main__":
    main()
