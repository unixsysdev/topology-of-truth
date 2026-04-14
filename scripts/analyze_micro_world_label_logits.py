from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


LABELS = ["True", "False", "Unknown"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze verdict-step label log-probs for True/False/Unknown."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    manifest = manifest[manifest["status"] == "ok"].copy()
    if args.limit > 0:
        manifest = manifest.head(int(args.limit)).copy()
    if manifest.empty:
        raise RuntimeError(f"No successful rows in {args.manifest}")

    device = resolve_device()
    model, tokenizer = load_model_and_tokenizer(args.model_id, device=device)
    model.eval()

    label_token_candidates = build_label_first_token_candidates(tokenizer)

    rows: list[dict[str, Any]] = []
    for _, rec in manifest.iterrows():
        sample_path = Path(str(rec["sample_path"]))
        if not sample_path.exists():
            continue
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        prompt = str(sample.get("prompt", ""))
        if not prompt:
            continue

        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
        with torch.no_grad():
            out = model(prompt_ids, use_cache=False)
            first_log_probs = torch.log_softmax(out.logits[:, -1, :], dim=-1)[0]

        label_logps: dict[str, float] = {}
        label_best_variant: dict[str, str] = {}
        for label in LABELS:
            best_lp = None
            best_variant = ""
            for variant_text, first_token_id in label_token_candidates[label]:
                lp = float(first_log_probs[int(first_token_id)].item())
                if best_lp is None or lp > best_lp:
                    best_lp = lp
                    best_variant = variant_text
            label_logps[label] = float(best_lp) if best_lp is not None else float("-inf")
            label_best_variant[label] = best_variant

        label_probs = softmax_dict(label_logps)
        argmax_label = max(label_logps, key=label_logps.get)
        non_unknown_best = max(label_logps["True"], label_logps["False"])
        sorted_labels = sorted(LABELS, key=lambda x: label_logps[x], reverse=True)
        unknown_rank = sorted_labels.index("Unknown") + 1

        decoder_pred = normalize_pred_label(rec.get("pred_label"))
        gold = str(rec["label"])
        rows.append(
            {
                "example_id": str(rec["example_id"]),
                "world_id": str(rec["world_id"]),
                "gold_label": gold,
                "decoder_pred_label": decoder_pred,
                "decoder_correct": decoder_pred == gold,
                "argmax_logit_label": argmax_label,
                "argmax_correct": argmax_label == gold,
                "logp_true": label_logps["True"],
                "logp_false": label_logps["False"],
                "logp_unknown": label_logps["Unknown"],
                "prob_true": label_probs["True"],
                "prob_false": label_probs["False"],
                "prob_unknown": label_probs["Unknown"],
                "unknown_minus_best_nonunknown_logp": float(label_logps["Unknown"] - non_unknown_best),
                "unknown_rank": int(unknown_rank),
                "unknown_top2": bool(unknown_rank <= 2),
                "variant_true": label_best_variant["True"],
                "variant_false": label_best_variant["False"],
                "variant_unknown": label_best_variant["Unknown"],
                "prompt_variant": sample.get("prompt_variant", ""),
                "sample_path": str(sample_path),
            }
        )

    detailed = pd.DataFrame(rows)
    if detailed.empty:
        raise RuntimeError("No rows produced during label-logit analysis.")

    detailed.to_parquet(out_dir / "label_logits_detailed.parquet", index=False)
    detailed.to_csv(out_dir / "label_logits_detailed.csv", index=False)

    summary = build_summary(detailed)
    summary.to_csv(out_dir / "label_logits_summary.csv", index=False)

    unknown_fail = detailed[
        (detailed["gold_label"] == "Unknown") & (detailed["decoder_pred_label"] != "Unknown")
    ].copy()
    if not unknown_fail.empty:
        unknown_fail.to_csv(out_dir / "unknown_gold_decoder_nonunknown.csv", index=False)

    pair = detailed.groupby("gold_label", as_index=False).agg(
        n=("example_id", "count"),
        mean_prob_true=("prob_true", "mean"),
        mean_prob_false=("prob_false", "mean"),
        mean_prob_unknown=("prob_unknown", "mean"),
        argmax_unknown_rate=("argmax_logit_label", lambda s: float((s == "Unknown").mean())),
        decoder_unknown_rate=("decoder_pred_label", lambda s: float((s == "Unknown").mean())),
    )
    pair.to_csv(out_dir / "label_logits_by_gold.csv", index=False)


def build_label_first_token_candidates(tokenizer) -> dict[str, list[tuple[str, int]]]:
    variants: dict[str, list[tuple[str, int]]] = {}
    for label in LABELS:
        texts = [f" {label}", label]
        seen_first_ids: set[int] = set()
        kept: list[tuple[str, int]] = []
        for txt in texts:
            ids = tokenizer.encode(txt, add_special_tokens=False)
            if not ids:
                continue
            first_id = int(ids[0])
            if first_id in seen_first_ids:
                continue
            seen_first_ids.add(first_id)
            kept.append((txt, first_id))
        if not kept:
            raise RuntimeError(f"No token variants for label {label}")
        variants[label] = kept
    return variants


def softmax_dict(values: dict[str, float]) -> dict[str, float]:
    arr = np.asarray([values[k] for k in LABELS], dtype=np.float64)
    arr = np.exp(arr - np.max(arr))
    arr = arr / np.sum(arr)
    return {k: float(arr[i]) for i, k in enumerate(LABELS)}


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


def build_summary(detailed: pd.DataFrame) -> pd.DataFrame:
    n = len(detailed)
    unknown_gold = detailed[detailed["gold_label"] == "Unknown"]
    unknown_fail = unknown_gold[unknown_gold["decoder_pred_label"] != "Unknown"]

    row = {
        "n_examples": int(n),
        "decoder_accuracy": float(detailed["decoder_correct"].mean()),
        "argmax_logit_accuracy": float(detailed["argmax_correct"].mean()),
        "decoder_unknown_recall": float(
            (unknown_gold["decoder_pred_label"] == "Unknown").mean() if len(unknown_gold) else np.nan
        ),
        "argmax_unknown_recall_on_gold_unknown": float(
            (unknown_gold["argmax_logit_label"] == "Unknown").mean() if len(unknown_gold) else np.nan
        ),
        "unknown_gold_count": int(len(unknown_gold)),
        "unknown_gold_decoder_nonunknown_count": int(len(unknown_fail)),
        "unknown_fail_mean_prob_unknown": float(unknown_fail["prob_unknown"].mean()) if len(unknown_fail) else np.nan,
        "unknown_fail_mean_unknown_minus_best_nonunknown_logp": float(
            unknown_fail["unknown_minus_best_nonunknown_logp"].mean()
        )
        if len(unknown_fail)
        else np.nan,
        "unknown_fail_unknown_argmax_rate": float(
            (unknown_fail["argmax_logit_label"] == "Unknown").mean()
        )
        if len(unknown_fail)
        else np.nan,
        "unknown_fail_unknown_top2_rate": float(unknown_fail["unknown_top2"].mean()) if len(unknown_fail) else np.nan,
        "unknown_fail_unknown_prob_ge_0_20_rate": float((unknown_fail["prob_unknown"] >= 0.20).mean())
        if len(unknown_fail)
        else np.nan,
        "unknown_fail_unknown_prob_ge_0_33_rate": float((unknown_fail["prob_unknown"] >= 0.33).mean())
        if len(unknown_fail)
        else np.nan,
    }
    return pd.DataFrame([row])


if __name__ == "__main__":
    main()
