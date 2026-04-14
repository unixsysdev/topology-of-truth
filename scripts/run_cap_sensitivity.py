from __future__ import annotations

import argparse
import csv
import json
import random
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from _common import EventLogger, ensure_dir, prepare_hf_env, save_env_dump, timed_step, write_json
from topology_paper.src.config import load_config
from topology_paper.src.data.load_dataset import load_configured_dataset, sample_record
from topology_paper.src.eval.correctness import score_generation
from topology_paper.src.features.lexical_baselines import save_logits_npz
from topology_paper.src.models.generate import generate_response
from topology_paper.src.models.hidden_state_extract import extract_hidden_states
from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


MANIFEST_FIELDS = [
    "question_id",
    "dataset_index",
    "dataset_sample_id",
    "sample_id",
    "seed",
    "status",
    "source_stop_reason",
    "source_correct",
    "source_token_count",
    "correct",
    "stop_reason",
    "token_count",
    "answer_length",
    "pred_answer",
    "gold_answer",
    "extractor_confident",
    "runtime_sec",
    "hidden_path",
    "logits_path",
    "error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun capped samples at a higher token budget.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--source-artifact-dir", default="artifacts/qwen35_2b_within_question")
    parser.add_argument("--out-dir", default="artifacts/qwen35_2b_cap_sensitivity_640")
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=1200)
    parser.add_argument("--resume-skip-existing", action="store_true")
    args = parser.parse_args()

    prepare_hf_env()
    source_dir = Path(args.source_artifact_dir)
    out_dir = ensure_dir(args.out_dir)
    logger = EventLogger(out_dir / "run.log")
    save_env_dump(out_dir / "env.txt", model_id=args.model_id)

    source_manifest = pd.read_csv(source_dir / "run_manifest.csv")
    capped = source_manifest[source_manifest["stop_reason"] == "max_new_tokens"].copy()
    capped = capped.sort_values(["dataset_index", "sample_id"])
    if args.limit > 0:
        capped = capped.head(args.limit)

    config = load_config(args.config)
    with timed_step(logger, "load_dataset", args.timeout_sec):
        dataset = load_configured_dataset(config["dataset"])
    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())

    run_rows: list[dict[str, Any]] = []
    for _, source_row in capped.iterrows():
        question_id = str(source_row["question_id"])
        sample_name = str(source_row["sample_id"])
        dataset_idx = int(source_row["dataset_index"])
        seed = int(source_row["seed"])
        sample_dir = ensure_dir(out_dir / question_id / sample_name)
        record = sample_record(dataset[dataset_idx], dataset_idx, config["generation"]["prompt_style"])

        if args.resume_skip_existing:
            existing = load_existing_row(sample_dir, source_row)
            if existing is not None:
                logger.log(f"{question_id}_{sample_name}", "skip_existing")
                run_rows.append(existing)
                continue

        try:
            set_all_seeds(seed)
            generation_config = build_generation_config(config["generation"], args.temperature, args.top_p, args.max_new_tokens)
            with timed_step(logger, f"generate_{question_id}_{sample_name}", args.timeout_sec):
                start_evt = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_evt = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if start_evt is not None:
                    start_evt.record()
                generation = generate_response(
                    model,
                    tokenizer,
                    record["prompt"],
                    generation_config,
                    save_logits=True,
                    save_full_logits=False,
                )
                if end_evt is not None:
                    end_evt.record()
                    torch.cuda.synchronize()
                    runtime_sec = start_evt.elapsed_time(end_evt) / 1000.0
                else:
                    runtime_sec = float("nan")

            with timed_step(logger, f"hidden_{question_id}_{sample_name}", args.timeout_sec):
                hidden = extract_hidden_states(
                    model,
                    generation.full_token_ids,
                    generation.input_length,
                    config["extraction"].get("layers", ["last"]),
                )["last"]

            hidden_path = sample_dir / "hidden.npy"
            save_npy_atomic(hidden_path, hidden)
            np.load(hidden_path)
            logits_path = save_logits_npz(sample_dir / "logits_summary.npz", generation.logits_summary, generation.full_logits)

            score = score_generation(generation.generated_text, record["gold_answer"])
            answer_length = len(generation.generated_text.split())
            answer_token_indices = find_answer_token_indices(tokenizer, generation.generated_token_ids, score["pred_answer"])
            payload = {
                "question_id": question_id,
                "dataset_sample_id": record["sample_id"],
                "sample_id": sample_name,
                "seed": seed,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_new_tokens": int(args.max_new_tokens),
                "source_artifact_dir": str(source_dir),
                "source_stop_reason": source_row["stop_reason"],
                "source_correct": bool(source_row["correct"]),
                "source_token_count": int(source_row["token_count"]),
                "question": record["question"],
                "prompt": record["prompt"],
                "gold_answer": record["gold_answer"],
                "generated_text": generation.generated_text,
                "pred_answer": score["pred_answer"],
                "correct": bool(score["correct"]),
                "stop_reason": generation.stop_reason,
                "token_count": len(generation.generated_token_ids),
                "answer_length": answer_length,
                "extractor_confident": bool(score["pred_answer"]),
                "runtime_sec": runtime_sec,
                "hidden_shape": list(hidden.shape),
                "generated_token_ids": generation.generated_token_ids,
                "answer_token_indices": answer_token_indices,
                "hidden_path": str(hidden_path),
                "logits_path": str(logits_path) if logits_path else "",
            }
            write_json(sample_dir / "sample.json", payload)
            row = row_from_payload(payload, source_row, hidden_path, logits_path, runtime_sec, "")
        except Exception as exc:
            logger.log(f"{question_id}_{sample_name}", "error", error=repr(exc), traceback=traceback.format_exc())
            row = error_row(source_row, repr(exc))
        run_rows.append(row)
        write_manifest(out_dir / "run_manifest.csv", run_rows)

    write_manifest(out_dir / "run_manifest.csv", run_rows)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_generation_config(base: dict, temperature: float, top_p: float, max_new_tokens: int) -> dict:
    cfg = deepcopy(base)
    cfg["deterministic"] = False
    cfg["temperature"] = float(temperature)
    cfg["top_p"] = float(top_p)
    cfg["max_new_tokens"] = int(max_new_tokens)
    return cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.save(f, array)
    tmp_path.replace(path)


def load_existing_row(sample_dir: Path, source_row: pd.Series) -> dict[str, Any] | None:
    sample_path = sample_dir / "sample.json"
    hidden_path = sample_dir / "hidden.npy"
    logits_path = sample_dir / "logits_summary.npz"
    if not sample_path.exists() or not hidden_path.exists() or not logits_path.exists():
        return None
    if sample_path.stat().st_size == 0 or hidden_path.stat().st_size == 0 or logits_path.stat().st_size == 0:
        return None
    try:
        with sample_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        np.load(hidden_path, mmap_mode="r")
        with np.load(logits_path) as logits:
            if not logits.files:
                return None
    except Exception:
        return None
    return row_from_payload(payload, source_row, hidden_path, logits_path, float(payload.get("runtime_sec", 0.0)), "")


def row_from_payload(
    payload: dict[str, Any],
    source_row: pd.Series,
    hidden_path: Path,
    logits_path: Path | str | None,
    runtime_sec: float,
    error: str,
) -> dict[str, Any]:
    return {
        "question_id": payload["question_id"],
        "dataset_index": int(source_row["dataset_index"]),
        "dataset_sample_id": payload.get("dataset_sample_id", source_row.get("dataset_sample_id", "")),
        "sample_id": payload["sample_id"],
        "seed": int(payload["seed"]),
        "status": "ok",
        "source_stop_reason": source_row["stop_reason"],
        "source_correct": bool(source_row["correct"]),
        "source_token_count": int(source_row["token_count"]),
        "correct": bool(payload["correct"]),
        "stop_reason": payload["stop_reason"],
        "token_count": int(payload["token_count"]),
        "answer_length": int(payload["answer_length"]),
        "pred_answer": payload.get("pred_answer", ""),
        "gold_answer": payload.get("gold_answer", ""),
        "extractor_confident": bool(payload.get("extractor_confident")),
        "runtime_sec": round(float(runtime_sec), 3),
        "hidden_path": str(hidden_path),
        "logits_path": str(logits_path) if logits_path else "",
        "error": error,
    }


def error_row(source_row: pd.Series, error: str) -> dict[str, Any]:
    return {
        "question_id": source_row["question_id"],
        "dataset_index": int(source_row["dataset_index"]),
        "dataset_sample_id": source_row["dataset_sample_id"],
        "sample_id": source_row["sample_id"],
        "seed": int(source_row["seed"]),
        "status": "error",
        "source_stop_reason": source_row["stop_reason"],
        "source_correct": bool(source_row["correct"]),
        "source_token_count": int(source_row["token_count"]),
        "correct": "",
        "stop_reason": "",
        "token_count": "",
        "answer_length": "",
        "pred_answer": "",
        "gold_answer": source_row["gold_answer"],
        "extractor_confident": "",
        "runtime_sec": "",
        "hidden_path": "",
        "logits_path": "",
        "error": error,
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"), index=False)


def find_answer_token_indices(tokenizer, generated_token_ids: list[int], answer_text: str | None) -> list[int]:
    if not answer_text:
        return []
    answer_ids = tokenizer.encode(str(answer_text), add_special_tokens=False)
    if not answer_ids:
        return []
    start = find_last_subsequence(generated_token_ids, answer_ids)
    if start < 0:
        return []
    return list(range(start, start + len(answer_ids)))


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


if __name__ == "__main__":
    main()
