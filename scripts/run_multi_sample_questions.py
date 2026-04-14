from __future__ import annotations

import argparse
import csv
import json
import random
import traceback
from copy import deepcopy
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple stochastic samples per GSM8K question.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-count", type=int, default=20)
    parser.add_argument("--question-indices", default="")
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--samples-per-question", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    parser.add_argument("--artifact-dir", default="artifacts/qwen35_2b_within_question")
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--resume-skip-existing", action="store_true")
    parser.add_argument("--rebuild-all-manifest", action="store_true")
    args = parser.parse_args()

    prepare_hf_env()
    config = load_config(args.config)
    artifact_dir = ensure_dir(args.artifact_dir)
    logger = EventLogger(Path(artifact_dir) / "run.log")
    save_env_dump(Path(artifact_dir) / "env.txt", model_id=args.model_id)

    with timed_step(logger, "load_dataset", args.timeout_sec):
        dataset = load_configured_dataset(config["dataset"])
    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())

    run_rows: list[dict[str, object]] = []
    question_rows: list[dict[str, object]] = []

    for dataset_idx in parse_question_indices(args.question_indices, args.question_start, args.question_count):
        question_id = f"q_{dataset_idx + 1:04d}"
        question_dir = ensure_dir(Path(artifact_dir) / question_id)
        record = sample_record(dataset[dataset_idx], dataset_idx, config["generation"]["prompt_style"])
        per_question_rows: list[dict[str, object]] = []

        for sample_idx in range(args.sample_start, args.sample_start + args.samples_per_question):
            sample_name = f"sample_{sample_idx:02d}"
            sample_dir = ensure_dir(question_dir / sample_name)
            seed = int(args.base_seed + dataset_idx * args.samples_per_question + sample_idx)

            if args.resume_skip_existing:
                existing_row = load_existing_sample_row(sample_dir, question_id, dataset_idx, record, sample_name, seed)
                if existing_row is not None:
                    logger.log(f"{question_id}_{sample_name}", "skip_existing")
                    per_question_rows.append(existing_row)
                    run_rows.append(existing_row)
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
                answer_token_indices = find_answer_token_indices(
                    tokenizer,
                    generation.generated_token_ids,
                    score["pred_answer"],
                )
                sample_payload = {
                    "question_id": question_id,
                    "dataset_sample_id": record["sample_id"],
                    "sample_id": sample_name,
                    "seed": seed,
                    "temperature": float(args.temperature),
                    "top_p": float(args.top_p),
                    "max_new_tokens": int(args.max_new_tokens),
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
                write_json(sample_dir / "sample.json", sample_payload)
                row = {
                    "question_id": question_id,
                    "dataset_index": dataset_idx,
                    "dataset_sample_id": record["sample_id"],
                    "sample_id": sample_name,
                    "seed": seed,
                    "status": "ok",
                    "correct": bool(score["correct"]),
                    "stop_reason": generation.stop_reason,
                    "token_count": len(generation.generated_token_ids),
                    "answer_length": answer_length,
                    "pred_answer": score["pred_answer"],
                    "gold_answer": record["gold_answer"],
                    "extractor_confident": bool(score["pred_answer"]),
                    "runtime_sec": round(runtime_sec, 3),
                    "hidden_path": str(hidden_path),
                    "logits_path": str(logits_path) if logits_path else "",
                    "error": "",
                }
            except Exception as exc:
                row = {
                    "question_id": question_id,
                    "dataset_index": dataset_idx,
                    "dataset_sample_id": record["sample_id"],
                    "sample_id": sample_name,
                    "seed": seed,
                    "status": "error",
                    "correct": "",
                    "stop_reason": "",
                    "token_count": "",
                    "answer_length": "",
                    "pred_answer": "",
                    "gold_answer": record["gold_answer"],
                    "extractor_confident": "",
                    "runtime_sec": "",
                    "hidden_path": "",
                    "logits_path": "",
                    "error": repr(exc),
                }
                logger.log(
                    f"{question_id}_{sample_name}",
                    "error",
                    error=repr(exc),
                    traceback=traceback.format_exc(),
                )

            per_question_rows.append(row)
            run_rows.append(row)

        question_rows.append(build_question_summary(question_id, dataset_idx, record, per_question_rows))

    if args.rebuild_all_manifest:
        run_rows, question_rows = load_all_existing_rows(Path(artifact_dir))

    write_csv(
        Path(artifact_dir) / "run_manifest.csv",
        run_rows,
        [
            "question_id",
            "dataset_index",
            "dataset_sample_id",
            "sample_id",
            "seed",
            "status",
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
        ],
    )
    write_csv(
        Path(artifact_dir) / "question_summary.csv",
        question_rows,
        [
            "question_id",
            "dataset_index",
            "dataset_sample_id",
            "gold_answer",
            "n_samples",
            "n_ok",
            "n_correct",
            "n_wrong",
            "n_capped",
            "n_eos",
            "n_extractor_confident",
            "mixed_labels",
        ],
    )

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


def parse_question_indices(question_indices: str, question_start: int, question_count: int) -> list[int]:
    if not question_indices.strip():
        return list(range(question_start, question_start + question_count))
    out = []
    for part in question_indices.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


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


def load_existing_sample_row(
    sample_dir: Path,
    question_id: str,
    dataset_idx: int,
    record: dict,
    sample_name: str,
    seed: int,
) -> dict[str, object] | None:
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

    return {
        "question_id": question_id,
        "dataset_index": dataset_idx,
        "dataset_sample_id": payload.get("dataset_sample_id", record["sample_id"]),
        "sample_id": sample_name,
        "seed": seed,
        "status": "ok",
        "correct": bool(payload.get("correct")),
        "stop_reason": payload.get("stop_reason", ""),
        "token_count": payload.get("token_count", ""),
        "answer_length": payload.get("answer_length", ""),
        "pred_answer": payload.get("pred_answer", ""),
        "gold_answer": payload.get("gold_answer", record["gold_answer"]),
        "extractor_confident": bool(payload.get("extractor_confident")),
        "runtime_sec": round(float(payload.get("runtime_sec", 0.0)), 3),
        "hidden_path": str(hidden_path),
        "logits_path": str(logits_path),
        "error": "",
    }


def load_all_existing_rows(artifact_dir: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    run_rows: list[dict[str, object]] = []
    for sample_path in sorted(artifact_dir.glob("q_*/sample_*/sample.json")):
        if sample_path.stat().st_size == 0:
            continue
        with sample_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        question_id = payload.get("question_id", sample_path.parents[1].name)
        dataset_index = int(str(question_id).removeprefix("q_")) - 1
        sample_dir = sample_path.parent
        hidden_path = sample_dir / "hidden.npy"
        logits_path = sample_dir / "logits_summary.npz"
        if not hidden_path.exists() or hidden_path.stat().st_size == 0:
            continue
        if not logits_path.exists() or logits_path.stat().st_size == 0:
            continue
        run_rows.append(
            {
                "question_id": question_id,
                "dataset_index": dataset_index,
                "dataset_sample_id": payload.get("dataset_sample_id", ""),
                "sample_id": payload.get("sample_id", sample_dir.name),
                "seed": payload.get("seed", ""),
                "status": "ok",
                "correct": bool(payload.get("correct")),
                "stop_reason": payload.get("stop_reason", ""),
                "token_count": payload.get("token_count", ""),
                "answer_length": payload.get("answer_length", ""),
                "pred_answer": payload.get("pred_answer", ""),
                "gold_answer": payload.get("gold_answer", ""),
                "extractor_confident": bool(payload.get("extractor_confident")),
                "runtime_sec": round(float(payload.get("runtime_sec", 0.0)), 3),
                "hidden_path": str(hidden_path),
                "logits_path": str(logits_path),
                "error": "",
            }
        )

    question_rows = []
    for question_id, group in pd.DataFrame(run_rows).groupby("question_id", sort=True):
        rows = group.to_dict("records")
        first = rows[0]
        ok_rows = [row for row in rows if row["status"] == "ok"]
        correct_rows = [row for row in ok_rows if row["correct"] is True]
        wrong_rows = [row for row in ok_rows if row["correct"] is False]
        question_rows.append(
            {
                "question_id": question_id,
                "dataset_index": int(first["dataset_index"]),
                "dataset_sample_id": first.get("dataset_sample_id", ""),
                "gold_answer": first.get("gold_answer", ""),
                "n_samples": len(rows),
                "n_ok": len(ok_rows),
                "n_correct": len(correct_rows),
                "n_wrong": len(wrong_rows),
                "n_capped": sum(row.get("stop_reason") == "max_new_tokens" for row in ok_rows),
                "n_eos": sum(row.get("stop_reason") == "eos_token" for row in ok_rows),
                "n_extractor_confident": sum(bool(row.get("extractor_confident")) for row in ok_rows),
                "mixed_labels": bool(correct_rows and wrong_rows),
            }
        )
    return run_rows, question_rows


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


def build_question_summary(question_id: str, dataset_idx: int, record: dict, rows: list[dict[str, object]]) -> dict[str, object]:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    correct_rows = [row for row in ok_rows if row["correct"] is True]
    wrong_rows = [row for row in ok_rows if row["correct"] is False]
    return {
        "question_id": question_id,
        "dataset_index": dataset_idx,
        "dataset_sample_id": record["sample_id"],
        "gold_answer": record["gold_answer"],
        "n_samples": len(rows),
        "n_ok": len(ok_rows),
        "n_correct": len(correct_rows),
        "n_wrong": len(wrong_rows),
        "n_capped": sum(row.get("stop_reason") == "max_new_tokens" for row in ok_rows),
        "n_eos": sum(row.get("stop_reason") == "eos_token" for row in ok_rows),
        "n_extractor_confident": sum(bool(row.get("extractor_confident")) for row in ok_rows),
        "mixed_labels": bool(correct_rows and wrong_rows),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(rows).to_parquet(path.with_suffix(".parquet"), index=False)


if __name__ == "__main__":
    main()
