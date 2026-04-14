from __future__ import annotations

import argparse
import csv
import traceback
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from _common import EventLogger, ensure_dir, prepare_hf_env, save_env_dump, timed_step, write_json
from topology_paper.src.config import load_config
from topology_paper.src.data.load_dataset import load_configured_dataset, sample_record
from topology_paper.src.eval.correctness import score_generation
from topology_paper.src.features.topology import compute_tda
from topology_paper.src.features.lexical_baselines import save_logits_npz
from topology_paper.src.models.generate import generate_response
from topology_paper.src.models.hidden_state_extract import extract_hidden_states
from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five GSM8K questions and save per-sample status.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    parser.add_argument("--artifact-dir", default="artifacts/five_question")
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    prepare_hf_env()
    config = load_config(args.config)
    artifact_dir = ensure_dir(args.artifact_dir)
    traces_dir = ensure_dir(Path(artifact_dir) / "traces")
    logger = EventLogger(Path(artifact_dir) / "run.log")
    save_env_dump(Path(artifact_dir) / "env.txt", model_id=args.model_id)

    with timed_step(logger, "load_dataset", args.timeout_sec):
        dataset = load_configured_dataset(config["dataset"])
    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())

    rows: list[dict[str, object]] = []
    for offset in range(args.count):
        idx = args.start_index + offset
        sample_name = f"sample_{idx + 1:04d}"
        sample_dir = ensure_dir(traces_dir / sample_name)
        try:
            record = sample_record(dataset[idx], idx, config["generation"]["prompt_style"])
            with timed_step(logger, f"generate_{sample_name}", args.timeout_sec):
                generation_started = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                generation_finished = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                if generation_started is not None:
                    generation_started.record()
                generation = generate_response(
                    model,
                    tokenizer,
                    record["prompt"],
                    config["generation"],
                    save_logits=True,
                    save_full_logits=False,
                )
                if generation_finished is not None:
                    generation_finished.record()
                    torch.cuda.synchronize()
                    runtime_sec = generation_started.elapsed_time(generation_finished) / 1000.0
                else:
                    runtime_sec = float("nan")
            with timed_step(logger, f"hidden_{sample_name}", args.timeout_sec):
                hidden = extract_hidden_states(
                    model,
                    generation.full_token_ids,
                    generation.input_length,
                    config["extraction"].get("layers", ["last"]),
                )["last"]
            hidden_path = sample_dir / "hidden.npy"
            np.save(hidden_path, hidden)
            np.load(hidden_path)
            logits_path = save_logits_npz(sample_dir / "logits_summary.npz", generation.logits_summary, generation.full_logits)
            score = score_generation(generation.generated_text, record["gold_answer"])
            topology_config = dict(config["topology"])
            topology_config["max_homology_dim"] = 0
            static_h0 = compute_tda(hidden, topology_config)
            if not np.isfinite(static_h0["h0_entropy"]) or not np.isfinite(static_h0["h0_total_persistence"]):
                raise RuntimeError(f"Non-finite topology features for {sample_name}")
            answer_length = len(generation.generated_text.split())
            write_json(
                sample_dir / "sample.json",
                {
                    "sample_id": sample_name,
                    "dataset_sample_id": record["sample_id"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "generated_text": generation.generated_text,
                    "pred_answer": score["pred_answer"],
                    "correct": score["correct"],
                    "token_count": len(generation.generated_token_ids),
                    "answer_length": answer_length,
                    "hidden_shape": list(hidden.shape),
                    "stop_reason": generation.stop_reason,
                    "runtime_sec": runtime_sec,
                    "h0_entropy": static_h0["h0_entropy"],
                    "h0_total_persistence": static_h0["h0_total_persistence"],
                    "logits_path": logits_path,
                },
            )
            rows.append(
                {
                    "sample_id": sample_name,
                    "sample_index": idx,
                    "status": "ok",
                    "pred_answer": score["pred_answer"],
                    "gold_answer": record["gold_answer"],
                    "correct": score["correct"],
                    "token_count": len(generation.generated_token_ids),
                    "answer_length": answer_length,
                    "runtime_sec": round(runtime_sec, 3),
                    "stop_reason": generation.stop_reason,
                    "h0_entropy": static_h0["h0_entropy"],
                    "h0_total_persistence": static_h0["h0_total_persistence"],
                    "logits_path": logits_path or "",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "sample_id": sample_name,
                    "sample_index": idx,
                    "status": "error",
                    "pred_answer": "",
                    "gold_answer": "",
                    "correct": "",
                    "token_count": "",
                    "answer_length": "",
                    "runtime_sec": "",
                    "stop_reason": "",
                    "h0_entropy": "",
                    "h0_total_persistence": "",
                    "logits_path": "",
                    "error": repr(exc),
                }
            )
            logger.log(f"sample_{sample_name}", "error", error=repr(exc), traceback=traceback.format_exc())

    with (Path(artifact_dir) / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "sample_index",
                "status",
                "pred_answer",
                "gold_answer",
                "correct",
                "token_count",
                "answer_length",
                "runtime_sec",
                "stop_reason",
                "h0_entropy",
                "h0_total_persistence",
                "logits_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    pd.DataFrame(rows).to_parquet(Path(artifact_dir) / "features.parquet", index=False)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
