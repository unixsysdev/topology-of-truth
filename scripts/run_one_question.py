from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from _common import EventLogger, ensure_dir, prepare_hf_env, reopen_json, save_env_dump, timed_step, write_json
from topology_paper.src.config import load_config
from topology_paper.src.data.load_dataset import load_configured_dataset, sample_record
from topology_paper.src.eval.correctness import score_generation
from topology_paper.src.models.generate import generate_response
from topology_paper.src.models.hidden_state_extract import extract_hidden_states
from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one GSM8K question end to end.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--config", default="topology_paper/configs/gsm8k_qwen35.yaml")
    parser.add_argument("--artifact-dir", default="artifacts/one_question")
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    prepare_hf_env()
    config = load_config(args.config)
    artifact_dir = ensure_dir(args.artifact_dir)
    logger = EventLogger(artifact_dir / "run.log")
    save_env_dump(artifact_dir / "env.txt", model_id=args.model_id)
    sample_name = f"sample_{args.sample_index + 1:04d}"

    with timed_step(logger, "load_dataset", args.timeout_sec):
        dataset = load_configured_dataset(config["dataset"])
        record = sample_record(dataset[args.sample_index], args.sample_index, config["generation"]["prompt_style"])

    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())

    with timed_step(logger, "generation", args.timeout_sec):
        gen_start = time.perf_counter()
        generation = generate_response(
            model,
            tokenizer,
            record["prompt"],
            config["generation"],
            save_logits=bool(config["extraction"].get("save_logits", True)),
            save_full_logits=bool(config["extraction"].get("save_full_logits", False)),
        )
        generation_runtime = time.perf_counter() - gen_start

    with timed_step(logger, "hidden_extraction", args.timeout_sec):
        hidden = extract_hidden_states(
            model,
            generation.full_token_ids,
            generation.input_length,
            config["extraction"].get("layers", ["last"]),
        )
        last_layer = hidden["last"]

    hidden_path = artifact_dir / f"{sample_name}_hidden.npy"
    np.save(hidden_path, last_layer)
    reopened_hidden = np.load(hidden_path)
    if reopened_hidden.shape != last_layer.shape:
        raise RuntimeError(f"Hidden-state integrity check failed: {reopened_hidden.shape} != {last_layer.shape}")

    score = score_generation(generation.generated_text, record["gold_answer"])
    sample_payload = {
        "sample_id": sample_name,
        "dataset_sample_id": record["sample_id"],
        "question": record["question"],
        "prompt_text": record["prompt"],
        "gold_answer": record["gold_answer"],
        "generated_text": generation.generated_text,
        "pred_answer": score["pred_answer"],
        "correct": score["correct"],
        "token_count": len(generation.generated_token_ids),
        "token_ids": generation.generated_token_ids,
        "hidden_shape": list(last_layer.shape),
        "model_id": args.model_id,
        "runtime_sec": round(generation_runtime, 3),
        "hit_max_new_tokens": generation.hit_max_new_tokens,
        "terminated_by_eos": generation.terminated_by_eos,
        "stop_reason": generation.stop_reason,
        "artifact_hidden_path": str(hidden_path),
        "events": logger.events,
    }
    sample_json_path = artifact_dir / f"{sample_name}.json"
    write_json(sample_json_path, sample_payload)
    reopened_json = reopen_json(sample_json_path)
    if reopened_json["hidden_shape"] != list(last_layer.shape):
        raise RuntimeError("JSON integrity check failed after write")

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
