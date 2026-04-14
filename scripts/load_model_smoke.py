from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from _common import EventLogger, ensure_dir, prepare_hf_env, save_env_dump, timed_step, write_json
from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Load model once, log environment, and exit.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--artifact-dir", default="artifacts/load_proof")
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    prepare_hf_env()
    artifact_dir = ensure_dir(args.artifact_dir)
    logger = EventLogger(artifact_dir / "load.log")
    save_env_dump(artifact_dir / "env.txt", model_id=args.model_id)

    started = time.perf_counter()
    device = resolve_device()
    with timed_step(logger, "load_config", args.timeout_sec):
        config = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)

    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=device)

    snapshot_path = snapshot_download(args.model_id, allow_patterns=["config.json"], local_files_only=True)
    parameter = next(model.parameters())
    payload = {
        "ts": logger.events[-1]["ts"],
        "model_id": args.model_id,
        "device": str(parameter.device),
        "dtype": str(parameter.dtype),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "wall_clock_sec": round(time.perf_counter() - started, 3),
        "hidden_size": getattr(getattr(config, "text_config", config), "hidden_size", None),
        "num_hidden_layers": getattr(getattr(config, "text_config", config), "num_hidden_layers", None),
        "hf_cache_path": snapshot_path,
        "memory_allocated": int(torch.cuda.memory_allocated(0)) if torch.cuda.is_available() else 0,
        "memory_reserved": int(torch.cuda.memory_reserved(0)) if torch.cuda.is_available() else 0,
        "events": logger.events,
    }
    write_json(Path(artifact_dir) / "load_log.json", payload)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
