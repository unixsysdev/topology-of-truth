from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from topology_paper.src.config import ensure_output_dirs, load_config, model_slug
from topology_paper.src.data.load_dataset import load_configured_dataset, sample_record
from topology_paper.src.eval.correctness import score_generation
from topology_paper.src.features.lexical_baselines import save_logits_npz
from topology_paper.src.models.generate import generate_response
from topology_paper.src.models.hidden_state_extract import extract_hidden_states
from topology_paper.src.models.loader import load_model_and_tokenizer


def run_generation(config_path: str | Path | None = None, models: list[str] | None = None, start: int = 0, end: int | None = None) -> None:
    config = load_config(config_path)
    dirs = ensure_output_dirs(config)
    dataset = load_configured_dataset(config["dataset"])
    if end is None:
        end = len(dataset)
    selected = range(start, min(end, len(dataset)))
    model_ids = models or config["models"]

    for model_id in model_ids:
        slug = model_slug(model_id)
        model_dir = dirs["traces"] / slug
        model_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load_model_and_tokenizer(model_id)

        for idx in tqdm(selected, desc=f"generate {model_id}"):
            item = dataset[idx]
            record = sample_record(item, idx, config["generation"].get("prompt_style", "math_boxed"))
            sample_dir = model_dir / record["sample_id"]
            trace_path = sample_dir / "trace.json"
            if trace_path.exists():
                continue
            sample_dir.mkdir(parents=True, exist_ok=True)

            generation = generate_response(
                model,
                tokenizer,
                record["prompt"],
                config["generation"],
                save_logits=bool(config["extraction"].get("save_logits", True)),
                save_full_logits=bool(config["extraction"].get("save_full_logits", False)),
            )
            hidden = extract_hidden_states(
                model,
                generation.full_token_ids,
                generation.input_length,
                config["extraction"].get("layers", ["last"]),
            )
            hidden_path = sample_dir / "last_layer.npy"
            if "last" not in hidden:
                first_key = sorted(hidden.keys())[-1]
                hidden_path = sample_dir / f"{first_key}.npy"
                np.save(hidden_path, hidden[first_key])
            else:
                np.save(hidden_path, hidden["last"])

            for key, value in hidden.items():
                if key != "last":
                    np.save(sample_dir / f"{key}.npy", value)

            logits_path = save_logits_npz(sample_dir / "logits_summary.npz", generation.logits_summary, generation.full_logits)
            score = score_generation(generation.generated_text, record["gold_answer"])
            trace = {
                "sample_id": record["sample_id"],
                "model_id": model_id,
                "question": record["question"],
                "gold_answer": record["gold_answer"],
                "generated_text": generation.generated_text,
                "pred_answer": score["pred_answer"],
                "correct": score["correct"],
                "n_generated_tokens": len(generation.generated_token_ids),
                "hit_max_new_tokens": generation.hit_max_new_tokens,
                "terminated_by_eos": generation.terminated_by_eos,
                "stop_reason": generation.stop_reason,
                "token_ids": generation.generated_token_ids,
                "hidden_path": str(hidden_path),
                "logits_path": logits_path,
            }
            with trace_path.open("w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GSM8K traces and hidden states.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    run_generation(args.config, models=args.models, start=args.start, end=args.end)


if __name__ == "__main__":
    main()
