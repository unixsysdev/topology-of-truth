from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import EventLogger, ensure_dir, prepare_hf_env, save_env_dump, timed_step, write_json
from topology_paper.src.models.generate import build_chat_prompt
from topology_paper.src.models.hidden_state_extract import extract_hidden_states
from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract teacher-forced short-answer references for within-question runs.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--artifact-dir", default="artifacts/qwen35_2b_within_question")
    parser.add_argument("--config-system-prompt", default="")
    parser.add_argument("--template", default="The answer is {answer}.")
    parser.add_argument("--timeout-sec", type=int, default=900)
    args = parser.parse_args()

    prepare_hf_env()
    artifact_dir = Path(args.artifact_dir)
    logger = EventLogger(artifact_dir / "reference.log")
    save_env_dump(artifact_dir / "reference_env.txt", model_id=args.model_id)

    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())

    for question_dir in sorted(artifact_dir.glob("q_*")):
        sample_json = first_sample_json(question_dir)
        if sample_json is None:
            continue
        payload = json.loads(sample_json.read_text(encoding="utf-8"))
        reference_dir = ensure_dir(question_dir / "reference")
        target_text = args.template.format(answer=payload["gold_answer"])
        rendered_prompt = build_chat_prompt(tokenizer, payload.get("prompt", payload["question"]), args.config_system_prompt or None)
        prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0].tolist()
        target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + target_ids

        with timed_step(logger, f"reference_{payload['question_id']}", args.timeout_sec):
            hidden = extract_hidden_states(model, full_ids, len(prompt_ids), ["last"])["last"]
        hidden_path = reference_dir / "reference_hidden.npy"
        np.save(hidden_path, hidden)
        np.load(hidden_path)

        answer_ids = tokenizer.encode(str(payload["gold_answer"]), add_special_tokens=False)
        answer_start = find_last_subsequence(target_ids, answer_ids)
        answer_indices = list(range(answer_start, answer_start + len(answer_ids))) if answer_start >= 0 else []
        write_json(
            reference_dir / "reference.json",
            {
                "question_id": payload["question_id"],
                "dataset_sample_id": payload["dataset_sample_id"],
                "gold_answer": payload["gold_answer"],
                "question": payload["question"],
                "prompt": payload.get("prompt", payload["question"]),
                "template": args.template,
                "target_text": target_text,
                "token_count": len(target_ids),
                "target_token_ids": target_ids,
                "answer_token_indices": answer_indices,
                "hidden_shape": list(hidden.shape),
                "hidden_path": str(hidden_path),
            },
        )

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def first_sample_json(question_dir: Path) -> Path | None:
    for sample_json in sorted(question_dir.glob("sample_*/sample.json")):
        return sample_json
    return None


def find_last_subsequence(sequence: list[int], subsequence: list[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


if __name__ == "__main__":
    main()
