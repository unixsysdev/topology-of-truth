from __future__ import annotations

from typing import Any

from datasets import load_dataset

from topology_paper.src.data.answer_extract import extract_gsm8k_answer


def load_configured_dataset(dataset_config: dict[str, Any]):
    name = dataset_config["name"]
    split = dataset_config.get("split", "test")
    ds_config = dataset_config.get("config")
    max_samples = dataset_config.get("max_samples")

    if ds_config:
        dataset = load_dataset(name, ds_config, split=split)
    else:
        dataset = load_dataset(name, split=split)

    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def sample_record(item: dict[str, Any], idx: int, prompt_style: str = "math_boxed") -> dict[str, Any]:
    question = item.get("question") or item.get("problem") or item.get("prompt") or ""
    raw_answer = item.get("answer") or item.get("solution") or ""
    gold = extract_gsm8k_answer(raw_answer) or str(raw_answer).strip()
    prompt = format_prompt(question, prompt_style)
    return {
        "sample_id": f"gsm8k_{idx:06d}",
        "question": question,
        "gold_answer": gold,
        "raw_answer": raw_answer,
        "prompt": prompt,
    }


def format_prompt(question: str, prompt_style: str) -> str:
    if prompt_style == "math_boxed":
        return f"{question}\n\nShow your work briefly and put the final answer in \\boxed{{}}."
    if prompt_style == "plain":
        return question
    raise ValueError(f"Unknown prompt_style: {prompt_style}")

