from __future__ import annotations

import argparse
import csv
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _common import EventLogger, ensure_dir, prepare_hf_env, save_env_dump, timed_step, write_json
from topology_paper.src.features.lexical_baselines import save_logits_npz
from topology_paper.src.models.generate import build_chat_prompt, generate_response
from topology_paper.src.models.loader import load_model_and_tokenizer, resolve_device


LABEL_RE = re.compile(r"\b(True|False|Unknown)\b", re.IGNORECASE)

MANIFEST_FIELDS = [
    "example_id",
    "world_id",
    "proposition_id",
    "paraphrase_id",
    "split",
    "label",
    "pred_label",
    "correct",
    "template_id",
    "template_family",
    "lf_type",
    "status",
    "stop_reason",
    "token_count",
    "runtime_sec",
    "sample_path",
    "states_path",
    "logits_path",
    "error",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run short-output inference for the micro-world dataset.")
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--dataset", default="artifacts/micro_world_v1/dataset/eval.jsonl")
    parser.add_argument("--artifact-dir", default="artifacts/micro_world_v1/generations/Qwen__Qwen3_5_2B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--timeout-sec", type=int, default=900)
    parser.add_argument("--resume-skip-existing", action="store_true")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--force-raw-prompt", action="store_true")
    parser.add_argument("--constrained-label-decoding", action="store_true")
    parser.add_argument(
        "--prompt-variant",
        choices=["default", "base_label"],
        default="default",
        help="Optional prompt rendering variant. 'base_label' appends an explicit answer cue.",
    )
    args = parser.parse_args()

    prepare_hf_env()
    artifact_dir = ensure_dir(args.artifact_dir)
    examples_dir = ensure_dir(artifact_dir / "examples")
    logger = EventLogger(artifact_dir / "run.log")
    save_env_dump(artifact_dir / "env.txt", model_id=args.model_id)

    examples = load_jsonl(Path(args.dataset))
    if args.offset:
        examples = examples[args.offset :]
    if args.limit > 0:
        examples = examples[: args.limit]

    with timed_step(logger, "load_model", args.timeout_sec):
        model, tokenizer = load_model_and_tokenizer(args.model_id, device=resolve_device())

    rows: list[dict[str, Any]] = []
    generation_config = {
        "deterministic": True,
        "temperature": 0.0,
        "max_new_tokens": int(args.max_new_tokens),
    }
    if args.system_prompt:
        generation_config["system_prompt"] = str(args.system_prompt)
    if args.disable_thinking:
        generation_config["enable_thinking"] = False
    if args.force_raw_prompt:
        generation_config["force_raw_prompt"] = True
    if args.constrained_label_decoding:
        generation_config["constrained_labels"] = ["True", "False", "Unknown"]
    for example in examples:
        example_id = example["example_id"]
        example_dir = ensure_dir(examples_dir / example_id)
        sample_path = example_dir / "sample.json"
        states_path = example_dir / "verdict_states.npz"
        logits_path = example_dir / "logits_summary.npz"

        if args.resume_skip_existing and sample_path.exists() and states_path.exists() and sample_path.stat().st_size > 0 and states_path.stat().st_size > 0:
            try:
                payload = json.loads(sample_path.read_text(encoding="utf-8"))
                rows.append(row_from_payload(payload, sample_path, states_path, logits_path, ""))
                logger.log(example_id, "skip_existing")
                continue
            except Exception:
                pass

        try:
            with timed_step(logger, f"generate_{example_id}", args.timeout_sec):
                start = time.perf_counter()
                prompt_text = render_prompt(example["prompt"], args.prompt_variant)
                generation = generate_response(
                    model,
                    tokenizer,
                    prompt_text,
                    generation_config,
                    save_logits=True,
                    save_full_logits=False,
                )
                runtime_sec = time.perf_counter() - start

            rendered_prompt = build_chat_prompt(
                tokenizer,
                prompt_text,
                generation_config.get("system_prompt"),
                generation_config.get("enable_thinking"),
                bool(generation_config.get("force_raw_prompt", False)),
            )
            prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")["input_ids"][0].tolist()
            save_verdict_states(model, generation.full_token_ids, len(prompt_ids), states_path)
            saved_logits_path = save_logits_npz(logits_path, generation.logits_summary, generation.full_logits)

            pred_label = extract_pred_label(generation.generated_text)
            payload = {
                "example_id": example_id,
                "world_id": example["world_id"],
                "proposition_id": example["proposition_id"],
                "paraphrase_id": example["paraphrase_id"],
                "split": example["split"],
                "label": example["label"],
                "pred_label": pred_label,
                "correct": pred_label == example["label"],
                "template_id": example["template_id"],
                "template_family": example["template_family"],
                "logical_form": example["logical_form"],
                "statement": example["statement"],
                "prompt": prompt_text,
                "prompt_variant": args.prompt_variant,
                "generated_text": generation.generated_text,
                "generated_token_ids": generation.generated_token_ids,
                "stop_reason": generation.stop_reason,
                "token_count": len(generation.generated_token_ids),
                "runtime_sec": runtime_sec,
                "model_id": args.model_id,
                "states_path": str(states_path),
                "logits_path": str(saved_logits_path) if saved_logits_path else "",
            }
            write_json(sample_path, payload)
            row = row_from_payload(payload, sample_path, states_path, logits_path, "")
        except Exception as exc:
            logger.log(example_id, "error", error=repr(exc), traceback=traceback.format_exc())
            row = error_row(example, repr(exc))
        rows.append(row)
        write_manifest(artifact_dir / "manifest.csv", rows)

    write_manifest(artifact_dir / "manifest.csv", rows)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@torch.no_grad()
def save_verdict_states(model, full_token_ids: list[int], input_length: int, path: Path) -> None:
    input_ids = torch.tensor([full_token_ids], device=model.device)
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden = outputs.hidden_states[-1][0].detach().cpu().float().numpy().astype(np.float32)
    prompt = hidden[:input_length]
    generated = hidden[input_length:]
    final_prompt = prompt[-1]
    prompt_tail = prompt[-min(5, len(prompt)) :].mean(axis=0)
    if len(generated):
        verdict_token = generated[0]
        verdict_span = generated[: min(3, len(generated))].mean(axis=0)
    else:
        verdict_token = np.zeros_like(final_prompt)
        verdict_span = np.zeros_like(final_prompt)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("wb") as f:
        np.savez_compressed(
            f,
            final_prompt=final_prompt,
            prompt_tail_mean=prompt_tail,
            verdict_token=verdict_token,
            verdict_span_mean=verdict_span,
        )
    tmp_path.replace(path)


def extract_pred_label(text: str) -> str:
    match = LABEL_RE.search(text)
    if not match:
        return ""
    value = match.group(1).lower()
    if value == "true":
        return "True"
    if value == "false":
        return "False"
    return "Unknown"


def render_prompt(prompt: str, variant: str) -> str:
    if variant == "base_label":
        trimmed = prompt.rstrip()
        if trimmed.endswith("Answer:"):
            return trimmed
        return f"{trimmed}\nAnswer:"
    return prompt


def row_from_payload(payload: dict[str, Any], sample_path: Path, states_path: Path, logits_path: Path, error: str) -> dict[str, Any]:
    return {
        "example_id": payload["example_id"],
        "world_id": payload["world_id"],
        "proposition_id": payload["proposition_id"],
        "paraphrase_id": payload["paraphrase_id"],
        "split": payload["split"],
        "label": payload["label"],
        "pred_label": payload["pred_label"],
        "correct": bool(payload["correct"]),
        "template_id": payload["template_id"],
        "template_family": payload["template_family"],
        "lf_type": payload["logical_form"]["type"],
        "status": "ok",
        "stop_reason": payload["stop_reason"],
        "token_count": int(payload["token_count"]),
        "runtime_sec": round(float(payload["runtime_sec"]), 3),
        "sample_path": str(sample_path),
        "states_path": str(states_path),
        "logits_path": str(logits_path),
        "error": error,
    }


def error_row(example: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "example_id": example["example_id"],
        "world_id": example["world_id"],
        "proposition_id": example["proposition_id"],
        "paraphrase_id": example["paraphrase_id"],
        "split": example["split"],
        "label": example["label"],
        "pred_label": "",
        "correct": "",
        "template_id": example["template_id"],
        "template_family": example["template_family"],
        "lf_type": example["logical_form"]["type"],
        "status": "error",
        "stop_reason": "",
        "token_count": "",
        "runtime_sec": "",
        "sample_path": "",
        "states_path": "",
        "logits_path": "",
        "error": error,
    }


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
