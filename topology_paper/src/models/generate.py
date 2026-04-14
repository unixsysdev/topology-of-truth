from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class GenerationResult:
    generated_text: str
    generated_token_ids: list[int]
    full_token_ids: list[int]
    input_length: int
    logits_summary: dict[str, np.ndarray] | None
    full_logits: np.ndarray | None
    hit_max_new_tokens: bool
    terminated_by_eos: bool
    stop_reason: str


def build_chat_prompt(
    tokenizer,
    prompt: str,
    system_prompt: str | None = None,
    enable_thinking: bool | None = None,
    force_raw_prompt: bool = False,
) -> str:
    if force_raw_prompt:
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}\n"
        return prompt
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        template_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        return tokenizer.apply_chat_template(messages, **template_kwargs)
    if system_prompt:
        return f"{system_prompt}\n\n{prompt}\n"
    return prompt


@torch.no_grad()
def generate_response(
    model,
    tokenizer,
    prompt: str,
    generation_config: dict,
    save_logits: bool = True,
    save_full_logits: bool = False,
) -> GenerationResult:
    rendered = build_chat_prompt(
        tokenizer,
        prompt,
        generation_config.get("system_prompt"),
        generation_config.get("enable_thinking"),
        bool(generation_config.get("force_raw_prompt", False)),
    )
    inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
    input_length = int(inputs["input_ids"].shape[1])

    deterministic = generation_config.get("deterministic", True)
    temperature = float(generation_config.get("temperature", 0.0))
    do_sample = (not deterministic) and temperature > 0.0

    generate_kwargs = {
        "max_new_tokens": int(generation_config.get("max_new_tokens", 512)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": save_logits,
    }
    # Some chat models (e.g., Gemma) can greedily emit PAD forever when PAD is
    # a normal vocab token distinct from EOS; block PAD from generation.
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        eos_id = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple, set)):
            eos_ids = {int(x) for x in eos_id}
        elif eos_id is None:
            eos_ids = set()
        else:
            eos_ids = {int(eos_id)}
        if int(pad_id) not in eos_ids:
            generate_kwargs["bad_words_ids"] = [[int(pad_id)]]
    if temperature > 0:
        generate_kwargs["temperature"] = temperature
    if do_sample:
        generate_kwargs["top_p"] = float(generation_config.get("top_p", 1.0))

    max_new_tokens = int(generation_config.get("max_new_tokens", 512))
    eos_ids = _eos_token_ids(model, tokenizer)
    constrained_labels = generation_config.get("constrained_labels")
    if constrained_labels:
        allowed_sequences = _build_allowed_label_sequences(tokenizer, constrained_labels)
        if not allowed_sequences:
            raise ValueError("constrained_labels produced no token sequences")
        generate_kwargs["prefix_allowed_tokens_fn"] = _build_prefix_allowed_tokens_fn(
            allowed_sequences,
            eos_ids,
            input_length,
        )

    outputs = model.generate(
        **inputs,
        **generate_kwargs,
    )

    sequence = outputs.sequences[0]
    generated_ids = sequence[input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    terminated_by_eos = bool(len(generated_ids) > 0 and int(generated_ids[-1]) in eos_ids)
    hit_max_new_tokens = bool(len(generated_ids) >= max_new_tokens and not terminated_by_eos)
    stop_reason = "max_new_tokens" if hit_max_new_tokens else ("eos_token" if terminated_by_eos else "unknown")

    logits_summary = None
    full_logits = None
    if save_logits and outputs.scores:
        scores = torch.stack(outputs.scores, dim=0).float().cpu()
        chosen_ids = generated_ids[: scores.shape[0]].cpu()
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).squeeze(1).numpy()
        max_prob = probs.max(dim=-1).values.squeeze(1).numpy()
        chosen_logprob = torch.log_softmax(scores, dim=-1).squeeze(1).gather(
            1, chosen_ids[:, None]
        ).squeeze(1).numpy()
        top2 = torch.topk(probs, k=2, dim=-1).values.squeeze(1)
        margin = (top2[:, 0] - top2[:, 1]).numpy()
        logits_summary = {
            "entropy": entropy.astype(np.float32),
            "max_prob": max_prob.astype(np.float32),
            "chosen_logprob": chosen_logprob.astype(np.float32),
            "top2_margin": margin.astype(np.float32),
        }
        if save_full_logits:
            full_logits = scores.squeeze(1).numpy().astype(np.float16)

    return GenerationResult(
        generated_text=generated_text,
        generated_token_ids=generated_ids.cpu().tolist(),
        full_token_ids=sequence.cpu().tolist(),
        input_length=input_length,
        logits_summary=logits_summary,
        full_logits=full_logits,
        hit_max_new_tokens=hit_max_new_tokens,
        terminated_by_eos=terminated_by_eos,
        stop_reason=stop_reason,
    )


def _eos_token_ids(model, tokenizer) -> set[int]:
    eos = getattr(model.generation_config, "eos_token_id", None)
    if eos is None:
        eos = tokenizer.eos_token_id
    if eos is None:
        return set()
    if isinstance(eos, (list, tuple, set)):
        return {int(x) for x in eos}
    return {int(eos)}


def _build_allowed_label_sequences(tokenizer, labels: list[str]) -> list[tuple[int, ...]]:
    variants: set[tuple[int, ...]] = set()
    for label in labels:
        for text in [label, f" {label}", f"{label}\n", f" {label}\n"]:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if token_ids:
                variants.add(tuple(int(t) for t in token_ids))
    return sorted(variants)


def _build_prefix_allowed_tokens_fn(
    allowed_sequences: list[tuple[int, ...]],
    eos_ids: set[int],
    input_length: int,
):
    allowed_set = set(allowed_sequences)

    def prefix_allowed_tokens_fn(batch_id, input_ids):  # noqa: ARG001
        generated = tuple(int(t) for t in input_ids[input_length:].tolist())
        candidates: set[int] = set()
        for seq in allowed_sequences:
            if len(generated) < len(seq) and seq[: len(generated)] == generated:
                candidates.add(seq[len(generated)])
            elif generated == seq:
                candidates.update(eos_ids)
        if candidates:
            return sorted(candidates)
        if generated in allowed_set and eos_ids:
            return sorted(eos_ids)
        if eos_ids:
            return sorted(eos_ids)
        # Fallback should be unreachable, but keep generation alive if no EOS is configured.
        return [allowed_sequences[0][0]]

    return prefix_allowed_tokens_fn
