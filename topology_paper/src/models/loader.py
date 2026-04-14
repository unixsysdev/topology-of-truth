from __future__ import annotations

import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")


def resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model_and_tokenizer(model_id: str, device: str | None = None):
    device = device or resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if device in {"cuda", "mps"}:
        dtype = torch.float16
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            cfg_dtype = getattr(cfg, "torch_dtype", None)
            if isinstance(cfg_dtype, str):
                cfg_dtype = getattr(torch, cfg_dtype, None)
            if cfg_dtype in {torch.float16, torch.bfloat16, torch.float32}:
                dtype = cfg_dtype
        except Exception:
            pass
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map={"": device} if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer
