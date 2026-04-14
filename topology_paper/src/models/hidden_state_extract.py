from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def extract_hidden_states(model, token_ids: list[int], input_length: int, layers: list[str | int]) -> dict[str, np.ndarray]:
    input_ids = torch.tensor([token_ids], device=model.device)
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states

    result = {}
    for layer in layers:
        key, idx = _resolve_layer(layer, hidden_states)
        values = hidden_states[idx][0, input_length:, :].detach().cpu().float().numpy()
        result[key] = values.astype(np.float32)
    return result


def _resolve_layer(layer: str | int, hidden_states) -> tuple[str, int]:
    if layer == "last":
        return "last", -1
    if layer == "middle":
        idx = len(hidden_states) // 2
        return f"layer_{idx}", idx
    if isinstance(layer, int):
        return f"layer_{layer}", layer
    if isinstance(layer, str) and layer.startswith("layer_"):
        return layer, int(layer.split("_", 1)[1])
    raise ValueError(f"Unsupported layer specifier: {layer}")

