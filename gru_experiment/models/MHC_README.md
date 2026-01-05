# Manifold-Constrained Hyper-Connections (mHC)

## Overview

This module implements **Manifold-Constrained Hyper-Connections** for expanding the residual stream width of transformer models while maintaining training stability. Based on DeepSeek-AI's Hyper-Connections paper.

## Key Concepts

### Standard Residual Connection
```
x_{l+1} = x_l + F(x_l, W_l)
```

### Hyper-Connections (HC)
```
x_{l+1} = H_l^res @ x_l + H_l^post.T @ F(H_l^pre @ x_l, W_l)
```

### Manifold-Constrained HC (mHC)
```
x_{l+1} = P_M(H_l^res) @ x_l + H_l^post.T @ F(H_l^pre @ x_l, W_l)

where P_M projects onto doubly stochastic manifold
```

## Architecture

```
Input (batch, seq, C)
         │
         ▼
    ┌─────────┐
    │ Expand  │  C → n×C (default n=4)
    └────┬────┘
         │
         ▼
    ┌─────────────────────────────────────┐
    │         For each layer l:           │
    │                                     │
    │  block_input = H^pre @ x_expanded   │
    │  block_out = TransformerBlock(...)  │
    │  x_expanded = P_M(H^res) @ x        │
    │             + H^post @ block_out    │
    │                                     │
    └─────────────────────────────────────┘
         │
         ▼
    ┌──────────┐
    │ Compress │  n×C → C
    └────┬─────┘
         │
         ▼
Output (batch, seq, C)
```

## Components

### `SinkhornProjection`
Projects matrices onto the doubly stochastic manifold using the Sinkhorn-Knopp algorithm. A doubly stochastic matrix has all rows and columns summing to 1.

### `HyperConnectionMatrix`
Learnable matrix with optional manifold constraint. Used for H^res, H^pre, and H^post.

### `ExpandedResidualStream`
Handles expansion (C → n×C) and compression (n×C → C) of the residual stream.

### `HyperConnectionLayer`
Single layer hyper-connection module with H^res, H^pre, and H^post matrices.

### `ManifoldHyperConnections`
Complete mHC module managing all layers and stream expansion/compression.

## Usage

### Basic Usage

```python
from gru_experiment.models import ManifoldHyperConnections, HyperConnectionConfig

# Create for Qwen3-0.6B
config = HyperConnectionConfig(
    hidden_size=1024,
    num_layers=28,
    expansion_rate=4
)
mhc = ManifoldHyperConnections(config).to(device='cuda', dtype=torch.bfloat16)

# Forward pass
x_expanded = mhc.expand_input(hidden_states)

for layer_idx in range(num_layers):
    block_input = mhc.get_block_input(layer_idx, x_expanded)
    block_output = transformer_block(block_input)
    x_expanded = mhc.apply_connection(layer_idx, x_expanded, block_output)

output = mhc.compress_output(x_expanded)
```

### Factory Function

```python
from gru_experiment.models import create_mhc_for_model

mhc = create_mhc_for_model(
    hidden_size=1024,
    num_layers=28,
    expansion_rate=4,
    sinkhorn_iterations=10,
    warm_start=True
)
```

### Integration with HookedModel

```python
# Future work: integrate mHC with the HookedModel for steering
# The expanded residual stream provides more dimensions for intervention
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_size` | 1024 | Base model hidden dimension (C) |
| `num_layers` | 28 | Number of transformer layers |
| `expansion_rate` | 4 | Residual expansion factor (n) |
| `sinkhorn_iterations` | 10 | Iterations for doubly stochastic projection |
| `sinkhorn_temperature` | 1.0 | Temperature for Sinkhorn softmax |
| `init_scale` | 0.01 | Scale for off-diagonal initialization |
| `warm_start` | True | Initialize to match original model |

## Warm Start

When `warm_start=True`, the mHC matrices are initialized so that the expanded model is mathematically equivalent to the original:

- **H^res**: Identity matrix (residual passes through unchanged)
- **H^pre**: Selects from first slot of expanded stream
- **H^post**: Writes to first slot of expanded stream
- **Expand**: Puts input in first slot, zeros elsewhere
- **Compress**: Takes from first slot

This allows you to start from a pretrained model and gradually train the hyper-connections.

## Training Tips

1. **Invalidate caches after optimizer step**:
   ```python
   optimizer.step()
   mhc.invalidate_caches()
   ```

2. **Monitor manifold violation**:
   ```python
   stats = mhc.get_stats()
   print(f"Manifold violation: {stats['manifold_violation']}")
   ```

3. **Use bfloat16** for efficient training on modern GPUs

4. **Start with warm_start=True** when fine-tuning from pretrained models

## Parameter Count

For Qwen3-0.6B (hidden_size=1024, num_layers=28, expansion_rate=4):
- Expanded size: 4096
- H^res per layer: 4096 × 4096 = 16.8M
- H^pre per layer: 1024 × 4096 = 4.2M  
- H^post per layer: 4096 × 1024 = 4.2M
- **Total: ~700M additional parameters**

This is a significant addition, but provides much richer representational capacity for the residual stream.

## Future Work

- [ ] Integration with GRU controller for adaptive steering in expanded space
- [ ] Efficient fused kernels for Sinkhorn projection
- [ ] Sparse hyper-connection matrices for reduced memory
- [ ] Per-head expansion for attention layers
- [ ] Gradient checkpointing for memory efficiency

## References

- DeepSeek-AI Hyper-Connections paper
- Sinkhorn-Knopp algorithm for doubly stochastic matrices
