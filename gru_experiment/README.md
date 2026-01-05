# GRU Reasoning Intervention Experiment

**Goal**: Improve small model (0.6B) reasoning by detecting and correcting "topological drift" in real-time using a GRU-based meta-controller.

## Core Hypothesis

Based on our TDA findings:
- **Low H₀ entropy** = coherent reasoning = correct answers
- **High H₀ entropy** = fragmented "dust" = wrong answers
- **H₁ loops are noise**, not signal (wrong answers have MORE loops)

We can train a lightweight GRU to:
1. **Detect** when activations start fragmenting (early entropy spike)
2. **Intervene** by injecting a steering vector to restore coherence
3. **Validate** that intervention reduces trajectory H₀ entropy AND improves accuracy

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Qwen3-0.6B (Frozen)                                                        │
│                                                                             │
│  Layer 0 → Layer 1 → ... → Layer L_int ──┬──► Layer L_int+1 → ... → Output │
│                                          │                                  │
│                              ┌───────────┴───────────┐                      │
│                              │    GRU Meta-Controller │                      │
│                              │                        │                      │
│                              │  ┌──────────────────┐ │                      │
│                              │  │   GRU Encoder    │ │ ← Encodes recent    │
│                              │  │   (hidden=256)   │ │   activation context │
│                              │  └────────┬─────────┘ │                      │
│                              │           │           │                      │
│                              │  ┌────────▼─────────┐ │                      │
│                              │  │  Entropy Gate    │ │ ← Opens when local  │
│                              │  │     (σ)          │ │   entropy spikes    │
│                              │  └────────┬─────────┘ │                      │
│                              │           │           │                      │
│                              │  ┌────────▼─────────┐ │                      │
│                              │  │   GRU Decoder    │ │ ← Generates steering │
│                              │  │  (low-rank=16)   │ │   vector             │
│                              │  └────────┬─────────┘ │                      │
│                              │           │           │                      │
│                              └───────────┼───────────┘                      │
│                                          │                                  │
│                                          ▼                                  │
│                              steering_vector (added to residual)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### Intervention Point: Layer L_int (≈1/3 depth)

For Qwen3-0.6B with 28 layers → **Layer 9 or 10**

Why?
- Early enough that intervention can help downstream layers
- Late enough that model has started "reasoning" (not just embedding)
- Matches Google's meta-controller placement strategy

### What the GRU Sees

**Input**: Residual stream at layer L_int, projected to lower dimension
- Raw residual: ~1024 dim (0.6B hidden size)
- Projected: 256 dim (for efficiency)
- Sequence: Last N tokens (e.g., 32-64 tokens sliding window)

**The GRU maintains**:
- Hidden state encoding "reasoning trajectory so far"
- Learned patterns of coherent vs fragmented reasoning

### Gate Trigger: Local Entropy Estimate

We don't compute full TDA at inference (too slow). Instead, we use a **local entropy proxy**:

```python
# Fast local entropy estimate
def local_entropy_proxy(activations, window=16):
    """
    Estimate fragmentation from activation variance/clustering
    High variance across tokens = high entropy = fragmentation
    """
    # Option 1: Variance-based
    token_variance = activations.var(dim=0).mean()
    
    # Option 2: Cosine similarity based  
    cos_sim = F.cosine_similarity(activations[:-1], activations[1:], dim=-1)
    coherence = cos_sim.mean()
    
    return 1 - coherence  # High value = fragmented
```

The GRU learns to predict when this proxy indicates trouble.

### Steering Vector: Low-Rank Injection

Following Google's approach:
- Decoder outputs low-rank matrices A (d×r) and B (r×d), where r=16
- Steering vector = A @ B @ latent_plan
- Added to residual stream: `h_new = h_old + gate * steering_vector`

## Training Strategy

### Phase 1: Pattern Discovery (Unsupervised)

1. Run 1.7B model on GSM8K → collect activation trajectories
2. Cluster successful reasoning patterns at layer L_int
3. Learn latent codes for "coherent reasoning phases"

This gives us implicit sub-goals without manual labeling.

### Phase 2: Intervention Training (RL or Supervised)

**Option A: Supervised Distillation**
- Train GRU to push 0.6B activations toward 1.7B's coherent patterns
- Loss = MSE(intervened_0.6B_trajectory, 1.7B_trajectory)

**Option B: RL with TDA Reward**
- Reward = -H₀_entropy + accuracy_bonus
- Gate learns when to intervene
- Decoder learns what correction to apply

**Option C: Hybrid**
- Phase 2a: Supervised warmup (distill from 1.7B)
- Phase 2b: RL fine-tuning (optimize for actual accuracy)

## Computational Requirements

### Local (RTX 3090/4090, 24GB VRAM)
- 0.6B model fits easily (~1.5GB fp16)
- GRU controller is tiny (~10MB)
- Batch size 1-4, gradient checkpointing
- TDA computation on CPU (slow but works)

### Cloud (H100/A100, 40-80GB VRAM)
- Can fit 0.6B + 1.7B simultaneously for distillation
- Larger batches (8-16)
- Faster TDA with GPU ripser

### Optimizations for Local
- Use fp16/bf16 throughout
- Gradient checkpointing on base model
- Compute TDA only for validation, not training
- Use local entropy proxy during training
- Small sliding window (32 tokens)

## Directory Structure

```
gru_experiment/
├── configs/
│   ├── base.yaml           # Base configuration
│   ├── local.yaml          # Optimized for local GPU
│   └── cloud.yaml          # Full-scale cloud training
├── models/
│   ├── gru_controller.py   # GRU encoder-decoder + gate
│   ├── steering.py         # Low-rank steering vector
│   └── hooked_model.py     # Qwen with intervention hooks
├── data/
│   ├── prepare_trajectories.py  # Extract activations from models
│   └── trajectory_dataset.py    # DataLoader for training
├── scripts/
│   ├── train_phase1.py     # Pattern discovery
│   ├── train_phase2.py     # Intervention training
│   ├── evaluate.py         # Test accuracy + TDA metrics
│   └── visualize.py        # Compare before/after topology
├── checkpoints/            # Saved models
└── results/                # Evaluation outputs
```

## Success Metrics

| Metric | 0.6B Baseline | Target (with GRU) |
|--------|---------------|-------------------|
| GSM8K Accuracy | 63.3% | >70% |
| H₀ Entropy (correct) | 3.41 | <3.2 |
| H₀ Entropy (wrong) | 3.83 | <3.5 |
| Inference overhead | - | <10% slowdown |

## References

- [Google: Emergent Temporal Abstraction](https://arxiv.org/) - Meta-controller architecture
- [TDA for LLM Reasoning](https://arxiv.org/) - Topology of reasoning chains
- Our results: `../results_thinking/` - H₀ entropy predicts correctness
