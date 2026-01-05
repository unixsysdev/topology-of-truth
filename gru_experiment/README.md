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

### Intervention Point: Layer L_int (configurable, default=4)

For Qwen3-0.6B with 28 layers → **Layer 4** (default) or configurable via ratio

**Why early (layer 4)?**
- Gives ~24 remaining layers to use the correction
- Problems often start early in the reasoning trajectory
- More room for the steering to propagate and influence output

**Configurable:**
```python
# Absolute layer
intervention_layer: int = 4

# Or as ratio of model depth
intervention_layer_ratio: float = 0.15  # 15% into model
```

**Trade-offs:**
| Layer | Pros | Cons |
|-------|------|------|
| Early (4) | More correction room | May be before "reasoning" starts |
| Middle (9) | Model has started reasoning | Less room to correct |
| Late (14+) | Clear signal of what's wrong | Too late to fix |

We default to early (layer 4) because we want maximum leverage from the intervention.

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

### Main Signal: KL Divergence from Teacher Logits

The key insight: **logit gradients already encode what "good topology" looks like**. 

We don't need TDA as a training signal - the gradient from matching teacher logits teaches the controller:
- **Gate**: learns WHEN to intervene (when student would diverge from teacher)
- **Decoder**: learns WHAT steering helps (what correction aligns with teacher)

```
┌─────────────────────────────────────────────────────────────┐
│  Training Flow                                              │
│                                                             │
│  Input ──► Student (frozen) ──► Layer 9 activations         │
│                                      │                      │
│                                      ▼                      │
│                               GRU Controller                │
│                               (trainable)                   │
│                                      │                      │
│                                steering vector              │
│                                      │                      │
│                                      ▼                      │
│  Input ──► Student + steering ──► Student Logits            │
│                                      │                      │
│  Input ──► Teacher (frozen) ────► Teacher Logits            │
│                                      │                      │
│                                      ▼                      │
│                              KL Divergence Loss             │
│                                      │                      │
│                                      ▼                      │
│                         Backprop through Controller         │
└─────────────────────────────────────────────────────────────┘
```

### TDA Role: Offline Verification (Not Training)

TDA is used to **verify** the intervention works, not to train:

| Phase | TDA Role |
|-------|----------|
| Training | Not used (too slow, redundant with logit signal) |
| Validation | Periodic check that H₀ entropy is actually reducing |
| Analysis | Visualize before/after topology for interpretability |

### Why This Design?

1. **Simpler**: One clean loss function (KL divergence)
2. **Faster**: No TDA computation during training
3. **Principled**: Logits are the ground truth for "good reasoning"
4. **Verifiable**: TDA confirms we're not just matching logits superficially

### Optional: TDA Auxiliary Loss

If enabled (`use_tda_aux_loss=True`), adds a pre-trained TDA predictor as auxiliary loss. 
**Disabled by default** - the logit signal is sufficient and cleaner.

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
