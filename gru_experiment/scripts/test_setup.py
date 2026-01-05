#!/usr/bin/env python
"""
Quick sanity check for GRU intervention setup.

Tests that all components work together without full training.
Run this first to catch any setup issues.

Usage:
    python scripts/test_setup.py
    python scripts/test_setup.py --device cpu  # If no GPU
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse


def test_controller():
    """Test GRU controller in isolation."""
    print("\n[1/4] Testing GRU Controller...")
    
    from models import GRUController, ControllerConfig
    
    config = ControllerConfig(
        model_hidden_size=1024,  # Qwen3-0.6B
        encoder_hidden_dim=128,  # Smaller for test
        latent_dim=32,
        decoder_rank=8
    )
    
    controller = GRUController(config)
    
    # Simulate input
    batch_size = 2
    seq_len = 64
    activations = torch.randn(batch_size, seq_len, 1024)
    
    # Forward pass
    result = controller(activations, return_diagnostics=True)
    
    assert result['steering'].shape == (batch_size, 1024), "Steering shape wrong"
    assert result['gate'].shape == (batch_size, 1), "Gate shape wrong"
    assert result['latent'].shape == (batch_size, 32), "Latent shape wrong"
    
    num_params = sum(p.numel() for p in controller.parameters())
    print(f"  ✓ Controller works! Parameters: {num_params:,}")
    
    return True


def test_hooked_model(device: str = "cuda"):
    """Test model with hooks."""
    print(f"\n[2/4] Testing Hooked Model (device={device})...")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping...")
        return True
    
    from models import HookedModel, HookConfig
    
    # Use tiny config for speed
    model = HookedModel(
        model_name="Qwen/Qwen3-0.6B",
        config=HookConfig(intervention_layer=4),
        device=device,
        dtype=torch.float16 if device == "cuda" else torch.float32
    )
    
    # Test forward pass
    text = "What is 2 + 2?"
    inputs = model.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Check activations cached
    activations = model.get_activations()
    assert activations is not None, "No activations cached"
    
    print(f"  ✓ Hooked model works! Activation shape: {activations.shape}")
    
    # Test steering injection
    steering = torch.randn(1, model.hidden_size, device=device, dtype=model.dtype) * 0.01
    
    with model.steering_context(steering):
        outputs_steered = model(**inputs)
    
    print(f"  ✓ Steering injection works!")
    
    model.remove_hooks()
    del model
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return True


def test_data_loading():
    """Test dataset loading."""
    print("\n[3/4] Testing Data Loading...")
    
    from datasets import load_dataset
    
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    sample = dataset[0]
    
    assert 'question' in sample, "Missing 'question' field"
    assert 'answer' in sample, "Missing 'answer' field"
    
    print(f"  ✓ Dataset loads! Total samples: {len(dataset)}")
    print(f"  Sample question: {sample['question'][:50]}...")
    
    return True


def test_integration(device: str = "cuda"):
    """Test full forward pass with intervention."""
    print(f"\n[4/4] Testing Full Integration (device={device})...")
    
    if device == "cuda" and not torch.cuda.is_available():
        print("  ⚠ CUDA not available, skipping...")
        return True
    
    from models import GRUController, ControllerConfig, HookedModel, HookConfig
    
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load model
    model = HookedModel(
        model_name="Qwen/Qwen3-0.6B",
        config=HookConfig(intervention_layer=4),
        device=device,
        dtype=dtype
    )
    
    # Create controller
    controller_config = ControllerConfig(
        model_hidden_size=model.hidden_size,
        encoder_hidden_dim=128,
        latent_dim=32,
        decoder_rank=8
    )
    controller = GRUController(controller_config).to(device)
    
    # Prepare input
    text = "What is 2 + 2? Let me solve this step by step."
    inputs = model.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward without intervention
    with torch.no_grad():
        _ = model(**inputs)
        activations = model.get_activations()
    
    # Get steering from controller
    with torch.no_grad():
        ctrl_out = controller(activations.to(dtype))
        steering = ctrl_out['steering']
        gate = ctrl_out['gate']
    
    # Forward with intervention
    with model.steering_context(steering):
        with torch.no_grad():
            outputs = model(**inputs)
    
    print(f"  ✓ Full integration works!")
    print(f"    Gate value: {gate.item():.4f}")
    print(f"    Steering norm: {steering.norm().item():.4f}")
    print(f"    Output shape: {outputs.logits.shape}")
    
    model.remove_hooks()
    del model, controller
    torch.cuda.empty_cache() if device == "cuda" else None
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Test GRU intervention setup")
    parser.add_argument("--device", type=str, default="cuda", 
                        choices=["cuda", "cpu"],
                        help="Device to test on")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip model loading tests (faster)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("GRU Intervention Setup Test")
    print("=" * 60)
    
    results = []
    
    # Test 1: Controller (always runs, CPU only)
    results.append(("Controller", test_controller()))
    
    # Test 2: Data loading
    results.append(("Data Loading", test_data_loading()))
    
    if not args.skip_model:
        # Test 3: Hooked model
        results.append(("Hooked Model", test_hooked_model(args.device)))
        
        # Test 4: Integration
        results.append(("Integration", test_integration(args.device)))
    else:
        print("\n⚠ Skipping model tests (--skip-model)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n✓ All tests passed! Ready to train.")
        print("\nQuick start:")
        print("  python scripts/train_phase2.py --debug  # Fast sanity check")
        print("  python scripts/train_phase2.py --fast   # Quick training")
        print("  python scripts/train_phase2.py          # Full training")
    else:
        print("\n✗ Some tests failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
