"""
Prepare Activation Trajectories

Extract and save activation trajectories from models for:
1. Phase 1: Pattern discovery (cluster successful reasoning patterns)
2. Phase 2: Training data (pairs of 0.6B and 1.7B activations)
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory extraction."""
    # Models
    student_model: str = "Qwen/Qwen3-0.6B"
    teacher_model: str = "Qwen/Qwen3-1.7B"
    
    # Layers to extract (relative to model depth)
    extraction_layers_rel: List[float] = None  # e.g., [0.2, 0.33, 0.5]
    
    # Data
    dataset: str = "openai/gsm8k"
    split: str = "train"
    max_samples: int = 500
    max_seq_length: int = 2048
    
    # Output
    output_dir: str = "gru_experiment/data/trajectories"
    
    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = 1  # Process one at a time for activation extraction
    
    def __post_init__(self):
        if self.extraction_layers_rel is None:
            self.extraction_layers_rel = [0.2, 0.33, 0.5]


def get_extraction_layers(num_layers: int, rel_positions: List[float]) -> List[int]:
    """Convert relative positions to absolute layer indices."""
    return [int(num_layers * pos) for pos in rel_positions]


def setup_model_with_hooks(
    model_name: str,
    extraction_layers: List[int],
    device: str,
    dtype: torch.dtype
) -> Tuple[AutoModelForCausalLM, AutoTokenizer, Dict]:
    """Load model and setup activation extraction hooks."""
    
    print(f"Loading {model_name}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Storage for activations
    activation_storage = {}
    hooks = []
    
    # Get layers
    if hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    else:
        raise ValueError("Cannot find transformer layers")
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            activation_storage[f"layer_{layer_idx}"] = hidden.detach().cpu()
        return hook
    
    for layer_idx in extraction_layers:
        if layer_idx < len(layers):
            h = layers[layer_idx].register_forward_hook(make_hook(layer_idx))
            hooks.append(h)
            print(f"  Hooked layer {layer_idx}")
    
    return model, tokenizer, activation_storage, hooks


def extract_trajectory(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    activation_storage: Dict,
    prompt: str,
    max_length: int,
    device: str
) -> Tuple[str, Dict[str, np.ndarray]]:
    """
    Generate response and extract activation trajectories.
    
    Returns:
        generated_text: The model's output
        trajectories: Dict mapping layer names to activation arrays
    """
    activation_storage.clear()
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
    
    # Decode output
    generated_ids = outputs.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Note: For full trajectory we'd need to hook during generation
    # This simplified version just gets final activations
    # For training, we'll do a forward pass on the full generated sequence
    
    # Do a forward pass to get activations
    activation_storage.clear()
    with torch.no_grad():
        model(generated_ids.unsqueeze(0))
    
    # Convert to numpy
    trajectories = {
        k: v.squeeze(0).numpy()  # Remove batch dim
        for k, v in activation_storage.items()
    }
    
    return generated_text, trajectories


def extract_answer(text: str) -> Optional[str]:
    """Extract final numerical answer from generated text."""
    import re
    
    # Try boxed format
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        return boxed[-1].strip()
    
    # Try #### format
    hash_match = re.search(r'####\s*(-?[\d,]+(?:\.\d+)?)', text)
    if hash_match:
        return hash_match.group(1).replace(',', '')
    
    return None


def compare_answers(pred: Optional[str], gold: str) -> bool:
    """Check if predicted answer matches gold."""
    if pred is None:
        return False
    
    try:
        pred_num = float(pred.replace(',', ''))
        gold_num = float(gold.replace(',', ''))
        return abs(pred_num - gold_num) < 1e-6
    except ValueError:
        return pred.strip().lower() == gold.strip().lower()


def prepare_trajectories(config: TrajectoryConfig):
    """Main function to extract trajectories from both models."""
    
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
    
    # Load dataset
    print(f"Loading dataset {config.dataset}...")
    dataset = load_dataset(config.dataset, "main", split=config.split)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    print(f"Processing {len(dataset)} samples")
    
    # Process each model
    for model_name, model_key in [
        (config.student_model, "student"),
        (config.teacher_model, "teacher")
    ]:
        print(f"\n{'='*60}")
        print(f"Processing {model_key}: {model_name}")
        print('='*60)
        
        # Get layer count
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        num_layers = model_config.num_hidden_layers
        
        extraction_layers = get_extraction_layers(
            num_layers, 
            config.extraction_layers_rel
        )
        print(f"Extraction layers: {extraction_layers}")
        
        # Setup model
        model, tokenizer, activation_storage, hooks = setup_model_with_hooks(
            model_name,
            extraction_layers,
            config.device,
            dtype
        )
        
        # Storage for results
        results = []
        
        # Process samples
        for idx, sample in enumerate(tqdm(dataset, desc=f"Extracting {model_key}")):
            question = sample['question']
            gold_answer = sample['answer'].split('####')[-1].strip()
            
            # Format prompt
            prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer: Let me think through this step by step."
            
            try:
                # Extract trajectory
                generated_text, trajectories = extract_trajectory(
                    model,
                    tokenizer,
                    activation_storage,
                    prompt,
                    config.max_seq_length,
                    config.device
                )
                
                # Extract and compare answer
                pred_answer = extract_answer(generated_text)
                correct = compare_answers(pred_answer, gold_answer)
                
                # Save trajectory
                traj_file = output_dir / f"{model_key}_q{idx:04d}.npz"
                np.savez_compressed(
                    traj_file,
                    **trajectories
                )
                
                # Record metadata
                results.append({
                    'question_id': f"q_{idx:04d}",
                    'question': question,
                    'gold_answer': gold_answer,
                    'pred_answer': pred_answer,
                    'correct': correct,
                    'trajectory_file': str(traj_file),
                    'seq_length': list(trajectories.values())[0].shape[0] if trajectories else 0
                })
                
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                results.append({
                    'question_id': f"q_{idx:04d}",
                    'error': str(e)
                })
        
        # Save metadata
        meta_file = output_dir / f"{model_key}_metadata.jsonl"
        with open(meta_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        # Summary statistics
        correct_count = sum(1 for r in results if r.get('correct', False))
        print(f"\n{model_key} accuracy: {correct_count}/{len(results)} ({100*correct_count/len(results):.1f}%)")
        
        # Cleanup
        for h in hooks:
            h.remove()
        del model
        torch.cuda.empty_cache()
    
    # Save config
    config_file = output_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"\nTrajectories saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract activation trajectories")
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="gru_experiment/data/trajectories")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    
    args = parser.parse_args()
    
    config = TrajectoryConfig(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        device=args.device,
        dtype=args.dtype
    )
    
    prepare_trajectories(config)
