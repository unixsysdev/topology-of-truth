"""
Module A: Trace Generator (The Miner) - FAST VERSION

Generates reasoning traces from all 3 models on LIMO dataset,
extracting both the text output AND the residual stream activations.

PHASE 1: Last layer only - cleanest signal for topology
PHASE 2 (later): Multi-layer to see where divergence happens

Key optimizations:
1. NO HOOKS during generation - they kill performance
2. Single forward pass AFTER generation to get activations
3. Last layer only (Phase 1)
4. AMD ROCm compatible (no SDPA)
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional, List
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
import os

from config import MODELS, ExperimentConfig, ModelConfig


# AMD ROCm optimizations
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:True")


@dataclass
class TraceResult:
    """Result from a single model on a single question."""
    question_id: str
    model_name: str
    model_params: str
    question: str
    gold_answer: str
    generated_text: str
    final_answer: Optional[str]
    num_tokens: int
    num_reasoning_steps: int
    activations_path: Optional[str] = None


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{...} format."""
    import re
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract final answer from GSM8K format (#### NUMBER)."""
    import re
    pattern = r'####\s*(-?[\d,]+(?:\.\d+)?)'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].replace(',', '').strip()
    return None


def count_reasoning_steps(text: str) -> int:
    """Count reasoning steps by newlines and markers."""
    import re
    steps = re.split(r'\n+|(?:^|\n)\s*\d+[\.\)]\s*|(?:^|\n)\s*[-•]\s*', text)
    steps = [s.strip() for s in steps if s.strip() and len(s.strip()) > 10]
    return len(steps)


def load_model(model_config: ModelConfig, device: str = "cuda"):
    """Load model optimized for AMD ROCm."""
    print(f"Loading {model_config.hf_id}...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.hf_id,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_config.hf_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",  # Disable SDPA for AMD
    )
    model.eval()
    
    return model, tokenizer


@torch.no_grad()
def generate_and_extract(
    model,
    tokenizer,
    prompt: str,
    config: ExperimentConfig,
) -> tuple[str, np.ndarray]:
    """
    Generate text, then do ONE forward pass to extract LAST LAYER activations.
    
    Returns:
        generated_text: The model's response
        activations: (seq_len, hidden_dim) array - last layer only
    """
    
    # Format as chat
    messages = [
        {"role": "system", "content": config.system_prompt},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]
    
    # === PHASE 1: Generate (fast, no hooks) ===
    outputs = model.generate(
        **inputs,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Decode generated text
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # === PHASE 2: Single forward pass for LAST LAYER activations ===
    full_sequence = outputs[0:1]  # Keep batch dim
    
    forward_outputs = model(
        full_sequence,
        output_hidden_states=True,
        use_cache=False,
    )
    
    # Get LAST LAYER hidden states for GENERATED tokens only
    hidden_states = forward_outputs.hidden_states  # Tuple of (batch, seq, hidden)
    last_layer = hidden_states[-1]  # (1, seq_len, hidden_dim)
    
    # Extract only the generated part (not the prompt)
    activations = last_layer[0, input_length:, :].cpu().float().numpy()  # (gen_len, hidden_dim)
    
    # Clean up
    del forward_outputs, hidden_states, last_layer
    
    return generated_text, activations


def save_activations(activations: np.ndarray, question_id: str, model_name: str, output_dir: Path) -> str:
    """Save activations to compressed numpy file."""
    filename = f"{question_id}_{model_name}_activations.npz"
    filepath = output_dir / filename
    # Save as single "last_layer" key for consistency with TDA pipeline
    np.savez_compressed(filepath, last_layer=activations)
    return str(filepath)


def run_trace_generation(
    config: ExperimentConfig,
    models_to_run: Optional[list[str]] = None,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
):
    """
    Main entry point: Generate traces for all models on LIMO.
    """
    # Load dataset
    print(f"Loading dataset: {config.dataset_name}")
    dataset = load_dataset(config.dataset_name, "main", split=config.dataset_split) if "gsm8k" in config.dataset_name else load_dataset(config.dataset_name, split=config.dataset_split)
    
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
    
    if end_idx is None:
        end_idx = len(dataset)
    
    dataset = dataset.select(range(start_idx, min(end_idx, len(dataset))))
    print(f"Processing questions {start_idx} to {end_idx} ({len(dataset)} total)")
    
    # Filter models
    models_list = MODELS
    if models_to_run:
        models_list = [m for m in MODELS if m.params in models_to_run]
    
    # Process each model
    for model_config in models_list:
        print(f"\n{'='*60}")
        print(f"Processing: {model_config.name} ({model_config.params})")
        print(f"{'='*60}")
        
        # Load model
        model, tokenizer = load_model(model_config)
        
        n_layers = model.config.num_hidden_layers
        print(f"Model has {n_layers} layers. Extracting LAST LAYER only (Phase 1).")
        
        # Output file
        output_file = config.traces_dir / f"traces_{model_config.short_name}.jsonl"
        
        # Check for existing progress
        existing_ids = set()
        if output_file.exists():
            with open(output_file) as f:
                for line in f:
                    try:
                        existing_ids.add(json.loads(line)['question_id'])
                    except:
                        pass
            print(f"Resuming: {len(existing_ids)} already processed")
        
        # Process questions
        with open(output_file, "a") as f:
            for idx, item in enumerate(tqdm(dataset, desc=f"Generating {model_config.params}")):
                question_id = f"q_{start_idx + idx:04d}"
                
                # Skip if already done
                if question_id in existing_ids:
                    continue
                
                # Get question and gold answer
                question = item.get("question", item.get("problem", ""))
                gold_answer_raw = item.get("answer", item.get("solution", ""))
                # For GSM8K, extract the number after ####
                gold_answer = extract_gsm8k_answer(gold_answer_raw) or gold_answer_raw
                
                try:
                    # Generate + extract activations (FAST - last layer only)
                    generated_text, activations = generate_and_extract(
                        model, tokenizer, question, config
                    )
                    
                    # Save activations
                    act_path = save_activations(
                        activations, question_id, model_config.short_name,
                        config.embeddings_dir
                    )
                    
                    # Create result
                    result = TraceResult(
                        question_id=question_id,
                        model_name=model_config.name,
                        model_params=model_config.params,
                        question=question,
                        gold_answer=gold_answer,
                        generated_text=generated_text,
                        final_answer=extract_boxed_answer(generated_text),
                        num_tokens=len(activations),  # = number of generated tokens
                        num_reasoning_steps=count_reasoning_steps(generated_text),
                        activations_path=act_path,
                    )
                    
                    # Write to JSONL
                    f.write(json.dumps(asdict(result)) + "\n")
                    f.flush()
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"\nOOM on {question_id}, clearing cache and skipping...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                    
                except Exception as e:
                    print(f"\nError on {question_id}: {e}")
                    continue
                
                # Periodic cache clear
                if (idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()
        
        # Free memory before next model
        print(f"Cleaning up {model_config.params}...")
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    print("\nTrace generation complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate reasoning traces (FAST)")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to run (e.g., 4B 1.7B 0.6B)")
    parser.add_argument("--start", type=int, default=0,
                       help="Starting question index")
    parser.add_argument("--end", type=int, default=None,
                       help="Ending question index")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Max samples to process")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    if args.max_samples:
        config.max_samples = args.max_samples
    
    run_trace_generation(
        config,
        models_to_run=args.models,
        start_idx=args.start,
        end_idx=args.end
    )
