#!/usr/bin/env python
"""
Generate answers with the student model and label correctness.

Runs the 0.6B model on GSM8K, extracts answers, compares to gold.
Outputs a JSON file that training can load to apply selective intervention.

Usage:
    python gru_experiment/scripts/label_correctness.py
    python gru_experiment/scripts/label_correctness.py --max-samples 500
    python gru_experiment/scripts/label_correctness.py --model Qwen/Qwen3-0.6B --output labels.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import re
import argparse
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def extract_answer(text: str) -> str:
    """
    Extract numerical answer from model output.
    
    Only accepts properly formatted answers:
    1. \\boxed{...} format (preferred)
    2. "answer is X" pattern
    
    Returns empty string if no clear answer found - this counts as wrong.
    """
    # Try \boxed{} first - this is the format we ask for
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return clean_number(boxed_match.group(1))
    
    # Try "the answer is X" or "final answer: X" pattern
    answer_match = re.search(
        r'(?:final\s+)?(?:answer|result)\s*(?:is|=|:)\s*\$?([+-]?[\d,]+\.?\d*)',
        text, 
        re.IGNORECASE
    )
    if answer_match:
        return clean_number(answer_match.group(1))
    
    # No clear answer found - don't guess
    return ""


def clean_number(s: str) -> str:
    """Clean a number string for comparison."""
    # Remove commas, whitespace, dollar signs
    s = s.replace(',', '').replace('$', '').strip()
    
    # Try to normalize to float then back to clean string
    try:
        num = float(s)
        # If it's a whole number, return as int string
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        return s


def extract_gold_answer(answer_text: str) -> str:
    """Extract gold answer from GSM8K answer field."""
    # GSM8K format: "... #### 42"
    if '####' in answer_text:
        return clean_number(answer_text.split('####')[-1].strip())
    return clean_number(answer_text)


def main():
    parser = argparse.ArgumentParser(description="Label student model correctness on GSM8K")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model to evaluate")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k",
                        help="Dataset to use")
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Number of samples to evaluate")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens to generate (512 is enough for most GSM8K)")
    parser.add_argument("--output", type=str, default="gru_experiment/data/correctness_labels.json",
                        help="Output JSON file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    print("=" * 60)
    print("Labeling Student Model Correctness")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device,
        trust_remote_code=True
    )
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, "main", split="train")
    
    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
    
    print(f"Evaluating {len(dataset)} samples")
    
    # Generate and evaluate
    results = {}
    correct_count = 0
    
    for idx, item in enumerate(tqdm(dataset, desc="Generating")):
        question = item['question']
        gold_answer = extract_gold_answer(item['answer'])
        
        # Format prompt - ask for boxed answer explicitly
        prompt = f"Solve this math problem step by step. Put your final answer in \\boxed{{}}.\n\nQuestion: {question}\n\nAnswer:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(args.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        
        # Generate with early stopping
        # Use greedy decoding for consistency (no sampling)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0,
                do_sample=False,  # Greedy - faster and deterministic
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode
        generated = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        predicted_answer = extract_answer(generated)
        
        # Check correctness
        is_correct = (predicted_answer == gold_answer)
        if is_correct:
            correct_count += 1
        
        # Store result
        question_id = f"q_{idx:04d}"
        results[question_id] = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
            "correct": is_correct,
            "generated_text": generated[:500]  # Truncate to save space
        }
        
        # Print progress every 50
        if (idx + 1) % 50 == 0:
            acc = correct_count / (idx + 1) * 100
            print(f"\n  Progress: {idx + 1}/{len(dataset)}, Accuracy so far: {acc:.1f}%")
    
    # Summary
    accuracy = correct_count / len(dataset) * 100
    num_correct = correct_count
    num_incorrect = len(dataset) - correct_count
    
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(dataset)}")
    print(f"Correct: {num_correct} ({accuracy:.1f}%)")
    print(f"Incorrect: {num_incorrect} ({100-accuracy:.1f}%)")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "model": args.model,
            "dataset": args.dataset,
            "num_samples": len(dataset),
            "num_correct": num_correct,
            "num_incorrect": num_incorrect,
            "accuracy": accuracy
        },
        "labels": results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved to: {output_path}")
    
    # Show some examples
    print(f"\n{'=' * 60}")
    print("EXAMPLES")
    print(f"{'=' * 60}")
    
    # Show 2 correct and 2 incorrect
    correct_examples = [(k, v) for k, v in results.items() if v['correct']][:2]
    incorrect_examples = [(k, v) for k, v in results.items() if not v['correct']][:2]
    
    print("\n--- CORRECT ---")
    for qid, data in correct_examples:
        print(f"\n{qid}: {data['question'][:80]}...")
        print(f"  Gold: {data['gold_answer']}, Predicted: {data['predicted_answer']} ✓")
    
    print("\n--- INCORRECT ---")
    for qid, data in incorrect_examples:
        print(f"\n{qid}: {data['question'][:80]}...")
        print(f"  Gold: {data['gold_answer']}, Predicted: {data['predicted_answer']} ✗")


if __name__ == "__main__":
    main()
