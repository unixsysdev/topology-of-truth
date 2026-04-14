"""
Direct Answer Verifier

Verifies model answers against GSM8K gold answers WITHOUT using an external LLM.
Compares extracted numerical answers directly.

Enhanced: Also checks reasoning text for correct answers when boxed answer is missing.
"""

import json
import re
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass, asdict

from config import ExperimentConfig, MODELS, get_bucket, BUCKET_NAMES


def normalize_number(s: str) -> Optional[float]:
    """
    Normalize a string to a number for comparison.
    Handles: integers, decimals, fractions, percentages, commas, negatives.
    """
    if s is None:
        return None
    
    s = s.strip()
    
    # Remove commas (e.g., "1,000" -> "1000")
    s = s.replace(',', '')
    
    # Remove dollar signs, percent signs at the end
    s = s.replace('$', '').replace('%', '').strip()
    
    # Handle fractions like "3/4"
    if '/' in s and not any(c.isalpha() for c in s):
        try:
            parts = s.split('/')
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except (ValueError, ZeroDivisionError):
            pass
    
    # Handle mixed numbers like "1 1/2" -> 1.5
    mixed_match = re.match(r'^(-?\d+)\s+(\d+)/(\d+)$', s)
    if mixed_match:
        try:
            whole = int(mixed_match.group(1))
            num = int(mixed_match.group(2))
            denom = int(mixed_match.group(3))
            sign = -1 if whole < 0 else 1
            return sign * (abs(whole) + num / denom)
        except (ValueError, ZeroDivisionError):
            pass
    
    # Try direct float conversion
    try:
        return float(s)
    except ValueError:
        pass
    
    # Try extracting just the number part
    number_match = re.search(r'-?\d+\.?\d*', s)
    if number_match:
        try:
            return float(number_match.group())
        except ValueError:
            pass
    
    return None


def extract_final_number(text: str) -> Optional[str]:
    """
    Extract the final numerical answer from model output.
    Tries multiple formats: \\boxed{}, #### format, or last number.
    """
    if text is None:
        return None
    
    # Try \\boxed{...} format first (what models are asked to produce)
    boxed_pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    boxed_matches = re.findall(boxed_pattern, text)
    if boxed_matches:
        return boxed_matches[-1].strip()
    
    # Try #### format (GSM8K native format)
    hash_pattern = r'####\s*(-?[\d,]+(?:\.\d+)?)'
    hash_matches = re.findall(hash_pattern, text)
    if hash_matches:
        return hash_matches[-1].replace(',', '').strip()
    
    # Try "answer is X" or "= X" patterns
    answer_patterns = [
        r'(?:answer|result|total)\s*(?:is|=|:)\s*\$?\s*(-?[\d,]+(?:\.\d+)?)',
        r'=\s*\$?\s*(-?[\d,]+(?:\.\d+)?)\s*$',
        r'\$?\s*(-?[\d,]+(?:\.\d+)?)\s*(?:dollars?|clips?|items?|people|friends|hours?|minutes?|days?)?\.?\s*$'
    ]
    
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return matches[-1].replace(',', '').strip()
    
    return None


def check_answer_in_reasoning(text: str, gold_answer: str) -> bool:
    """
    Check if the correct answer appears as the FINAL CONCLUSION in the model's reasoning,
    even if it didn't produce a final \\boxed{} answer.
    
    This handles cases where the model was cut off before concluding.
    
    STRICT but FAIR: 
    - Look for clear conclusion patterns
    - If model confirms the answer (e.g., "So that's X"), credit it
    - If model is still exploring alternatives without settling, don't credit
    """
    if text is None or gold_answer is None:
        return False
    
    gold_num = normalize_number(gold_answer)
    if gold_num is None:
        return False
    
    # Convert to int string if it's a whole number for matching
    if gold_num == int(gold_num):
        gold_str = str(int(gold_num))
    else:
        gold_str = str(gold_num)
    
    # Look at the last 1000 characters for context
    last_part = text[-1000:] if len(text) > 1000 else text
    
    # Strong conclusion patterns - these indicate the model settled on an answer
    strong_patterns = [
        # "Therefore, the answer is X"
        rf'(?:Therefore|So|Thus|Hence)[,\s]+(?:the\s+)?answer\s+(?:is|=|:)\s*\$?{gold_str}\b',
        # "the answer is X"
        rf'\bthe\s+answer\s+is\s+\$?{gold_str}\b',
        # "So that's X hours/dollars" - confirmation pattern
        rf"(?:So\s+)?that'?s?\s+\$?{gold_str}\s+(?:hours?|dollars?|beetles?|ml|pages?)",
        # "= X hours" as final calculation
        rf'=\s*{gold_str}\s+(?:hours?|dollars?|beetles?|ml|pages?)\s*[.,\n]',
        # "total is X" or "result is X"
        rf'\b(?:total|result)\s+(?:is|=)\s*\$?{gold_str}\b',
    ]
    
    for pattern in strong_patterns:
        match = re.search(pattern, last_part, re.IGNORECASE)
        if match:
            # Check what comes IMMEDIATELY after (within 50 chars)
            after_pos = match.end()
            immediately_after = last_part[after_pos:after_pos+50]
            
            # If it's followed by confirmation, definitely correct
            if re.search(r"(?:so\s+)?that'?s?\s+(?:the\s+)?(?:same\s+)?answer", immediately_after, re.IGNORECASE):
                return True
            
            # If it's followed by "wait" or "but" IMMEDIATELY, it's doubt
            if re.search(r'^[\s.,]*(?:wait|but|hold on|actually|no,|hmm)', immediately_after, re.IGNORECASE):
                continue  # Try other patterns
            
            # Otherwise accept it - "Alternatively" later doesn't negate a clear statement
            return True
    
    return False


def answers_match(answer1: str, answer2: str, tolerance: float = 1e-6) -> bool:
    """
    Check if two answers are equivalent.
    """
    if answer1 is None or answer2 is None:
        return False
    
    num1 = normalize_number(answer1)
    num2 = normalize_number(answer2)
    
    if num1 is None or num2 is None:
        # Fall back to string comparison (case-insensitive, whitespace-normalized)
        return answer1.strip().lower() == answer2.strip().lower()
    
    # Numerical comparison with tolerance
    if num2 == 0:
        return abs(num1) < tolerance
    
    return abs(num1 - num2) < tolerance or abs((num1 - num2) / num2) < tolerance


@dataclass
class DirectVerificationResult:
    """Verification result for a single question."""
    question_id: str
    question: str
    gold_answer: str
    gold_normalized: Optional[float]
    model_4b_answer: Optional[str]
    model_4b_normalized: Optional[float]
    model_4b_found_in_reasoning: bool
    model_17b_answer: Optional[str]
    model_17b_normalized: Optional[float]
    model_17b_found_in_reasoning: bool
    model_06b_answer: Optional[str]
    model_06b_normalized: Optional[float]
    model_06b_found_in_reasoning: bool
    correct_4b: bool
    correct_17b: bool
    correct_06b: bool
    bucket: str
    bucket_name: str


def load_traces_with_text(config: ExperimentConfig) -> dict:
    """Load all traces with full generated text and organize by question_id."""
    traces_by_question = {}
    
    for model in MODELS:
        trace_file = config.traces_dir / f"traces_{model.short_name}.jsonl"
        if not trace_file.exists():
            print(f"Warning: {trace_file} not found")
            continue
        
        with open(trace_file) as f:
            for line in f:
                trace = json.loads(line)
                qid = trace['question_id']
                
                if qid not in traces_by_question:
                    traces_by_question[qid] = {
                        'question_id': qid,
                        'question': trace['question'],
                        'gold_answer': trace['gold_answer'],
                        'answers': {},
                        'generated_texts': {}
                    }
                
                traces_by_question[qid]['answers'][model.params] = trace.get('final_answer')
                traces_by_question[qid]['generated_texts'][model.params] = trace.get('generated_text', '')
    
    return traces_by_question


def run_direct_verification(config: ExperimentConfig, check_reasoning: bool = True):
    """Run direct verification without external LLM."""
    
    print("=" * 70)
    print("DIRECT VERIFICATION (No External LLM)")
    if check_reasoning:
        print("  + Checking reasoning for correct answers (strict mode)")
    print("=" * 70)
    
    # Load traces with full text
    print("\nLoading traces...")
    traces = load_traces_with_text(config)
    print(f"Found {len(traces)} questions")
    
    # Verify each question
    results = []
    correct_counts = {'4B': 0, '1.7B': 0, '0.6B': 0}
    reasoning_recoveries = {'4B': 0, '1.7B': 0, '0.6B': 0}
    bucket_counts = {}
    
    print("\nVerifying answers...")
    for qid in sorted(traces.keys()):
        data = traces[qid]
        
        gold = data['gold_answer']
        gold_norm = normalize_number(gold)
        
        ans_4b = data['answers'].get('4B')
        ans_17b = data['answers'].get('1.7B')
        ans_06b = data['answers'].get('0.6B')
        
        text_4b = data['generated_texts'].get('4B', '')
        text_17b = data['generated_texts'].get('1.7B', '')
        text_06b = data['generated_texts'].get('0.6B', '')
        
        # Check boxed answers first
        correct_4b = answers_match(ans_4b, gold) if ans_4b else False
        correct_17b = answers_match(ans_17b, gold) if ans_17b else False
        correct_06b = answers_match(ans_06b, gold) if ans_06b else False
        
        # Track if we found answer in reasoning (for cut-off cases)
        found_in_reasoning_4b = False
        found_in_reasoning_17b = False
        found_in_reasoning_06b = False
        
        # If no boxed answer, check reasoning
        if check_reasoning:
            if not correct_4b and ans_4b is None:
                found_in_reasoning_4b = check_answer_in_reasoning(text_4b, gold)
                if found_in_reasoning_4b:
                    correct_4b = True
                    reasoning_recoveries['4B'] += 1
            
            if not correct_17b and ans_17b is None:
                found_in_reasoning_17b = check_answer_in_reasoning(text_17b, gold)
                if found_in_reasoning_17b:
                    correct_17b = True
                    reasoning_recoveries['1.7B'] += 1
            
            if not correct_06b and ans_06b is None:
                found_in_reasoning_06b = check_answer_in_reasoning(text_06b, gold)
                if found_in_reasoning_06b:
                    correct_06b = True
                    reasoning_recoveries['0.6B'] += 1
        
        bucket = get_bucket(correct_4b, correct_17b, correct_06b)
        bucket_name = BUCKET_NAMES.get(bucket, '')
        
        result = DirectVerificationResult(
            question_id=qid,
            question=data['question'][:200] + "..." if len(data['question']) > 200 else data['question'],
            gold_answer=gold,
            gold_normalized=gold_norm,
            model_4b_answer=ans_4b,
            model_4b_normalized=normalize_number(ans_4b) if ans_4b else None,
            model_4b_found_in_reasoning=found_in_reasoning_4b,
            model_17b_answer=ans_17b,
            model_17b_normalized=normalize_number(ans_17b) if ans_17b else None,
            model_17b_found_in_reasoning=found_in_reasoning_17b,
            model_06b_answer=ans_06b,
            model_06b_normalized=normalize_number(ans_06b) if ans_06b else None,
            model_06b_found_in_reasoning=found_in_reasoning_06b,
            correct_4b=correct_4b,
            correct_17b=correct_17b,
            correct_06b=correct_06b,
            bucket=bucket,
            bucket_name=bucket_name
        )
        results.append(result)
        
        # Update counts
        if correct_4b:
            correct_counts['4B'] += 1
        if correct_17b:
            correct_counts['1.7B'] += 1
        if correct_06b:
            correct_counts['0.6B'] += 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
    
    # Save results
    output_file = config.output_dir / "verification_results.jsonl"
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + '\n')
    
    print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    n = len(results)
    
    print("\nModel Accuracy:")
    for model, count in correct_counts.items():
        pct = 100 * count / n if n > 0 else 0
        recovered = reasoning_recoveries[model]
        if recovered > 0:
            print(f"  {model}: {count}/{n} ({pct:.1f}%) [+{recovered} recovered from reasoning]")
        else:
            print(f"  {model}: {count}/{n} ({pct:.1f}%)")
    
    if any(reasoning_recoveries.values()):
        print(f"\n  * 'Recovered from reasoning' = correct answer found in thinking,")
        print(f"    but model was cut off before producing \\boxed{'{}'} answer")
    
    print("\nTruth Bucket Distribution:")
    for bucket in sorted(bucket_counts.keys()):
        count = bucket_counts[bucket]
        pct = 100 * count / n if n > 0 else 0
        name = BUCKET_NAMES.get(bucket, '')
        print(f"  {bucket} ({name}): {count} ({pct:.1f}%)")
    
    # Show detailed results
    print("\n" + "=" * 60)
    print("DETAILED RESULTS")
    print("=" * 60)
    
    for r in results:
        status_4b = "✓" if r.correct_4b else "✗"
        status_17b = "✓" if r.correct_17b else "✗"
        status_06b = "✓" if r.correct_06b else "✗"
        
        # Add indicator for reasoning recovery
        note_4b = " (from reasoning)" if r.model_4b_found_in_reasoning else ""
        note_17b = " (from reasoning)" if r.model_17b_found_in_reasoning else ""
        note_06b = " (from reasoning)" if r.model_06b_found_in_reasoning else ""
        
        print(f"\n{r.question_id} | Gold: {r.gold_answer} (={r.gold_normalized})")
        print(f"  4B:   {r.model_4b_answer} (={r.model_4b_normalized}) {status_4b}{note_4b}")
        print(f"  1.7B: {r.model_17b_answer} (={r.model_17b_normalized}) {status_17b}{note_17b}")
        print(f"  0.6B: {r.model_06b_answer} (={r.model_06b_normalized}) {status_06b}{note_06b}")
        print(f"  Bucket: {r.bucket} - {r.bucket_name}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Direct answer verification")
    parser.add_argument("--output-dir", type=str, default="./results_thinking",
                       help="Directory containing traces")
    parser.add_argument("--no-reasoning-check", action="store_true",
                       help="Don't check reasoning for answers (strict mode)")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    config.output_dir = Path(args.output_dir)
    config.traces_dir = config.output_dir / "traces"
    config.embeddings_dir = config.output_dir / "embeddings"
    config.tda_dir = config.output_dir / "tda"
    config.viz_dir = config.output_dir / "visualizations"
    
    run_direct_verification(config, check_reasoning=not args.no_reasoning_check)
