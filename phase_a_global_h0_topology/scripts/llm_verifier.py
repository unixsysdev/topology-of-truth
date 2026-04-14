"""
LLM-based Answer Verifier (Batch Version)

Uses a large model to verify answers in batches, comparing model output
against gold answer. This provides a second opinion on our direct verification.
"""

import json
import os
import time
import requests
from pathlib import Path
from typing import Optional, List

from config import ExperimentConfig, MODELS, get_bucket, BUCKET_NAMES


# API Configuration
CHUTES_API_URL = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.environ.get("CHUTES_API_TOKEN", "")
VERIFIER_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"


def build_batch_prompt(batch: List[dict]) -> str:
    """Build a prompt for batch verification of multiple question-answer pairs."""
    
    prompt = """You are a precise math answer verifier. I will give you several math problems with:
- The question
- The correct (gold) answer  
- The student's final boxed answer (may be None if they didn't finish)
- A snippet of the student's reasoning (may be truncated)

For each problem, determine if the student got the correct answer. Consider:
1. If student has a boxed answer, compare it numerically to gold (72 = 72.0 = 72.00)
2. If no boxed answer (None), check if they clearly arrived at the correct answer in reasoning
3. Be strict: only mark correct if the answer clearly matches

Return ONLY a JSON array with your verdicts (no other text):
```json
[
  {"id": "q_0000_4B", "correct": true, "student_value": "72"},
  {"id": "q_0000_1.7B", "correct": false, "student_value": "36"},
  ...
]
```

Here are the problems:

"""
    
    for item in batch:
        # Truncate reasoning to last 800 chars
        reasoning = item.get('reasoning', '')
        if len(reasoning) > 800:
            reasoning = "..." + reasoning[-800:]
        
        prompt += f"""
---
ID: {item['id']}
Question: {item['question'][:300]}{'...' if len(item['question']) > 300 else ''}
Gold Answer: {item['gold']}
Student's Boxed Answer: {item['answer'] if item['answer'] else 'None (no boxed answer)'}
Student's Reasoning (snippet): {reasoning[-500:] if reasoning else 'N/A'}

"""
    
    prompt += "\nNow provide the JSON array with your verdicts:"
    return prompt


def call_verifier_api(prompt: str, max_retries: int = 3) -> Optional[List[dict]]:
    """Call the API and parse the response."""
    
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": VERIFIER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 4000,
        "temperature": 0.1,
    }
    
    for attempt in range(max_retries):
        try:
            print(f"    API call attempt {attempt+1}...")
            response = requests.post(CHUTES_API_URL, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            data = response.json()
            
            # Debug: print response structure
            if "choices" not in data:
                print(f"    Warning: No 'choices' in response. Keys: {data.keys()}")
                if "error" in data:
                    print(f"    Error: {data['error']}")
                continue
            
            if not data["choices"]:
                print(f"    Warning: Empty choices array")
                continue
            
            message = data["choices"][0].get("message", {})
            content = message.get("content")
            
            if content is None:
                print(f"    Warning: Content is None. Message keys: {message.keys()}")
                # Check if there's a refusal or other issue
                if "refusal" in message:
                    print(f"    Refusal: {message['refusal']}")
                # Try waiting and retrying
                time.sleep(5)
                continue
            
            # Extract JSON from response (handle thinking tags if present)
            if "</think>" in content:
                content = content.split("</think>")[-1].strip()
            
            # Find JSON array in response
            import re
            json_match = re.search(r'\[[\s\S]*?\]', content)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    return result
                except json.JSONDecodeError as e:
                    print(f"    JSON parse error: {e}")
                    print(f"    Matched text: {json_match.group()[:200]}...")
            else:
                print(f"    Warning: Could not find JSON array in response")
                print(f"    Response preview: {content[:500]}...")
                
        except requests.exceptions.Timeout:
            print(f"    Attempt {attempt+1} timed out")
        except requests.exceptions.RequestException as e:
            print(f"    Attempt {attempt+1} failed (request error): {e}")
        except Exception as e:
            print(f"    Attempt {attempt+1} failed: {type(e).__name__}: {e}")
        
        if attempt < max_retries - 1:
            wait_time = (attempt + 1) * 5
            print(f"    Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return None


def load_traces_with_text(config: ExperimentConfig) -> dict:
    """Load all traces with full generated text."""
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


def load_direct_verification(config: ExperimentConfig) -> dict:
    """Load our direct verification results for comparison."""
    results = {}
    results_file = config.output_dir / "verification_results.jsonl"
    
    if results_file.exists():
        with open(results_file) as f:
            for line in f:
                r = json.loads(line)
                results[r['question_id']] = r
    
    return results


def run_llm_verification(config: ExperimentConfig, batch_size: int = 10):
    """Run LLM-based verification in batches and compare with direct verification."""
    
    print("=" * 70)
    print("LLM-BASED ANSWER VERIFICATION (Batch Mode)")
    print("=" * 70)
    
    # Load traces
    print("\nLoading traces...")
    traces = load_traces_with_text(config)
    print(f"Found {len(traces)} questions")
    
    # Load direct verification for comparison
    direct_results = load_direct_verification(config)
    print(f"Loaded {len(direct_results)} direct verification results")
    
    # Build list of all items to verify
    all_items = []
    for qid in sorted(traces.keys()):
        data = traces[qid]
        for model_params in ['4B', '1.7B', '0.6B']:
            item = {
                'id': f"{qid}_{model_params}",
                'qid': qid,
                'model': model_params,
                'question': data['question'],
                'gold': data['gold_answer'],
                'answer': data['answers'].get(model_params),
                'reasoning': data['generated_texts'].get(model_params, '')
            }
            all_items.append(item)
    
    print(f"Total items to verify: {len(all_items)}")
    print(f"Batch size: {batch_size}")
    print(f"Number of batches: {(len(all_items) + batch_size - 1) // batch_size}")
    
    # Process in batches
    llm_verdicts = {}  # id -> verdict
    
    for i in range(0, len(all_items), batch_size):
        batch = all_items[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(all_items) + batch_size - 1) // batch_size
        
        print(f"\nProcessing batch {batch_num}/{total_batches}...")
        
        prompt = build_batch_prompt(batch)
        results = call_verifier_api(prompt)
        
        if results:
            for r in results:
                item_id = r.get('id')
                if item_id:
                    llm_verdicts[item_id] = {
                        'correct': r.get('correct'),
                        'student_value': r.get('student_value')
                    }
            print(f"  Got {len(results)} verdicts")
        else:
            print(f"  WARNING: Batch {batch_num} failed!")
        
        # Rate limiting
        time.sleep(1)
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON: LLM vs Direct Verification")
    print("=" * 70)
    
    model_stats = {
        '4B': {'llm_correct': 0, 'direct_correct': 0, 'agree': 0, 'total': 0},
        '1.7B': {'llm_correct': 0, 'direct_correct': 0, 'agree': 0, 'total': 0},
        '0.6B': {'llm_correct': 0, 'direct_correct': 0, 'agree': 0, 'total': 0}
    }
    
    disagreements = []
    
    for item in all_items:
        item_id = item['id']
        qid = item['qid']
        model = item['model']
        
        llm_verdict = llm_verdicts.get(item_id, {})
        llm_correct = llm_verdict.get('correct')
        
        # Get direct verification
        direct = direct_results.get(qid, {})
        if model == '4B':
            direct_correct = direct.get('correct_4b')
        elif model == '1.7B':
            direct_correct = direct.get('correct_17b')
        else:
            direct_correct = direct.get('correct_06b')
        
        model_stats[model]['total'] += 1
        if llm_correct:
            model_stats[model]['llm_correct'] += 1
        if direct_correct:
            model_stats[model]['direct_correct'] += 1
        if llm_correct == direct_correct:
            model_stats[model]['agree'] += 1
        else:
            disagreements.append({
                'id': item_id,
                'qid': qid,
                'model': model,
                'gold': item['gold'],
                'answer': item['answer'],
                'llm_correct': llm_correct,
                'llm_value': llm_verdict.get('student_value'),
                'direct_correct': direct_correct
            })
    
    # Print summary table
    print("\n{:<8} {:>12} {:>12} {:>12} {:>12}".format(
        "Model", "LLM Correct", "Direct Corr", "Agreement", "Total"))
    print("-" * 60)
    
    for model in ['4B', '1.7B', '0.6B']:
        stats = model_stats[model]
        n = stats['total']
        llm_pct = 100 * stats['llm_correct'] / n if n else 0
        direct_pct = 100 * stats['direct_correct'] / n if n else 0
        agree_pct = 100 * stats['agree'] / n if n else 0
        
        print("{:<8} {:>5}/{} ({:>5.1f}%) {:>5}/{} ({:>5.1f}%) {:>5}/{} ({:>5.1f}%)".format(
            model,
            stats['llm_correct'], n, llm_pct,
            stats['direct_correct'], n, direct_pct,
            stats['agree'], n, agree_pct
        ))
    
    # Print disagreements
    if disagreements:
        print(f"\n" + "=" * 70)
        print(f"DISAGREEMENTS ({len(disagreements)} cases)")
        print("=" * 70)
        
        for d in disagreements:
            print(f"\n{d['id']}:")
            print(f"  Gold: {d['gold']}")
            print(f"  Model answer: {d['answer']}")
            print(f"  LLM says: {d['llm_correct']} (value: {d['llm_value']})")
            print(f"  Direct says: {d['direct_correct']}")
    else:
        print("\n✓ Perfect agreement between LLM and direct verification!")
    
    # Save full results
    output = {
        'llm_verdicts': llm_verdicts,
        'model_stats': model_stats,
        'disagreements': disagreements
    }
    
    output_file = config.output_dir / "llm_verification_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull results saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM-based answer verification")
    parser.add_argument("--output-dir", type=str, default="./results_thinking",
                       help="Directory containing traces")
    parser.add_argument("--batch-size", type=int, default=5,
                       help="Number of items per API call")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    config.output_dir = Path(args.output_dir)
    config.traces_dir = config.output_dir / "traces"
    
    run_llm_verification(config, batch_size=args.batch_size)
