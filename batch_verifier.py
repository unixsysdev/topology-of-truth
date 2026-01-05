"""
Module B: Batch Verifier (The Judge)

Sends batches of answers to Qwen3-235B via Chutes API to determine
which models got each question correct.

Outputs the 8 truth buckets for each question.
"""

import json
import os
import time
import requests
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional

from config import ExperimentConfig, MODELS, get_bucket, BUCKET_NAMES


# API Configuration
CHUTES_API_URL = "https://llm.chutes.ai/v1/chat/completions"
CHUTES_API_KEY = os.environ.get(
    "CHUTES_API_TOKEN",
    "cpk_17f9b37e060244088f988d19519233e0.370d8ad91ab3510687f98806820f1062.5gbrZsdtzbeXmwgETzAJlTdH2aKw3Dwb"
)
VERIFIER_MODEL = "Qwen/Qwen3-235B-A22B-Thinking-2507"


@dataclass
class VerificationResult:
    """Verification result for a single question."""
    question_id: str
    question: str
    gold_answer: str
    model_4b_answer: Optional[str]
    model_17b_answer: Optional[str]
    model_06b_answer: Optional[str]
    correct_4b: bool
    correct_17b: bool
    correct_06b: bool
    bucket: str
    bucket_name: str
    verifier_reasoning: str


def build_verification_prompt(batch: list[dict]) -> str:
    """
    Build a prompt for batch verification of multiple questions.
    
    Each item in batch should have:
      - question_id
      - question
      - gold_answer
      - answers: {model_param: answer_text}
    """
    prompt = """You are a precise mathematical answer verifier. I will give you multiple math problems, each with:
- The original question
- The correct (gold) answer
- Three student answers from models of different sizes (4B, 1.7B, 0.6B parameters)

For each problem, determine if each student's final answer is mathematically equivalent to the gold answer.
- Focus ONLY on the final numerical/expression answer, not the reasoning process
- Consider answers equivalent if they simplify to the same value
- If a student answer is missing or unclear, mark it as incorrect

Return your response as a JSON array with one object per question:
```json
[
  {
    "question_id": "q_0001",
    "4B": true,
    "1.7B": false,
    "0.6B": false,
    "reasoning": "Brief explanation of your judgments"
  },
  ...
]
```

Here are the problems to verify:

"""
    
    for i, item in enumerate(batch):
        prompt += f"""
---
PROBLEM {i+1} (ID: {item['question_id']})

Question: {item['question'][:500]}...

Gold Answer: {item['gold_answer']}

Student Answers:
- 4B Model: {item['answers'].get('4B', 'NO ANSWER')}
- 1.7B Model: {item['answers'].get('1.7B', 'NO ANSWER')}
- 0.6B Model: {item['answers'].get('0.6B', 'NO ANSWER')}

"""
    
    prompt += "\nNow provide the JSON verification array:"
    return prompt


def call_verifier_api(prompt: str, max_retries: int = 3) -> dict:
    """Call the Chutes API for verification."""
    
    headers = {
        "Authorization": f"Bearer {CHUTES_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": VERIFIER_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False,
        "max_tokens": 4096,
        "temperature": 0.1,  # Low temp for consistent judgments
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                CHUTES_API_URL,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"API call failed (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))  # Exponential backoff
            else:
                raise


def parse_verification_response(response_text: str) -> list[dict]:
    """Parse JSON array from verifier response."""
    import re
    
    # Try to find JSON array in response
    # Handle markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON array
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            raise ValueError(f"Could not find JSON array in response: {response_text[:500]}")
    
    return json.loads(json_str)


def load_traces(traces_dir: Path) -> dict[str, dict]:
    """
    Load all traces and organize by question_id.
    
    Returns:
        {question_id: {
            'question': str,
            'gold_answer': str,
            'answers': {model_param: answer},
            'traces': {model_param: full_trace}
        }}
    """
    questions = {}
    
    for model in MODELS:
        trace_file = traces_dir / f"traces_{model.short_name}.jsonl"
        if not trace_file.exists():
            print(f"Warning: {trace_file} not found")
            continue
            
        with open(trace_file) as f:
            for line in f:
                trace = json.loads(line)
                qid = trace['question_id']
                
                if qid not in questions:
                    questions[qid] = {
                        'question_id': qid,
                        'question': trace['question'],
                        'gold_answer': trace['gold_answer'],
                        'answers': {},
                        'traces': {},
                    }
                
                questions[qid]['answers'][model.params] = trace.get('final_answer', '')
                questions[qid]['traces'][model.params] = trace
    
    return questions


def run_batch_verification(
    config: ExperimentConfig,
    batch_size: int = 10,
    dry_run: bool = False,
):
    """
    Main entry point: Verify all traces in batches.
    
    Args:
        config: Experiment configuration
        batch_size: Questions per API call
        dry_run: If True, don't call API, just show what would be sent
    """
    # Load all traces
    print("Loading traces...")
    questions = load_traces(config.traces_dir)
    print(f"Found {len(questions)} questions with traces")
    
    # Check which questions have all 3 models
    complete_questions = {
        qid: q for qid, q in questions.items()
        if len(q['answers']) == 3
    }
    print(f"{len(complete_questions)} questions have all 3 model answers")
    
    # Sort by question_id for reproducibility
    sorted_qids = sorted(complete_questions.keys())
    
    # Process in batches
    results = []
    output_file = config.output_dir / "verification_results.jsonl"
    
    for batch_start in tqdm(range(0, len(sorted_qids), batch_size), desc="Verifying"):
        batch_qids = sorted_qids[batch_start:batch_start + batch_size]
        batch = [complete_questions[qid] for qid in batch_qids]
        
        # Build prompt
        prompt = build_verification_prompt(batch)
        
        if dry_run:
            print(f"\n{'='*60}")
            print(f"BATCH {batch_start // batch_size + 1}")
            print(f"{'='*60}")
            print(prompt[:2000] + "...")
            continue
        
        # Call API
        try:
            response = call_verifier_api(prompt)
            response_text = response['choices'][0]['message']['content']
            
            # Parse results
            verifications = parse_verification_response(response_text)
            
            # Match with questions and create results
            for v in verifications:
                qid = v.get('question_id', '')
                if qid not in complete_questions:
                    # Try to match by index
                    continue
                
                q = complete_questions[qid]
                
                correct_4b = v.get('4B', False)
                correct_17b = v.get('1.7B', False)
                correct_06b = v.get('0.6B', False)
                
                bucket = get_bucket(correct_4b, correct_17b, correct_06b)
                
                result = VerificationResult(
                    question_id=qid,
                    question=q['question'][:500],
                    gold_answer=q['gold_answer'],
                    model_4b_answer=q['answers'].get('4B'),
                    model_17b_answer=q['answers'].get('1.7B'),
                    model_06b_answer=q['answers'].get('0.6B'),
                    correct_4b=correct_4b,
                    correct_17b=correct_17b,
                    correct_06b=correct_06b,
                    bucket=bucket,
                    bucket_name=BUCKET_NAMES[bucket],
                    verifier_reasoning=v.get('reasoning', ''),
                )
                
                results.append(result)
                
                # Write immediately
                with open(output_file, 'a') as f:
                    f.write(json.dumps(asdict(result)) + "\n")
            
        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
            continue
        
        # Rate limiting
        time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    bucket_counts = {}
    for r in results:
        bucket_counts[r.bucket] = bucket_counts.get(r.bucket, 0) + 1
    
    for bucket in sorted(bucket_counts.keys()):
        count = bucket_counts[bucket]
        name = BUCKET_NAMES[bucket]
        pct = 100 * count / len(results) if results else 0
        print(f"  {bucket} ({name}): {count} ({pct:.1f}%)")
    
    print(f"\nResults saved to: {output_file}")
    return results


def load_verification_results(config: ExperimentConfig) -> list[VerificationResult]:
    """Load verification results from file."""
    results = []
    results_file = config.output_dir / "verification_results.jsonl"
    
    if not results_file.exists():
        return results
    
    with open(results_file) as f:
        for line in f:
            data = json.loads(line)
            results.append(VerificationResult(**data))
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify model answers")
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Questions per API call")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show prompts without calling API")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    run_batch_verification(
        config,
        batch_size=args.batch_size,
        dry_run=args.dry_run
    )
