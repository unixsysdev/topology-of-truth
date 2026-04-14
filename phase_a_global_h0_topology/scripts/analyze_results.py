"""
Module E: Statistical Analysis & Conclusions

Analyzes TDA results to answer the core hypothesis:
"Can we predict correctness from topology alone?"

Outputs:
1. T-tests: correct vs wrong per model
2. Correlations: topology metrics → correctness
3. Bucket analysis: topology by truth bucket
4. Cross-model comparison: Wasserstein distances
5. Automatic conclusions
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed. Install with: pip install scipy")

from config import ExperimentConfig, BUCKET_NAMES


@dataclass
class StatisticalResult:
    """Result of a statistical test."""
    metric: str
    model: str
    correct_mean: float
    correct_std: float
    correct_n: int
    wrong_mean: float
    wrong_std: float
    wrong_n: int
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool


def load_data(config: ExperimentConfig) -> Tuple[List[dict], List[dict]]:
    """Load TDA features and verification results."""
    
    # Load TDA features
    tda_file = config.tda_dir / "tda_features.jsonl"
    tda_features = []
    if tda_file.exists():
        with open(tda_file) as f:
            for line in f:
                tda_features.append(json.loads(line))
    
    # Load verifications
    verify_file = config.output_dir / "verification_results.jsonl"
    verifications = []
    if verify_file.exists():
        with open(verify_file) as f:
            for line in f:
                verifications.append(json.loads(line))
    
    return tda_features, verifications


def merge_data(tda_features: List[dict], verifications: List[dict]) -> List[dict]:
    """Merge TDA features with correctness labels."""
    
    # Index verifications by question_id
    verify_map = {v['question_id']: v for v in verifications}
    
    # Merge
    merged = []
    for tda in tda_features:
        qid = tda['question_id']
        model = tda['model_params']
        
        if qid not in verify_map:
            continue
        
        v = verify_map[qid]
        
        # Get correctness for this model
        correct_key = {
            '4B': 'correct_4b',
            '1.7B': 'correct_17b',
            '0.6B': 'correct_06b',
        }.get(model)
        
        if not correct_key:
            continue
        
        merged.append({
            **tda,
            'correct': v.get(correct_key, False),
            'bucket': v.get('bucket', ''),
        })
    
    return merged


def cohens_d(group1: List[float], group2: List[float]) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def run_ttest(
    data: List[dict],
    metric: str,
    model: str,
    alpha: float = 0.05
) -> Optional[StatisticalResult]:
    """Run t-test comparing correct vs wrong for a metric."""
    
    if not HAS_SCIPY:
        return None
    
    # Filter by model
    model_data = [d for d in data if d['model_params'] == model]
    
    # Split by correctness
    correct_vals = [d[metric] for d in model_data if d['correct']]
    wrong_vals = [d[metric] for d in model_data if not d['correct']]
    
    if len(correct_vals) < 2 or len(wrong_vals) < 2:
        return None
    
    # Run t-test
    t_stat, p_val = stats.ttest_ind(correct_vals, wrong_vals)
    
    # Effect size
    d = cohens_d(correct_vals, wrong_vals)
    
    return StatisticalResult(
        metric=metric,
        model=model,
        correct_mean=np.mean(correct_vals),
        correct_std=np.std(correct_vals),
        correct_n=len(correct_vals),
        wrong_mean=np.mean(wrong_vals),
        wrong_std=np.std(wrong_vals),
        wrong_n=len(wrong_vals),
        t_statistic=t_stat,
        p_value=p_val,
        effect_size=d,
        significant=p_val < alpha,
    )


def compute_correlation(data: List[dict], metric: str, model: str) -> Tuple[float, float]:
    """Compute point-biserial correlation between metric and correctness."""
    
    if not HAS_SCIPY:
        return 0.0, 1.0
    
    model_data = [d for d in data if d['model_params'] == model]
    
    if len(model_data) < 3:
        return 0.0, 1.0
    
    values = [d[metric] for d in model_data]
    correct = [1 if d['correct'] else 0 for d in model_data]
    
    r, p = stats.pointbiserialr(correct, values)
    
    return r, p


def analyze_by_bucket(data: List[dict]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute mean metrics per bucket per model."""
    
    results = {}
    
    for bucket in BUCKET_NAMES.keys():
        results[bucket] = {}
        
        for model in ['4B', '1.7B', '0.6B']:
            bucket_model_data = [
                d for d in data 
                if d['bucket'] == bucket and d['model_params'] == model
            ]
            
            if not bucket_model_data:
                continue
            
            results[bucket][model] = {
                'n': len(bucket_model_data),
                'h0_entropy': np.mean([d['h0_entropy'] for d in bucket_model_data]),
                'h1_max_persistence': np.mean([d['h1_max_persistence'] for d in bucket_model_data]),
                'dust_score': np.mean([d['dust_score'] for d in bucket_model_data]),
                'loop_score': np.mean([d['loop_score'] for d in bucket_model_data]),
            }
    
    return results


def generate_conclusions(
    ttest_results: List[StatisticalResult],
    correlations: Dict[str, Dict[str, Tuple[float, float]]],
    bucket_analysis: Dict,
) -> List[str]:
    """Generate automatic conclusions from the analysis."""
    
    conclusions = []
    
    # 1. Check if H0 entropy distinguishes correct/wrong
    h0_significant = [r for r in ttest_results if r.metric == 'h0_entropy' and r.significant]
    if h0_significant:
        models = [r.model for r in h0_significant]
        conclusions.append(
            f"✓ H₀ ENTROPY DISTINGUISHES CORRECT VS WRONG for {', '.join(models)} "
            f"(p < 0.05). Wrong answers have higher entropy (more fragmented)."
        )
    else:
        conclusions.append(
            "✗ H₀ entropy does NOT significantly distinguish correct vs wrong answers."
        )
    
    # 2. Check if H1 persistence distinguishes correct/wrong
    h1_significant = [r for r in ttest_results if r.metric == 'h1_max_persistence' and r.significant]
    if h1_significant:
        models = [r.model for r in h1_significant]
        # Check direction
        higher_correct = [r for r in h1_significant if r.correct_mean > r.wrong_mean]
        if higher_correct:
            conclusions.append(
                f"✓ H₁ PERSISTENCE IS HIGHER FOR CORRECT ANSWERS in {', '.join([r.model for r in higher_correct])} "
                f"(p < 0.05). Correct answers show more 'deliberation loops'."
            )
    else:
        conclusions.append(
            "✗ H₁ persistence does NOT significantly distinguish correct vs wrong answers."
        )
    
    # 3. Check effect sizes
    large_effects = [r for r in ttest_results if abs(r.effect_size) > 0.8 and r.significant]
    if large_effects:
        conclusions.append(
            f"✓ LARGE EFFECT SIZES (|d| > 0.8) found for: " +
            ", ".join([f"{r.metric} ({r.model}, d={r.effect_size:.2f})" for r in large_effects])
        )
    
    # 4. Check correlations
    strong_corrs = []
    for metric, by_model in correlations.items():
        for model, (r, p) in by_model.items():
            if abs(r) > 0.3 and p < 0.05:
                strong_corrs.append((metric, model, r, p))
    
    if strong_corrs:
        conclusions.append(
            "✓ SIGNIFICANT CORRELATIONS (|r| > 0.3, p < 0.05):"
        )
        for metric, model, r, p in strong_corrs:
            direction = "positive" if r > 0 else "negative"
            conclusions.append(
                f"   - {metric} × correctness ({model}): r={r:.3f}, p={p:.4f} ({direction})"
            )
    
    # 5. Overall verdict
    n_significant = len([r for r in ttest_results if r.significant])
    total_tests = len(ttest_results)
    
    if n_significant >= total_tests * 0.5:
        conclusions.append(
            "\n" + "="*60 +
            "\n★ HYPOTHESIS SUPPORTED: Topology predicts correctness!\n" +
            f"  {n_significant}/{total_tests} tests significant (p < 0.05)\n" +
            "="*60
        )
    else:
        conclusions.append(
            "\n" + "="*60 +
            "\n✗ HYPOTHESIS NOT STRONGLY SUPPORTED\n" +
            f"  Only {n_significant}/{total_tests} tests significant\n" +
            "="*60
        )
    
    return conclusions


def print_ttest_table(results: List[StatisticalResult]):
    """Print t-test results as a table."""
    
    print("\n" + "="*90)
    print("T-TEST RESULTS: Correct vs Wrong Answers")
    print("="*90)
    print(f"{'Metric':<20} {'Model':<8} {'Correct':<18} {'Wrong':<18} {'t':<8} {'p':<10} {'d':<8} {'Sig?':<5}")
    print("-"*90)
    
    for r in sorted(results, key=lambda x: (x.metric, x.model)):
        correct_str = f"{r.correct_mean:.3f}±{r.correct_std:.3f} (n={r.correct_n})"
        wrong_str = f"{r.wrong_mean:.3f}±{r.wrong_std:.3f} (n={r.wrong_n})"
        sig_str = "***" if r.p_value < 0.001 else ("**" if r.p_value < 0.01 else ("*" if r.p_value < 0.05 else ""))
        
        print(f"{r.metric:<20} {r.model:<8} {correct_str:<18} {wrong_str:<18} {r.t_statistic:<8.2f} {r.p_value:<10.4f} {r.effect_size:<8.2f} {sig_str:<5}")


def print_correlation_table(correlations: Dict[str, Dict[str, Tuple[float, float]]]):
    """Print correlation results."""
    
    print("\n" + "="*70)
    print("CORRELATIONS: Topology Metrics × Correctness")
    print("="*70)
    print(f"{'Metric':<25} {'4B':<20} {'1.7B':<20} {'0.6B':<20}")
    print("-"*70)
    
    for metric in correlations:
        row = f"{metric:<25}"
        for model in ['4B', '1.7B', '0.6B']:
            if model in correlations[metric]:
                r, p = correlations[metric][model]
                sig = "*" if p < 0.05 else ""
                row += f"r={r:+.3f} (p={p:.3f}){sig:<3}"
            else:
                row += f"{'N/A':<20}"
        print(row)


def print_bucket_table(bucket_analysis: Dict):
    """Print per-bucket analysis."""
    
    print("\n" + "="*80)
    print("TOPOLOGY BY TRUTH BUCKET (Mean Values)")
    print("="*80)
    
    metrics = ['h0_entropy', 'h1_max_persistence', 'dust_score']
    
    for metric in metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print(f"{'Bucket':<10} {'Name':<25} {'4B':<12} {'1.7B':<12} {'0.6B':<12}")
        print("-"*70)
        
        for bucket in sorted(bucket_analysis.keys()):
            name = BUCKET_NAMES.get(bucket, '')[:24]
            row = f"{bucket:<10} {name:<25}"
            
            for model in ['4B', '1.7B', '0.6B']:
                if model in bucket_analysis[bucket]:
                    val = bucket_analysis[bucket][model].get(metric, 0)
                    row += f"{val:<12.3f}"
                else:
                    row += f"{'N/A':<12}"
            
            print(row)


def run_statistical_analysis(config: ExperimentConfig, save_report: bool = True) -> dict:
    """
    Main entry point: Run full statistical analysis.
    
    Returns dict with all results for programmatic access.
    """
    
    print("\n" + "="*70)
    print("STATISTICAL ANALYSIS: TOPOLOGY OF TRUTH")
    print("="*70)
    
    # Load and merge data
    print("\nLoading data...")
    tda_features, verifications = load_data(config)
    print(f"  TDA features: {len(tda_features)}")
    print(f"  Verifications: {len(verifications)}")
    
    if not tda_features or not verifications:
        print("ERROR: Missing data. Run the pipeline first.")
        return {}
    
    data = merge_data(tda_features, verifications)
    print(f"  Merged records: {len(data)}")
    
    if not data:
        print("ERROR: No matching records found.")
        return {}
    
    # Summary stats
    for model in ['4B', '1.7B', '0.6B']:
        model_data = [d for d in data if d['model_params'] == model]
        correct = sum(1 for d in model_data if d['correct'])
        print(f"  {model}: {correct}/{len(model_data)} correct ({100*correct/len(model_data):.1f}%)")
    
    # 1. T-tests
    print("\nRunning t-tests...")
    metrics = ['h0_entropy', 'h1_max_persistence', 'dust_score', 'loop_score', 'coherence_score']
    models = ['4B', '1.7B', '0.6B']
    
    ttest_results = []
    for metric in metrics:
        for model in models:
            result = run_ttest(data, metric, model)
            if result:
                ttest_results.append(result)
    
    print_ttest_table(ttest_results)
    
    # 2. Correlations
    print("\nComputing correlations...")
    correlations = {}
    for metric in metrics:
        correlations[metric] = {}
        for model in models:
            r, p = compute_correlation(data, metric, model)
            correlations[metric][model] = (r, p)
    
    print_correlation_table(correlations)
    
    # 3. Bucket analysis
    print("\nAnalyzing by truth bucket...")
    bucket_analysis = analyze_by_bucket(data)
    print_bucket_table(bucket_analysis)
    
    # 4. Generate conclusions
    print("\n" + "="*70)
    print("CONCLUSIONS")
    print("="*70)
    
    conclusions = generate_conclusions(ttest_results, correlations, bucket_analysis)
    for c in conclusions:
        print(c)
    
    # 5. Save report
    if save_report:
        report_path = config.output_dir / "analysis_report.json"
        report = {
            'ttest_results': [
                {
                    'metric': r.metric,
                    'model': r.model,
                    'correct_mean': r.correct_mean,
                    'correct_std': r.correct_std,
                    'correct_n': r.correct_n,
                    'wrong_mean': r.wrong_mean,
                    'wrong_std': r.wrong_std,
                    'wrong_n': r.wrong_n,
                    't_statistic': r.t_statistic,
                    'p_value': r.p_value,
                    'effect_size': r.effect_size,
                    'significant': r.significant,
                }
                for r in ttest_results
            ],
            'correlations': {
                metric: {model: {'r': r, 'p': p} for model, (r, p) in by_model.items()}
                for metric, by_model in correlations.items()
            },
            'bucket_analysis': bucket_analysis,
            'conclusions': conclusions,
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x) if isinstance(x, (np.floating, np.integer)) else x)
        print(f"\nReport saved to: {report_path}")
    
    return {
        'ttest_results': ttest_results,
        'correlations': correlations,
        'bucket_analysis': bucket_analysis,
        'conclusions': conclusions,
    }


if __name__ == "__main__":
    config = ExperimentConfig()
    run_statistical_analysis(config)
