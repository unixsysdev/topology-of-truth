"""
Audit Report: Main Orchestrator

Runs the full Topology of Truth pipeline:
1. Generate traces + extract activations (trace_generator.py)
2. Verify answers via 235B API (batch_verifier.py)
3. Compute TDA features (topological_engine.py)
4. Run statistical analysis (analyze_results.py)
5. Generate visualizations (visualizer.py + visualize_comparisons.py)
6. Print summary and conclusions
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

from config import ExperimentConfig, MODELS, BUCKET_NAMES


def print_banner(text: str):
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_summary(config: ExperimentConfig):
    """Print summary statistics from the audit."""
    
    # Load verification results
    results_file = config.output_dir / "verification_results.jsonl"
    if not results_file.exists():
        print("No verification results found.")
        return
    
    verifications = []
    with open(results_file) as f:
        for line in f:
            verifications.append(json.loads(line))
    
    print(f"\nTotal questions verified: {len(verifications)}")
    
    # Bucket distribution
    print("\n" + "-" * 50)
    print("TRUTH BUCKET DISTRIBUTION")
    print("-" * 50)
    
    bucket_counts = {}
    for v in verifications:
        b = v['bucket']
        bucket_counts[b] = bucket_counts.get(b, 0) + 1
    
    for bucket in sorted(bucket_counts.keys()):
        count = bucket_counts[bucket]
        pct = 100 * count / len(verifications)
        name = BUCKET_NAMES.get(bucket, '')
        bar = "█" * int(pct / 2)
        print(f"  {bucket} {name:<25} {count:>4} ({pct:>5.1f}%) {bar}")
    
    # Per-model accuracy
    print("\n" + "-" * 50)
    print("MODEL ACCURACY")
    print("-" * 50)
    
    for model_params, key in [("4B", "correct_4b"), ("1.7B", "correct_17b"), ("0.6B", "correct_06b")]:
        correct = sum(1 for v in verifications if v.get(key, False))
        pct = 100 * correct / len(verifications) if verifications else 0
        print(f"  {model_params:<8} {correct:>4}/{len(verifications)} ({pct:.1f}%)")
    
    # Load TDA features for topology summary
    tda_file = config.tda_dir / "tda_features.jsonl"
    if tda_file.exists():
        print("\n" + "-" * 50)
        print("TOPOLOGY SUMMARY (Mean ± Std)")
        print("-" * 50)
        
        tda_features = []
        with open(tda_file) as f:
            for line in f:
                tda_features.append(json.loads(line))
        
        import numpy as np
        
        for model_params in ["4B", "1.7B", "0.6B"]:
            model_tda = [t for t in tda_features if t['model_params'] == model_params]
            if model_tda:
                h0_ent = [t['h0_entropy'] for t in model_tda]
                h1_max = [t['h1_max_persistence'] for t in model_tda]
                dust = [t['dust_score'] for t in model_tda]
                
                print(f"\n  {model_params}:")
                print(f"    H₀ Entropy:      {np.mean(h0_ent):.3f} ± {np.std(h0_ent):.3f}")
                print(f"    H₁ Max Persist:  {np.mean(h1_max):.3f} ± {np.std(h1_max):.3f}")
                print(f"    Dust Score:      {np.mean(dust):.3f} ± {np.std(dust):.3f}")


def run_full_pipeline(
    config: ExperimentConfig,
    skip_generation: bool = False,
    skip_verification: bool = False,
    skip_tda: bool = False,
    skip_analysis: bool = False,
    skip_viz: bool = False,
    skip_comparisons: bool = False,
    max_samples: int = None,
    max_comparisons: int = 5,
):
    """Run the complete topology audit pipeline."""
    
    print_banner("TOPOLOGY OF TRUTH: GEOMETRIC AUDIT")
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {config.output_dir}")
    
    if max_samples:
        config.max_samples = max_samples
        print(f"Max samples: {max_samples}")
    
    # =========================================================================
    # PHASE 1: Generate Traces + Extract Activations
    # =========================================================================
    if not skip_generation:
        print_banner("PHASE 1: TRACE GENERATION")
        from trace_generator import run_trace_generation
        run_trace_generation(config)
    else:
        print("\n[Skipping trace generation]")
    
    # =========================================================================
    # PHASE 2: Verify Answers via API
    # =========================================================================
    if not skip_verification:
        print_banner("PHASE 2: ANSWER VERIFICATION")
        from batch_verifier import run_batch_verification
        run_batch_verification(config)
    else:
        print("\n[Skipping verification]")
    
    # =========================================================================
    # PHASE 3: Compute TDA Features
    # =========================================================================
    if not skip_tda:
        print_banner("PHASE 3: TOPOLOGICAL ANALYSIS")
        from topological_engine import run_topological_analysis
        run_topological_analysis(config)
    else:
        print("\n[Skipping TDA computation]")
    
    # =========================================================================
    # PHASE 4: Statistical Analysis
    # =========================================================================
    if not skip_analysis:
        print_banner("PHASE 4: STATISTICAL ANALYSIS")
        from analyze_results import run_statistical_analysis
        run_statistical_analysis(config)
    else:
        print("\n[Skipping statistical analysis]")
    
    # =========================================================================
    # PHASE 5: Summary Visualizations
    # =========================================================================
    if not skip_viz:
        print_banner("PHASE 5: SUMMARY VISUALIZATIONS")
        from visualizer import TopologyVisualizer
        viz = TopologyVisualizer(config)
        viz.load_data()
        
        # Generate summary plots only (not per-question)
        viz_dir = config.viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        print("Generating summary plots...")
        viz.plot_entropy_by_bucket(viz_dir / "01_h0_entropy_by_bucket.png")
        viz.plot_h1_persistence_kde(viz_dir / "02_h1_persistence_kde.png")
        viz.plot_coherence_scatter(viz_dir / "03_coherence_scatter.png")
        viz.plot_bucket_summary_heatmap(viz_dir / "04_bucket_heatmap.png")
        print(f"Summary plots saved to: {viz_dir}")
    else:
        print("\n[Skipping summary visualizations]")
    
    # =========================================================================
    # PHASE 6: Per-Question Comparison Figures
    # =========================================================================
    if not skip_comparisons:
        print_banner("PHASE 6: COMPARISON FIGURES")
        import subprocess
        
        comparisons_dir = config.output_dir / "comparisons"
        cmd = [
            "python", "visualize_comparisons.py",
            "--max", str(max_comparisons),
            "--output-dir", str(comparisons_dir),
            "--results-dir", str(config.output_dir)
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print("\n[Skipping comparison figures]")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print_banner("FINAL SUMMARY")
    print_summary(config)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print output locations
    print("\n" + "-" * 50)
    print("OUTPUT LOCATIONS")
    print("-" * 50)
    print(f"  Traces:        {config.traces_dir}")
    print(f"  Activations:   {config.embeddings_dir}")
    print(f"  TDA Features:  {config.tda_dir}")
    print(f"  Visualizations:{config.viz_dir}")
    print(f"  Comparisons:   results/comparisons/")
    print(f"  Analysis:      {config.output_dir / 'analysis_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Topology of Truth audit pipeline"
    )
    
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip trace generation (use existing traces)")
    parser.add_argument("--skip-verification", action="store_true",
                       help="Skip answer verification (use existing results)")
    parser.add_argument("--skip-tda", action="store_true",
                       help="Skip TDA computation (use existing features)")
    parser.add_argument("--skip-analysis", action="store_true",
                       help="Skip statistical analysis")
    parser.add_argument("--skip-viz", action="store_true",
                       help="Skip summary visualization generation")
    parser.add_argument("--skip-comparisons", action="store_true",
                       help="Skip per-question comparison figures")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to process")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Dataset to use (e.g., 'openai/gsm8k', 'GAIR/LIMO')")
    parser.add_argument("--max-comparisons", type=int, default=5,
                       help="Max comparison figures per bucket")
    parser.add_argument("--output-dir", type=str, default="./results",
                       help="Output directory for all results")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    if args.max_samples:
        config.max_samples = args.max_samples
    if args.dataset:
        config.dataset_name = args.dataset
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
        config.traces_dir = config.output_dir / "traces"
        config.embeddings_dir = config.output_dir / "embeddings"
        config.tda_dir = config.output_dir / "tda"
        config.viz_dir = config.output_dir / "visualizations"
    
    run_full_pipeline(
        config,
        skip_generation=args.skip_generation,
        skip_verification=args.skip_verification,
        skip_tda=args.skip_tda,
        skip_analysis=args.skip_analysis,
        skip_viz=args.skip_viz,
        skip_comparisons=args.skip_comparisons,
        max_samples=args.max_samples,
        max_comparisons=args.max_comparisons,
    )
