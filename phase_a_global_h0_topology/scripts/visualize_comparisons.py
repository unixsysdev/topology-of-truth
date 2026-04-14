"""
Unified Comparison Visualizer

Standalone script to generate ONE figure per question showing all 3 models side-by-side.
Run this AFTER the main pipeline has completed.

Usage:
    python visualize_comparisons.py                    # All questions
    python visualize_comparisons.py --bucket 100      # Only bucket 100
    python visualize_comparisons.py --question q_0042 # Single question
    python visualize_comparisons.py --max 5           # First 5 per bucket

Output structure:
    results/comparisons/
    ├── bucket_000_void/
    │   ├── q_0001_comparison.png
    │   └── ...
    ├── bucket_100_gap/
    │   └── ...
    └── summary_grid.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import warnings

# Optional imports
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap not installed. Install with: pip install umap-learn")

from config import ExperimentConfig, BUCKET_NAMES


# =============================================================================
# CONSTANTS
# =============================================================================

MODEL_ORDER = ["4B", "1.7B", "0.6B"]

MODEL_COLORS = {
    "4B": "#2ecc71",      # Green - largest
    "1.7B": "#3498db",    # Blue - middle
    "0.6B": "#e74c3c",    # Red - smallest
}

MODEL_CMAPS = {
    "4B": "Greens",
    "1.7B": "Blues",
    "0.6B": "Reds",
}

BUCKET_COLORS = {
    "000": "#7f8c8d",
    "001": "#9b59b6",
    "010": "#f39c12",
    "011": "#1abc9c",
    "100": "#e74c3c",
    "101": "#3498db",
    "110": "#2ecc71",
    "111": "#27ae60",
}

BUCKET_NAMES_SHORT = {
    "000": "Void (All Wrong)",
    "001": "Anomaly (Only 0.6B ✓)",
    "010": "Upset (Only 1.7B ✓)",
    "011": "4B Failure",
    "100": "Gap (Only 4B ✓)",
    "101": "Middle Collapse",
    "110": "Distillation",
    "111": "Trivial (All ✓)",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_jsonl(path: Path) -> List[dict]:
    """Load JSONL file."""
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_activations(path: Path) -> Optional[np.ndarray]:
    """Load residual stream activations (Phase 1: last layer only)."""
    if not path.exists():
        return None
    try:
        data = np.load(path)
        
        # New format: 'last_layer' key
        if 'last_layer' in data.files:
            return data['last_layer'].astype(np.float32)
        
        # Old format: layer_X keys
        keys = sorted([k for k in data.files if k.startswith('layer_')])
        if keys:
            return data[keys[-1]].astype(np.float32)
        
        return None
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def load_diagrams(path: Path) -> Dict[str, np.ndarray]:
    """Load persistence diagrams."""
    if not path.exists():
        return {}
    try:
        data = np.load(path)
        return {k: data[k] for k in ['h0', 'h1'] if k in data.files and len(data[k]) > 0}
    except:
        return {}


def compute_betti_curve(dgm: np.ndarray, n_bins: int = 80) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Betti curve from persistence diagram."""
    if dgm is None or len(dgm) == 0:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins)
    
    births = dgm[:, 0]
    deaths = dgm[:, 1].copy()
    
    # Handle infinity
    finite_mask = np.isfinite(deaths)
    max_val = deaths[finite_mask].max() if finite_mask.any() else (births.max() if len(births) > 0 else 1.0)
    deaths[~finite_mask] = max_val * 1.1
    
    filt = np.linspace(0, max_val * 1.05, n_bins)
    betti = np.array([np.sum((births <= f) & (deaths > f)) for f in filt])
    
    return filt, betti


def apply_umap_3d(points: np.ndarray) -> Optional[np.ndarray]:
    """Reduce to 3D with UMAP."""
    if not HAS_UMAP or points is None or len(points) < 5:
        return None
    try:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(points) - 1),
            min_dist=0.1,
            metric='cosine',
            random_state=42,
            verbose=False
        )
        return reducer.fit_transform(points)
    except Exception as e:
        print(f"  UMAP failed: {e}")
        return None


# =============================================================================
# MAIN COMPARISON FIGURE
# =============================================================================

def create_comparison_figure(
    question_id: str,
    bucket: str,
    verification: dict,
    tda_by_model: Dict[str, dict],
    config: ExperimentConfig,
    save_path: Path,
) -> bool:
    """
    Create ONE unified figure comparing all 3 models.
    
    Layout (4 rows × 3 cols):
        Row 0: UMAP 3D trajectories
        Row 1: Betti curves H₀ (components)
        Row 2: Betti curves H₁ (loops)
        Row 3: Persistence diagrams (birth-death)
        
    Plus title bar and stats footer.
    
    Returns True if successful.
    """
    
    # Correctness per model
    correct = {
        "4B": verification.get('correct_4b', False),
        "1.7B": verification.get('correct_17b', False),
        "0.6B": verification.get('correct_06b', False),
    }
    
    # Load data for each model
    model_data = {}
    for model in MODEL_ORDER:
        short = model.lower().replace(".", "")
        
        # Activations
        act_path = config.embeddings_dir / f"{question_id}_{short}_activations.npz"
        points = load_activations(act_path)
        
        # Diagrams
        dgm_path = config.tda_dir / "diagrams" / f"{question_id}_{model}_diagrams.npz"
        diagrams = load_diagrams(dgm_path)
        
        # TDA stats
        tda = tda_by_model.get(model, {})
        
        model_data[model] = {
            'points': points,
            'diagrams': diagrams,
            'tda': tda,
            'correct': correct[model],
        }
    
    # Check we have some data
    has_any_data = any(d['points'] is not None or d['diagrams'] for d in model_data.values())
    if not has_any_data:
        print(f"  No data found for {question_id}, skipping")
        return False
    
    # === CREATE FIGURE ===
    fig = plt.figure(figsize=(18, 22))
    
    # GridSpec: 5 rows (title + 4 plot rows)
    gs = GridSpec(5, 3, figure=fig, 
                  height_ratios=[0.15, 1.0, 0.7, 0.7, 0.7],
                  hspace=0.3, wspace=0.25)
    
    # === ROW 0: TITLE BAR ===
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis('off')
    
    status_parts = []
    for m in MODEL_ORDER:
        symbol = "✓" if correct[m] else "✗"
        status_parts.append(f"{m}: {symbol}")
    status_str = "  |  ".join(status_parts)
    
    title_text = f"{question_id}  —  Bucket {bucket}: {BUCKET_NAMES_SHORT.get(bucket, '')}\n{status_str}"
    
    ax_title.text(0.5, 0.5, title_text,
                  ha='center', va='center',
                  fontsize=14, fontweight='bold',
                  bbox=dict(boxstyle='round,pad=0.5',
                           facecolor=BUCKET_COLORS.get(bucket, '#cccccc'),
                           alpha=0.3, edgecolor='gray'))
    
    # === ROW 1: UMAP 3D TRAJECTORIES ===
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[1, col], projection='3d')
        
        data = model_data[model]
        points = data['points']
        color = MODEL_COLORS[model]
        cmap = MODEL_CMAPS[model]
        status = "✓ CORRECT" if data['correct'] else "✗ WRONG"
        status_color = "#27ae60" if data['correct'] else "#c0392b"
        
        if points is not None and len(points) >= 5:
            emb = apply_umap_3d(points)
            
            if emb is not None:
                n = len(emb)
                positions = np.arange(n)
                
                # Trajectory line
                ax.plot(emb[:, 0], emb[:, 1], emb[:, 2],
                       color=color, alpha=0.4, linewidth=1.5)
                
                # Points colored by position (time)
                scatter = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2],
                                    c=positions, cmap=cmap, s=25, alpha=0.8)
                
                # Start (triangle) and End (square)
                ax.scatter(*emb[0], c='black', s=120, marker='^', 
                          edgecolors='white', linewidths=1, zorder=10, label='Start')
                ax.scatter(*emb[-1], c='black', s=120, marker='s',
                          edgecolors='white', linewidths=1, zorder=10, label='End')
                
                # Colorbar
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
                cbar.set_label('Token Position', fontsize=8)
                
                ax.set_xlabel('UMAP 1', fontsize=8)
                ax.set_ylabel('UMAP 2', fontsize=8)
                ax.set_zlabel('UMAP 3', fontsize=8)
            else:
                ax.text2D(0.5, 0.5, "UMAP failed", ha='center', va='center',
                         transform=ax.transAxes, fontsize=10, color='gray')
        else:
            ax.text2D(0.5, 0.5, "No activation data", ha='center', va='center',
                     transform=ax.transAxes, fontsize=10, color='gray')
        
        ax.set_title(f"{model}\n{status}", fontsize=12, fontweight='bold', color=status_color)
    
    # === ROW 2: BETTI CURVES H₀ (Components) ===
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[2, col])
        
        data = model_data[model]
        dgm = data['diagrams'].get('h0')
        color = MODEL_COLORS[model]
        tda = data['tda']
        
        if dgm is not None and len(dgm) > 0:
            filt, betti = compute_betti_curve(dgm)
            
            ax.fill_between(filt, betti, alpha=0.3, color=color)
            ax.plot(filt, betti, color=color, linewidth=2)
            
            # Mark peak (fragmentation point)
            if betti.max() > 1:
                peak_idx = np.argmax(betti)
                ax.axvline(filt[peak_idx], color='red', linestyle='--', alpha=0.6, linewidth=1)
                ax.scatter([filt[peak_idx]], [betti[peak_idx]], color='red', s=50, zorder=5)
                ax.annotate(f'Peak: {int(betti[peak_idx])}',
                           xy=(filt[peak_idx], betti[peak_idx]),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=9, color='red',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Show H0 entropy in corner
            h0_ent = tda.get('h0_entropy', 0)
            ax.text(0.95, 0.95, f'H₀ Entropy: {h0_ent:.2f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No H₀ data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
        
        ax.set_xlabel('Filtration ε', fontsize=9)
        ax.set_ylabel('β₀ (Components)', fontsize=9)
        ax.set_title('H₀: Connected Components', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # === ROW 3: BETTI CURVES H₁ (Loops) ===
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[3, col])
        
        data = model_data[model]
        dgm = data['diagrams'].get('h1')
        color = MODEL_COLORS[model]
        tda = data['tda']
        
        if dgm is not None and len(dgm) > 0:
            filt, betti = compute_betti_curve(dgm)
            
            ax.fill_between(filt, betti, alpha=0.3, color=color)
            ax.plot(filt, betti, color=color, linewidth=2)
            
            # Area under curve
            area = np.trapz(betti, filt)
            
            # H1 max persistence
            h1_max = tda.get('h1_max_persistence', 0)
            
            ax.text(0.95, 0.95, f'H₁ Max: {h1_max:.2f}\nArea: {area:.1f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, "No H₁ features\n(Topologically Flat)",
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=10, color='gray', style='italic')
            
            # Still show the zero
            ax.text(0.95, 0.95, 'H₁ Max: 0.00',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Filtration ε', fontsize=9)
        ax.set_ylabel('β₁ (Loops)', fontsize=9)
        ax.set_title('H₁: Deliberation Loops', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # === ROW 4: PERSISTENCE DIAGRAMS ===
    for col, model in enumerate(MODEL_ORDER):
        ax = fig.add_subplot(gs[4, col])
        
        data = model_data[model]
        diagrams = data['diagrams']
        color = MODEL_COLORS[model]
        
        # Find max value for diagonal
        max_val = 0
        
        # Plot H0 (circles)
        dgm_h0 = diagrams.get('h0')
        if dgm_h0 is not None and len(dgm_h0) > 0:
            finite_h0 = dgm_h0[np.isfinite(dgm_h0[:, 1])]
            if len(finite_h0) > 0:
                ax.scatter(finite_h0[:, 0], finite_h0[:, 1],
                          c=color, s=40, alpha=0.6, label='H₀', marker='o')
                max_val = max(max_val, finite_h0.max())
        
        # Plot H1 (triangles)
        dgm_h1 = diagrams.get('h1')
        if dgm_h1 is not None and len(dgm_h1) > 0:
            finite_h1 = dgm_h1[np.isfinite(dgm_h1[:, 1])]
            if len(finite_h1) > 0:
                ax.scatter(finite_h1[:, 0], finite_h1[:, 1],
                          c='purple', s=60, alpha=0.7, label='H₁', marker='^')
                max_val = max(max_val, finite_h1.max())
        
        # Diagonal line (birth = death)
        if max_val > 0:
            ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 
                   'k--', alpha=0.3, linewidth=1, label='Birth=Death')
            ax.set_xlim(-0.02, max_val * 1.1)
            ax.set_ylim(-0.02, max_val * 1.1)
        else:
            ax.text(0.5, 0.5, "No persistence data", ha='center', va='center',
                   transform=ax.transAxes, fontsize=10, color='gray')
        
        ax.set_xlabel('Birth', fontsize=9)
        ax.set_ylabel('Death', fontsize=9)
        ax.set_title('Persistence Diagram', fontsize=10)
        ax.legend(fontsize=8, loc='lower right')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
    
    # === SAVE ===
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate unified comparison figures")
    parser.add_argument("--bucket", type=str, default=None,
                       help="Only process this bucket (e.g., '100')")
    parser.add_argument("--question", type=str, default=None,
                       help="Only process this question (e.g., 'q_0042')")
    parser.add_argument("--max", type=int, default=None,
                       help="Max questions per bucket")
    parser.add_argument("--output-dir", type=str, default="results/comparisons",
                       help="Output directory for comparison figures")
    parser.add_argument("--results-dir", type=str, default="./results",
                       help="Directory containing traces, embeddings, TDA results")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    # Override paths if custom results dir specified
    results_dir = Path(args.results_dir)
    config.output_dir = results_dir
    config.traces_dir = results_dir / "traces"
    config.embeddings_dir = results_dir / "embeddings"
    config.tda_dir = results_dir / "tda"
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("UNIFIED COMPARISON VISUALIZER")
    print("="*70)
    
    # Load verification results
    print("\nLoading verification results...")
    verifications = load_jsonl(config.output_dir / "verification_results.jsonl")
    print(f"  Found {len(verifications)} verified questions")
    
    if not verifications:
        print("ERROR: No verification results found. Run the pipeline first.")
        return
    
    # Load TDA features
    print("Loading TDA features...")
    tda_features = load_jsonl(config.tda_dir / "tda_features.jsonl")
    print(f"  Found {len(tda_features)} TDA feature sets")
    
    # Index TDA by (question_id, model)
    tda_index = {}
    for t in tda_features:
        key = (t['question_id'], t['model_params'])
        tda_index[key] = t
    
    # Organize by bucket
    by_bucket = {}
    for v in verifications:
        bucket = v['bucket']
        if bucket not in by_bucket:
            by_bucket[bucket] = []
        by_bucket[bucket].append(v)
    
    print(f"\nBucket distribution:")
    for bucket in sorted(by_bucket.keys()):
        print(f"  {bucket} ({BUCKET_NAMES_SHORT.get(bucket, '')}): {len(by_bucket[bucket])} questions")
    
    # Filter if requested
    if args.question:
        # Single question mode
        v = next((v for v in verifications if v['question_id'] == args.question), None)
        if not v:
            print(f"ERROR: Question {args.question} not found")
            return
        to_process = [(v['bucket'], [v])]
    elif args.bucket:
        # Single bucket mode
        if args.bucket not in by_bucket:
            print(f"ERROR: Bucket {args.bucket} not found")
            return
        questions = by_bucket[args.bucket]
        if args.max:
            questions = questions[:args.max]
        to_process = [(args.bucket, questions)]
    else:
        # All buckets
        to_process = []
        for bucket in sorted(by_bucket.keys()):
            questions = by_bucket[bucket]
            if args.max:
                questions = questions[:args.max]
            to_process.append((bucket, questions))
    
    # Generate figures
    total = sum(len(qs) for _, qs in to_process)
    print(f"\nGenerating {total} comparison figures...")
    
    success_count = 0
    
    for bucket, questions in to_process:
        bucket_name = BUCKET_NAMES_SHORT.get(bucket, '').replace(' ', '_').replace('(', '').replace(')', '').replace('✓', 'Y').replace('✗', 'X')
        bucket_dir = output_dir / f"bucket_{bucket}_{bucket_name}"
        bucket_dir.mkdir(exist_ok=True)
        
        print(f"\nBucket {bucket}: {len(questions)} questions")
        
        for v in questions:
            qid = v['question_id']
            
            # Get TDA for each model
            tda_by_model = {}
            for model in MODEL_ORDER:
                key = (qid, model)
                if key in tda_index:
                    tda_by_model[model] = tda_index[key]
            
            save_path = bucket_dir / f"{qid}_comparison.png"
            
            print(f"  {qid}...", end=" ", flush=True)
            
            success = create_comparison_figure(
                question_id=qid,
                bucket=bucket,
                verification=v,
                tda_by_model=tda_by_model,
                config=config,
                save_path=save_path,
            )
            
            if success:
                print("✓")
                success_count += 1
            else:
                print("✗")
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success_count}/{total} figures generated")
    print(f"Output: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
