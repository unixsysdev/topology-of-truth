"""
Module D: Visualizer (The Gallery Creator)

Creates the visual "Geometric Portfolio" showing:
1. H0 Entropy Heatmap (dust detection)
2. H1 Lifetime KDE (deliberation loops)
3. UMAP 3D trajectories (manifold shape)
4. Persistence Landscapes (mountain overlays)
5. Betti Curves (connectivity over scale)
6. Wasserstein Distance Matrix (model similarity)
7. Per-bucket galleries
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from typing import Optional, Dict, List
import warnings

# Dimensionality reduction
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# TDA visualization
try:
    from persim import plot_diagrams, sliced_wasserstein
    HAS_PERSIM = True
except ImportError:
    HAS_PERSIM = False

from config import ExperimentConfig, MODELS, BUCKET_NAMES


def normalize_persistence_diagram(dgm: np.ndarray) -> np.ndarray:
    """
    Normalize persistence diagram to [0,1] range.
    This allows cross-model Wasserstein comparison despite different hidden dimensions.
    """
    if len(dgm) == 0:
        return dgm
    
    dgm_copy = dgm.copy()
    
    # Get max value for normalization (use max death time)
    max_val = dgm_copy[:, 1].max()
    min_val = dgm_copy[:, 0].min()
    
    if max_val - min_val < 1e-10:
        return dgm_copy
    
    # Normalize to [0, 1]
    dgm_copy[:, 0] = (dgm_copy[:, 0] - min_val) / (max_val - min_val)
    dgm_copy[:, 1] = (dgm_copy[:, 1] - min_val) / (max_val - min_val)
    
    return dgm_copy



# =============================================================================
# COLOR SCHEMES
# =============================================================================

MODEL_COLORS = {
    "4B": "#2ecc71",      # Green - largest
    "1.7B": "#3498db",    # Blue - middle  
    "0.6B": "#e74c3c",    # Red - smallest
}

CORRECT_COLOR = "#27ae60"   # Green
INCORRECT_COLOR = "#c0392b" # Red

BUCKET_COLORS = {
    "000": "#7f8c8d",  # Gray - all wrong (The Void)
    "001": "#9b59b6",  # Purple - only 0.6B (The Anomaly)
    "010": "#f39c12",  # Orange - only 1.7B (Middle Upset)
    "011": "#1abc9c",  # Teal - 1.7B + 0.6B
    "100": "#e74c3c",  # Red - only 4B (The Gap)
    "101": "#3498db",  # Blue - 4B + 0.6B
    "110": "#2ecc71",  # Green - 4B + 1.7B (Distillation)
    "111": "#27ae60",  # Bright green - all correct (Trivial)
}


def set_plot_style():
    """Set consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'legend.fontsize': 10,
        'figure.figsize': (12, 8),
        'figure.dpi': 150,
    })


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_betti_curve(diagram: np.ndarray, n_bins: int = 100):
    """
    Compute Betti curve: number of features alive at each filtration value.
    """
    if len(diagram) == 0:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins)
    
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    
    # Handle infinite deaths
    finite_mask = np.isfinite(deaths)
    max_finite = deaths[finite_mask].max() if np.any(finite_mask) else births.max()
    deaths = np.where(np.isfinite(deaths), deaths, max_finite * 1.1)
    
    # Create filtration range
    f_min = max(0, births.min())
    f_max = max(deaths.max(), births.max()) * 1.05
    filtration = np.linspace(f_min, f_max, n_bins)
    
    # Count alive features at each filtration value
    betti = np.zeros(n_bins)
    for i, f in enumerate(filtration):
        betti[i] = np.sum((births <= f) & (deaths > f))
    
    return filtration, betti


def load_tda_features(config: ExperimentConfig) -> List[dict]:
    """Load TDA features from JSONL file."""
    features = []
    features_file = config.tda_dir / "tda_features.jsonl"
    
    if not features_file.exists():
        print(f"Warning: {features_file} not found")
        return features
    
    with open(features_file) as f:
        for line in f:
            features.append(json.loads(line))
    
    return features


def load_verification_results(config: ExperimentConfig) -> List[dict]:
    """Load verification results from JSONL file."""
    results = []
    results_file = config.output_dir / "verification_results.jsonl"
    
    if not results_file.exists():
        print(f"Warning: {results_file} not found")
        return results
    
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))
    
    return results


def extract_point_cloud_from_activations(
    activations_path: str,
    method: str = "last_layer",
    subsample: Optional[int] = None,
) -> np.ndarray:
    """Extract point cloud from saved activations."""
    data = np.load(activations_path)
    layer_keys = sorted([k for k in data.files if k.startswith('layer_')])
    
    if not layer_keys:
        raise ValueError(f"No layer data found in {activations_path}")
    
    if method == "last_layer":
        point_cloud = data[layer_keys[-1]]
    elif method == "all_layers":
        arrays = [data[k] for k in layer_keys]
        point_cloud = np.vstack(arrays)
    else:
        point_cloud = data[layer_keys[-1]]
    
    if subsample and len(point_cloud) > subsample:
        indices = np.random.choice(len(point_cloud), subsample, replace=False)
        indices.sort()
        point_cloud = point_cloud[indices]
    
    return point_cloud.astype(np.float32)


# =============================================================================
# MAIN VISUALIZER CLASS
# =============================================================================

class TopologyVisualizer:
    """Main visualization class for the topology audit."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.tda_features: List[dict] = []
        self.verifications: List[dict] = []
        self.merged_data: Dict[str, dict] = {}
        
        set_plot_style()
    
    def load_data(self):
        """Load TDA features and verification results."""
        print("Loading TDA features...")
        self.tda_features = load_tda_features(self.config)
        print(f"  Loaded {len(self.tda_features)} TDA feature sets")
        
        print("Loading verification results...")
        self.verifications = load_verification_results(self.config)
        print(f"  Loaded {len(self.verifications)} verifications")
        
        self._merge_data()
    
    def _merge_data(self):
        """Merge TDA features with verification results."""
        # Index verifications by question_id
        verification_map = {v['question_id']: v for v in self.verifications}
        
        # Index TDA features by (question_id, model_params)
        tda_map = {}
        for f in self.tda_features:
            key = (f['question_id'], f['model_params'])
            tda_map[key] = f
        
        # Merge
        for qid, v in verification_map.items():
            models_tda = {}
            for model_params in ["4B", "1.7B", "0.6B"]:
                key = (qid, model_params)
                if key in tda_map:
                    models_tda[model_params] = tda_map[key]
            
            if len(models_tda) == 3:
                self.merged_data[qid] = {
                    'verification': v,
                    'tda': models_tda,
                }
        
        print(f"  Merged data for {len(self.merged_data)} questions")
    
    # =========================================================================
    # CHART 1: H0 Entropy Boxplot by Truth Bucket
    # =========================================================================
    
    def plot_entropy_by_bucket(self, save_path: Optional[Path] = None):
        """
        H0 Entropy boxplot across truth buckets.
        Shows which buckets have "dusty" (high entropy) vs "coherent" (low entropy) traces.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, model_params in enumerate(["4B", "1.7B", "0.6B"]):
            ax = axes[idx]
            
            bucket_data = {b: [] for b in BUCKET_NAMES.keys()}
            
            for qid, data in self.merged_data.items():
                bucket = data['verification']['bucket']
                tda = data['tda'].get(model_params)
                if tda:
                    bucket_data[bucket].append(tda['h0_entropy'])
            
            # Prepare boxplot data
            boxes = []
            labels = []
            colors = []
            
            for bucket in sorted(bucket_data.keys()):
                if bucket_data[bucket]:
                    boxes.append(bucket_data[bucket])
                    labels.append(f"{bucket}\n(n={len(bucket_data[bucket])})")
                    colors.append(BUCKET_COLORS[bucket])
            
            if not boxes:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            bp = ax.boxplot(boxes, labels=labels, patch_artist=True)
            
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel("Truth Bucket (count)")
            ax.set_ylabel("H₀ Persistent Entropy")
            ax.set_title(f"{model_params} Model")
            ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle("H₀ Entropy by Truth Bucket\n(Higher Entropy = More Fragmented 'Dust')", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 2: H1 Max Persistence KDE (Deliberation Depth)
    # =========================================================================
    
    def plot_h1_persistence_kde(self, save_path: Optional[Path] = None):
        """
        KDE of H1 max persistence (deliberation loops).
        Correct answers should have a "fat tail" of long-lived loops.
        """
        if not HAS_SCIPY:
            print("scipy not installed, skipping KDE plot")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, model_params in enumerate(["4B", "1.7B", "0.6B"]):
            ax = axes[idx]
            
            correct_h1 = []
            incorrect_h1 = []
            
            for qid, data in self.merged_data.items():
                v = data['verification']
                tda = data['tda'].get(model_params)
                
                if not tda:
                    continue
                
                # Determine if this model was correct
                is_correct = {
                    "4B": v['correct_4b'],
                    "1.7B": v['correct_17b'],
                    "0.6B": v['correct_06b'],
                }[model_params]
                
                h1_val = tda['h1_max_persistence']
                
                if is_correct:
                    correct_h1.append(h1_val)
                else:
                    incorrect_h1.append(h1_val)
            
            # Plot KDEs
            all_vals = correct_h1 + incorrect_h1
            if not all_vals:
                continue
            
            x_max = max(all_vals) * 1.1 + 0.01
            x_range = np.linspace(0, x_max, 200)
            
            # Check for sufficient variance before KDE (avoid LinAlgError on constant data)
            if len(correct_h1) > 1 and np.std(correct_h1) > 1e-10:
                try:
                    kde_correct = stats.gaussian_kde(correct_h1)
                    ax.fill_between(x_range, kde_correct(x_range), alpha=0.5,
                                   color=CORRECT_COLOR, label=f'Correct (n={len(correct_h1)})')
                    ax.plot(x_range, kde_correct(x_range), color=CORRECT_COLOR, linewidth=2)
                except np.linalg.LinAlgError:
                    ax.axvline(np.mean(correct_h1), color=CORRECT_COLOR, linestyle='--',
                              label=f'Correct (n={len(correct_h1)}, const)')
            elif len(correct_h1) == 1:
                ax.axvline(correct_h1[0], color=CORRECT_COLOR, linestyle='--',
                          label=f'Correct (n=1)')
            
            if len(incorrect_h1) > 1 and np.std(incorrect_h1) > 1e-10:
                try:
                    kde_incorrect = stats.gaussian_kde(incorrect_h1)
                    ax.fill_between(x_range, kde_incorrect(x_range), alpha=0.5,
                                   color=INCORRECT_COLOR, label=f'Incorrect (n={len(incorrect_h1)})')
                    ax.plot(x_range, kde_incorrect(x_range), color=INCORRECT_COLOR, linewidth=2)
                except np.linalg.LinAlgError:
                    ax.axvline(np.mean(incorrect_h1), color=INCORRECT_COLOR, linestyle='--',
                              label=f'Incorrect (n={len(incorrect_h1)}, const)')
            elif len(incorrect_h1) == 1:
                ax.axvline(incorrect_h1[0], color=INCORRECT_COLOR, linestyle='--',
                          label=f'Incorrect (n=1)')
            
            ax.set_xlabel("H₁ Max Persistence (Deliberation Depth)")
            ax.set_ylabel("Density")
            ax.set_title(f"{model_params} Model")
            ax.legend()
        
        plt.suptitle("H₁ Persistence Distribution: Correct vs Incorrect\n(Higher = More 'Thinking Loops')", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 3: UMAP 3D Trajectory
    # =========================================================================
    
    def plot_umap_trajectory(
        self,
        question_id: str,
        model_params: str = "4B",
        save_path: Optional[Path] = None
    ):
        """
        3D UMAP projection of residual stream trajectory.
        Color-coded by token position to show the "path" of reasoning.
        """
        if not HAS_UMAP:
            print("UMAP not installed. Install with: pip install umap-learn")
            return None
        
        # Find activation file
        model_short = model_params.lower().replace(".", "")
        act_path = self.config.embeddings_dir / f"{question_id}_{model_short}_activations.npz"
        
        if not act_path.exists():
            print(f"Activation file not found: {act_path}")
            return None
        
        # Load point cloud
        point_cloud = extract_point_cloud_from_activations(str(act_path), method="last_layer")
        
        print(f"Point cloud shape: {point_cloud.shape}")
        
        # Apply UMAP
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=min(15, len(point_cloud) - 1),
            min_dist=0.1,
            metric='cosine'
        )
        
        embedding = reducer.fit_transform(point_cloud)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by position (time step)
        n_points = len(embedding)
        colors = np.arange(n_points)
        
        scatter = ax.scatter(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            c=colors, cmap='viridis', s=30, alpha=0.8
        )
        
        # Draw trajectory line
        ax.plot(
            embedding[:, 0], embedding[:, 1], embedding[:, 2],
            'k-', alpha=0.3, linewidth=1
        )
        
        # Mark start and end
        ax.scatter(*embedding[0], c='green', s=200, marker='^', label='Start', edgecolors='black', zorder=10)
        ax.scatter(*embedding[-1], c='red', s=200, marker='s', label='End', edgecolors='black', zorder=10)
        
        plt.colorbar(scatter, ax=ax, shrink=0.6, label='Token Position')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.set_title(f"Residual Stream Trajectory\n{question_id} - {model_params}")
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 4: Persistence Landscape Overlay (3 Models Same Question)
    # =========================================================================
    
    def plot_persistence_landscape_overlay(
        self,
        question_id: str,
        save_path: Optional[Path] = None
    ):
        """
        Overlay persistence landscapes from all 3 models on same question.
        Shows "mountain range collapse" as model size decreases.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for dim_idx, (key, dim_name) in enumerate([('h0', 'H₀ (Connected Components)'), ('h1', 'H₁ (Loops)')]):
            ax = axes[dim_idx]
            
            for model_params in ["4B", "1.7B", "0.6B"]:
                color = MODEL_COLORS[model_params]
                
                # Load persistence diagram
                dgm_path = self.config.tda_dir / "diagrams" / f"{question_id}_{model_params}_diagrams.npz"
                
                if not dgm_path.exists():
                    print(f"Diagram not found: {dgm_path}")
                    continue
                
                dgm_data = np.load(dgm_path)
                
                if key not in dgm_data.files or len(dgm_data[key]) == 0:
                    continue
                
                dgm = dgm_data[key]
                filt, betti = compute_betti_curve(dgm, n_bins=100)
                
                ax.fill_between(filt, betti, alpha=0.3, color=color, label=model_params)
                ax.plot(filt, betti, color=color, linewidth=2)
            
            ax.set_xlabel('Filtration Parameter (ε)')
            ax.set_ylabel(f'Betti Number (β{dim_idx})')
            ax.set_title(dim_name)
            ax.legend()
        
        plt.suptitle(f"Persistence Landscape Overlay: {question_id}\n(Comparing Manifold Shape Across Model Scales)", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 5: Betti Curves with Collapse Detection
    # =========================================================================
    
    def plot_betti_curves_comparison(
        self,
        question_id: str,
        save_path: Optional[Path] = None
    ):
        """
        Plot Betti curves showing where logic "collapses" (H0 spikes).
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        for col_idx, model_params in enumerate(["4B", "1.7B", "0.6B"]):
            dgm_path = self.config.tda_dir / "diagrams" / f"{question_id}_{model_params}_diagrams.npz"
            
            if not dgm_path.exists():
                continue
            
            dgm_data = np.load(dgm_path)
            
            for row_idx, (key, label) in enumerate([('h0', 'H₀ (Components)'), ('h1', 'H₁ (Loops)')]):
                ax = axes[row_idx, col_idx]
                
                if key in dgm_data.files and len(dgm_data[key]) > 0:
                    dgm = dgm_data[key]
                    filt, betti = compute_betti_curve(dgm, n_bins=100)
                    
                    ax.fill_between(filt, betti, alpha=0.3, color=MODEL_COLORS[model_params])
                    ax.plot(filt, betti, color=MODEL_COLORS[model_params], linewidth=2)
                    
                    # Mark the "collapse point" (max betti for H0)
                    if key == 'h0' and betti.max() > 0:
                        max_idx = np.argmax(betti)
                        ax.axvline(filt[max_idx], color='red', linestyle='--', alpha=0.5)
                        ax.annotate(f'Peak: {int(betti[max_idx])}',
                                   xy=(filt[max_idx], betti[max_idx]),
                                   xytext=(10, 10), textcoords='offset points',
                                   fontsize=10, color='red')
                
                ax.set_xlabel('Filtration ε')
                ax.set_ylabel(f'β{row_idx}')
                ax.set_title(f'{model_params} - {label}')
        
        plt.suptitle(f"Betti Curves: {question_id}\n(H₀ Peak = Logic Fragmentation Point)", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 6: Model Coherence Scatter (Dust Score vs Loop Score)
    # =========================================================================
    
    def plot_coherence_scatter(self, save_path: Optional[Path] = None):
        """
        Scatter plot: Dust Score (x) vs Loop Score (y).
        Ideal reasoning = low dust, high loops (top-left quadrant).
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for model_params in ["4B", "1.7B", "0.6B"]:
            dust_correct = []
            loop_correct = []
            dust_incorrect = []
            loop_incorrect = []
            
            for qid, data in self.merged_data.items():
                v = data['verification']
                tda = data['tda'].get(model_params)
                
                if not tda:
                    continue
                
                is_correct = {
                    "4B": v['correct_4b'],
                    "1.7B": v['correct_17b'],
                    "0.6B": v['correct_06b'],
                }[model_params]
                
                if is_correct:
                    dust_correct.append(tda['dust_score'])
                    loop_correct.append(tda['loop_score'])
                else:
                    dust_incorrect.append(tda['dust_score'])
                    loop_incorrect.append(tda['loop_score'])
            
            color = MODEL_COLORS[model_params]
            
            # Plot correct (filled) and incorrect (hollow)
            if dust_correct:
                ax.scatter(dust_correct, loop_correct, c=color, s=50, alpha=0.6,
                          label=f'{model_params} Correct', marker='o')
            if dust_incorrect:
                ax.scatter(dust_incorrect, loop_incorrect, c=color, s=50, alpha=0.6,
                          label=f'{model_params} Incorrect', marker='x')
        
        ax.set_xlabel("Dust Score (H₀ Entropy / log(Components))")
        ax.set_ylabel("Loop Score (H₁ Max Persistence × log(Loops))")
        ax.set_title("Coherence Space: Dust vs Deliberation\n(Top-Left = Coherent Reasoning, Bottom-Right = Fragmented Guessing)")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=1, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 7: Truth Bucket Summary Heatmap
    # =========================================================================
    
    def plot_bucket_summary_heatmap(self, save_path: Optional[Path] = None):
        """
        Heatmap showing mean TDA metrics for each model × bucket combination.
        """
        metrics = ['h0_entropy', 'h1_max_persistence', 'dust_score', 'loop_score']
        buckets = sorted(BUCKET_NAMES.keys())
        models = ["4B", "1.7B", "0.6B"]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            # Build matrix: rows = buckets, cols = models
            matrix = np.zeros((len(buckets), len(models)))
            
            for b_idx, bucket in enumerate(buckets):
                for m_idx, model_params in enumerate(models):
                    values = []
                    for qid, data in self.merged_data.items():
                        if data['verification']['bucket'] == bucket:
                            tda = data['tda'].get(model_params)
                            if tda and metric in tda:
                                values.append(tda[metric])
                    
                    matrix[b_idx, m_idx] = np.mean(values) if values else np.nan
            
            # Plot heatmap
            im = ax.imshow(matrix, cmap='RdYlGn_r' if 'dust' in metric or 'entropy' in metric else 'RdYlGn',
                          aspect='auto')
            
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models)
            ax.set_yticks(range(len(buckets)))
            ax.set_yticklabels([f"{b}" for b in buckets])
            
            # Add values as text
            for i in range(len(buckets)):
                for j in range(len(models)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=9, color='black' if 0.3 < val < 0.7 else 'white')
            
            ax.set_title(metric.replace('_', ' ').title())
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle("TDA Metrics by Truth Bucket × Model\n(Green = Better Reasoning, Red = Worse)", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # CHART 8: Wasserstein Distance Matrix
    # =========================================================================
    
    def plot_wasserstein_matrix(
        self,
        question_id: str,
        save_path: Optional[Path] = None
    ):
        """
        Compute and visualize pairwise Wasserstein distances between models.
        """
        if not HAS_PERSIM:
            print("persim not installed, skipping Wasserstein matrix")
            return None
        
        models = ["4B", "1.7B", "0.6B"]
        
        # Load diagrams
        diagrams = {}
        for model_params in models:
            dgm_path = self.config.tda_dir / "diagrams" / f"{question_id}_{model_params}_diagrams.npz"
            if dgm_path.exists():
                diagrams[model_params] = np.load(dgm_path)
        
        if len(diagrams) < 2:
            print(f"Not enough diagrams found for {question_id}")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for dim_idx, key in enumerate(['h0', 'h1']):
            ax = axes[dim_idx]
            
            # Compute pairwise distances
            n = len(models)
            dist_matrix = np.zeros((n, n))
            
            for i, m1 in enumerate(models):
                for j, m2 in enumerate(models):
                    if i == j:
                        dist_matrix[i, j] = 0
                    elif m1 in diagrams and m2 in diagrams:
                        dgm1 = diagrams[m1][key] if key in diagrams[m1].files else np.array([])
                        dgm2 = diagrams[m2][key] if key in diagrams[m2].files else np.array([])
                        
                        if len(dgm1) > 0 and len(dgm2) > 0:
                            # Filter out infinite values
                            dgm1 = dgm1[np.isfinite(dgm1[:, 1])]
                            dgm2 = dgm2[np.isfinite(dgm2[:, 1])]
                            if len(dgm1) > 0 and len(dgm2) > 0:
                                # Normalize diagrams to [0,1] for cross-model comparison
                                dgm1_norm = normalize_persistence_diagram(dgm1)
                                dgm2_norm = normalize_persistence_diagram(dgm2)
                                dist_matrix[i, j] = sliced_wasserstein(dgm1_norm, dgm2_norm)
            
            # Plot
            im = ax.imshow(dist_matrix, cmap='Blues')
            ax.set_xticks(range(n))
            ax.set_xticklabels(models)
            ax.set_yticks(range(n))
            ax.set_yticklabels(models)
            
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, f'{dist_matrix[i, j]:.3f}', ha='center', va='center',
                           fontsize=12, color='white' if dist_matrix[i, j] > dist_matrix.max()/2 else 'black')
            
            ax.set_title(f'{"H₀" if dim_idx == 0 else "H₁"} Sliced Wasserstein Distance')
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle(f"Manifold Similarity: {question_id}\n(Lower = More Similar Topology)", fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    # =========================================================================
    # GENERATE FULL REPORT
    # =========================================================================
    
    def generate_full_report(self):
        """Generate all visualizations for the audit report."""
        print("\n" + "="*70)
        print("GENERATING TOPOLOGY AUDIT REPORT")
        print("="*70)
        
        viz_dir = self.config.viz_dir
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Summary plots
        print("\n[1/7] Generating H₀ Entropy by Bucket...")
        self.plot_entropy_by_bucket(viz_dir / "01_h0_entropy_by_bucket.png")
        
        print("[2/7] Generating H₁ Persistence KDE...")
        self.plot_h1_persistence_kde(viz_dir / "02_h1_persistence_kde.png")
        
        print("[3/7] Generating Coherence Scatter...")
        self.plot_coherence_scatter(viz_dir / "03_coherence_scatter.png")
        
        print("[4/7] Generating Bucket Summary Heatmap...")
        self.plot_bucket_summary_heatmap(viz_dir / "04_bucket_heatmap.png")
        
        # 2. Per-question detailed plots (sample from each bucket)
        print("\n[5/7] Generating per-bucket sample visualizations...")
        
        bucket_samples = {}
        for qid, data in self.merged_data.items():
            bucket = data['verification']['bucket']
            if bucket not in bucket_samples:
                bucket_samples[bucket] = []
            bucket_samples[bucket].append(qid)
        
        # Take first 2 from each bucket
        for bucket, qids in bucket_samples.items():
            bucket_dir = viz_dir / f"bucket_{bucket}"
            bucket_dir.mkdir(exist_ok=True)
            
            for qid in qids[:2]:
                print(f"  Processing {qid} (bucket {bucket})...")
                
                # Persistence landscapes
                self.plot_persistence_landscape_overlay(
                    qid, bucket_dir / f"{qid}_landscapes.png"
                )
                
                # Betti curves
                self.plot_betti_curves_comparison(
                    qid, bucket_dir / f"{qid}_betti_curves.png"
                )
                
                # UMAP trajectories for each model
                for model_params in ["4B", "1.7B", "0.6B"]:
                    self.plot_umap_trajectory(
                        qid, model_params,
                        bucket_dir / f"{qid}_{model_params}_umap.png"
                    )
                
                # Wasserstein matrix
                self.plot_wasserstein_with_diagrams(
                    qid, bucket_dir / f"{qid}_wasserstein.png"
                )
        
        print("\n" + "="*70)
        print("REPORT GENERATION COMPLETE")
        print(f"Visualizations saved to: {viz_dir}")
        print("="*70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    config = ExperimentConfig()
    
    viz = TopologyVisualizer(config)
    viz.load_data()
    viz.generate_full_report()


    def plot_wasserstein_with_diagrams(
        self,
        question_id: str,
        save_path: Optional[Path] = None
    ):
        """
        Enhanced visualization showing persistence diagrams for all models
        AND the Wasserstein distance matrix side-by-side.
        """
        if not HAS_PERSIM:
            print("persim not installed, skipping Wasserstein visualization")
            return None
        
        models = ["4B", "1.7B", "0.6B"]
        model_colors = {"4B": "#2ecc71", "1.7B": "#3498db", "0.6B": "#e74c3c"}
        
        # Load diagrams
        diagrams = {}
        for model_params in models:
            dgm_path = self.config.tda_dir / "diagrams" / f"{question_id}_{model_params}_diagrams.npz"
            if dgm_path.exists():
                diagrams[model_params] = np.load(dgm_path)
        
        if len(diagrams) < 2:
            print(f"Not enough diagrams found for {question_id}")
            return None
        
        # Get correctness info
        verification = self.merged_data.get(question_id, {}).get('verification', {})
        bucket = verification.get('bucket', '???')
        
        # Create figure: 2 rows x 4 cols
        # Top row: H0 diagrams for each model + H0 Wasserstein matrix
        # Bottom row: H1 diagrams for each model + H1 Wasserstein matrix
        fig = plt.figure(figsize=(20, 10))
        
        for dim_idx, (key, dim_name) in enumerate([('h0', 'H₀ (Components)'), ('h1', 'H₁ (Loops)')]):
            # Plot persistence diagrams for each model
            for m_idx, model_params in enumerate(models):
                ax = fig.add_subplot(2, 4, dim_idx * 4 + m_idx + 1)
                
                if model_params in diagrams and key in diagrams[model_params].files:
                    dgm = diagrams[model_params][key]
                    dgm_finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else np.array([])
                    
                    if len(dgm_finite) > 0:
                        # Normalize for display
                        dgm_norm = normalize_persistence_diagram(dgm_finite)
                        
                        # Plot points
                        ax.scatter(dgm_norm[:, 0], dgm_norm[:, 1], 
                                   c=model_colors[model_params], alpha=0.6, s=50)
                        
                        # Plot diagonal
                        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
                        
                        # Stats
                        lifetimes = dgm_norm[:, 1] - dgm_norm[:, 0]
                        ax.text(0.05, 0.95, f"n={len(dgm_norm)}\nmax={lifetimes.max():.2f}",
                                transform=ax.transAxes, fontsize=9, va='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    else:
                        ax.text(0.5, 0.5, "No features", ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                
                ax.set_xlim(-0.05, 1.05)
                ax.set_ylim(-0.05, 1.05)
                ax.set_xlabel("Birth (normalized)")
                ax.set_ylabel("Death (normalized)")
                
                # Check correctness for this model
                correct_map = {'4B': bucket[0], '1.7B': bucket[1], '0.6B': bucket[2]}
                is_correct = correct_map.get(model_params, '?') == '1'
                status = "✓" if is_correct else "✗"
                
                ax.set_title(f"{model_params} {status}\n{dim_name}", fontsize=11,
                            color='green' if is_correct else 'red')
            
            # Wasserstein matrix (4th column)
            ax = fig.add_subplot(2, 4, dim_idx * 4 + 4)
            
            n = len(models)
            dist_matrix = np.zeros((n, n))
            
            for i, m1 in enumerate(models):
                for j, m2 in enumerate(models):
                    if i == j:
                        dist_matrix[i, j] = 0
                    elif m1 in diagrams and m2 in diagrams:
                        dgm1 = diagrams[m1][key] if key in diagrams[m1].files else np.array([])
                        dgm2 = diagrams[m2][key] if key in diagrams[m2].files else np.array([])
                        
                        if len(dgm1) > 0 and len(dgm2) > 0:
                            dgm1 = dgm1[np.isfinite(dgm1[:, 1])]
                            dgm2 = dgm2[np.isfinite(dgm2[:, 1])]
                            if len(dgm1) > 0 and len(dgm2) > 0:
                                dgm1_norm = normalize_persistence_diagram(dgm1)
                                dgm2_norm = normalize_persistence_diagram(dgm2)
                                dist_matrix[i, j] = sliced_wasserstein(dgm1_norm, dgm2_norm)
            
            im = ax.imshow(dist_matrix, cmap='Blues')
            ax.set_xticks(range(n))
            ax.set_xticklabels(models)
            ax.set_yticks(range(n))
            ax.set_yticklabels(models)
            
            for i in range(n):
                for j in range(n):
                    color = 'white' if dist_matrix[i, j] > dist_matrix.max() / 2 else 'black'
                    ax.text(j, i, f'{dist_matrix[i, j]:.3f}', ha='center', va='center',
                           fontsize=11, color=color, fontweight='bold')
            
            ax.set_title(f"{dim_name}\nWasserstein Distance", fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.8)
        
        plt.suptitle(f"Topology Comparison: {question_id} | Bucket {bucket} ({BUCKET_NAMES.get(bucket, '')})",
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {save_path}")
        
        return fig
