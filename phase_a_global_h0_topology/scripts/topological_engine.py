"""
Module C: Topological Engine (The Mapper)

Computes TDA features from residual stream activations:
- Persistence Diagrams (H0, H1)
- Betti Curves
- Persistent Entropy
- Total Persistence

Phase 1: Last layer only - clean signal
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import Optional
import warnings

# TDA imports
try:
    from ripser import ripser
    from persim import sliced_wasserstein
    HAS_RIPSER = True
except ImportError:
    HAS_RIPSER = False
    warnings.warn("ripser/persim not installed. Install with: pip install ripser persim")

from config import ExperimentConfig, MODELS

# Dimensionality reduction
try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    warnings.warn("sklearn not installed. Install with: pip install scikit-learn")


@dataclass
class TDAFeatures:
    """Topological features extracted from a single trace."""
    question_id: str
    model_name: str
    model_params: str
    
    # Point cloud info
    num_points: int
    
    # H0 features (connected components)
    h0_num_components: int
    h0_total_persistence: float
    h0_max_persistence: float
    h0_entropy: float
    
    # H1 features (loops/holes)
    h1_num_loops: int
    h1_total_persistence: float
    h1_max_persistence: float  # "Deliberation depth"
    h1_entropy: float
    
    # Derived metrics
    dust_score: float  # High H0 entropy = "dust"
    loop_score: float  # H1 presence = deliberation
    coherence_score: float  # Combined metric
    
    # Betti curve stats
    betti0_area: float
    betti1_area: float
    
    # Paths
    persistence_diagram_path: Optional[str] = None


def compute_persistent_entropy(diagram: np.ndarray) -> float:
    """
    Compute persistent entropy.
    E = -sum(p_i * log(p_i)) where p_i = lifetime_i / total_lifetime
    """
    if len(diagram) == 0:
        return 0.0
    
    # Filter out infinite deaths
    finite_mask = np.isfinite(diagram[:, 1])
    finite_diagram = diagram[finite_mask]
    
    if len(finite_diagram) == 0:
        return 0.0
    
    # Compute lifetimes
    lifetimes = finite_diagram[:, 1] - finite_diagram[:, 0]
    lifetimes = lifetimes[lifetimes > 0]
    
    if len(lifetimes) == 0:
        return 0.0
    
    total = lifetimes.sum()
    if total == 0:
        return 0.0
    
    p = lifetimes / total
    entropy = -np.sum(p * np.log(p + 1e-10))
    
    return float(entropy)


def compute_betti_curve(diagram: np.ndarray, n_bins: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Compute Betti curve: features alive at each filtration value."""
    if len(diagram) == 0:
        return np.linspace(0, 1, n_bins), np.zeros(n_bins)
    
    births = diagram[:, 0]
    deaths = diagram[:, 1].copy()
    
    # Handle infinite deaths
    finite_mask = np.isfinite(deaths)
    max_finite = deaths[finite_mask].max() if finite_mask.any() else births.max()
    deaths[~finite_mask] = max_finite * 1.1
    
    f_min = births.min()
    f_max = max(deaths.max(), births.max()) * 1.05
    filtration = np.linspace(f_min, f_max, n_bins)
    
    betti = np.array([np.sum((births <= f) & (deaths > f)) for f in filtration])
    
    return filtration, betti


def extract_point_cloud(activations_path: str, subsample: Optional[int] = None) -> np.ndarray:
    """
    Extract point cloud from saved activations.
    
    Phase 1: Expects 'last_layer' key (new format)
    Also supports old 'layer_X' format for backwards compatibility.
    """
    data = np.load(activations_path)
    
    # New format: single 'last_layer' key
    if 'last_layer' in data.files:
        point_cloud = data['last_layer']
    else:
        # Old format: layer_X keys, take the last one
        layer_keys = sorted([k for k in data.files if k.startswith('layer_')])
        if not layer_keys:
            raise ValueError(f"No activation data found in {activations_path}")
        point_cloud = data[layer_keys[-1]]
    
    # Subsample for computational efficiency
    if subsample and len(point_cloud) > subsample:
        indices = np.random.choice(len(point_cloud), subsample, replace=False)
        indices.sort()
        point_cloud = point_cloud[indices]
    
    return point_cloud.astype(np.float32)


def compute_tda_features(
    point_cloud: np.ndarray,
    question_id: str,
    model_name: str,
    model_params: str,
    max_dim: int = 1,
    max_edge_length: float = 2.0,
    n_bins: int = 100,
    output_dir: Optional[Path] = None,
) -> TDAFeatures:
    """
    Compute all TDA features from a point cloud.
    """
    if not HAS_RIPSER:
        raise RuntimeError("ripser not installed. Run: pip install ripser persim")
    
    num_points, num_dims = point_cloud.shape
    
    # Reduce dimensionality if we have more dimensions than points
    # This is critical for TDA - can't find topology in sparse high-dim space
    if num_dims > num_points and HAS_SKLEARN:
        # Project to num_points - 1 dimensions (or 50, whichever is smaller)
        n_components = min(num_points - 1, 50)
        if n_components > 1:
            pca = PCA(n_components=n_components)
            point_cloud = pca.fit_transform(point_cloud)
            # print(f"  PCA: {num_dims}d -> {n_components}d (explained var: {pca.explained_variance_ratio_.sum():.2%})")
    
    # Normalize point cloud (important for consistent filtration)
    point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-8)
    
    # Compute persistence with ripser
    result = ripser(
        point_cloud,
        maxdim=max_dim,
        thresh=max_edge_length,
    )
    
    diagrams = result['dgms']
    
    # === H0 features (connected components) ===
    h0_dgm = diagrams[0]
    h0_finite = h0_dgm[np.isfinite(h0_dgm[:, 1])]
    h0_lifetimes = h0_finite[:, 1] - h0_finite[:, 0] if len(h0_finite) > 0 else np.array([0])
    
    h0_num_components = len(h0_dgm)
    h0_total_persistence = float(h0_lifetimes.sum())
    h0_max_persistence = float(h0_lifetimes.max()) if len(h0_lifetimes) > 0 else 0.0
    h0_entropy = compute_persistent_entropy(h0_dgm)
    
    # === H1 features (loops) ===
    if max_dim >= 1 and len(diagrams) > 1 and len(diagrams[1]) > 0:
        h1_dgm = diagrams[1]
        h1_finite = h1_dgm[np.isfinite(h1_dgm[:, 1])]
        h1_lifetimes = h1_finite[:, 1] - h1_finite[:, 0] if len(h1_finite) > 0 else np.array([0])
        
        h1_num_loops = len(h1_dgm)
        h1_total_persistence = float(h1_lifetimes.sum())
        h1_max_persistence = float(h1_lifetimes.max()) if len(h1_lifetimes) > 0 else 0.0
        h1_entropy = compute_persistent_entropy(h1_dgm)
    else:
        h1_dgm = np.array([])
        h1_num_loops = 0
        h1_total_persistence = 0.0
        h1_max_persistence = 0.0
        h1_entropy = 0.0
    
    # === Betti curves ===
    _, betti0 = compute_betti_curve(h0_dgm, n_bins)
    betti0_area = float(np.trapz(betti0))
    
    if len(h1_dgm) > 0:
        _, betti1 = compute_betti_curve(h1_dgm, n_bins)
        betti1_area = float(np.trapz(betti1))
    else:
        betti1_area = 0.0
    
    # === Derived scores ===
    # Dust score: high entropy relative to component count = fragmented
    dust_score = h0_entropy / (np.log(h0_num_components + 1) + 1e-8)
    
    # Loop score: persistent loops = deliberation
    loop_score = h1_max_persistence * (1 + np.log(h1_num_loops + 1))
    
    # Coherence score: low dust + high loops = coherent reasoning
    coherence_score = (1 / (dust_score + 0.1)) + loop_score
    
    # === Save diagrams ===
    pd_path = None
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd_path = str(output_dir / f"{question_id}_{model_params}_diagrams.npz")
        np.savez(pd_path, h0=h0_dgm, h1=h1_dgm if len(h1_dgm) > 0 else np.array([]))
    
    return TDAFeatures(
        question_id=question_id,
        model_name=model_name,
        model_params=model_params,
        num_points=num_points,
        h0_num_components=h0_num_components,
        h0_total_persistence=h0_total_persistence,
        h0_max_persistence=h0_max_persistence,
        h0_entropy=h0_entropy,
        h1_num_loops=h1_num_loops,
        h1_total_persistence=h1_total_persistence,
        h1_max_persistence=h1_max_persistence,
        h1_entropy=h1_entropy,
        dust_score=float(dust_score),
        loop_score=float(loop_score),
        coherence_score=float(coherence_score),
        betti0_area=betti0_area,
        betti1_area=betti1_area,
        persistence_diagram_path=pd_path,
    )


def run_topological_analysis(
    config: ExperimentConfig,
    subsample: Optional[int] = 500,
):
    """
    Main entry point: Compute TDA features for all activation files.
    """
    print("Scanning for activation files...")
    
    activation_files = list(config.embeddings_dir.glob("*_activations.npz"))
    print(f"Found {len(activation_files)} activation files")
    
    if not activation_files:
        print("No activation files found. Run trace_generator.py first.")
        return []
    
    # Check for existing results to enable resume
    output_file = config.tda_dir / "tda_features.jsonl"
    existing_keys = set()
    
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    key = f"{data['question_id']}_{data['model_params']}"
                    existing_keys.add(key)
                except:
                    pass
        print(f"Resuming: {len(existing_keys)} already processed")
    
    # Process each file
    results = []
    
    for act_file in tqdm(activation_files, desc="Computing TDA"):
        # Parse filename: q_0001_4b_activations.npz
        parts = act_file.stem.split('_')
        question_id = f"{parts[0]}_{parts[1]}"
        model_short = parts[2]
        
        # Map short name to params
        model_params_map = {"4b": "4B", "17b": "1.7B", "06b": "0.6B"}
        model_params = model_params_map.get(model_short, model_short.upper())
        model_name = f"qwen3-{model_params.lower()}"
        
        # Skip if already done
        key = f"{question_id}_{model_params}"
        if key in existing_keys:
            continue
        
        try:
            # Extract point cloud
            point_cloud = extract_point_cloud(str(act_file), subsample=subsample)
            
            # Compute TDA
            features = compute_tda_features(
                point_cloud,
                question_id=question_id,
                model_name=model_name,
                model_params=model_params,
                max_dim=config.tda_max_dim,
                max_edge_length=config.tda_max_edge_length,
                n_bins=config.tda_n_bins,
                output_dir=config.tda_dir / "diagrams",
            )
            
            results.append(features)
            
            # Write immediately (append)
            with open(output_file, 'a') as f:
                f.write(json.dumps(asdict(features)) + "\n")
                
        except Exception as e:
            print(f"\nError processing {act_file.name}: {e}")
            continue
    
    print(f"\nTDA analysis complete!")
    print(f"Processed {len(results)} new traces")
    print(f"Results: {output_file}")
    
    # Summary statistics
    all_results = load_tda_features(config)
    if all_results:
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        for model_params in ["4B", "1.7B", "0.6B"]:
            model_results = [r for r in all_results if r.model_params == model_params]
            if model_results:
                h0_ent = [r.h0_entropy for r in model_results]
                h1_max = [r.h1_max_persistence for r in model_results]
                dust = [r.dust_score for r in model_results]
                
                print(f"\n{model_params} (n={len(model_results)}):")
                print(f"  H₀ Entropy:       {np.mean(h0_ent):.3f} ± {np.std(h0_ent):.3f}")
                print(f"  H₁ Max Persist:   {np.mean(h1_max):.3f} ± {np.std(h1_max):.3f}")
                print(f"  Dust Score:       {np.mean(dust):.3f} ± {np.std(dust):.3f}")
    
    return results


def load_tda_features(config: ExperimentConfig) -> list[TDAFeatures]:
    """Load TDA features from file."""
    features = []
    features_file = config.tda_dir / "tda_features.jsonl"
    
    if not features_file.exists():
        return features
    
    with open(features_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                features.append(TDAFeatures(**data))
            except:
                continue
    
    return features


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute TDA features")
    parser.add_argument("--subsample", type=int, default=500,
                       help="Max points per cloud (0 for all)")
    
    args = parser.parse_args()
    
    config = ExperimentConfig()
    run_topological_analysis(
        config,
        subsample=args.subsample if args.subsample > 0 else None,
    )
