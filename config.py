"""
Configuration for Topology of Truth experiment.

We analyze the residual stream topology of 3 Qwen3 models (0.6B, 1.7B, 4B)
on the LIMO dataset, using TDA to measure manifold coherence.

The 8 Truth Buckets (4B, 1.7B, 0.6B):
  000: All wrong (The Void)
  001: Only 0.6B correct (The Anomaly - smallest got it?) 
  010: Only 1.7B correct (Middle Upset)
  011: 1.7B + 0.6B correct (4B Failure)
  100: Only 4B correct (The Gap) - largest-only success
  101: 4B + 0.6B correct (Middle Collapse)
  110: 4B + 1.7B correct (Baseline Failure)
  111: All correct (Triviality)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    hf_id: str
    params: str  # e.g., "4B", "1.7B", "0.6B"
    
    @property
    def short_name(self) -> str:
        return self.params.lower().replace(".", "")  # "4b", "17b", "06b"


# The 3 models we have locally
MODELS = [
    ModelConfig(name="qwen3-4b", hf_id="Qwen/Qwen3-4B", params="4B"),
    ModelConfig(name="qwen3-1.7b", hf_id="Qwen/Qwen3-1.7B", params="1.7B"),
    ModelConfig(name="qwen3-0.6b", hf_id="Qwen/Qwen3-0.6B", params="0.6B"),
]


@dataclass
class ExperimentConfig:
    """Master configuration for the topology audit."""
    
    # === Paths ===
    output_dir: Path = Path("./results")
    traces_dir: Path = Path("./results/traces")
    embeddings_dir: Path = Path("./results/embeddings")
    tda_dir: Path = Path("./results/tda")
    viz_dir: Path = Path("./results/visualizations")
    
    # === Dataset ===
    dataset_name: str = "GAIR/LIMO"
    dataset_split: str = "train"
    max_samples: Optional[int] = None  # None = use all 817
    
    # === Generation ===
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.95
    system_prompt: str = (
        "/nothink Please reason step by step, and put your final answer within \\boxed{}."
    )
    
    # === Residual Stream Extraction ===
    extract_layers: Literal["all", "last", "every_4th"] = "all"
    extract_positions: Literal["all", "last", "reasoning_tokens"] = "all"
    
    # === TDA Parameters ===
    tda_max_dim: int = 1  # Compute H0 and H1
    tda_max_edge_length: float = 2.0  # Max filtration radius
    tda_n_bins: int = 100  # For Betti curves
    
    # === Verification API ===
    verifier_api_url: str = "https://llm.chutes.ai/v1/chat/completions"
    verifier_model: str = "Qwen/Qwen3-235B-A22B-Thinking-2507"
    verifier_batch_size: int = 10  # Questions per API call
    
    # === Visualization ===
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 3
    
    def __post_init__(self):
        """Create output directories."""
        for d in [self.output_dir, self.traces_dir, self.embeddings_dir, 
                  self.tda_dir, self.viz_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Truth bucket utilities
BUCKET_NAMES = {
    "000": "The Void (All Wrong)",
    "001": "The Anomaly (Only 0.6B)",
    "010": "Middle Upset (Only 1.7B)",
    "011": "4B Failure",
    "100": "The Gap (Only 4B)",
    "101": "Middle Collapse",
    "110": "Baseline Failure",
    "111": "Triviality (All Correct)",
}


def get_bucket(correct_4b: bool, correct_17b: bool, correct_06b: bool) -> str:
    """Get truth bucket code from correctness booleans."""
    return f"{int(correct_4b)}{int(correct_17b)}{int(correct_06b)}"


def bucket_to_bools(bucket: str) -> tuple[bool, bool, bool]:
    """Convert bucket code to (4B, 1.7B, 0.6B) correctness."""
    return bool(int(bucket[0])), bool(int(bucket[1])), bool(int(bucket[2]))
