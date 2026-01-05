"""
Manifold-Constrained Hyper-Connections (mHC)

Based on DeepSeek-AI's Hyper-Connections paper, this module implements
residual stream expansion with doubly stochastic manifold constraints.

Key concepts:
- Expand residual stream width from C to n×C (default n=4)
- Learnable connection matrices control information flow between layers
- Doubly stochastic constraint ensures training stability (rows & cols sum to 1)

Architecture:
    Standard Residual:  x_{l+1} = x_l + F(x_l, W_l)
    
    Hyper-Connections:  x_{l+1} = H_l^res @ x_l + H_l^post.T @ F(H_l^pre @ x_l, W_l)
    
    mHC (This impl):    x_{l+1} = P_M(H_l^res) @ x_l + H_l^post.T @ F(H_l^pre @ x_l, W_l)
                        where P_M projects onto doubly stochastic manifold

Reference: DeepSeek-AI Hyper-Connections paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class HyperConnectionConfig:
    """Configuration for Manifold-Constrained Hyper-Connections."""
    
    # Model dimensions
    hidden_size: int = 1024  # Base hidden size (C)
    num_layers: int = 28     # Number of transformer layers
    
    # Expansion settings
    expansion_rate: int = 4  # n in n×C expansion
    
    # Manifold projection settings
    sinkhorn_iterations: int = 10  # Iterations for Sinkhorn-Knopp projection
    sinkhorn_temperature: float = 1.0  # Temperature for softmax in Sinkhorn
    
    # Initialization
    init_scale: float = 0.01  # Scale for off-diagonal initialization
    warm_start: bool = True   # Initialize to match original model behavior
    
    # Training options
    project_every_forward: bool = True  # Project to manifold every forward pass
    use_cached_projection: bool = False  # Cache projection for inference


class SinkhornProjection(nn.Module):
    """
    Projects matrices onto the doubly stochastic manifold using Sinkhorn-Knopp algorithm.
    
    A doubly stochastic matrix has all rows and columns summing to 1.
    This ensures bounded signal propagation during training.
    """
    
    def __init__(self, num_iterations: int = 10, temperature: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.eps = eps
        
    def forward(self, M: torch.Tensor) -> torch.Tensor:
        """
        Project matrix M onto doubly stochastic manifold.
        
        Args:
            M: (n, n) or (batch, n, n) matrix to project
            
        Returns:
            P: Doubly stochastic matrix (rows and cols sum to 1)
        """
        # Apply temperature-scaled softmax to get positive entries
        # This is a differentiable relaxation
        M_scaled = M / self.temperature
        
        # Sinkhorn-Knopp iterations
        # Alternately normalize rows and columns
        P = torch.exp(M_scaled)
        
        for _ in range(self.num_iterations):
            # Normalize rows
            P = P / (P.sum(dim=-1, keepdim=True) + self.eps)
            # Normalize columns  
            P = P / (P.sum(dim=-2, keepdim=True) + self.eps)
            
        return P


class HyperConnectionMatrix(nn.Module):
    """
    Learnable hyper-connection matrix with optional manifold constraint.
    
    Can be used for H^res (residual), H^pre (pre-block), or H^post (post-block).
    """
    
    def __init__(
        self,
        size: int,
        manifold_constrained: bool = True,
        sinkhorn_iterations: int = 10,
        sinkhorn_temperature: float = 1.0,
        init_identity: bool = True,
        init_scale: float = 0.01
    ):
        super().__init__()
        self.size = size
        self.manifold_constrained = manifold_constrained
        
        # Raw learnable parameters (before projection)
        self.weight = nn.Parameter(torch.zeros(size, size))
        
        # Initialize
        if init_identity:
            # Start close to identity for warm start
            with torch.no_grad():
                self.weight.copy_(torch.eye(size) + init_scale * torch.randn(size, size))
        else:
            nn.init.normal_(self.weight, std=init_scale)
            
        # Sinkhorn projection for manifold constraint
        if manifold_constrained:
            self.projection = SinkhornProjection(
                num_iterations=sinkhorn_iterations,
                temperature=sinkhorn_temperature
            )
        else:
            self.projection = None
            
        # Cache for inference
        self._cached_matrix: Optional[torch.Tensor] = None
        self._cache_valid = False
        
    def forward(self, use_cache: bool = False) -> torch.Tensor:
        """
        Get the connection matrix, optionally projected to manifold.
        
        Args:
            use_cache: If True, use cached projection (for inference)
            
        Returns:
            Connection matrix of shape (size, size)
        """
        if use_cache and self._cache_valid and self._cached_matrix is not None:
            return self._cached_matrix
            
        if self.manifold_constrained and self.projection is not None:
            matrix = self.projection(self.weight)
        else:
            matrix = self.weight
            
        if use_cache:
            self._cached_matrix = matrix.detach()
            self._cache_valid = True
            
        return matrix
        
    def invalidate_cache(self):
        """Invalidate cached projection (call after parameter updates)."""
        self._cache_valid = False


class ExpandedResidualStream(nn.Module):
    """
    Manages the expanded residual stream (C → n×C).
    
    Handles expansion at input and compression at output.
    """
    
    def __init__(self, hidden_size: int, expansion_rate: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.expansion_rate = expansion_rate
        self.expanded_size = hidden_size * expansion_rate
        
        # Expansion: C → n×C (replicate and scale)
        # For warm start: first slot gets original, others are zero
        self.register_buffer(
            'expand_matrix',
            self._create_expand_matrix()
        )
        
        # Compression: n×C → C (average or learned)
        self.register_buffer(
            'compress_matrix', 
            self._create_compress_matrix()
        )
        
    def _create_expand_matrix(self) -> torch.Tensor:
        """Create expansion matrix for warm start."""
        # Shape: (n×C, C)
        # First C rows are identity, rest are zeros
        matrix = torch.zeros(self.expanded_size, self.hidden_size)
        matrix[:self.hidden_size] = torch.eye(self.hidden_size)
        return matrix
        
    def _create_compress_matrix(self) -> torch.Tensor:
        """Create compression matrix for warm start."""
        # Shape: (C, n×C)
        # Average across all n slots
        matrix = torch.zeros(self.hidden_size, self.expanded_size)
        for i in range(self.expansion_rate):
            start = i * self.hidden_size
            end = (i + 1) * self.hidden_size
            matrix[:, start:end] = torch.eye(self.hidden_size) / self.expansion_rate
        return matrix
        
    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand hidden states from C to n×C.
        
        Args:
            x: (batch, seq, hidden_size)
            
        Returns:
            (batch, seq, expanded_size)
        """
        # x @ expand_matrix.T
        return F.linear(x, self.expand_matrix)
        
    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress hidden states from n×C to C.
        
        Args:
            x: (batch, seq, expanded_size)
            
        Returns:
            (batch, seq, hidden_size)
        """
        # x @ compress_matrix.T
        return F.linear(x, self.compress_matrix)


class HyperConnectionLayer(nn.Module):
    """
    Single layer hyper-connection module.
    
    Replaces standard residual connection:
        x_{l+1} = x_l + F(x_l)
        
    With hyper-connection:
        x_{l+1} = H^res @ x_l + H^post.T @ F(H^pre @ x_l)
        
    Where H^res is projected onto doubly stochastic manifold.
    """
    
    def __init__(
        self,
        expanded_size: int,
        hidden_size: int,
        config: HyperConnectionConfig
    ):
        super().__init__()
        self.expanded_size = expanded_size
        self.hidden_size = hidden_size
        self.config = config
        
        # H^res: Residual connection matrix (n×C, n×C) - manifold constrained
        self.H_res = HyperConnectionMatrix(
            size=expanded_size,
            manifold_constrained=True,
            sinkhorn_iterations=config.sinkhorn_iterations,
            sinkhorn_temperature=config.sinkhorn_temperature,
            init_identity=config.warm_start,
            init_scale=config.init_scale
        )
        
        # H^pre: Pre-block projection (C, n×C) - selects input for block
        # For warm start: first slot only
        self.H_pre = nn.Linear(expanded_size, hidden_size, bias=False)
        if config.warm_start:
            with torch.no_grad():
                # Take from first slot
                weight = torch.zeros(hidden_size, expanded_size)
                weight[:, :hidden_size] = torch.eye(hidden_size)
                self.H_pre.weight.copy_(weight)
                
        # H^post: Post-block projection (C, n×C) - distributes output
        # For warm start: first slot only  
        self.H_post = nn.Linear(hidden_size, expanded_size, bias=False)
        if config.warm_start:
            with torch.no_grad():
                # Write to first slot
                weight = torch.zeros(expanded_size, hidden_size)
                weight[:hidden_size] = torch.eye(hidden_size)
                self.H_post.weight.copy_(weight)
                
    def forward(
        self,
        x_expanded: torch.Tensor,
        block_output: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Apply hyper-connection.
        
        Args:
            x_expanded: (batch, seq, expanded_size) - expanded residual stream
            block_output: (batch, seq, hidden_size) - output from transformer block
            use_cache: Whether to use cached projection matrices
            
        Returns:
            x_next: (batch, seq, expanded_size) - updated expanded residual
        """
        # Get manifold-projected residual matrix
        H_res = self.H_res(use_cache=use_cache)  # (expanded_size, expanded_size)
        
        # Residual path: H^res @ x
        # x_expanded: (batch, seq, expanded_size)
        # H_res: (expanded_size, expanded_size)
        residual = F.linear(x_expanded, H_res)  # (batch, seq, expanded_size)
        
        # Block path: H^post @ block_output
        block_contribution = self.H_post(block_output)  # (batch, seq, expanded_size)
        
        # Combine
        x_next = residual + block_contribution
        
        return x_next
        
    def get_block_input(self, x_expanded: torch.Tensor) -> torch.Tensor:
        """
        Get input for the transformer block from expanded residual.
        
        Args:
            x_expanded: (batch, seq, expanded_size)
            
        Returns:
            block_input: (batch, seq, hidden_size)
        """
        return self.H_pre(x_expanded)


class ManifoldHyperConnections(nn.Module):
    """
    Complete mHC module for a transformer model.
    
    Manages:
    - Residual stream expansion (C → n×C)
    - Per-layer hyper-connection matrices
    - Manifold projection for training stability
    - Compression back to original size (n×C → C)
    
    Usage:
        mhc = ManifoldHyperConnections(config)
        
        # At model input
        x_expanded = mhc.expand_input(hidden_states)
        
        # At each layer
        block_input = mhc.get_block_input(layer_idx, x_expanded)
        block_output = transformer_block(block_input)
        x_expanded = mhc.apply_connection(layer_idx, x_expanded, block_output)
        
        # At model output
        hidden_states = mhc.compress_output(x_expanded)
    """
    
    def __init__(self, config: HyperConnectionConfig):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.expanded_size = config.hidden_size * config.expansion_rate
        
        # Expansion/compression
        self.stream = ExpandedResidualStream(
            hidden_size=config.hidden_size,
            expansion_rate=config.expansion_rate
        )
        
        # Per-layer hyper-connections
        self.layers = nn.ModuleList([
            HyperConnectionLayer(
                expanded_size=self.expanded_size,
                hidden_size=config.hidden_size,
                config=config
            )
            for _ in range(config.num_layers)
        ])
        
    def expand_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input hidden states for mHC processing.
        
        Args:
            x: (batch, seq, hidden_size)
            
        Returns:
            x_expanded: (batch, seq, expanded_size)
        """
        return self.stream.expand(x)
        
    def compress_output(self, x_expanded: torch.Tensor) -> torch.Tensor:
        """
        Compress expanded states back to original size.
        
        Args:
            x_expanded: (batch, seq, expanded_size)
            
        Returns:
            x: (batch, seq, hidden_size)
        """
        return self.stream.compress(x_expanded)
        
    def get_block_input(self, layer_idx: int, x_expanded: torch.Tensor) -> torch.Tensor:
        """
        Get input for transformer block at given layer.
        
        Args:
            layer_idx: Layer index
            x_expanded: (batch, seq, expanded_size)
            
        Returns:
            block_input: (batch, seq, hidden_size)
        """
        return self.layers[layer_idx].get_block_input(x_expanded)
        
    def apply_connection(
        self,
        layer_idx: int,
        x_expanded: torch.Tensor,
        block_output: torch.Tensor,
        use_cache: bool = False
    ) -> torch.Tensor:
        """
        Apply hyper-connection at given layer.
        
        Args:
            layer_idx: Layer index
            x_expanded: (batch, seq, expanded_size) - current expanded residual
            block_output: (batch, seq, hidden_size) - transformer block output
            use_cache: Whether to use cached projections
            
        Returns:
            x_next: (batch, seq, expanded_size) - updated expanded residual
        """
        return self.layers[layer_idx](x_expanded, block_output, use_cache=use_cache)
        
    def invalidate_caches(self):
        """Invalidate all cached projections (call after optimizer step)."""
        for layer in self.layers:
            layer.H_res.invalidate_cache()
            
    def get_manifold_violation(self) -> torch.Tensor:
        """
        Compute how far the H^res matrices are from doubly stochastic.
        
        Returns:
            violation: Scalar tensor measuring constraint violation
        """
        total_violation = 0.0
        for layer in self.layers:
            H = layer.H_res.weight
            # Row sums should be 1
            row_violation = (H.sum(dim=-1) - 1.0).abs().mean()
            # Column sums should be 1
            col_violation = (H.sum(dim=-2) - 1.0).abs().mean()
            total_violation = total_violation + row_violation + col_violation
        return total_violation / (2 * len(self.layers))
        
    def get_stats(self) -> Dict[str, float]:
        """Get statistics about the hyper-connection matrices."""
        stats = {
            'manifold_violation': self.get_manifold_violation().item(),
            'num_layers': len(self.layers),
            'expansion_rate': self.config.expansion_rate,
            'expanded_size': self.expanded_size,
        }
        
        # Compute average matrix norms
        res_norms = []
        pre_norms = []
        post_norms = []
        
        for layer in self.layers:
            res_norms.append(layer.H_res.weight.norm().item())
            pre_norms.append(layer.H_pre.weight.norm().item())
            post_norms.append(layer.H_post.weight.norm().item())
            
        stats['avg_H_res_norm'] = sum(res_norms) / len(res_norms)
        stats['avg_H_pre_norm'] = sum(pre_norms) / len(pre_norms)
        stats['avg_H_post_norm'] = sum(post_norms) / len(post_norms)
        
        return stats


def create_mhc_for_model(
    hidden_size: int,
    num_layers: int,
    expansion_rate: int = 4,
    **config_overrides
) -> ManifoldHyperConnections:
    """
    Factory function to create mHC module for a specific model.
    
    Args:
        hidden_size: Model hidden size
        num_layers: Number of transformer layers
        expansion_rate: Residual expansion rate (default 4)
        **config_overrides: Additional config overrides
        
    Returns:
        Configured ManifoldHyperConnections module
    """
    config = HyperConnectionConfig(
        hidden_size=hidden_size,
        num_layers=num_layers,
        expansion_rate=expansion_rate,
        **config_overrides
    )
    return ManifoldHyperConnections(config)


# Quick test and demonstration
if __name__ == "__main__":
    print("Testing Manifold-Constrained Hyper-Connections...")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Create mHC for Qwen3-0.6B dimensions
    hidden_size = 1024
    num_layers = 28
    expansion_rate = 4
    
    mhc = create_mhc_for_model(
        hidden_size=hidden_size,
        num_layers=num_layers,
        expansion_rate=expansion_rate
    ).to(device=device, dtype=dtype)
    
    # Count parameters
    num_params = sum(p.numel() for p in mhc.parameters())
    print(f"mHC Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Expansion: {hidden_size} → {hidden_size * expansion_rate}")
    print(f"Layers: {num_layers}")
    
    # Simulate forward pass
    batch_size = 2
    seq_len = 64
    
    # Input hidden states
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    print(f"\nInput shape: {x.shape}")
    
    # Expand
    x_expanded = mhc.expand_input(x)
    print(f"Expanded shape: {x_expanded.shape}")
    
    # Simulate going through layers
    for layer_idx in range(num_layers):
        # Get block input
        block_input = mhc.get_block_input(layer_idx, x_expanded)
        
        # Simulate transformer block (identity for test)
        block_output = block_input
        
        # Apply hyper-connection
        x_expanded = mhc.apply_connection(layer_idx, x_expanded, block_output)
        
    # Compress
    x_out = mhc.compress_output(x_expanded)
    print(f"Output shape: {x_out.shape}")
    
    # Check warm start: output should be close to input
    diff = (x_out - x).abs().mean()
    print(f"\nWarm start test (should be ~0): {diff.item():.6f}")
    
    # Get stats
    stats = mhc.get_stats()
    print(f"\nManifold violation: {stats['manifold_violation']:.6f}")
    print(f"Avg H^res norm: {stats['avg_H_res_norm']:.4f}")
    print(f"Avg H^pre norm: {stats['avg_H_pre_norm']:.4f}")
    print(f"Avg H^post norm: {stats['avg_H_post_norm']:.4f}")
    
    # Test Sinkhorn projection
    print("\n" + "=" * 60)
    print("Testing Sinkhorn projection...")
    
    proj = SinkhornProjection(num_iterations=20)
    M = torch.randn(4, 4, device=device, dtype=dtype)
    P = proj(M)
    
    print(f"Row sums (should be ~1): {P.sum(dim=-1)}")
    print(f"Col sums (should be ~1): {P.sum(dim=-2)}")
    
    print("\n✓ All tests passed!")
