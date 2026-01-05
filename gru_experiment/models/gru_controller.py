"""
GRU Meta-Controller for Reasoning Intervention

Architecture inspired by Google's "Emergent Temporal Abstraction" paper.
Adapted for topology-guided reasoning correction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Configuration for GRU controller."""
    # Input projection (from model hidden size)
    model_hidden_size: int = 1024  # Qwen3-0.6B hidden size
    projected_dim: int = 256
    
    # GRU Encoder
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.1
    
    # Latent space
    latent_dim: int = 64
    
    # Gate
    gate_hidden_dim: int = 128
    gate_threshold: float = 0.5
    
    # Decoder (low-rank)
    decoder_rank: int = 16
    
    # Entropy proxy
    entropy_window: int = 32
    entropy_ema_decay: float = 0.9


class EntropyProxy(nn.Module):
    """
    Fast local entropy estimator.
    
    Instead of computing full TDA (expensive), we estimate fragmentation
    from activation statistics. High variance / low coherence = high entropy.
    """
    
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.window_size = config.entropy_window
        self.ema_decay = config.entropy_ema_decay
        
        # Input projection (from model hidden size to projected dim)
        self.input_proj = nn.Linear(config.model_hidden_size, config.projected_dim)
        
        # Learnable entropy estimator
        self.entropy_net = nn.Sequential(
            nn.Linear(config.projected_dim * 2, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Running EMA of entropy
        self.register_buffer('ema_entropy', torch.tensor(0.5))
        
    def forward(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Estimate local entropy from activation window.
        
        Args:
            activations: (batch, seq_len, dim) - recent activations
            
        Returns:
            entropy_estimate: (batch,) - value in [0, 1], higher = more fragmented
        """
        batch_size = activations.shape[0]
        
        # Use last window_size tokens
        window = activations[:, -self.window_size:, :]
        
        # Project to lower dimension (cast to network dtype first)
        window = self.input_proj(window.to(self.input_proj.weight.dtype))  # (batch, window, projected_dim)
        
        # Feature 1: Mean activation
        mean_act = window.mean(dim=1)  # (batch, projected_dim)
        
        # Feature 2: Variance across tokens (high = fragmented)
        var_act = window.var(dim=1)  # (batch, projected_dim)
        
        # Feature 3: Cosine coherence (low = fragmented) - computed but not used currently
        if window.shape[1] > 1:
            cos_sim = F.cosine_similarity(
                window[:, :-1, :], 
                window[:, 1:, :], 
                dim=-1
            )  # (batch, seq-1)
            coherence = cos_sim.mean(dim=-1, keepdim=True)  # (batch, 1)
        else:
            coherence = torch.ones(batch_size, 1, device=activations.device)
        
        # Combine features
        features = torch.cat([mean_act, var_act], dim=-1)  # (batch, projected_dim*2)
        
        # Learned entropy estimate
        entropy = self.entropy_net(features).squeeze(-1)  # (batch,)
        
        # Update EMA (during training)
        if self.training:
            with torch.no_grad():
                batch_entropy = entropy.mean()
                self.ema_entropy = (
                    self.ema_decay * self.ema_entropy + 
                    (1 - self.ema_decay) * batch_entropy
                )
        
        return entropy


class GRUEncoder(nn.Module):
    """
    GRU encoder that maintains a rolling hidden state.
    
    Compresses the reasoning trajectory into a latent representation,
    independent of context window limitations.
    """
    
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(
            config.model_hidden_size, 
            config.projected_dim
        )
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=config.projected_dim,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            dropout=config.encoder_dropout if config.encoder_num_layers > 1 else 0,
            batch_first=True
        )
        
        # Project to latent space
        self.to_latent = nn.Linear(
            config.encoder_hidden_dim, 
            config.latent_dim
        )
        
    def forward(
        self, 
        activations: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode activations into latent space.
        
        Args:
            activations: (batch, seq_len, model_hidden_size)
            hidden: Optional previous hidden state
            
        Returns:
            latent: (batch, latent_dim) - compressed representation
            hidden: (num_layers, batch, hidden_dim) - for next step
        """
        # Project input (cast to network dtype)
        x = self.input_proj(activations.to(self.input_proj.weight.dtype))  # (batch, seq, projected_dim)
        
        # Run through GRU
        output, hidden = self.gru(x, hidden)  # output: (batch, seq, hidden_dim)
        
        # Take final hidden state and project to latent
        final_hidden = hidden[-1]  # (batch, hidden_dim) - last layer
        latent = self.to_latent(final_hidden)  # (batch, latent_dim)
        
        return latent, hidden


class InterventionGate(nn.Module):
    """
    Learned gate that decides when to intervene.
    
    Opens (β→1) when:
    - Local entropy is high (fragmentation detected)
    - Trajectory is diverging from coherent patterns
    
    Stays closed (β→0) when:
    - Reasoning is coherent
    - Current plan is working
    """
    
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.threshold = config.gate_threshold
        
        # Gate network: takes latent + entropy estimate
        self.gate_net = nn.Sequential(
            nn.Linear(config.latent_dim + 1, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, config.gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gate_hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        latent: torch.Tensor, 
        entropy: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gate activation.
        
        Args:
            latent: (batch, latent_dim) - encoded trajectory
            entropy: (batch,) - local entropy estimate
            
        Returns:
            gate: (batch, 1) - intervention strength in [0, 1]
        """
        # Concatenate latent and entropy
        gate_input = torch.cat([
            latent, 
            entropy.unsqueeze(-1)
        ], dim=-1)  # (batch, latent_dim + 1)
        
        # Cast to network dtype for consistency
        gate_input = gate_input.to(self.gate_net[0].weight.dtype)
        
        gate = self.gate_net(gate_input)  # (batch, 1)
        
        return gate


class LowRankDecoder(nn.Module):
    """
    Decoder that generates steering vectors using low-rank decomposition.
    
    Instead of outputting a full (model_hidden_size,) vector directly,
    we output two smaller matrices A and B such that:
        steering_vector = A @ z @ B
    
    This reduces parameters and constrains the steering to meaningful subspaces.
    """
    
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        
        # MLP to process latent
        self.latent_processor = nn.Sequential(
            nn.Linear(config.latent_dim, config.encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.ReLU()
        )
        
        # Low-rank decomposition matrices
        # A: (hidden_dim, rank), B: (rank, model_hidden_size)
        self.A = nn.Linear(config.encoder_hidden_dim, config.decoder_rank, bias=False)
        self.B = nn.Linear(config.decoder_rank, config.model_hidden_size, bias=False)
        
        # Small initialization to start with minimal intervention
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.normal_(self.B.weight, std=0.01)
        
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Generate steering vector from latent code.
        
        Args:
            latent: (batch, latent_dim)
            
        Returns:
            steering: (batch, model_hidden_size) - vector to add to residual stream
        """
        # Process latent
        h = self.latent_processor(latent)  # (batch, hidden_dim)
        
        # Low-rank projection
        low_rank = self.A(h)  # (batch, rank)
        steering = self.B(low_rank)  # (batch, model_hidden_size)
        
        return steering


class GRUController(nn.Module):
    """
    Complete GRU Meta-Controller.
    
    Combines:
    - Entropy proxy (detect fragmentation)
    - GRU encoder (compress trajectory)
    - Gate (decide when to intervene)
    - Low-rank decoder (generate steering vector)
    """
    
    def __init__(self, config: ControllerConfig):
        super().__init__()
        self.config = config
        
        # Components
        self.entropy_proxy = EntropyProxy(config)
        self.encoder = GRUEncoder(config)
        self.gate = InterventionGate(config)
        self.decoder = LowRankDecoder(config)
        
        # Track intervention statistics
        self.register_buffer('intervention_rate', torch.tensor(0.0))
        self.register_buffer('num_steps', torch.tensor(0))
        
    def forward(
        self,
        activations: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Process activations and generate intervention.
        
        Args:
            activations: (batch, seq_len, model_hidden_size) - residual stream
            hidden: Optional GRU hidden state from previous step
            return_diagnostics: Whether to return detailed info
            
        Returns:
            Dictionary with:
            - steering: (batch, model_hidden_size) - vector to add (may be zeros)
            - gate: (batch, 1) - gate activation
            - hidden: GRU hidden state for next step
            - entropy: (batch,) - entropy estimate
            - latent: (batch, latent_dim) - encoded trajectory
        """
        batch_size = activations.shape[0]
        device = activations.device
        
        # 1. Estimate local entropy
        entropy = self.entropy_proxy(activations)  # (batch,)
        
        # 2. Encode trajectory
        latent, hidden = self.encoder(activations, hidden)  # (batch, latent_dim)
        
        # 3. Compute gate
        gate_value = self.gate(latent, entropy)  # (batch, 1)
        
        # 4. Generate steering vector
        raw_steering = self.decoder(latent)  # (batch, model_hidden_size)
        
        # 5. Apply gate
        steering = gate_value * raw_steering  # Gated steering
        
        # Update statistics
        if self.training:
            with torch.no_grad():
                intervened = (gate_value > self.config.gate_threshold).float().mean()
                self.num_steps += 1
                alpha = 1.0 / self.num_steps.clamp(min=1)
                self.intervention_rate = (
                    (1 - alpha) * self.intervention_rate + 
                    alpha * intervened
                )
        
        result = {
            'steering': steering,
            'gate': gate_value,
            'hidden': hidden,
            'entropy': entropy,
            'latent': latent
        }
        
        if return_diagnostics:
            result['raw_steering_norm'] = raw_steering.norm(dim=-1)
            result['steering_norm'] = steering.norm(dim=-1)
            result['intervention_rate'] = self.intervention_rate
            
        return result
    
    def reset_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state for new sequences."""
        return torch.zeros(
            self.config.encoder_num_layers,
            batch_size,
            self.config.encoder_hidden_dim,
            device=device
        )
    
    def get_intervention_stats(self) -> Dict[str, float]:
        """Get intervention statistics."""
        return {
            'intervention_rate': self.intervention_rate.item(),
            'num_steps': self.num_steps.item(),
            'ema_entropy': self.entropy_proxy.ema_entropy.item()
        }


def create_controller(
    model_hidden_size: int = 1024,
    config_overrides: Optional[Dict] = None
) -> GRUController:
    """
    Factory function to create a controller.
    
    Args:
        model_hidden_size: Hidden size of the base model
        config_overrides: Optional dict to override default config
        
    Returns:
        Initialized GRUController
    """
    config = ControllerConfig(model_hidden_size=model_hidden_size)
    
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
    return GRUController(config)


# Quick test
if __name__ == "__main__":
    print("Testing GRU Controller...")
    
    # Create controller for Qwen3-0.6B (hidden_size=1024)
    controller = create_controller(model_hidden_size=1024)
    
    # Simulate input
    batch_size = 2
    seq_len = 64
    hidden_size = 1024
    
    activations = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    result = controller(activations, return_diagnostics=True)
    
    print(f"Steering shape: {result['steering'].shape}")
    print(f"Gate values: {result['gate'].squeeze()}")
    print(f"Entropy estimates: {result['entropy']}")
    print(f"Latent shape: {result['latent'].shape}")
    print(f"Steering norm: {result['steering_norm']}")
    
    # Count parameters
    num_params = sum(p.numel() for p in controller.parameters())
    print(f"\nTotal parameters: {num_params:,} ({num_params/1e6:.2f}M)")
