"""
Phase 2 Training: Intervention Training

Train the GRU controller to:
1. Detect when student model is fragmenting (high entropy)
2. Generate steering vectors to nudge toward teacher's coherent patterns
3. Learn sparse gating (only intervene when needed)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm
import argparse

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import GRUController, ControllerConfig, create_controller
from data.trajectory_dataset import create_dataloader


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    data_dir: str = "gru_experiment/data/trajectories"
    layer_name: str = "layer_9"
    max_seq_length: int = 512
    
    # Model
    model_hidden_size: int = 1024  # Qwen3-0.6B
    encoder_hidden_dim: int = 256
    latent_dim: int = 64
    decoder_rank: int = 16
    
    # Training
    epochs: int = 20
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    
    # Loss weights
    distillation_weight: float = 1.0  # MSE to teacher trajectory
    entropy_weight: float = 0.5  # Encourage low entropy interventions
    gate_sparsity_weight: float = 0.1  # Encourage sparse gating
    
    # Hardware
    device: str = "cuda"
    dtype: str = "float16"
    gradient_checkpointing: bool = True
    
    # Logging
    log_every: int = 10
    save_every: int = 5
    checkpoint_dir: str = "gru_experiment/checkpoints"
    
    # Evaluation
    eval_tda: bool = False  # Expensive, do separately
    

class InterventionTrainer:
    """Trainer for GRU intervention controller."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.float16 if config.dtype == "float16" else torch.bfloat16
        
        # Create controller
        controller_config = ControllerConfig(
            model_hidden_size=config.model_hidden_size,
            encoder_hidden_dim=config.encoder_hidden_dim,
            latent_dim=config.latent_dim,
            decoder_rank=config.decoder_rank
        )
        self.controller = GRUController(controller_config).to(self.device)
        
        print(f"Controller parameters: {sum(p.numel() for p in self.controller.parameters()):,}")
        
        # Create dataloader
        self.train_loader = create_dataloader(
            data_dir=config.data_dir,
            layer_name=config.layer_name,
            batch_size=config.batch_size,
            max_seq_length=config.max_seq_length,
            shuffle=True
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self.controller.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler
        total_steps = len(self.train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Tracking
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = []
        
        # Create checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
    def compute_loss(
        self, 
        student_traj: torch.Tensor,
        teacher_traj: torch.Tensor,
        ctrl_output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            student_traj: (batch, seq, hidden) - student activations
            teacher_traj: (batch, seq, hidden) - teacher activations  
            ctrl_output: Controller output dict
            
        Returns:
            Dict with individual losses and total
        """
        steering = ctrl_output['steering']  # (batch, hidden)
        gate = ctrl_output['gate']  # (batch, 1)
        entropy = ctrl_output['entropy']  # (batch,)
        
        # 1. Distillation loss: Push student toward teacher
        # Apply steering to student's last position and compare to teacher
        # Simplified: compare trajectory statistics
        
        student_mean = student_traj.mean(dim=1)  # (batch, hidden)
        teacher_mean = teacher_traj.mean(dim=1)  # (batch, hidden)
        
        # Intervened student
        intervened_mean = student_mean + steering
        
        # MSE to teacher
        distill_loss = F.mse_loss(intervened_mean, teacher_mean)
        
        # 2. Entropy loss: Encourage interventions that reduce entropy
        # We want the steering to push toward coherence
        # Use the entropy proxy as a soft target
        entropy_loss = entropy.mean()  # Lower is better
        
        # 3. Gate sparsity: Encourage sparse intervention
        # Only intervene when really needed
        gate_loss = gate.mean()  # Lower = less intervention
        
        # Total loss
        total_loss = (
            self.config.distillation_weight * distill_loss +
            self.config.entropy_weight * entropy_loss +
            self.config.gate_sparsity_weight * gate_loss
        )
        
        return {
            'total': total_loss,
            'distillation': distill_loss,
            'entropy': entropy_loss,
            'gate_sparsity': gate_loss
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.controller.train()
        
        # Move to device
        student_traj = batch['student_trajectories'].to(self.device, dtype=self.dtype)
        teacher_traj = batch['teacher_trajectories'].to(self.device, dtype=self.dtype)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        ctrl_output = self.controller(student_traj, return_diagnostics=True)
        
        # Compute losses
        losses = self.compute_loss(student_traj, teacher_traj, ctrl_output)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.controller.parameters(), 
                self.config.gradient_clip
            )
        
        # Update
        self.optimizer.step()
        self.scheduler.step()
        
        self.step += 1
        
        # Return losses as floats
        return {k: v.item() for k, v in losses.items()}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {
            'total': 0.0,
            'distillation': 0.0,
            'entropy': 0.0,
            'gate_sparsity': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}")
        
        for batch in pbar:
            losses = self.train_step(batch)
            
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1
            
            # Update progress bar
            if self.step % self.config.log_every == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total']:.4f}",
                    'dist': f"{losses['distillation']:.4f}",
                    'gate': f"{losses['gate_sparsity']:.4f}"
                })
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        self.epoch += 1
        
        return epoch_losses
    
    def save_checkpoint(self, name: str = "latest"):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'controller_state': self.controller.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.__dict__,
            'history': self.history
        }
        
        path = Path(self.config.checkpoint_dir) / f"checkpoint_{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint['history']
        
        self.controller.load_state_dict(checkpoint['controller_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Full training loop."""
        print(f"\nStarting training for {self.config.epochs} epochs")
        print(f"  Batches per epoch: {len(self.train_loader)}")
        print(f"  Total steps: {len(self.train_loader) * self.config.epochs}")
        
        for epoch in range(self.config.epochs):
            # Train epoch
            losses = self.train_epoch()
            
            # Log
            print(f"\nEpoch {self.epoch} complete:")
            print(f"  Total loss: {losses['total']:.4f}")
            print(f"  Distillation: {losses['distillation']:.4f}")
            print(f"  Entropy: {losses['entropy']:.4f}")
            print(f"  Gate sparsity: {losses['gate_sparsity']:.4f}")
            
            # Get intervention stats
            stats = self.controller.get_intervention_stats()
            print(f"  Intervention rate: {stats['intervention_rate']:.2%}")
            print(f"  Avg entropy: {stats['ema_entropy']:.4f}")
            
            # Track history
            self.history.append({
                'epoch': self.epoch,
                'losses': losses,
                'stats': stats
            })
            
            # Save checkpoint
            if self.epoch % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{self.epoch}")
            
            # Save best
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                self.save_checkpoint("best")
        
        # Final save
        self.save_checkpoint("final")
        
        # Save training history
        history_path = Path(self.config.checkpoint_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoints saved to: {self.config.checkpoint_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GRU intervention controller")
    
    # Data args
    parser.add_argument("--data-dir", type=str, default="gru_experiment/data/trajectories")
    parser.add_argument("--layer-name", type=str, default="layer_9")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    # Model args
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--decoder-rank", type=int, default=16)
    
    # Hardware args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16")
    
    # Other
    parser.add_argument("--checkpoint-dir", type=str, default="gru_experiment/checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig(
        data_dir=args.data_dir,
        layer_name=args.layer_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_hidden_size=args.hidden_size,
        encoder_hidden_dim=args.encoder_dim,
        latent_dim=args.latent_dim,
        decoder_rank=args.decoder_rank,
        device=args.device,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Create trainer
    trainer = InterventionTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
