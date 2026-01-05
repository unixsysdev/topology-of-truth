"""
Phase 2 Training: Intervention Training with Logit Distillation

Train the GRU controller using KL divergence from teacher logits as main signal.
The gate and decoder learn implicitly what "good topology" looks like from gradients.

Training Flow:
1. Student generates with GRU intervention at layer L
2. Compare student logits to teacher logits (KL divergence)
3. Backprop through GRU controller (gate learns WHEN, decoder learns WHAT)
4. TDA validates offline that intervention reduces H₀ entropy

Key insight: Logit gradient already encodes topology - TDA is verification, not training signal.
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
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import argparse
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import GRUController, ControllerConfig
from models.hooked_model import HookedModel, HookConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Models
    student_model: str = "Qwen/Qwen3-0.6B"
    teacher_model: str = "Qwen/Qwen3-1.7B"
    intervention_layer: int = 4  # Early intervention (~1/7 into model) gives more room to correct
    intervention_layer_ratio: float = None  # Alternative: set as ratio of model depth (e.g., 0.15)
    
    # GRU Controller
    encoder_hidden_dim: int = 256
    latent_dim: int = 64
    decoder_rank: int = 16
    
    # Data
    dataset: str = "openai/gsm8k"
    max_samples: int = 500
    max_seq_length: int = 1024
    
    # Training
    epochs: int = 20
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip: float = 1.0
    gradient_accumulation_steps: int = 4
    
    # Loss weights
    kl_weight: float = 1.0           # Main: KL divergence to teacher logits
    gate_sparsity_weight: float = 0.1  # Encourage sparse intervention
    
    # Optional: TDA auxiliary loss (disabled by default)
    use_tda_aux_loss: bool = False
    tda_aux_weight: float = 0.1
    
    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"
    gradient_checkpointing: bool = True
    
    # Logging & Checkpoints
    log_every: int = 10
    save_every: int = 5
    eval_every: int = 5  # TDA evaluation frequency (epochs)
    checkpoint_dir: str = "gru_experiment/checkpoints"
    
    # TDA Validation (offline)
    tda_eval_samples: int = 50  # Subset for TDA eval (expensive)


class TDAValidator:
    """
    Offline TDA validation to verify intervention reduces entropy.
    
    Not used for training - just verification that the learned
    intervention actually improves topological coherence.
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        # Import TDA tools from parent project
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from topological_engine import compute_tda_features
            self.compute_tda = compute_tda_features
            self.available = True
        except ImportError:
            print("Warning: TDA tools not available. Validation disabled.")
            self.available = False
    
    def compute_entropy(self, activations: np.ndarray) -> Dict[str, float]:
        """Compute H₀ entropy and dust score from activations."""
        if not self.available:
            return {'h0_entropy': -1, 'dust_score': -1}
        
        try:
            features = self.compute_tda(activations)
            return {
                'h0_entropy': features.get('h0_entropy', -1),
                'dust_score': features.get('dust_score', -1),
                'h1_max_persistence': features.get('h1_max_persistence', -1)
            }
        except Exception as e:
            print(f"TDA computation failed: {e}")
            return {'h0_entropy': -1, 'dust_score': -1}
    
    def compare_trajectories(
        self,
        baseline_activations: np.ndarray,
        intervened_activations: np.ndarray
    ) -> Dict[str, float]:
        """Compare TDA metrics before/after intervention."""
        baseline = self.compute_entropy(baseline_activations)
        intervened = self.compute_entropy(intervened_activations)
        
        return {
            'baseline_h0': baseline['h0_entropy'],
            'intervened_h0': intervened['h0_entropy'],
            'h0_reduction': baseline['h0_entropy'] - intervened['h0_entropy'],
            'baseline_dust': baseline['dust_score'],
            'intervened_dust': intervened['dust_score'],
            'dust_reduction': baseline['dust_score'] - intervened['dust_score']
        }


class InterventionTrainer:
    """
    Trainer for GRU intervention controller.
    
    Main training signal: KL divergence from teacher logits
    Verification: Offline TDA to confirm entropy reduction
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16 if config.dtype == "bfloat16" else torch.float16
        
        print("=" * 60)
        print("Initializing Intervention Trainer")
        print("=" * 60)
        
        # Load student model with hooks
        print(f"\nLoading student: {config.student_model}")
        
        # Determine intervention layer
        if config.intervention_layer_ratio is not None:
            # Get model config to determine num layers
            from transformers import AutoConfig
            model_config = AutoConfig.from_pretrained(config.student_model, trust_remote_code=True)
            num_layers = model_config.num_hidden_layers
            intervention_layer = int(num_layers * config.intervention_layer_ratio)
            print(f"  Using ratio {config.intervention_layer_ratio} -> layer {intervention_layer}/{num_layers}")
        else:
            intervention_layer = config.intervention_layer
        
        self.intervention_layer = intervention_layer
        
        self.student = HookedModel(
            model_name=config.student_model,
            config=HookConfig(intervention_layer=intervention_layer),
            device=config.device,
            dtype=self.dtype
        )
        
        # Load teacher model (frozen, no hooks needed for training)
        print(f"\nLoading teacher: {config.teacher_model}")
        from transformers import AutoModelForCausalLM
        self.teacher = AutoModelForCausalLM.from_pretrained(
            config.teacher_model,
            torch_dtype=self.dtype,
            device_map=config.device,
            trust_remote_code=True
        )
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        # Create GRU controller
        print("\nCreating GRU controller...")
        controller_config = ControllerConfig(
            model_hidden_size=self.student.hidden_size,
            encoder_hidden_dim=config.encoder_hidden_dim,
            latent_dim=config.latent_dim,
            decoder_rank=config.decoder_rank
        )
        self.controller = GRUController(controller_config).to(self.device)
        
        num_params = sum(p.numel() for p in self.controller.parameters())
        print(f"  Controller parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        
        # Freeze student model - only train controller
        for p in self.student.model.parameters():
            p.requires_grad = False
        
        # Optimizer (only controller parameters)
        self.optimizer = AdamW(
            self.controller.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Load dataset
        print(f"\nLoading dataset: {config.dataset}")
        self.train_data = self._load_dataset()
        print(f"  Samples: {len(self.train_data)}")
        
        # Scheduler
        total_steps = (len(self.train_data) // config.batch_size) * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # TDA validator (for offline verification)
        self.tda_validator = TDAValidator(device="cpu")
        
        # Tracking
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        self.history = []
        self.tda_history = []
        
        # Create checkpoint dir
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print("\nInitialization complete!")
        print("=" * 60)
    
    def _load_dataset(self) -> List[Dict]:
        """Load and prepare GSM8K dataset."""
        from datasets import load_dataset
        
        dataset = load_dataset(self.config.dataset, "main", split="train")
        
        if self.config.max_samples:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        # Format prompts
        data = []
        for item in dataset:
            prompt = f"Solve this math problem step by step.\n\nQuestion: {item['question']}\n\nAnswer:"
            gold = item['answer'].split('####')[-1].strip()
            data.append({
                'prompt': prompt,
                'gold_answer': gold,
                'question': item['question']
            })
        
        return data
    
    def _get_batch(self, indices: List[int]) -> Dict[str, torch.Tensor]:
        """Prepare a batch of data."""
        prompts = [self.train_data[i]['prompt'] for i in indices]
        
        # Tokenize
        encodings = self.student.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length
        )
        
        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device)
        }
    
    def compute_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gate_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses.
        
        Args:
            student_logits: (batch, seq, vocab) - student output
            teacher_logits: (batch, seq, vocab) - teacher output
            gate_values: (batch, 1) - intervention gate activation
            
        Returns:
            Dict with individual losses and total
        """
        # 1. KL Divergence: Main distillation loss
        # Align sequence lengths (take minimum)
        min_len = min(student_logits.shape[1], teacher_logits.shape[1])
        student_logits = student_logits[:, :min_len, :]
        teacher_logits = teacher_logits[:, :min_len, :]
        
        # KL divergence (teacher as target)
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean'
        )
        
        # 2. Gate sparsity: Encourage sparse intervention
        gate_loss = gate_values.mean()
        
        # Total loss
        total_loss = (
            self.config.kl_weight * kl_loss +
            self.config.gate_sparsity_weight * gate_loss
        )
        
        return {
            'total': total_loss,
            'kl': kl_loss,
            'gate_sparsity': gate_loss
        }
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with intervention.
        
        Flow:
        1. Run student forward to intervention layer
        2. GRU controller generates steering vector
        3. Continue student forward with steering injected
        4. Get teacher logits (no intervention)
        5. Compute KL loss, backprop through controller
        """
        self.controller.train()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        batch_size = input_ids.shape[0]
        
        # Get teacher logits (no gradient)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            teacher_logits = teacher_outputs.logits
        
        # Forward student to get activations at intervention layer
        # First pass without intervention to get activations
        with torch.no_grad():
            _ = self.student(input_ids=input_ids, attention_mask=attention_mask)
            activations = self.student.get_activations()
        
        # GRU controller processes activations
        ctrl_output = self.controller(activations, return_diagnostics=True)
        steering = ctrl_output['steering']
        gate = ctrl_output['gate']
        
        # Forward student with steering intervention
        with self.student.steering_context(steering):
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        student_logits = student_outputs.logits
        
        # Compute loss
        losses = self.compute_loss(student_logits, teacher_logits, gate)
        
        # Backward (only controller has gradients)
        loss = losses['total'] / self.config.gradient_accumulation_steps
        loss.backward()
        
        self.step += 1
        
        # Gradient accumulation
        if self.step % self.config.gradient_accumulation_steps == 0:
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.controller.parameters(),
                    self.config.gradient_clip
                )
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {'total': 0.0, 'kl': 0.0, 'gate_sparsity': 0.0}
        num_batches = 0
        
        # Shuffle data
        indices = np.random.permutation(len(self.train_data))
        
        pbar = tqdm(
            range(0, len(indices), self.config.batch_size),
            desc=f"Epoch {self.epoch + 1}"
        )
        
        for start_idx in pbar:
            batch_indices = indices[start_idx:start_idx + self.config.batch_size]
            if len(batch_indices) < self.config.batch_size:
                continue  # Skip incomplete batch
            
            batch = self._get_batch(batch_indices.tolist())
            losses = self.train_step(batch)
            
            for k, v in losses.items():
                epoch_losses[k] += v
            num_batches += 1
            
            # Update progress bar
            if self.step % self.config.log_every == 0:
                pbar.set_postfix({
                    'loss': f"{losses['total']:.4f}",
                    'kl': f"{losses['kl']:.4f}",
                    'gate': f"{losses['gate_sparsity']:.3f}"
                })
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(num_batches, 1)
        
        self.epoch += 1
        return epoch_losses
    
    def validate_tda(self, num_samples: int = None) -> Dict[str, float]:
        """
        Offline TDA validation.
        
        Compare H₀ entropy before/after intervention to verify
        the controller is learning to improve coherence.
        """
        if not self.tda_validator.available:
            return {'status': 'TDA not available'}
        
        num_samples = num_samples or self.config.tda_eval_samples
        num_samples = min(num_samples, len(self.train_data))
        
        print(f"\nRunning TDA validation on {num_samples} samples...")
        
        self.controller.eval()
        
        results = {
            'h0_reductions': [],
            'dust_reductions': [],
            'baseline_h0s': [],
            'intervened_h0s': []
        }
        
        indices = np.random.choice(len(self.train_data), num_samples, replace=False)
        
        for idx in tqdm(indices, desc="TDA validation"):
            batch = self._get_batch([idx])
            
            with torch.no_grad():
                # Baseline (no intervention)
                self.student.disable_steering()
                _ = self.student(**batch)
                baseline_act = self.student.get_activations().cpu().numpy()
                
                # With intervention
                ctrl_output = self.controller(
                    self.student.get_activations()
                )
                self.student.set_steering(ctrl_output['steering'])
                self.student.enable_steering()
                _ = self.student(**batch)
                intervened_act = self.student.get_activations().cpu().numpy()
                
                self.student.disable_steering()
            
            # Compute TDA comparison
            comparison = self.tda_validator.compare_trajectories(
                baseline_act.squeeze(0),
                intervened_act.squeeze(0)
            )
            
            if comparison['baseline_h0'] > 0:  # Valid computation
                results['h0_reductions'].append(comparison['h0_reduction'])
                results['dust_reductions'].append(comparison['dust_reduction'])
                results['baseline_h0s'].append(comparison['baseline_h0'])
                results['intervened_h0s'].append(comparison['intervened_h0'])
        
        # Aggregate results
        if results['h0_reductions']:
            summary = {
                'mean_h0_reduction': np.mean(results['h0_reductions']),
                'mean_dust_reduction': np.mean(results['dust_reductions']),
                'mean_baseline_h0': np.mean(results['baseline_h0s']),
                'mean_intervened_h0': np.mean(results['intervened_h0s']),
                'pct_improved': np.mean([r > 0 for r in results['h0_reductions']]) * 100,
                'num_samples': len(results['h0_reductions'])
            }
        else:
            summary = {'status': 'No valid TDA computations'}
        
        return summary
    
    def save_checkpoint(self, name: str = "latest"):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'step': self.step,
            'controller_state': self.controller.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': asdict(self.config) if hasattr(self.config, '__dataclass_fields__') else self.config.__dict__,
            'history': self.history,
            'tda_history': self.tda_history
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
        self.history = checkpoint.get('history', [])
        self.tda_history = checkpoint.get('tda_history', [])
        
        self.controller.load_state_dict(checkpoint['controller_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        print(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self):
        """Full training loop with periodic TDA validation."""
        print(f"\n{'='*60}")
        print("Starting Training")
        print(f"{'='*60}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  TDA validation every: {self.config.eval_every} epochs")
        
        for epoch in range(self.config.epochs):
            # Train epoch
            losses = self.train_epoch()
            
            # Log
            print(f"\n{'='*40}")
            print(f"Epoch {self.epoch} complete:")
            print(f"  Total loss: {losses['total']:.4f}")
            print(f"  KL divergence: {losses['kl']:.4f}")
            print(f"  Gate sparsity: {losses['gate_sparsity']:.4f}")
            
            # Controller stats
            stats = self.controller.get_intervention_stats()
            print(f"  Intervention rate: {stats['intervention_rate']:.2%}")
            print(f"  Avg entropy proxy: {stats['ema_entropy']:.4f}")
            
            # Track history
            self.history.append({
                'epoch': self.epoch,
                'losses': losses,
                'stats': stats
            })
            
            # TDA validation (periodic)
            if self.epoch % self.config.eval_every == 0:
                tda_results = self.validate_tda()
                self.tda_history.append({
                    'epoch': self.epoch,
                    'results': tda_results
                })
                
                if 'mean_h0_reduction' in tda_results:
                    print(f"\n  TDA Validation:")
                    print(f"    H₀ reduction: {tda_results['mean_h0_reduction']:.4f}")
                    print(f"    Dust reduction: {tda_results['mean_dust_reduction']:.4f}")
                    print(f"    % improved: {tda_results['pct_improved']:.1f}%")
                    print(f"    Baseline H₀: {tda_results['mean_baseline_h0']:.3f}")
                    print(f"    Intervened H₀: {tda_results['mean_intervened_h0']:.3f}")
            
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
            json.dump({
                'training': self.history,
                'tda_validation': self.tda_history
            }, f, indent=2)
        
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Checkpoints: {self.config.checkpoint_dir}")
        
        # Final TDA summary
        if self.tda_history:
            final_tda = self.tda_history[-1]['results']
            if 'mean_h0_reduction' in final_tda:
                print(f"\nFinal TDA Results:")
                print(f"  H₀ entropy reduction: {final_tda['mean_h0_reduction']:.4f}")
                print(f"  Samples improved: {final_tda['pct_improved']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Train GRU intervention controller")
    
    # Model args
    parser.add_argument("--student-model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-model", type=str, default="Qwen/Qwen3-1.7B")
    parser.add_argument("--intervention-layer", type=int, default=4,
                        help="Layer to intervene at (default: 4, early intervention)")
    parser.add_argument("--intervention-layer-ratio", type=float, default=None,
                        help="Set intervention layer as ratio of model depth (e.g., 0.15). Overrides --intervention-layer")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=4)
    
    # Controller args
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--decoder-rank", type=int, default=16)
    
    # Data args
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    
    # Hardware args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    
    # Other
    parser.add_argument("--checkpoint-dir", type=str, default="gru_experiment/checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval-every", type=int, default=5, help="TDA validation frequency")
    parser.add_argument("--tda-samples", type=int, default=50, help="Samples for TDA validation")
    
    # Debug/fast iteration mode
    parser.add_argument("--debug", action="store_true", 
                        help="Fast debug mode: 20 samples, 2 epochs, no TDA validation")
    parser.add_argument("--fast", action="store_true",
                        help="Fast iteration mode: 100 samples, 5 epochs, TDA every 5 epochs")
    
    args = parser.parse_args()
    
    # Apply debug/fast presets
    if args.debug:
        print("=" * 60)
        print("DEBUG MODE: Fast iteration with minimal data")
        print("=" * 60)
        args.max_samples = 20
        args.epochs = 2
        args.eval_every = 99  # Effectively disable TDA validation
        args.tda_samples = 5
        args.batch_size = 2
        args.grad_accum = 1
        args.save_every = 1
    elif args.fast:
        print("=" * 60)
        print("FAST MODE: Quick training with reduced data")
        print("=" * 60)
        args.max_samples = 100
        args.epochs = 5
        args.eval_every = 5
        args.tda_samples = 20
    
    # Create config
    config = TrainingConfig(
        student_model=args.student_model,
        teacher_model=args.teacher_model,
        intervention_layer=args.intervention_layer,
        intervention_layer_ratio=args.intervention_layer_ratio,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        gradient_accumulation_steps=args.grad_accum,
        encoder_hidden_dim=args.encoder_dim,
        latent_dim=args.latent_dim,
        decoder_rank=args.decoder_rank,
        max_samples=args.max_samples,
        max_seq_length=args.max_seq_length,
        device=args.device,
        dtype=args.dtype,
        checkpoint_dir=args.checkpoint_dir,
        eval_every=args.eval_every,
        tda_eval_samples=args.tda_samples
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
