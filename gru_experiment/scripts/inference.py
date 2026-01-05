#!/usr/bin/env python
"""
Inference script to compare baseline vs intervened model outputs.

Loads a trained controller checkpoint and generates responses with/without intervention
so you can see the actual difference in outputs.

Usage:
    python gru_experiment/scripts/inference.py
    python gru_experiment/scripts/inference.py --checkpoint path/to/checkpoint.pt
    python gru_experiment/scripts/inference.py --question "What is 15 + 27?"
    python gru_experiment/scripts/inference.py --interactive
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from typing import Optional, List, Dict

from models import GRUController, ControllerConfig, HookedModel, HookConfig


class IntervenedGenerator:
    """Generate text with optional GRU controller intervention."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        checkpoint_path: Optional[str] = None,
        intervention_layer: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        self.device = device
        self.dtype = dtype
        
        print(f"Loading model: {model_name}")
        self.model = HookedModel(
            model_name=model_name,
            config=HookConfig(intervention_layer=intervention_layer),
            device=device,
            dtype=dtype
        )
        
        self.controller = None
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load controller from checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Get config from checkpoint or use defaults
        config_dict = checkpoint.get('config', {})
        
        # Check if intervention layer matches
        saved_layer = config_dict.get('intervention_layer', None)
        if saved_layer is not None and saved_layer != self.model.config.intervention_layer:
            print(f"  WARNING: Checkpoint was trained on layer {saved_layer}, but model is using layer {self.model.config.intervention_layer}")
            print(f"  Updating model to use layer {saved_layer}")
            # Re-register hooks at correct layer
            self.model.remove_hooks()
            self.model.config.intervention_layer = saved_layer
            self.model._register_hooks()
        
        controller_config = ControllerConfig(
            model_hidden_size=self.model.hidden_size,
            encoder_hidden_dim=config_dict.get('encoder_hidden_dim', 256),
            latent_dim=config_dict.get('latent_dim', 64),
            decoder_rank=config_dict.get('decoder_rank', 16)
        )
        
        self.controller = GRUController(controller_config).to(
            device=self.device, 
            dtype=self.dtype
        )
        self.controller.load_state_dict(checkpoint['controller_state'])
        self.controller.eval()
        
        print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}")
        print(f"  Best loss: {checkpoint.get('best_loss', '?'):.4f}" if isinstance(checkpoint.get('best_loss'), float) else f"  Best loss: {checkpoint.get('best_loss', '?')}")
        print(f"  Intervention layer: {config_dict.get('intervention_layer', 'unknown')}")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        use_intervention: bool = True
    ) -> Dict[str, str]:
        """
        Generate response with and without intervention.
        
        Returns dict with 'baseline' and 'intervened' outputs.
        """
        # Tokenize
        inputs = self.model.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = inputs['input_ids'].shape[1]
        
        results = {}
        
        # Baseline generation (no intervention)
        self.model.disable_steering()
        with torch.no_grad():
            baseline_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.model.tokenizer.pad_token_id
            )
        
        baseline_text = self.model.tokenizer.decode(
            baseline_output[0][input_len:], 
            skip_special_tokens=True
        )
        results['baseline'] = baseline_text
        
        # Intervened generation (if controller loaded)
        if self.controller is not None and use_intervention:
            # Get activations from a forward pass
            with torch.no_grad():
                _ = self.model(**inputs)
                activations = self.model.get_activations()
                
                # Get steering from controller
                ctrl_out = self.controller(activations, return_diagnostics=True)
                steering = ctrl_out['steering']
                gate = ctrl_out['gate']
            
            # Generate with steering
            self.model.set_steering(steering)
            self.model.enable_steering()
            
            with torch.no_grad():
                intervened_output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.model.tokenizer.pad_token_id
                )
            
            self.model.disable_steering()
            
            intervened_text = self.model.tokenizer.decode(
                intervened_output[0][input_len:],
                skip_special_tokens=True
            )
            results['intervened'] = intervened_text
            results['gate_value'] = gate.item()
            results['steering_norm'] = steering.norm().item()
        
        return results
    
    def compare_on_questions(
        self, 
        questions: List[str],
        max_new_tokens: int = 512
    ):
        """Run comparison on multiple questions."""
        
        for i, question in enumerate(questions):
            prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
            
            print(f"\n{'='*70}")
            print(f"Question {i+1}: {question}")
            print('='*70)
            
            results = self.generate(prompt, max_new_tokens=max_new_tokens)
            
            print(f"\n--- BASELINE (no intervention) ---")
            print(results['baseline'][:1000])
            if len(results['baseline']) > 1000:
                print("... [truncated]")
            
            if 'intervened' in results:
                print(f"\n--- INTERVENED (gate={results['gate_value']:.3f}, steering_norm={results['steering_norm']:.3f}) ---")
                print(results['intervened'][:1000])
                if len(results['intervened']) > 1000:
                    print("... [truncated]")
    
    def interactive(self, max_new_tokens: int = 512):
        """Interactive mode - ask questions and see both outputs."""
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("Enter math questions to see baseline vs intervened outputs.")
        print("Type 'quit' to exit.")
        print("="*70)
        
        while True:
            try:
                question = input("\nQuestion: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not question:
                    continue
                
                prompt = f"Solve this math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
                results = self.generate(prompt, max_new_tokens=max_new_tokens)
                
                print(f"\n--- BASELINE ---")
                print(results['baseline'])
                
                if 'intervened' in results:
                    print(f"\n--- INTERVENED (gate={results['gate_value']:.3f}) ---")
                    print(results['intervened'])
                    
            except KeyboardInterrupt:
                break
        
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs intervened generation")
    
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model to use")
    parser.add_argument("--checkpoint", type=str, 
                        default="gru_experiment/checkpoints/checkpoint_best.pt",
                        help="Path to controller checkpoint")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Run without loading checkpoint (baseline only)")
    parser.add_argument("--intervention-layer", type=int, default=4,
                        help="Layer to intervene at")
    
    parser.add_argument("--question", type=str, default=None,
                        help="Single question to test")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode")
    
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    
    args = parser.parse_args()
    
    # Set dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map[args.dtype]
    
    # Check checkpoint exists
    checkpoint_path = None if args.no_checkpoint else args.checkpoint
    if checkpoint_path and not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Running in baseline-only mode")
        checkpoint_path = None
    
    # Create generator
    generator = IntervenedGenerator(
        model_name=args.model,
        checkpoint_path=checkpoint_path,
        intervention_layer=args.intervention_layer,
        device=args.device,
        dtype=dtype
    )
    
    if args.interactive:
        generator.interactive(max_new_tokens=args.max_tokens)
    elif args.question:
        generator.compare_on_questions([args.question], max_new_tokens=args.max_tokens)
    else:
        # Default test questions
        test_questions = [
            "What is 15 + 27?",
            "A store sells apples for $2 each. If John buys 5 apples and pays with a $20 bill, how much change does he get?",
            "If a train travels at 60 mph for 2.5 hours, how far does it go?",
            "Sally has 3 times as many marbles as Tom. If Tom has 12 marbles, how many do they have together?",
        ]
        generator.compare_on_questions(test_questions, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
