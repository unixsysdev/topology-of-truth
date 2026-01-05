"""
Hooked Model Wrapper

Wraps a HuggingFace model with hooks to:
1. Extract activations at intervention layer
2. Inject steering vectors from GRU controller
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Callable, Any, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager


@dataclass 
class HookConfig:
    """Configuration for model hooks."""
    intervention_layer: int = 9  # Layer to intercept
    monitor_layers: List[int] = None  # Additional layers to monitor
    extract_before_intervention: bool = True  # Extract before or after layer
    
    def __post_init__(self):
        if self.monitor_layers is None:
            self.monitor_layers = [self.intervention_layer]


class ActivationCache:
    """Cache for storing activations during forward pass."""
    
    def __init__(self):
        self.activations: Dict[str, torch.Tensor] = {}
        self.enabled = True
        
    def store(self, name: str, tensor: torch.Tensor):
        if self.enabled:
            self.activations[name] = tensor.detach()
            
    def get(self, name: str) -> Optional[torch.Tensor]:
        return self.activations.get(name)
    
    def clear(self):
        self.activations.clear()
        
    def disable(self):
        self.enabled = False
        
    def enable(self):
        self.enabled = True


class HookedModel(nn.Module):
    """
    Wrapper around a causal LM that allows intervention.
    
    Features:
    - Extract activations at specified layers
    - Inject steering vectors into residual stream
    - Track activation statistics
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        config: Optional[HookConfig] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False
    ):
        super().__init__()
        
        self.config = config or HookConfig()
        self.device = device
        self.dtype = dtype
        
        # Load model
        print(f"Loading {model_name}...")
        
        load_kwargs = {
            "torch_dtype": dtype,
            "device_map": device if not (load_in_8bit or load_in_4bit) else "auto",
            "trust_remote_code": True,
        }
        
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            load_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Model info
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Intervention layer: {self.config.intervention_layer}")
        
        # Activation cache
        self.cache = ActivationCache()
        
        # Steering vector (set externally by controller)
        self._steering_vector: Optional[torch.Tensor] = None
        self._steering_enabled = False
        
        # Register hooks
        self._hooks = []
        self._register_hooks()
        
    def _get_layers(self):
        """Get the transformer layers (handles different model architectures)."""
        if hasattr(self.model, 'model'):
            # Qwen, Llama style
            if hasattr(self.model.model, 'layers'):
                return self.model.model.layers
        if hasattr(self.model, 'transformer'):
            # GPT style
            if hasattr(self.model.transformer, 'h'):
                return self.model.transformer.h
        raise ValueError("Cannot find transformer layers in model")
    
    def _register_hooks(self):
        """Register forward hooks on specified layers."""
        layers = self._get_layers()
        
        for layer_idx in set([self.config.intervention_layer] + self.config.monitor_layers):
            if layer_idx >= len(layers):
                print(f"Warning: Layer {layer_idx} doesn't exist (model has {len(layers)} layers)")
                continue
                
            layer = layers[layer_idx]
            
            # Hook to extract activations and optionally intervene
            hook = layer.register_forward_hook(
                self._create_hook(layer_idx)
            )
            self._hooks.append(hook)
            
    def _create_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a specific layer."""
        
        def hook(module, input, output):
            # Output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
                
            # Store activations
            self.cache.store(f"layer_{layer_idx}", hidden_states)
            
            # Apply steering at intervention layer
            if (layer_idx == self.config.intervention_layer and 
                self._steering_enabled and 
                self._steering_vector is not None):
                
                # Add steering vector to residual stream
                # steering_vector: (batch, hidden_size) or (batch, seq, hidden_size)
                steering = self._steering_vector
                
                if steering.dim() == 2:
                    # Expand to all positions: (batch, hidden) -> (batch, seq, hidden)
                    steering = steering.unsqueeze(1).expand_as(hidden_states)
                
                modified = hidden_states + steering
                
                # Return modified output
                if isinstance(output, tuple):
                    return (modified,) + output[1:]
                else:
                    return modified
                    
            return output
            
        return hook
    
    def set_steering(self, steering_vector: torch.Tensor):
        """Set the steering vector to be applied at intervention layer."""
        self._steering_vector = steering_vector
        
    def enable_steering(self):
        """Enable steering vector application."""
        self._steering_enabled = True
        
    def disable_steering(self):
        """Disable steering vector application."""
        self._steering_enabled = False
        
    @contextmanager
    def steering_context(self, steering_vector: torch.Tensor):
        """Context manager for temporary steering."""
        self.set_steering(steering_vector)
        self.enable_steering()
        try:
            yield
        finally:
            self.disable_steering()
            self._steering_vector = None
            
    def get_activations(self, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Get cached activations from a layer."""
        if layer_idx is None:
            layer_idx = self.config.intervention_layer
        return self.cache.get(f"layer_{layer_idx}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Any:
        """Forward pass with hooks active."""
        self.cache.clear()
        
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> torch.Tensor:
        """Generate with hooks active."""
        self.cache.clear()
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
    
    def generate_with_intervention(
        self,
        input_ids: torch.Tensor,
        controller: nn.Module,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, List]]:
        """
        Generate with GRU controller intervention.
        
        At each generation step:
        1. Get activations at intervention layer
        2. Feed to controller
        3. Apply steering if gate is open
        4. Continue generation
        
        Note: This is a simplified version. For full implementation,
        we'd need custom generation loop.
        """
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize controller hidden state
        hidden = controller.reset_hidden(batch_size, device)
        
        # Tracking
        diagnostics = {
            'gate_values': [],
            'entropy_values': [],
            'steering_norms': []
        }
        
        # Simple greedy generation with intervention
        generated = input_ids.clone()
        
        for step in range(max_new_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    generated,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
            
            # Get activations at intervention layer
            activations = self.get_activations()
            
            if activations is not None:
                # Get controller output
                ctrl_out = controller(activations, hidden, return_diagnostics=True)
                hidden = ctrl_out['hidden']
                
                # Track diagnostics
                diagnostics['gate_values'].append(ctrl_out['gate'].mean().item())
                diagnostics['entropy_values'].append(ctrl_out['entropy'].mean().item())
                diagnostics['steering_norms'].append(ctrl_out['steering_norm'].mean().item())
                
                # Apply steering for next forward pass
                self.set_steering(ctrl_out['steering'])
                self.enable_steering()
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device)
                ], dim=-1)
            
            # Check for EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
                
        self.disable_steering()
        
        return generated, diagnostics
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        
    def __del__(self):
        self.remove_hooks()


def create_hooked_model(
    model_name: str = "Qwen/Qwen3-0.6B",
    intervention_layer: Optional[int] = None,
    device: str = "cuda",
    dtype: str = "bfloat16"
) -> HookedModel:
    """
    Factory function to create hooked model.
    
    Args:
        model_name: HuggingFace model name
        intervention_layer: Layer for intervention (default: 1/3 into model)
        device: Device to load on
        dtype: Data type (bfloat16, float16, float32)
        
    Returns:
        HookedModel instance
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    
    # Load model temporarily to get config
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.num_hidden_layers
    
    # Default intervention layer: 1/3 into model
    if intervention_layer is None:
        intervention_layer = num_layers // 3
        
    hook_config = HookConfig(
        intervention_layer=intervention_layer,
        monitor_layers=[intervention_layer - 2, intervention_layer, intervention_layer + 2]
    )
    
    return HookedModel(
        model_name=model_name,
        config=hook_config,
        device=device,
        dtype=dtype_map.get(dtype, torch.bfloat16)
    )


# Quick test
if __name__ == "__main__":
    print("Testing Hooked Model...")
    
    # Create model (use small model for testing)
    model = create_hooked_model(
        model_name="Qwen/Qwen3-0.6B",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16"
    )
    
    # Test forward pass
    text = "What is 2 + 2?"
    inputs = model.tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    print(f"Output logits shape: {outputs.logits.shape}")
    
    # Check activations were cached
    activations = model.get_activations()
    if activations is not None:
        print(f"Cached activations shape: {activations.shape}")
    else:
        print("No activations cached")
        
    # Test with steering
    steering = torch.randn(1, model.hidden_size, device=model.device, dtype=model.dtype) * 0.01
    
    with model.steering_context(steering):
        outputs_steered = model(**inputs)
        
    print(f"Steered output shape: {outputs_steered.logits.shape}")
    
    # Clean up
    model.remove_hooks()
    print("Done!")
