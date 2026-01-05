"""Models package for GRU intervention experiment."""

from .gru_controller import (
    GRUController,
    ControllerConfig,
    create_controller,
    EntropyProxy,
    GRUEncoder,
    InterventionGate,
    LowRankDecoder
)

from .hooked_model import (
    HookedModel,
    HookConfig,
    create_hooked_model,
    ActivationCache
)

__all__ = [
    # Controller
    'GRUController',
    'ControllerConfig', 
    'create_controller',
    'EntropyProxy',
    'GRUEncoder',
    'InterventionGate',
    'LowRankDecoder',
    # Hooked model
    'HookedModel',
    'HookConfig',
    'create_hooked_model',
    'ActivationCache'
]
