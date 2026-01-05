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

from .hyper_connections import (
    ManifoldHyperConnections,
    HyperConnectionConfig,
    HyperConnectionLayer,
    HyperConnectionMatrix,
    SinkhornProjection,
    ExpandedResidualStream,
    create_mhc_for_model
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
    'ActivationCache',
    # Hyper-connections (mHC)
    'ManifoldHyperConnections',
    'HyperConnectionConfig',
    'HyperConnectionLayer',
    'HyperConnectionMatrix',
    'SinkhornProjection',
    'ExpandedResidualStream',
    'create_mhc_for_model'
]
