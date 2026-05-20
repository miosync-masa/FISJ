"""FISJ engines — pure computational kernels."""

from .nnnu import NNNUEngine, NNNUResult
from .regime import GenericRegimeDetector, GenericRegimeConfig, RegimeSegment

# Imported from existing modules (preserved from previous version)
from .inverse import InverseCausalEngine, InverseCausalEngineConfig
from .network_core import NetworkAnalyzerCore, NetworkResult, DimensionLink

__all__ = [
    "NNNUEngine",
    "NNNUResult",
    "GenericRegimeDetector",
    "GenericRegimeConfig",
    "RegimeSegment",
    "InverseCausalEngine",
    "InverseCausalEngineConfig",
    "NetworkAnalyzerCore",
    "NetworkResult",
    "DimensionLink",
]
