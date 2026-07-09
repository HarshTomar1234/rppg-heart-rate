"""
Deep Learning Models for rPPG
"""

try:
    from .physnet import (
        PhysNet3D,
        PhysNetLite,
        PhysNetPreprocessor,
        PhysNetInference,
        TORCH_AVAILABLE,
    )
except ImportError:
    TORCH_AVAILABLE = False
    PhysNet3D = None
    PhysNetLite = None
    PhysNetPreprocessor = None
    PhysNetInference = None

__all__ = ["PhysNet3D", "PhysNetLite", "PhysNetPreprocessor", "PhysNetInference", "TORCH_AVAILABLE"]
