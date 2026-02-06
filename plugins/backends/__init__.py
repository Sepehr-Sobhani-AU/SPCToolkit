"""
Backend Implementations Package

This package contains backend implementations for various algorithms.
Each backend provides a consistent interface while using different libraries
(GPU vs CPU) for the actual computation.

Backends are selected automatically by BackendRegistry based on available hardware.
"""

# Base classes
from .base import (
    BaseBackend,
    DBSCANBackend,
    KNNBackend,
    MaskingBackend,
    EigenvalueBackend,
)

# DBSCAN backends
from .dbscan_backends import (
    CuMLDBSCAN,
    SklearnDBSCAN,
    Open3DDBSCAN,
)

# KNN backends
from .knn_backends import (
    CuMLKNN,
    ScipyKNN,
)

# Masking backends
from .masking_backends import (
    CuPyMasking,
    NumpyMasking,
)

# Eigenvalue backends
from .eigenvalue_backends import (
    PyTorchCUDAEigen,
    PyTorchCPUEigen,
)

__all__ = [
    # Base classes
    'BaseBackend',
    'DBSCANBackend',
    'KNNBackend',
    'MaskingBackend',
    'EigenvalueBackend',
    # DBSCAN
    'CuMLDBSCAN',
    'SklearnDBSCAN',
    'Open3DDBSCAN',
    # KNN
    'CuMLKNN',
    'ScipyKNN',
    # Masking
    'CuPyMasking',
    'NumpyMasking',
    # Eigenvalue
    'PyTorchCUDAEigen',
    'PyTorchCPUEigen',
]
