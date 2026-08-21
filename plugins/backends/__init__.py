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
    HDBSCANBackend,
    KNNBackend,
    MaskingBackend,
    ScreenSelectionBackend,
    SpatialGridBackend,
    EigenvalueBackend,
    NormalEstimationBackend,
)

# DBSCAN backends
from .dbscan_backends import (
    CuMLDBSCAN,
    SklearnDBSCAN,
    Open3DDBSCAN,
)

# HDBSCAN backends
from .hdbscan_backends import (
    CuMLHDBSCAN,
    SklearnHDBSCAN,
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

# Screen-space selection backends
from .selection_backends import (
    CuPySelection,
    NumpySelection,
)

# Spatial grid (cell numbering) backends
from .grid_backends import (
    CuPyGrid,
    NumpyGrid,
)

# Eigenvalue backends
from .eigenvalue_backends import (
    PyTorchCUDAEigen,
    PyTorchCPUEigen,
)

# Normal estimation backends
from .normal_estimation_backends import (
    PyTorchCUDANormals,
    Open3DCUDANormals,
    Open3DNormals,
)

__all__ = [
    # Base classes
    'BaseBackend',
    'DBSCANBackend',
    'HDBSCANBackend',
    'KNNBackend',
    'MaskingBackend',
    'ScreenSelectionBackend',
    'SpatialGridBackend',
    'EigenvalueBackend',
    # DBSCAN
    'CuMLDBSCAN',
    'SklearnDBSCAN',
    'Open3DDBSCAN',
    # HDBSCAN
    'CuMLHDBSCAN',
    'SklearnHDBSCAN',
    # KNN
    'CuMLKNN',
    'ScipyKNN',
    # Masking
    'CuPyMasking',
    'NumpyMasking',
    # Screen-space selection
    'CuPySelection',
    'NumpySelection',
    'CuPyGrid',
    'NumpyGrid',
    # Eigenvalue
    'PyTorchCUDAEigen',
    'PyTorchCPUEigen',
    # Normal Estimation
    'NormalEstimationBackend',
    'PyTorchCUDANormals',
    'Open3DCUDANormals',
    'Open3DNormals',
]
