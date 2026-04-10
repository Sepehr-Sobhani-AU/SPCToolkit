"""
Backend Registry

Selects the best available backend for each algorithm based on detected hardware.
The registry is initialized once at startup and provides access to backends
throughout the application.
"""

import logging
from typing import Dict

from infrastructure.hardware_detector import HardwareInfo
from plugins.backends import (
    DBSCANBackend,
    HDBSCANBackend,
    KNNBackend,
    MaskingBackend,
    EigenvalueBackend,
    NormalEstimationBackend,
    CuMLDBSCAN,
    SklearnDBSCAN,
    CuMLHDBSCAN,
    SklearnHDBSCAN,
    CuMLKNN,
    ScipyKNN,
    CuPyMasking,
    NumpyMasking,
    PyTorchCUDAEigen,
    PyTorchCPUEigen,
    PyTorchCUDANormals,
    Open3DCUDANormals,
    Open3DNormals,
)

logger = logging.getLogger(__name__)


class BackendRegistry:
    """
    Selects and provides access to the best available backend for each algorithm.

    Three scenarios are supported:
    - FULL GPU: Linux + NVIDIA + RAPIDS (cuML, CuPy, PyTorch CUDA)
    - PARTIAL GPU: NVIDIA without RAPIDS (sklearn, CuPy, PyTorch CUDA)
    - CPU ONLY: No NVIDIA GPU (sklearn, NumPy, PyTorch CPU)

    Usage:
        registry = BackendRegistry(hardware_info)
        dbscan_backend = registry.get_dbscan()
        labels = dbscan_backend.run(points, eps=0.5, min_samples=10)
    """

    def __init__(self, hardware_info: HardwareInfo):
        """
        Initialize the backend registry based on detected hardware.

        Args:
            hardware_info: Hardware information from HardwareDetector
        """
        self.hardware = hardware_info
        self._backends: Dict[str, object] = {}
        self._register_backends()

    def _register_backends(self) -> None:
        """Choose the best backend for each algorithm based on hardware scenario."""

        # Determine scenario
        if self.hardware.nvidia_gpu and self.hardware.cuml_available:
            self.scenario = "FULL GPU"  # Linux + NVIDIA + RAPIDS
            logger.info("Backend scenario: FULL GPU (Linux + NVIDIA + RAPIDS)")
        elif self.hardware.nvidia_gpu:
            self.scenario = "PARTIAL GPU"  # NVIDIA but no RAPIDS (Windows or Linux)
            logger.info("Backend scenario: PARTIAL GPU (NVIDIA without RAPIDS)")
        else:
            self.scenario = "CPU ONLY"  # No GPU / AMD / Intel
            logger.info("Backend scenario: CPU ONLY")

        # DBSCAN: cuML (RAPIDS only) > sklearn
        if self.scenario == "FULL GPU":
            self._backends['dbscan'] = CuMLDBSCAN()
            logger.info("DBSCAN backend: cuML (GPU)")
        else:
            self._backends['dbscan'] = SklearnDBSCAN()
            logger.info("DBSCAN backend: scikit-learn (CPU)")

        # HDBSCAN: cuML (RAPIDS only) > sklearn
        if self.scenario == "FULL GPU":
            self._backends['hdbscan'] = CuMLHDBSCAN()
            logger.info("HDBSCAN backend: cuML (GPU)")
        else:
            self._backends['hdbscan'] = SklearnHDBSCAN()
            logger.info("HDBSCAN backend: scikit-learn (CPU)")

        # KNN: cuML (RAPIDS only) > scipy
        if self.scenario == "FULL GPU":
            self._backends['knn'] = CuMLKNN()
            logger.info("KNN backend: cuML (GPU)")
        else:
            self._backends['knn'] = ScipyKNN()
            logger.info("KNN backend: scipy (CPU)")

        # Masking: CuPy (NVIDIA GPU) > NumPy
        if self.scenario in ["FULL GPU", "PARTIAL GPU"] and self.hardware.cupy_available:
            self._backends['masking'] = CuPyMasking()
            logger.info("Masking backend: CuPy (GPU)")
        else:
            self._backends['masking'] = NumpyMasking()
            logger.info("Masking backend: NumPy (CPU)")

        # Eigenvalues: PyTorch CUDA (NVIDIA GPU) > PyTorch CPU
        if self.scenario in ["FULL GPU", "PARTIAL GPU"] and self.hardware.pytorch_cuda:
            self._backends['eigenvalue'] = PyTorchCUDAEigen()
            logger.info("Eigenvalue backend: PyTorch CUDA (GPU)")
        else:
            self._backends['eigenvalue'] = PyTorchCPUEigen()
            logger.info("Eigenvalue backend: PyTorch CPU")

        # Normal Estimation: PyTorch CUDA > Open3D CUDA tensor > Open3D CPU
        if self.scenario in ["FULL GPU", "PARTIAL GPU"] and self.hardware.pytorch_cuda:
            self._backends['normal_estimation'] = PyTorchCUDANormals()
            logger.info("Normal Estimation backend: PyTorch CUDA (GPU)")
        elif self.scenario in ["FULL GPU", "PARTIAL GPU"] and self._has_open3d_cuda():
            self._backends['normal_estimation'] = Open3DCUDANormals()
            logger.info("Normal Estimation backend: Open3D CUDA (GPU)")
        else:
            self._backends['normal_estimation'] = Open3DNormals()
            logger.info("Normal Estimation backend: Open3D (CPU)")

    def get_dbscan(self) -> DBSCANBackend:
        """Get the DBSCAN clustering backend."""
        return self._backends['dbscan']

    def get_hdbscan(self) -> HDBSCANBackend:
        """Get the HDBSCAN clustering backend."""
        return self._backends['hdbscan']

    def get_knn(self) -> KNNBackend:
        """Get the K-Nearest Neighbors backend."""
        return self._backends['knn']

    def get_masking(self) -> MaskingBackend:
        """Get the point cloud masking backend."""
        return self._backends['masking']

    def get_eigenvalue(self) -> EigenvalueBackend:
        """Get the eigenvalue computation backend."""
        return self._backends['eigenvalue']

    def get_normal_estimation(self) -> NormalEstimationBackend:
        """Get the normal estimation backend."""
        return self._backends['normal_estimation']

    @staticmethod
    def _has_open3d_cuda() -> bool:
        """Check if Open3D has CUDA tensor pipeline support."""
        try:
            import open3d.core as o3c
            return o3c.Device.is_available("CUDA:0")
        except Exception:
            return False

    def get_status_report(self) -> Dict[str, str]:
        """
        Get a dictionary of algorithm -> backend name for display.

        Returns:
            Dict mapping algorithm names to their backend names
        """
        return {
            'DBSCAN': self._backends['dbscan'].name,
            'HDBSCAN': self._backends['hdbscan'].name,
            'KNN': self._backends['knn'].name,
            'Masking': self._backends['masking'].name,
            'Eigenvalues': self._backends['eigenvalue'].name,
            'Normal Estimation': self._backends['normal_estimation'].name,
        }

    def get_scenario(self) -> str:
        """Get the current hardware scenario."""
        return self.scenario

    def get_summary(self) -> str:
        """Get a human-readable summary of registered backends."""
        report = self.get_status_report()
        lines = [
            f"Backend Configuration ({self.scenario}):",
            f"  DBSCAN:             {report['DBSCAN']}",
            f"  HDBSCAN:            {report['HDBSCAN']}",
            f"  KNN:                {report['KNN']}",
            f"  Masking:            {report['Masking']}",
            f"  Eigenvalues:        {report['Eigenvalues']}",
            f"  Normal Estimation:  {report['Normal Estimation']}",
        ]
        return "\n".join(lines)
