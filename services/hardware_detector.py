"""
Hardware Detection Module

Detects system hardware (OS, GPU) and available libraries at startup.
Used by BackendRegistry to select the best backend for each algorithm.
"""

import platform
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Stores detected hardware and library availability information."""

    # System info
    os: str  # "Windows", "Linux", "Darwin"

    # GPU info
    gpu_available: bool  # Any GPU detected
    gpu_vendor: Optional[str]  # "NVIDIA", "AMD", "Intel", None
    gpu_name: Optional[str]  # "RTX 3080", etc.
    gpu_memory_mb: int  # VRAM in MB (0 if not available)

    # NVIDIA-specific (for CUDA-based libraries)
    nvidia_gpu: bool  # True only if NVIDIA GPU detected
    cuda_available: bool  # CUDA toolkit available
    pytorch_cuda: bool  # PyTorch can use CUDA
    cupy_available: bool  # CuPy installed and working
    cuml_available: bool  # RAPIDS cuML installed (Linux only)


class HardwareDetector:
    """
    Singleton that detects hardware once at startup.

    Usage:
        hardware_info = HardwareDetector.detect()
        print(HardwareDetector.get_summary())
    """

    _hardware_info: Optional[HardwareInfo] = None
    _nvml_initialized: bool = False
    _nvml_handle = None

    @classmethod
    def detect(cls) -> HardwareInfo:
        """
        Detect hardware and return info (cached after first call).

        Returns:
            HardwareInfo: Detected hardware information
        """
        if cls._hardware_info is not None:
            return cls._hardware_info

        logger.info("Detecting hardware configuration...")

        # Detect OS
        os_name = platform.system()  # "Windows", "Linux", "Darwin"

        # Detect GPU and CUDA availability via PyTorch
        nvidia_gpu = False
        cuda_available = False
        pytorch_cuda = False
        gpu_name = None
        gpu_memory_mb = 0

        try:
            import torch

            if torch.cuda.is_available():
                pytorch_cuda = True
                cuda_available = True
                nvidia_gpu = True
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                logger.info(f"NVIDIA GPU detected: {gpu_name} ({gpu_memory_mb} MB)")
        except ImportError:
            logger.warning("PyTorch not installed, cannot detect CUDA GPU")
        except Exception as e:
            logger.warning(f"Error detecting GPU via PyTorch: {e}")

        # Determine GPU vendor from name
        gpu_vendor = None
        gpu_available = False
        if gpu_name:
            gpu_available = True
            if nvidia_gpu:
                gpu_vendor = "NVIDIA"
            # Note: AMD/Intel detection would require additional libraries
            # For now, we only detect NVIDIA via CUDA

        # Check CuPy availability
        cupy_available = False
        if nvidia_gpu:
            try:
                import cupy as cp

                # Try a simple operation to verify it works
                test_array = cp.array([1, 2, 3])
                _ = cp.asnumpy(test_array)
                cupy_available = True
                logger.info("CuPy is available and working")
            except ImportError as e:
                logger.warning(f"CuPy import failed: {e}")
            except Exception as e:
                logger.warning(f"CuPy error: {type(e).__name__}: {e}")

        # Check cuML availability (RAPIDS - Linux only)
        cuml_available = False
        if nvidia_gpu and os_name == "Linux":
            try:
                from cuml.cluster import DBSCAN as cumlDBSCAN

                cuml_available = True
                logger.info("cuML (RAPIDS) is available")
            except ImportError as e:
                logger.warning(f"cuML import failed: {e}")
            except Exception as e:
                logger.warning(f"cuML error: {type(e).__name__}: {e}")

        cls._hardware_info = HardwareInfo(
            os=os_name,
            gpu_available=gpu_available,
            gpu_vendor=gpu_vendor,
            gpu_name=gpu_name,
            gpu_memory_mb=gpu_memory_mb,
            nvidia_gpu=nvidia_gpu,
            cuda_available=cuda_available,
            pytorch_cuda=pytorch_cuda,
            cupy_available=cupy_available,
            cuml_available=cuml_available,
        )

        return cls._hardware_info

    @classmethod
    def get_summary(cls) -> str:
        """
        Get a human-readable summary of detected hardware.

        Returns:
            str: Formatted summary string
        """
        if cls._hardware_info is None:
            cls.detect()

        info = cls._hardware_info
        lines = [
            "=" * 50,
            "SPCToolkit - Hardware Detection",
            "=" * 50,
            f"OS: {info.os}",
        ]

        if info.gpu_available:
            lines.append(f"GPU: {info.gpu_name} ({info.gpu_memory_mb} MB)")
            lines.append(f"CUDA: {'Available' if info.cuda_available else 'Not available'}")
            lines.append(f"PyTorch CUDA: {'Available' if info.pytorch_cuda else 'Not available'}")
            lines.append(f"CuPy: {'Available' if info.cupy_available else 'Not available'}")
            lines.append(f"cuML (RAPIDS): {'Available' if info.cuml_available else 'Not available'}")
        else:
            lines.append("GPU: None detected (CPU mode)")

        lines.append("=" * 50)
        return "\n".join(lines)

    @classmethod
    def get_scenario(cls) -> str:
        """
        Determine the hardware scenario for backend selection.

        Returns:
            str: "FULL GPU", "PARTIAL GPU", or "CPU ONLY"
        """
        if cls._hardware_info is None:
            cls.detect()

        info = cls._hardware_info

        if info.nvidia_gpu and info.cuml_available:
            return "FULL GPU"
        elif info.nvidia_gpu:
            return "PARTIAL GPU"
        else:
            return "CPU ONLY"

    @classmethod
    def can_use_rapids(cls) -> bool:
        """Check if RAPIDS/cuML is available (Linux + NVIDIA + cuML)."""
        if cls._hardware_info is None:
            cls.detect()
        return cls._hardware_info.cuml_available

    @classmethod
    def can_use_cupy(cls) -> bool:
        """Check if CuPy is available (NVIDIA + CuPy)."""
        if cls._hardware_info is None:
            cls.detect()
        return cls._hardware_info.cupy_available

    @classmethod
    def can_use_pytorch_gpu(cls) -> bool:
        """Check if PyTorch CUDA is available."""
        if cls._hardware_info is None:
            cls.detect()
        return cls._hardware_info.pytorch_cuda

    @classmethod
    def reset(cls) -> None:
        """Reset cached hardware info (useful for testing)."""
        cls._hardware_info = None
        cls._shutdown_nvml()

    @classmethod
    def _init_nvml(cls) -> bool:
        """Initialize NVML once and keep handle for reuse."""
        if cls._nvml_initialized:
            return True

        try:
            import pynvml
            pynvml.nvmlInit()
            cls._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            cls._nvml_initialized = True
            logger.debug("NVML initialized successfully")
            return True
        except ImportError:
            logger.debug("pynvml not available")
            return False
        except Exception as e:
            logger.debug(f"Failed to initialize NVML: {e}")
            return False

    @classmethod
    def _shutdown_nvml(cls) -> None:
        """Shutdown NVML if initialized."""
        if cls._nvml_initialized:
            try:
                import pynvml
                pynvml.nvmlShutdown()
            except Exception:
                pass
            cls._nvml_initialized = False
            cls._nvml_handle = None

    @classmethod
    def get_free_gpu_memory_mb(cls) -> int:
        """
        Get current free GPU memory in MB.

        Returns:
            int: Free VRAM in MB (0 if no GPU or error)
        """
        if cls._hardware_info is None:
            cls.detect()

        if not cls._hardware_info.nvidia_gpu:
            return 0

        # Use persistent NVML handle
        if cls._init_nvml() and cls._nvml_handle is not None:
            try:
                import pynvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(cls._nvml_handle)
                return mem_info.free // (1024 * 1024)
            except Exception as e:
                logger.debug(f"Error getting free GPU memory: {e}")
                cls._nvml_initialized = False
                cls._nvml_handle = None

        # Fall back to PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                reserved = torch.cuda.memory_reserved(0)
                return (total - reserved) // (1024 * 1024)
        except Exception:
            pass

        return 0

    @classmethod
    def check_gpu_memory_available(cls, required_mb: int, safety_margin: float = 0.2) -> bool:
        """
        Check if enough GPU memory is available for an operation.

        Args:
            required_mb: Required memory in MB
            safety_margin: Extra margin to keep free (default 20%)

        Returns:
            bool: True if enough memory is available
        """
        free_mb = cls.get_free_gpu_memory_mb()
        if free_mb == 0:
            return False

        # Add safety margin
        required_with_margin = int(required_mb * (1 + safety_margin))
        return free_mb >= required_with_margin

    @classmethod
    def estimate_array_gpu_memory_mb(cls, array_or_shape, dtype=None) -> int:
        """
        Estimate GPU memory required for an array.

        Args:
            array_or_shape: NumPy array or tuple shape
            dtype: Data type (default: float32)

        Returns:
            int: Estimated memory in MB
        """
        import numpy as np

        if hasattr(array_or_shape, 'nbytes'):
            # It's an array
            return array_or_shape.nbytes // (1024 * 1024) + 1  # +1 to round up
        else:
            # It's a shape tuple
            if dtype is None:
                dtype = np.float32
            element_size = np.dtype(dtype).itemsize
            total_elements = 1
            for dim in array_or_shape:
                total_elements *= dim
            return (total_elements * element_size) // (1024 * 1024) + 1

    @classmethod
    def log_gpu_memory_status(cls, operation: str = "") -> None:
        """
        Log current GPU memory status.

        Args:
            operation: Description of the operation being performed
        """
        stats = cls.get_dynamic_stats()
        if stats['vram_total_mb'] > 0:
            used = stats['vram_used_mb']
            total = stats['vram_total_mb']
            free = total - used
            pct = stats['vram_percent']
            prefix = f"[{operation}] " if operation else ""
            logger.info(f"{prefix}GPU Memory: {used}/{total} MB used ({pct:.1f}%), {free} MB free")

    @classmethod
    def get_dynamic_stats(cls) -> dict:
        """
        Get real-time hardware statistics.

        Returns:
            dict with keys:
                - ram_used_gb: RAM used in GB
                - ram_total_gb: Total RAM in GB
                - ram_percent: RAM usage percentage
                - vram_used_mb: VRAM used in MB (0 if no GPU)
                - vram_total_mb: Total VRAM in MB (0 if no GPU)
                - vram_percent: VRAM usage percentage (0 if no GPU)
                - gpu_utilization: GPU utilization percentage (0 if no GPU)
                - gpu_temp_c: GPU temperature in Celsius (None if not available)
        """
        stats = {
            'ram_used_gb': 0.0,
            'ram_total_gb': 0.0,
            'ram_percent': 0.0,
            'vram_used_mb': 0,
            'vram_total_mb': 0,
            'vram_percent': 0.0,
            'gpu_utilization': 0,
            'gpu_temp_c': None,
        }

        # RAM statistics via psutil
        try:
            import psutil
            mem = psutil.virtual_memory()
            stats['ram_used_gb'] = mem.used / (1024 ** 3)
            stats['ram_total_gb'] = mem.total / (1024 ** 3)
            stats['ram_percent'] = mem.percent
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Error getting RAM stats: {e}")

        # GPU statistics via pynvml (more reliable than PyTorch for utilization/temp)
        if cls._hardware_info and cls._hardware_info.nvidia_gpu:
            # Use persistent NVML handle
            if cls._init_nvml() and cls._nvml_handle is not None:
                try:
                    import pynvml

                    # VRAM
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(cls._nvml_handle)
                    stats['vram_used_mb'] = mem_info.used // (1024 * 1024)
                    stats['vram_total_mb'] = mem_info.total // (1024 * 1024)
                    stats['vram_percent'] = (mem_info.used / mem_info.total) * 100

                    # GPU utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(cls._nvml_handle)
                    stats['gpu_utilization'] = util.gpu

                    # Temperature
                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(cls._nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                        stats['gpu_temp_c'] = temp
                    except Exception:
                        pass

                except Exception as e:
                    logger.debug(f"Error getting GPU stats via pynvml: {e}")
                    # Reset NVML on error to try reinitializing next time
                    cls._nvml_initialized = False
                    cls._nvml_handle = None
            else:
                # Fall back to PyTorch for basic VRAM info
                try:
                    import torch
                    if torch.cuda.is_available():
                        stats['vram_used_mb'] = torch.cuda.memory_allocated(0) // (1024 * 1024)
                        stats['vram_total_mb'] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
                        if stats['vram_total_mb'] > 0:
                            stats['vram_percent'] = (stats['vram_used_mb'] / stats['vram_total_mb']) * 100
                except Exception:
                    pass

        return stats
