#!/bin/bash
# GPU Mode Launcher for SPCToolkit
# This script runs the application in WSL2 with GPU acceleration

# Set CUDA library path for WSL2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Set DISPLAY for GUI (if not already set)
if [ -z "$DISPLAY" ]; then
    if [ -n "$WAYLAND_DISPLAY" ]; then
        # WSLg (Windows 11) - auto-configured
        :
    else
        # VcXsrv (Windows 10)
        export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
    fi
fi

# Navigate to project directory
cd "/mnt/c/Users/Sepeh/OneDrive/AI/SPCToolkit-Plugin Base 00"

# Activate RAPIDS environment
source ~/miniconda3/bin/activate rapids

# Display environment info
echo "============================================================"
echo "SPCToolkit - GPU Mode (WSL2 + RAPIDS)"
echo "============================================================"
echo "Environment: rapids"
python --version
echo ""
echo "GPU Libraries:"
python -c "import cuml; print(f'  cuML: {cuml.__version__}')" 2>/dev/null || echo "  cuML: Not available"
python -c "import cupy as cp; print(f'  CuPy: {cp.__version__}')" 2>/dev/null || echo "  CuPy: Not available"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || echo "  GPU: Not accessible"
echo ""
echo "Display: $DISPLAY"
echo "============================================================"
echo ""

# Run the application
python main.py
