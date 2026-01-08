#!/bin/bash
# Complete WSL2 X11 Setup Script for SPCToolkit

echo "============================================================"
echo "WSL2 X11 Setup for SPCToolkit"
echo "============================================================"
echo ""

# Step 1: Install X11 libraries
echo "[1/5] Installing X11 libraries..."
sudo apt-get update -qq
sudo apt-get install -y \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    x11-apps \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 > /dev/null 2>&1

echo "✓ X11 libraries installed"
echo ""

# Step 2: Configure DISPLAY variable
echo "[2/5] Configuring DISPLAY variable..."

# Check if WSLg is available (Windows 11)
if [ -n "$WAYLAND_DISPLAY" ]; then
    echo "✓ WSLg detected (Windows 11) - DISPLAY auto-configured"
else
    # Windows 10 - use VcXsrv
    if ! grep -q "DISPLAY=" ~/.bashrc; then
        echo 'export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '"'"'{print $2}'"'"'):0' >> ~/.bashrc
        echo "✓ DISPLAY variable added to ~/.bashrc"
    else
        echo "✓ DISPLAY variable already configured"
    fi
    export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
fi

echo ""

# Step 3: Set GPU library path
echo "[3/5] Configuring GPU libraries..."
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
if ! grep -q "LD_LIBRARY_PATH=/usr/lib/wsl/lib" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "✓ GPU library path added to ~/.bashrc"
else
    echo "✓ GPU library path already configured"
fi
echo ""

# Step 4: Verify GPU access
echo "[4/5] Verifying GPU access..."
if nvidia-smi > /dev/null 2>&1; then
    echo "✓ GPU accessible"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  /'
else
    echo "⚠ GPU not accessible - check NVIDIA drivers"
fi
echo ""

# Step 5: Test X11
echo "[5/5] Testing X11 display..."
if [ -n "$DISPLAY" ]; then
    echo "✓ DISPLAY is set to: $DISPLAY"

    # Try to connect to X server
    if timeout 2 xset q > /dev/null 2>&1; then
        echo "✓ X server is accessible"
    else
        echo "⚠ X server not responding"
        echo ""
        echo "Action required:"
        if [ -n "$WAYLAND_DISPLAY" ]; then
            echo "  - WSLg should work automatically on Windows 11"
            echo "  - Try restarting WSL: wsl --shutdown"
        else
            echo "  - Install VcXsrv on Windows: https://sourceforge.net/projects/vcxsrv/"
            echo "  - Run XLaunch and disable access control"
            echo "  - Allow VcXsrv through Windows Firewall"
        fi
    fi
else
    echo "⚠ DISPLAY variable not set"
fi

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To run SPCToolkit with GPU:"
echo "  1. Source your bashrc: source ~/.bashrc"
echo "  2. Activate rapids: source ~/miniconda3/bin/activate rapids"
echo "  3. Run app: python main.py"
echo ""
echo "Or use the convenience script:"
echo "  wsl bash run_gpu.sh"
echo "============================================================"
