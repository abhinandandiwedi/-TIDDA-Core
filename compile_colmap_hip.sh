#!/bin/bash
set -e

echo "==============================================="
echo " TIDDA COLMAP HIP Compilation Script"
echo "==============================================="
echo ""
echo "This script installs necessary system dependencies via apt-get"
echo "and compiles COLMAP from source with ROCm/HIP support."
echo ""

# 1. Install dependencies
echo "[1/3] Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libflann-dev \
    qtbase5-dev \
    libmetis-dev \
    liblz4-dev

# 2. Configure CMake
echo "[2/3] Configuring CMake for ROCm/HIP (gfx1200)..."
cd colmap_hip_build
mkdir -p build
cd build

cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_ENABLED=OFF \
    -DHIP_ENABLED=ON \
    -DCMAKE_HIP_ARCHITECTURES=gfx1200 \
    -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++

# 3. Build
echo "[3/3] Compiling COLMAP..."
ninja

echo "==============================================="
echo " Compilation Successful!"
echo " The COLMAP executable is located at: colmap_hip_build/build/src/colmap/exe/colmap"
echo "==============================================="
