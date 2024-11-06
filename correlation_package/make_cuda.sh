#!/usr/bin/env bash

# Set the CUDA path if not already set in the environment
CUDA_PATH=${CUDA_PATH:-/usr/local/cuda}

# Ensure CUDA is installed
if [ ! -d "$CUDA_PATH" ]; then
  echo "CUDA not found at $CUDA_PATH. Please install CUDA or set the correct CUDA_PATH."
  exit 1
fi

# Navigate to the directory where the CUDA kernels are located
cd correlation_package/src || exit 1
echo "Compiling correlation layer kernels by nvcc..."

# Compile CUDA kernels for correlation
nvcc -c -o corr_cuda_kernel.o corr_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52
nvcc -c -o corr1d_cuda_kernel.o corr1d_cuda_kernel.cu -x cu -Xcompiler -fPIC -arch=sm_52

# Move back to the project root directory
cd ../../

# Ensure setup.py exists in the root directory
if [ ! -f "setup.py" ]; then
  echo "setup.py not found in the project root directory. Ensure this script is run from the root."
  exit 1
fi

# Build and install the Python package
echo "Building and installing the Python package using setup.py..."
python setup.py build install