from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Check for CUDA availability and set compiler arguments accordingly
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Ensure CUDA is installed on your machine.")

# CUDA source files
cuda_sources = [
    'correlation_package/src/corr_cuda_kernel.cu', 
    'correlation_package/src/corr1d_cuda_kernel.cu'
]

setup(
    name='pwcnet',
    version='0.1',
    description='PWC-Net Optical Flow Estimation with Custom CUDA Correlation Layer',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='correlation_cuda',  # This is the extension name you’ll import in Python
            sources=cuda_sources,
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': [
                    '-O2',
                    '-Xcompiler', '-fPIC',
                    '-arch=sm_52'  # You may need to adjust this for your GPU architecture
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    install_requires=[
        'torch>=1.0.0',         # Adjust based on the PyTorch version you're using
        'torchvision',
        'opencv-python',
        'numpy',
        'scipy'
    ],
    python_requires='>=3.6'  # Adjust to the minimum Python version you want to support
)