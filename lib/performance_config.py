#!/usr/bin/env python3
"""
Performance Configuration for High-End Mac (M3 Ultra)

This module provides optimized settings for video processing on Apple Silicon,
specifically tuned for M3 Ultra with 96GB RAM.
"""

import os
import cv2
import torch
import multiprocessing as mp
import numpy as np
import psutil
from typing import Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from dot.env
load_dotenv('dot.env')

class PerformanceOptimizer:
    """
    Optimizes processing for Apple Silicon Macs with high memory.
    """

    def __init__(self, conservative_mode=None):
        """
        Initialize performance optimizer with hardware detection.

        Args:
            conservative_mode: None (auto-detect), True (conservative), False (aggressive)
        """
        # print("PerformanceOptimizer.__init__ starting...")
        self.cpu_count = mp.cpu_count()
        # print(f"CPU count: {self.cpu_count}")
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.is_apple_silicon = self._detect_apple_silicon()

        # Determine performance mode from env or parameter
        if conservative_mode is None:
            # Check env for performance mode
            env_mode = os.getenv('PERFORMANCE_MODE', 'auto').lower()
            if env_mode == 'conservative':
                self.conservative_mode = True
            elif env_mode == 'performance':
                self.conservative_mode = False
            else:  # auto
                self.conservative_mode = self._should_use_conservative_mode()
        else:
            self.conservative_mode = conservative_mode

        # Detect processor tier and set core counts
        self._detect_processor_tier()

        # Detect available acceleration
        self.mps_available = self._check_mps_availability()
        self.device = self._get_optimal_device()

        # if not self.conservative_mode:
        #     print(f"🚀 Performance Optimizer Initialized:")
        #     print(f"   Mode: {'Conservative' if self.conservative_mode else 'Performance'}")
        #     print(f"   CPU: {self.cpu_count} cores ({self.performance_cores}P + {self.efficiency_cores}E)")
        #     print(f"   Memory: {self.memory_gb:.1f}GB")
        #     print(f"   Device: {self.device}")
        #     print(f"   Apple Silicon: {self.is_apple_silicon}")
        #     print(f"   MPS Available: {self.mps_available}")

    def _should_use_conservative_mode(self) -> bool:
        """Auto-detect if conservative mode should be used."""
        # Check environment variable first
        if os.environ.get('CONSERVATIVE_MODE', '').lower() in ('true', '1', 'yes'):
            return True

        # Default thresholds (can be overridden by env vars in future)
        min_memory = 16
        min_cores = 8

        # Use conservative mode if below thresholds
        if self.memory_gb < min_memory or self.cpu_count < min_cores:
            return True

        # Use conservative on Intel Macs
        if not self._detect_apple_silicon():
            return True

        return False

    def _detect_processor_tier(self):
        """Detect processor tier and set appropriate core counts."""
        # Default conservative values
        self.performance_cores = max(1, self.cpu_count // 2)
        self.efficiency_cores = 0

        if self.is_apple_silicon:
            # Detect Apple Silicon tier based on core count and memory
            if self.cpu_count >= 20 and self.memory_gb >= 64:
                # M1/M2/M3 Ultra or Max
                self.performance_cores = min(20, self.cpu_count - 8)
                self.efficiency_cores = 8
                self.processor_tier = "ultra"
            elif self.cpu_count >= 10 and self.memory_gb >= 32:
                # M1/M2/M3 Pro
                self.performance_cores = min(10, self.cpu_count - 4)
                self.efficiency_cores = 4
                self.processor_tier = "pro"
            else:
                # M1/M2/M3 Base
                self.performance_cores = min(4, self.cpu_count - 4)
                self.efficiency_cores = 4
                self.processor_tier = "base"
        else:
            # Intel Mac - use half cores for safety
            self.performance_cores = max(1, self.cpu_count // 2)
            self.efficiency_cores = 0
            self.processor_tier = "intel"

        # Apply conservative mode scaling
        if self.conservative_mode:
            self.performance_cores = max(1, self.performance_cores // 2)

    def _detect_apple_silicon(self) -> bool:
        """Detect if running on Apple Silicon."""
        try:
            import platform
            return platform.processor() == 'arm' or 'Apple' in platform.processor()
        except:
            return False

    def _check_mps_availability(self) -> bool:
        """Check if Metal Performance Shaders (MPS) is available."""
        # Check if MPS is disabled via env
        if os.getenv('DISABLE_GPU', '').lower() in ('true', '1', 'yes'):
            return False

        if os.getenv('ENABLE_MPS', 'true').lower() in ('false', '0', 'no'):
            return False

        if not self.is_apple_silicon:
            return False

        try:
            if torch.backends.mps.is_available():
                # Test MPS with a small tensor to ensure it works
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError("MPS test timed out")

                # Set a 5-second timeout for MPS test
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)

                try:
                    test_tensor = torch.tensor([1.0]).to('mps')
                    _ = test_tensor * 2
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, old_handler)
                    return True
                except TimeoutError:
                    signal.alarm(0)  # Cancel the alarm
                    signal.signal(signal.SIGALRM, old_handler)
                    return False
        except Exception as e:
            pass
        return False

    def _get_optimal_device(self) -> torch.device:
        """Get the optimal torch device for processing."""
        if self.mps_available:
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def get_opencv_optimizations(self) -> dict:
        """Get OpenCV optimization settings scaled to hardware."""
        # Scale settings based on hardware tier
        if self.conservative_mode:
            num_threads = max(1, self.performance_cores)
            buffer_size = 100
            cache_size = 1000
            queue_size = 2
        else:
            num_threads = self.performance_cores
            buffer_size = min(1000, int(self.memory_gb * 10))
            cache_size = min(10000, int(self.memory_gb * 100))
            queue_size = 8 if self.processor_tier in ('ultra', 'pro') else 4

        optimizations = {
            'num_threads': num_threads,
            'opencv_queue_size': queue_size,
            'use_opencl': cv2.ocl.haveOpenCL() and not self.conservative_mode,  # Enable if available and not conservative
            'use_optimized': True,
            'video_backend': cv2.CAP_AVFOUNDATION if self.is_apple_silicon else cv2.CAP_ANY,
            'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
            'buffer_size': buffer_size,
            'cache_size': cache_size,
        }

        # Apply OpenCV thread settings
        cv2.setNumThreads(optimizations['num_threads'])
        cv2.setUseOptimized(optimizations['use_optimized'])

        # Enable OpenCL if available and requested
        if optimizations['use_opencl'] and cv2.ocl.haveOpenCL():
            cv2.ocl.setUseOpenCL(True)

        return optimizations

    def get_torch_optimizations(self) -> dict:
        """Get PyTorch optimization settings."""
        optimizations = {
            'device': self.device,
            'num_threads': self.performance_cores,
            'enable_mkldnn': False,  # MKL-DNN not optimal on Apple Silicon
            'enable_nnpack': True,   # NNPACK works well on ARM
            'pin_memory': False,     # Not needed for MPS
            'non_blocking': True,    # Enable async transfers
        }

        # Apply PyTorch settings
        torch.set_num_threads(optimizations['num_threads'])

        if self.mps_available:
            # MPS-specific optimizations
            torch.mps.set_per_process_memory_fraction(0.8)  # Use 80% of GPU memory

        return optimizations

    def get_multiprocessing_config(self) -> dict:
        """Get multiprocessing configuration for parallel video processing."""
        # Check for env override
        env_max_workers = os.getenv('MAX_WORKERS')

        if self.conservative_mode:
            # Conservative settings for lower-end systems
            worker_processes = max(1, min(self.performance_cores // 2, 4))
            chunk_size = 8
            prefetch_factor = 2
            max_queue_size = 20
        else:
            # Performance settings, scaled by tier
            if self.processor_tier == "ultra":
                worker_processes = min(self.performance_cores - 2, 16)
                chunk_size = 32
                prefetch_factor = 4
                max_queue_size = 100
            elif self.processor_tier == "pro":
                worker_processes = min(self.performance_cores - 1, 8)
                chunk_size = 16
                prefetch_factor = 3
                max_queue_size = 50
            else:
                worker_processes = min(self.performance_cores, 4)
                chunk_size = 8
                prefetch_factor = 2
                max_queue_size = 25

        # Apply env override if set
        if env_max_workers:
            try:
                worker_processes = min(int(env_max_workers), worker_processes)
            except ValueError:
                pass

        return {
            'num_workers': max(1, worker_processes),
            'chunk_size': chunk_size,
            'prefetch_factor': prefetch_factor,
            'persistent_workers': not self.conservative_mode,
            'multiprocessing_method': 'spawn',  # Better for Apple Silicon
            'max_queue_size': max_queue_size,
        }

    def get_memory_config(self) -> dict:
        """Get memory configuration scaled to available RAM."""
        available_memory = self.memory_gb

        # Check for env override
        env_max_memory = os.getenv('MAX_MEMORY_GB')
        if env_max_memory:
            try:
                available_memory = min(float(env_max_memory), available_memory)
            except ValueError:
                pass

        if self.conservative_mode:
            # Conservative memory usage
            memory_fraction = 0.5  # Use only 50% of RAM
            frame_cache = min(100, int(available_memory * 2))
            video_buffer = 50
            batch_size = 8
            memory_map_threshold = 500
        else:
            # Scale based on available memory
            if available_memory >= 64:
                # High memory systems (64GB+)
                memory_fraction = 0.8
                frame_cache = min(1000, int(available_memory * 10))
                video_buffer = 500
                batch_size = 64
                memory_map_threshold = 2000
            elif available_memory >= 32:
                # Medium memory systems (32-64GB)
                memory_fraction = 0.7
                frame_cache = min(500, int(available_memory * 8))
                video_buffer = 200
                batch_size = 32
                memory_map_threshold = 1000
            elif available_memory >= 16:
                # Standard memory systems (16-32GB)
                memory_fraction = 0.6
                frame_cache = min(200, int(available_memory * 5))
                video_buffer = 100
                batch_size = 16
                memory_map_threshold = 500
            else:
                # Low memory systems (<16GB)
                memory_fraction = 0.5
                frame_cache = min(50, int(available_memory * 3))
                video_buffer = 50
                batch_size = 8
                memory_map_threshold = 250

        return {
            'max_memory_usage_gb': available_memory * memory_fraction,
            'frame_cache_size': frame_cache,
            'video_buffer_frames': video_buffer,
            'batch_size': batch_size,
            'use_memory_mapping': available_memory < 32 or self.conservative_mode,
            'memory_map_threshold_mb': memory_map_threshold,
        }

    def optimize_for_optical_flow(self) -> dict:
        """Get optimal settings for optical flow processing."""
        return {
            # Algorithm parameters
            'pyr_scale': 0.5,
            'levels': 3,
            'winsize': 15,
            'iterations': 3,
            'poly_n': 5,
            'poly_sigma': 1.2,

            # Processing parameters
            'use_parallel': True,
            'tile_size': (512, 512) if self.memory_gb > 32 else (256, 256),
            'overlap': 64,

            # Quality vs speed trade-offs
            'quality_preset': 'high',  # 'low', 'medium', 'high', 'ultra'
            'gpu_acceleration': self.mps_available or torch.cuda.is_available(),
        }

    def get_video_processing_config(self) -> dict:
        """Get complete video processing configuration."""
        return {
            'opencv': self.get_opencv_optimizations(),
            'torch': self.get_torch_optimizations(),
            'multiprocessing': self.get_multiprocessing_config(),
            'memory': self.get_memory_config(),
            'optical_flow': self.optimize_for_optical_flow(),
            'device': str(self.device),
            'hardware': {
                'cpu_cores': self.cpu_count,
                'performance_cores': self.performance_cores,
                'memory_gb': self.memory_gb,
                'is_apple_silicon': self.is_apple_silicon,
                'mps_available': self.mps_available,
            }
        }

    def apply_optimizations(self):
        """Apply all optimizations to the current environment."""
        # Set environment variables for optimal performance
        os.environ['OMP_NUM_THREADS'] = str(self.performance_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.performance_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.performance_cores)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(self.performance_cores)

        # Disable MKL on Apple Silicon (not optimal)
        if self.is_apple_silicon:
            os.environ['MKL_DEBUG_CPU_TYPE'] = '5'

        # OpenCV optimizations
        self.get_opencv_optimizations()

        # PyTorch optimizations
        self.get_torch_optimizations()

        print("✅ Performance optimizations applied")

        return self.get_video_processing_config()


# Singleton instance
_optimizer = None

def get_optimizer(conservative_mode=None, force_reinit=False) -> PerformanceOptimizer:
    """
    Get or create the performance optimizer singleton.

    Args:
        conservative_mode: None (auto-detect), True (conservative), False (aggressive)
        force_reinit: Force reinitialization with new settings

    Returns:
        PerformanceOptimizer instance

    Environment Variables:
        CONSERVATIVE_MODE: Set to 'true' to force conservative mode
        DISABLE_MPS: Set to 'true' to disable MPS acceleration
        MAX_WORKERS: Override maximum worker processes
        MAX_MEMORY_GB: Override maximum memory usage
    """
    global _optimizer
    if _optimizer is None or force_reinit:
        _optimizer = PerformanceOptimizer(conservative_mode)
        _optimizer.apply_optimizations()
    return _optimizer


def get_optimal_batch_size(input_shape: Tuple[int, int], dtype=np.float32) -> int:
    """
    Calculate optimal batch size based on available memory and input shape.

    Args:
        input_shape: (height, width) of input frames
        dtype: Data type of the arrays

    Returns:
        Optimal batch size
    """
    optimizer = get_optimizer()
    memory_config = optimizer.get_memory_config()

    # Estimate memory per frame (rough estimate)
    bytes_per_element = np.dtype(dtype).itemsize
    memory_per_frame_mb = (input_shape[0] * input_shape[1] * 3 * bytes_per_element) / (1024**2)

    # Use 50% of available memory for batch processing
    available_mb = memory_config['max_memory_usage_gb'] * 1024 * 0.5

    # Calculate batch size
    batch_size = int(available_mb / memory_per_frame_mb)

    # Clamp to reasonable range
    return max(1, min(batch_size, memory_config['batch_size']))


def setup_parallel_processing(num_videos: int) -> mp.Pool:
    """
    Setup multiprocessing pool optimized for video processing.

    Args:
        num_videos: Number of videos to process

    Returns:
        Configured multiprocessing pool
    """
    optimizer = get_optimizer()
    config = optimizer.get_multiprocessing_config()

    # Adjust workers based on workload
    num_workers = min(config['num_workers'], num_videos)

    # Create pool with optimal settings
    mp.set_start_method('spawn', force=True)
    pool = mp.Pool(
        processes=num_workers,
        maxtasksperchild=10  # Restart workers periodically to free memory
    )

    return pool