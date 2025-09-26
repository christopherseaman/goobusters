#!/usr/bin/env python3
"""
Parallel Video Processing for M3 Ultra

This module provides optimized parallel processing for video optical flow
on high-end Apple Silicon systems.
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import numpy as np
import cv2
from typing import List, Dict, Any, Tuple, Optional
from functools import partial
import time
from tqdm import tqdm
import queue
import threading

from performance_config import get_optimizer, get_optimal_batch_size


class ParallelVideoProcessor:
    """
    Optimized parallel video processor for M3 Ultra.

    Uses a combination of multiprocessing and threading to maximize
    throughput on Apple Silicon with high memory.
    """

    def __init__(self, method: str = 'farneback'):
        """
        Initialize parallel processor.

        Args:
            method: Optical flow method to use
        """
        self.method = method
        self.optimizer = get_optimizer()
        self.config = self.optimizer.get_video_processing_config()

        # Processing settings
        self.num_workers = self.config['multiprocessing']['num_workers']
        self.chunk_size = self.config['multiprocessing']['chunk_size']
        self.prefetch_factor = self.config['multiprocessing']['prefetch_factor']

        print(f"🚀 Parallel Processor initialized:")
        print(f"   Workers: {self.num_workers}")
        print(f"   Chunk size: {self.chunk_size}")
        print(f"   Method: {self.method}")

    def process_videos_parallel(self, video_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple videos in parallel.

        Args:
            video_list: List of video processing tasks

        Returns:
            List of processing results
        """
        total_videos = len(video_list)
        print(f"\n📹 Processing {total_videos} videos with {self.num_workers} workers")

        # Split videos into optimal chunks
        chunks = self._create_optimal_chunks(video_list)

        # Create process pool with optimal settings
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunks
            futures = []
            for chunk in chunks:
                future = executor.submit(self._process_video_chunk, chunk)
                futures.append(future)

            # Collect results with progress bar
            results = []
            with tqdm(total=total_videos, desc="Processing videos") as pbar:
                for future in as_completed(futures):
                    try:
                        chunk_results = future.result(timeout=300)
                        results.extend(chunk_results)
                        pbar.update(len(chunk_results))
                    except Exception as e:
                        print(f"⚠️  Error processing chunk: {str(e)}")

        return results

    def _create_optimal_chunks(self, video_list: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Create optimal chunks for parallel processing.

        Balances chunk sizes based on video complexity and system resources.
        """
        # Sort videos by size/complexity if available
        sorted_videos = sorted(video_list, key=lambda x: x.get('frame_count', 0), reverse=True)

        # Create balanced chunks
        chunks = []
        current_chunk = []
        chunk_complexity = 0
        max_chunk_complexity = 1000  # Adjust based on testing

        for video in sorted_videos:
            video_complexity = video.get('frame_count', 100)

            if current_chunk and chunk_complexity + video_complexity > max_chunk_complexity:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_complexity = 0

            current_chunk.append(video)
            chunk_complexity += video_complexity

            if len(current_chunk) >= self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_complexity = 0

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _process_video_chunk(self, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a chunk of videos.

        Args:
            chunk: List of video processing tasks

        Returns:
            List of results
        """
        results = []

        for video_task in chunk:
            try:
                result = self._process_single_video(video_task)
                results.append(result)
            except Exception as e:
                print(f"⚠️  Error processing video {video_task.get('path', 'unknown')}: {str(e)}")
                results.append({'error': str(e), 'video': video_task})

        return results

    def _process_single_video(self, video_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single video with optimizations.

        Args:
            video_task: Video processing task

        Returns:
            Processing result
        """
        video_path = video_task['path']

        # Use optimized video capture settings
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['opencv']['buffer_size'])

        # Process frames with batching
        results = []
        frame_batch = []
        batch_size = get_optimal_batch_size((720, 1280))  # Assume HD for estimation

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_batch.append(frame)
            frame_count += 1

            if len(frame_batch) >= batch_size:
                # Process batch
                batch_results = self._process_frame_batch(frame_batch)
                results.extend(batch_results)
                frame_batch = []

        # Process remaining frames
        if frame_batch:
            batch_results = self._process_frame_batch(frame_batch)
            results.extend(batch_results)

        cap.release()

        return {
            'video_path': video_path,
            'frame_count': frame_count,
            'results': results,
            'method': self.method
        }

    def _process_frame_batch(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Process a batch of frames.

        Args:
            frames: List of frames to process

        Returns:
            List of processing results
        """
        # Placeholder for batch processing logic
        # In real implementation, this would call the optical flow processor
        results = []
        for i, frame in enumerate(frames):
            results.append({
                'frame_idx': i,
                'processed': True,
                # Add optical flow results here
            })
        return results


class AsyncVideoStreamer:
    """
    Asynchronous video streaming with prefetching for optimal performance.
    """

    def __init__(self, video_path: str, buffer_size: int = 100):
        """
        Initialize async video streamer.

        Args:
            video_path: Path to video file
            buffer_size: Number of frames to buffer
        """
        self.video_path = video_path
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()

        # Get video properties
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Start prefetch thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_frames)
        self.prefetch_thread.start()

    def _prefetch_frames(self):
        """Prefetch frames in background thread."""
        cap = cv2.VideoCapture(self.video_path)

        # Apply optimizations
        optimizer = get_optimizer()
        opencv_config = optimizer.get_opencv_optimizations()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, opencv_config['buffer_size'])

        frame_idx = 0
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                self.frame_queue.put((frame_idx, frame), timeout=1)
                frame_idx += 1
            except queue.Full:
                # Buffer is full, wait for consumer
                time.sleep(0.01)

        cap.release()
        # Signal end of stream
        self.frame_queue.put((None, None))

    def get_frames(self) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Get frames from the buffer.

        Returns:
            Tuple of (frame_index, frame) or (None, None) at end of stream
        """
        try:
            return self.frame_queue.get(timeout=5)
        except queue.Empty:
            return None, None

    def stop(self):
        """Stop the prefetch thread."""
        self.stop_event.set()
        self.prefetch_thread.join()


def benchmark_processing_methods():
    """
    Benchmark different processing methods to find optimal configuration.
    """
    import time
    import tempfile

    print("🔬 Benchmarking processing methods...")

    # Create test video
    temp_video = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, 30.0, (1920, 1080))

    print("Creating test video...")
    for _ in range(100):
        frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        out.write(frame)
    out.release()

    # Test configurations
    configs = [
        {'workers': 1, 'batch_size': 1},
        {'workers': 4, 'batch_size': 8},
        {'workers': 8, 'batch_size': 16},
        {'workers': 16, 'batch_size': 32},
    ]

    results = []
    for config in configs:
        print(f"\nTesting: Workers={config['workers']}, Batch={config['batch_size']}")

        start_time = time.time()

        # Simulate processing
        processor = ParallelVideoProcessor()
        processor.num_workers = config['workers']
        processor.chunk_size = config['batch_size']

        # Process test video
        video_list = [{'path': temp_video, 'frame_count': 100}] * 4
        processor.process_videos_parallel(video_list)

        elapsed = time.time() - start_time
        fps = (100 * 4) / elapsed

        results.append({
            'config': config,
            'time': elapsed,
            'fps': fps
        })

        print(f"  Time: {elapsed:.2f}s, FPS: {fps:.1f}")

    # Clean up
    os.remove(temp_video)

    # Find best configuration
    best = max(results, key=lambda x: x['fps'])
    print(f"\n✅ Best configuration: {best['config']}")
    print(f"   Performance: {best['fps']:.1f} FPS")

    return best


if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_processing_methods()