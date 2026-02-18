#!/usr/bin/env python
"""
Benchmark script for tracking worker performance.

Measures DIS optical flow presets, mask warping, per-frame tracking loop
(video decode + flow + warp), and full pipeline including I/O.

Usage:
    python scripts/benchmark_tracking.py [OPTIONS]

Options:
    --quick             3 iterations (micro), 1 run (pipeline)
    --iterations N      Micro-benchmark iterations (default: 10)
    --method METHOD     Comma-separated: dis-ultrafast,dis-fast,dis-medium,all (default: all)
    --output FILE       Write JSON to file instead of stdout
    --skip-pipeline     Skip end-to-end pipeline benchmarks

Env vars:
    DISABLE_GPU=true    Force CPU for all methods
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# Path setup (same pattern as rebuild_mask_archives.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib_server = os.path.join(PROJECT_ROOT, "lib", "server")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS = ["dis-ultrafast", "dis-fast", "dis-medium"]

# Test series: {label: (study_uid, series_uid, expected_frames, expected_annotations)}
TEST_SERIES = {
    "small": (
        "1.2.826.0.1.3680043.8.498.11903523154547667096601610925275921854",
        "1.2.826.0.1.3680043.8.498.96437439638255288983820464989087471222",
        60, 2,
    ),
    "medium": (
        "1.2.826.0.1.3680043.8.498.10152607431635338700140470803485154753",
        "1.2.826.0.1.3680043.8.498.10649434839423026912963496123749253318",
        109, 3,
    ),
    "large": (
        "1.2.826.0.1.3680043.8.498.87668743878037786459732261211254326279",
        "1.2.826.0.1.3680043.8.498.92308052977701030363540071511735819703",
        195, 18,
    ),
}


def log(msg: str) -> None:
    """Print progress to stderr so stdout stays clean for JSON."""
    print(msg, file=sys.stderr, flush=True)


def timing_stats(times_ms: list[float], iterations: int) -> dict:
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "std_ms": round(float(arr.std()), 2),
        "iterations": iterations,
    }


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

def collect_hardware_info() -> dict:
    import psutil

    from lib.performance_config import get_optimizer

    optimizer = get_optimizer()
    device = str(optimizer.device) if optimizer else "cpu"
    is_apple = hasattr(optimizer, "is_apple_silicon") and optimizer.is_apple_silicon
    mps = hasattr(optimizer, "mps_available") and optimizer.mps_available

    return {
        "platform": platform.platform(),
        "cpu_cores": psutil.cpu_count(logical=False) or os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 1),
        "device": device,
        "is_apple_silicon": is_apple,
        "mps_available": mps,
    }


# ---------------------------------------------------------------------------
# Frame / video helpers
# ---------------------------------------------------------------------------

def load_frame_pair(series_label: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load two consecutive frames from an existing series output directory."""
    study_uid, series_uid = TEST_SERIES[series_label][:2]
    frames_dir = Path(PROJECT_ROOT) / "output" / "dis" / f"{study_uid}_{series_uid}" / "frames"

    if not frames_dir.exists():
        return None

    frame_files = sorted(frames_dir.glob("frame_*.webp"))
    if len(frame_files) < 2:
        return None

    prev = cv2.imread(str(frame_files[0]), cv2.IMREAD_GRAYSCALE)
    curr = cv2.imread(str(frame_files[1]), cv2.IMREAD_GRAYSCALE)
    if prev is None or curr is None:
        return None

    return prev, curr


def find_video_path(series_label: str) -> str | None:
    """Find the video file for a test series."""
    from lib.config import load_config
    from track import find_images_dir

    study_uid, series_uid = TEST_SERIES[series_label][:2]
    try:
        config = load_config("server")
        images_dir = find_images_dir(
            str(config.data_dir), config.project_id, config.dataset_id
        )
        path = os.path.join(images_dir, study_uid, f"{series_uid}.mp4")
        return path if os.path.exists(path) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Micro-benchmarks
# ---------------------------------------------------------------------------

def benchmark_optical_flow(
    methods: list[str], iterations: int
) -> tuple[dict, tuple[int, int] | None]:
    """Benchmark optical flow calculation per frame pair for each DIS preset."""
    from lib.opticalflowprocessor import OpticalFlowProcessor

    pair = load_frame_pair("medium")
    if pair is None:
        log("  SKIP: no frames available for micro-benchmarks")
        return {}, None

    prev_frame, curr_frame = pair
    resolution = (prev_frame.shape[0], prev_frame.shape[1])
    results = {}

    for method_label in methods:
        preset = method_label.split("-", 1)[1]
        os.environ["DIS_PRESET"] = preset

        log(f"  {method_label}...")

        try:
            processor = OpticalFlowProcessor("dis")

            # Warmup (untimed)
            processor.calculate_flow(prev_frame, curr_frame)

            times_ms = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                processor.calculate_flow(prev_frame, curr_frame)
                elapsed = (time.perf_counter() - t0) * 1000
                times_ms.append(elapsed)

            results[method_label] = timing_stats(times_ms, iterations)
            processor.cleanup_memory()

        except Exception as exc:
            log(f"  ERROR ({method_label}): {exc}")
            results[method_label] = {"error": str(exc)}

    return results, resolution


def benchmark_mask_warp(iterations: int) -> dict | None:
    """Benchmark mask warping with optical flow."""
    from lib.multi_frame_tracker import MultiFrameTracker
    from lib.opticalflowprocessor import OpticalFlowProcessor

    pair = load_frame_pair("medium")
    if pair is None:
        log("  SKIP: no frames for mask warp benchmark")
        return None

    prev_frame, curr_frame = pair
    h, w = prev_frame.shape

    os.environ["DIS_PRESET"] = "fast"
    processor = OpticalFlowProcessor("dis")
    flow = processor.calculate_flow(prev_frame, curr_frame)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MultiFrameTracker(processor, tmpdir)

        # Warmup
        tracker._warp_mask_with_flow(mask, flow)

        times_ms = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            tracker._warp_mask_with_flow(mask, flow)
            elapsed = (time.perf_counter() - t0) * 1000
            times_ms.append(elapsed)

    processor.cleanup_memory()
    return timing_stats(times_ms, iterations)


def benchmark_video_decode(iterations: int) -> dict | None:
    """Benchmark raw video frame decode speed (cv2.VideoCapture.read)."""
    video_path = find_video_path("medium")
    if video_path is None:
        log("  SKIP: no video for decode benchmark")
        return None

    # Warmup: read one frame
    cap = cv2.VideoCapture(video_path)
    cap.read()
    cap.release()

    times_ms = []
    for _ in range(iterations):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        t0 = time.perf_counter()
        ret, _ = cap.read()
        elapsed = (time.perf_counter() - t0) * 1000
        cap.release()

        if ret:
            times_ms.append(elapsed)

    if not times_ms:
        return None
    return timing_stats(times_ms, iterations)


def benchmark_tracking_loop(series_label: str, preset: str = "fast") -> dict | None:
    """Benchmark the core tracking inner loop on a real video.

    Simulates what _process_segment does: sequential video decode, optical flow,
    mask warp for each frame — no file I/O. This is the compute-bound core that
    determines real tracking speed.

    Returns per-frame timing breakdown.
    """
    from lib.multi_frame_tracker import MultiFrameTracker
    from lib.opticalflowprocessor import OpticalFlowProcessor

    video_path = find_video_path(series_label)
    if video_path is None:
        log(f"  SKIP tracking loop ({series_label}): no video")
        return None

    study_uid, series_uid, expected_frames, _ = TEST_SERIES[series_label]

    os.environ["DIS_PRESET"] = preset
    processor = OpticalFlowProcessor("dis")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MultiFrameTracker(processor, tmpdir)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a starter mask (center rectangle, like a real annotation)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255

        # Read first frame (warmup)
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None

        decode_times = []
        flow_times = []
        warp_times = []
        frames_processed = 0

        for _ in range(total_frames - 1):
            # Video decode
            t0 = time.perf_counter()
            ret, curr_frame = cap.read()
            t_decode = time.perf_counter() - t0

            if not ret:
                break

            # Optical flow
            t0 = time.perf_counter()
            flow = processor.calculate_flow(prev_frame, curr_frame)
            t_flow = time.perf_counter() - t0

            # Mask warp
            t0 = time.perf_counter()
            mask = tracker._warp_mask_with_flow(mask, flow)
            t_warp = time.perf_counter() - t0

            decode_times.append(t_decode * 1000)
            flow_times.append(t_flow * 1000)
            warp_times.append(t_warp * 1000)

            prev_frame = curr_frame
            frames_processed += 1

        cap.release()

    processor.cleanup_memory()

    if frames_processed == 0:
        return None

    total_ms = sum(decode_times) + sum(flow_times) + sum(warp_times)

    return {
        "frames": frames_processed,
        "total_s": round(total_ms / 1000, 2),
        "per_frame_ms": round(total_ms / frames_processed, 2),
        "breakdown": {
            "decode": timing_stats(decode_times, frames_processed),
            "flow": timing_stats(flow_times, frames_processed),
            "warp": timing_stats(warp_times, frames_processed),
        },
        "pct": {
            "decode": round(sum(decode_times) / total_ms * 100, 1),
            "flow": round(sum(flow_times) / total_ms * 100, 1),
            "warp": round(sum(warp_times) / total_ms * 100, 1),
        },
        "dis_preset": preset,
    }


# ---------------------------------------------------------------------------
# Pipeline benchmarks
# ---------------------------------------------------------------------------

def benchmark_pipeline(quick: bool) -> dict:
    """Run full end-to-end tracking pipeline (compute + all I/O) on test series."""
    import pandas as pd

    from lib.config import load_config
    from lib.multi_frame_tracker import (
        process_video_with_multi_frame_tracking,
        set_label_ids,
    )
    from lib.opticalflowprocessor import OpticalFlowProcessor
    from track import find_annotations_file, find_images_dir

    try:
        config = load_config("server")
    except Exception as exc:
        log(f"  SKIP pipeline: config load failed: {exc}")
        return {k: {"skipped": True, "reason": str(exc)} for k in TEST_SERIES}

    try:
        annotations_path = find_annotations_file(
            str(config.data_dir), config.project_id, config.dataset_id
        )
        import mdai

        annotations_blob = mdai.common_utils.json_to_dataframe(annotations_path)
        all_annotations_df = pd.DataFrame(annotations_blob["annotations"])
    except Exception as exc:
        log(f"  SKIP pipeline: annotations load failed: {exc}")
        return {k: {"skipped": True, "reason": str(exc)} for k in TEST_SERIES}

    try:
        images_dir = find_images_dir(
            str(config.data_dir), config.project_id, config.dataset_id
        )
    except Exception as exc:
        log(f"  SKIP pipeline: images dir not found: {exc}")
        return {k: {"skipped": True, "reason": str(exc)} for k in TEST_SERIES}

    results = {}

    for label, (study_uid, series_uid, expected_frames, _) in TEST_SERIES.items():
        log(f"  pipeline: {label}...")

        series_df = all_annotations_df[
            (all_annotations_df["StudyInstanceUID"] == study_uid)
            & (all_annotations_df["SeriesInstanceUID"] == series_uid)
        ].copy()

        series_df = series_df[
            (series_df["labelId"] == config.label_id)
            | (series_df["labelId"] == config.empty_id)
        ].copy()

        if series_df.empty:
            results[label] = {"skipped": True, "reason": "no annotations"}
            continue

        video_path = os.path.join(images_dir, study_uid, f"{series_uid}.mp4")
        if not os.path.exists(video_path):
            results[label] = {"skipped": True, "reason": "video not found"}
            continue

        series_df["video_path"] = video_path

        if "labelName" not in series_df.columns:
            label_map = dict(
                all_annotations_df[["labelId", "labelName"]]
                .drop_duplicates()
                .values
            )
            series_df["labelName"] = series_df["labelId"].map(label_map)

        annotation_count = len(series_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            set_label_ids(config.label_id, config.empty_id)
            flow_processor = OpticalFlowProcessor(config.flow_method)

            t0 = time.perf_counter()
            try:
                process_video_with_multi_frame_tracking(
                    video_path=video_path,
                    annotations_df=series_df,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    flow_processor=flow_processor,
                    output_dir=tmpdir,
                )
                elapsed = time.perf_counter() - t0

                results[label] = {
                    "frames": expected_frames,
                    "annotations": annotation_count,
                    "elapsed_s": round(elapsed, 2),
                    "method": config.flow_method,
                    "skipped": False,
                }
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                results[label] = {
                    "frames": expected_frames,
                    "annotations": annotation_count,
                    "elapsed_s": round(elapsed, 2),
                    "method": config.flow_method,
                    "skipped": False,
                    "error": str(exc),
                }
            finally:
                flow_processor.cleanup_memory()

        if quick:
            for remaining in TEST_SERIES:
                if remaining not in results:
                    results[remaining] = {"skipped": True, "reason": "quick mode"}
            break

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark tracking worker performance")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (3 iters, 1 pipeline)")
    parser.add_argument("--iterations", type=int, default=10, help="Micro-benchmark iterations (default: 10)")
    parser.add_argument("--method", default="all", help="Comma-separated: dis-ultrafast,dis-fast,dis-medium,all")
    parser.add_argument("--output", help="Write JSON to file instead of stdout")
    parser.add_argument("--skip-pipeline", action="store_true", help="Skip end-to-end pipeline benchmarks")
    args = parser.parse_args()

    iterations = 3 if args.quick else args.iterations

    if args.method == "all":
        methods = ALL_METHODS
    else:
        methods = [m.strip() for m in args.method.split(",")]
        invalid = [m for m in methods if m not in ALL_METHODS]
        if invalid:
            parser.error(f"Unknown methods: {', '.join(invalid)}. Valid: {', '.join(ALL_METHODS)}")

    # Change to project root so config loading works
    os.chdir(PROJECT_ROOT)

    log("Collecting hardware info...")
    hardware = collect_hardware_info()
    log(f"  device: {hardware['device']}, cores: {hardware['cpu_cores']}, ram: {hardware['memory_gb']}GB")

    # --- Micro-benchmarks ---
    log("Running optical flow benchmarks...")
    flow_results, resolution = benchmark_optical_flow(methods, iterations)

    log("Running mask warp benchmark...")
    warp_result = benchmark_mask_warp(iterations)

    log("Running video decode benchmark...")
    decode_result = benchmark_video_decode(iterations)

    # --- Tracking loop (core compute, no I/O) ---
    log("Running tracking loop benchmarks...")
    tracking_loop_results = {}
    series_to_bench = ["small"] if args.quick else ["small", "medium", "large"]
    for label in series_to_bench:
        log(f"  tracking loop: {label}...")
        tracking_loop_results[label] = benchmark_tracking_loop(label)
    for label in TEST_SERIES:
        if label not in tracking_loop_results:
            tracking_loop_results[label] = {"skipped": True, "reason": "quick mode"}

    # --- Pipeline benchmarks (full I/O) ---
    if args.skip_pipeline:
        pipeline_results = {k: {"skipped": True, "reason": "--skip-pipeline"} for k in TEST_SERIES}
    else:
        log("Running pipeline benchmarks (full I/O)...")
        pipeline_results = benchmark_pipeline(args.quick)

    # Assemble output
    output = {
        "benchmark_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "hardware": hardware,
        "micro_benchmarks": {
            "frame_resolution": list(resolution) if resolution else None,
            "optical_flow": flow_results,
            "mask_warp": warp_result,
            "video_decode": decode_result,
        },
        "tracking_loop": tracking_loop_results,
        "pipeline_benchmarks": pipeline_results,
    }

    json_str = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(json_str + "\n")
        log(f"Results written to {args.output}")
    else:
        print(json_str)

    log("Done.")


if __name__ == "__main__":
    main()
