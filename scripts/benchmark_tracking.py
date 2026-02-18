#!/usr/bin/env python
"""
Benchmark script for tracking worker performance.

Runs end-to-end tracking loops with different optical flow backends,
pre-loading all frames to isolate compute from video I/O.

Usage:
    python scripts/benchmark_tracking.py [OPTIONS]

Options:
    --series LABEL      Which series: small,medium,large,all (default: medium)
    --method METHOD     Comma-separated flow methods (default: all)
    --output FILE       Write JSON to file instead of stdout

Available methods:
    dis-ultrafast, dis-fast, dis-medium     OpenCV DIS presets (CPU)
    vision-low, vision-medium, vision-high  Apple Vision framework (GPU/ANE)
    farneback                               OpenCV Farneback (CPU)

Env vars:
    DIS_SCALE=N         Resolution divisor for DIS (1=full, 2=half, 4=quarter)
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

# Path setup
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lib_server = os.path.join(PROJECT_ROOT, "lib", "server")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _lib_server not in sys.path:
    sys.path.insert(0, _lib_server)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIS_METHODS = ["dis-ultrafast", "dis-fast", "dis-medium"]
VISION_METHODS = ["vision-low", "vision-medium", "vision-high"]
OTHER_METHODS = ["farneback"]
ALL_METHODS = DIS_METHODS + VISION_METHODS + OTHER_METHODS

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
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Hardware
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
# Flow backends
# ---------------------------------------------------------------------------

def make_dis_flow_fn(preset: str):
    """Return a flow function using OpenCV DIS at the given preset."""
    from lib.opticalflowprocessor import OpticalFlowProcessor

    os.environ["DIS_PRESET"] = preset
    processor = OpticalFlowProcessor("dis")

    def calc(prev_bgr, curr_bgr):
        return processor.calculate_flow(prev_bgr, curr_bgr)

    return calc, processor


def make_farneback_flow_fn():
    """Return a flow function using OpenCV Farneback."""
    def calc(prev_bgr, curr_bgr):
        prev = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY) if len(prev_bgr.shape) == 3 else prev_bgr
        curr = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY) if len(curr_bgr.shape) == 3 else curr_bgr
        return cv2.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    return calc, None


def make_vision_flow_fn(accuracy_name: str):
    """Return a flow function using Apple Vision framework (GPU/ANE)."""
    try:
        import ctypes
        import Quartz
        import Vision
    except ImportError:
        return None, None

    accuracy_map = {
        "low": Vision.VNGenerateOpticalFlowRequestComputationAccuracyLow,
        "medium": Vision.VNGenerateOpticalFlowRequestComputationAccuracyMedium,
        "high": Vision.VNGenerateOpticalFlowRequestComputationAccuracyHigh,
    }
    accuracy = accuracy_map[accuracy_name]
    cs = Quartz.CGColorSpaceCreateDeviceRGB()

    def calc(prev_bgr, curr_bgr):
        h, w = prev_bgr.shape[:2]

        bgra_p = np.ascontiguousarray(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2BGRA))
        bgra_c = np.ascontiguousarray(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2BGRA))

        ci_p = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
            Quartz.NSData.dataWithBytes_length_(bgra_p.tobytes(), bgra_p.nbytes),
            w * 4, Quartz.CGSizeMake(w, h), Quartz.kCIFormatBGRA8, cs)
        ci_c = Quartz.CIImage.imageWithBitmapData_bytesPerRow_size_format_colorSpace_(
            Quartz.NSData.dataWithBytes_length_(bgra_c.tobytes(), bgra_c.nbytes),
            w * 4, Quartz.CGSizeMake(w, h), Quartz.kCIFormatBGRA8, cs)

        req = Vision.VNGenerateOpticalFlowRequest.alloc().initWithTargetedCIImage_options_(ci_c, {})
        req.setComputationAccuracy_(accuracy)
        hdl = Vision.VNSequenceRequestHandler.alloc().init()
        ok, err = hdl.performRequests_onCIImage_error_([req], ci_p, None)
        if not ok:
            raise RuntimeError(f"Vision failed: {err}")

        obs = req.results()[0]
        buf = obs.pixelBuffer()
        Quartz.CVPixelBufferLockBaseAddress(buf, 0)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(buf)
        ph = Quartz.CVPixelBufferGetHeight(buf)
        pw = Quartz.CVPixelBufferGetWidth(buf)
        base = Quartz.CVPixelBufferGetBaseAddress(buf)
        raw = bytes(base.as_buffer(bpr * ph))
        flow = np.frombuffer(raw, dtype=np.float32).reshape(ph, -1)[:, :pw * 2].reshape(ph, pw, 2).copy()
        Quartz.CVPixelBufferUnlockBaseAddress(buf, 0)
        return flow

    return calc, None


def make_flow_fn(method: str):
    """Factory: return (flow_fn, cleanup_obj_or_None) for a method name."""
    if method.startswith("dis-"):
        preset = method.split("-", 1)[1]
        return make_dis_flow_fn(preset)
    elif method == "farneback":
        return make_farneback_flow_fn()
    elif method.startswith("vision-"):
        acc = method.split("-", 1)[1]
        return make_vision_flow_fn(acc)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Pre-load frames
# ---------------------------------------------------------------------------

def preload_frames(series_label: str) -> tuple[list[np.ndarray], int, int] | None:
    """Read all video frames into memory. Returns (frames_bgr, width, height)."""
    from lib.config import load_config
    from track import find_images_dir

    study_uid, series_uid = TEST_SERIES[series_label][:2]
    try:
        config = load_config("server")
        images_dir = find_images_dir(
            str(config.data_dir), config.project_id, config.dataset_id
        )
        video_path = os.path.join(images_dir, study_uid, f"{series_uid}.mp4")
        if not os.path.exists(video_path):
            return None
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, w, h


# ---------------------------------------------------------------------------
# End-to-end tracking loop
# ---------------------------------------------------------------------------

def run_tracking_loop(
    method: str,
    frames: list[np.ndarray],
    width: int,
    height: int,
) -> dict:
    """Run flow+warp on pre-loaded frames. Returns timing results."""
    from lib.multi_frame_tracker import MultiFrameTracker
    from lib.opticalflowprocessor import OpticalFlowProcessor

    flow_fn, processor = make_flow_fn(method)
    if flow_fn is None:
        return {"skipped": True, "reason": f"{method} not available on this platform"}

    # Dummy processor for tracker (only used for warp grid cache)
    if processor is None:
        os.environ["DIS_PRESET"] = "fast"
        processor = OpticalFlowProcessor("dis")

    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = MultiFrameTracker(processor, tmpdir)

        mask = np.zeros((height, width), dtype=np.uint8)
        mask[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 255

        # Warmup: one flow + warp
        if len(frames) >= 2:
            wf = flow_fn(frames[0], frames[1])
            tracker._warp_mask_with_flow(mask, wf)

        flow_times = []
        warp_times = []

        for i in range(len(frames) - 1):
            t0 = time.perf_counter()
            flow = flow_fn(frames[i], frames[i + 1])
            flow_times.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            mask = tracker._warp_mask_with_flow(mask, flow)
            warp_times.append((time.perf_counter() - t0) * 1000)

    if processor and hasattr(processor, "cleanup_memory"):
        processor.cleanup_memory()

    n = len(flow_times)
    if n == 0:
        return {"skipped": True, "reason": "no frames processed"}

    total_ms = sum(flow_times) + sum(warp_times)
    flow_arr = np.array(flow_times)
    warp_arr = np.array(warp_times)

    return {
        "method": method,
        "frames": n,
        "total_s": round(total_ms / 1000, 2),
        "per_frame_ms": round(total_ms / n, 2),
        "flow_ms": round(float(flow_arr.mean()), 2),
        "warp_ms": round(float(warp_arr.mean()), 2),
        "flow_pct": round(sum(flow_times) / total_ms * 100, 1),
        "dis_scale": int(os.environ.get("DIS_SCALE", "1")),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark tracking with different optical flow backends"
    )
    parser.add_argument(
        "--series", default="medium",
        help="Series to test: small,medium,large,all (default: medium)",
    )
    parser.add_argument(
        "--method", default="all",
        help=f"Comma-separated methods (default: all). Available: {','.join(ALL_METHODS)}",
    )
    parser.add_argument("--output", help="Write JSON to file instead of stdout")
    args = parser.parse_args()

    if args.method == "all":
        methods = ALL_METHODS
    else:
        methods = [m.strip() for m in args.method.split(",")]
        invalid = [m for m in methods if m not in ALL_METHODS]
        if invalid:
            parser.error(
                f"Unknown methods: {', '.join(invalid)}. Valid: {', '.join(ALL_METHODS)}"
            )

    if args.series == "all":
        series_list = list(TEST_SERIES.keys())
    else:
        series_list = [s.strip() for s in args.series.split(",")]

    os.chdir(PROJECT_ROOT)

    log("Collecting hardware info...")
    hardware = collect_hardware_info()
    log(f"  {hardware['platform']}, {hardware['cpu_cores']} cores, "
        f"{hardware['memory_gb']}GB, device={hardware['device']}")

    # Pre-load all series frames
    series_data = {}
    for label in series_list:
        log(f"Pre-loading {label} series...")
        result = preload_frames(label)
        if result is None:
            log(f"  SKIP: frames not available for {label}")
            continue
        frames, w, h = result
        series_data[label] = (frames, w, h)
        log(f"  {len(frames)} frames, {w}x{h} ({w*h/1e6:.1f}MP)")

    # Run tracking loop for each method x series
    results = {}
    for label in series_list:
        if label not in series_data:
            results[label] = {m: {"skipped": True, "reason": "no frames"} for m in methods}
            continue

        frames, w, h = series_data[label]
        results[label] = {}

        for method in methods:
            log(f"  {label} x {method}...")
            try:
                results[label][method] = run_tracking_loop(method, frames, w, h)
            except Exception as exc:
                log(f"    ERROR: {exc}")
                results[label][method] = {"skipped": True, "reason": str(exc)}

    # Summary table to stderr
    log("")
    header_parts = [f"{'Method':<20s}"]
    for label in series_list:
        if label in series_data:
            n = len(series_data[label][0])
            header_parts.append(f"{label} ({n}f)")
    log("  ".join(header_parts))
    log("-" * (22 + 18 * len(series_list)))

    for method in methods:
        parts = [f"{method:<20s}"]
        for label in series_list:
            if label in results and method in results[label]:
                r = results[label][method]
                if r.get("skipped"):
                    parts.append(f"{'skip':>14s}")
                else:
                    parts.append(f"{r['total_s']:5.1f}s ({r['per_frame_ms']:5.1f}ms/f)")

            else:
                parts.append(f"{'--':>14s}")
        log("  ".join(parts))

    # JSON output
    output = {
        "benchmark_version": "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "hardware": hardware,
        "dis_scale": int(os.environ.get("DIS_SCALE", "1")),
        "results": results,
    }

    json_str = json.dumps(output, indent=2)

    if args.output:
        Path(args.output).write_text(json_str + "\n")
        log(f"\nResults written to {args.output}")
    else:
        print(json_str)

    log("Done.")


if __name__ == "__main__":
    main()
