#!/usr/bin/env python3
import os
import argparse
import json
from pathlib import Path
from src.validation.sparse_validation import create_sparse_validation_set
from src.validation.ground_truth_correction import GroundTruthCorrector

def main():
    parser = argparse.ArgumentParser(description="Run validation with sparse validation set and ground truth correction")
    parser.add_argument("--video_id", required=True, help="ID of the video being processed")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth masks")
    parser.add_argument("--algo_dir", required=True, help="Directory containing algorithm predicted masks")
    parser.add_argument("--frames_dir", required=True, help="Directory containing original video frames")
    parser.add_argument("--output_dir", required=True, help="Base directory for outputs")
    parser.add_argument("--num_frames", type=int, default=25, help="Number of frames for sparse validation")
    parser.add_argument("--selected_frames", help="Comma-separated list of specific frame numbers for validation (e.g. '30,67,84')")
    parser.add_argument("--mode", choices=["sparse", "correct", "both"], default="both",
                       help="Mode to run: sparse validation, ground truth correction, or both")
    
    args = parser.parse_args()
    
    # Create output directories
    base_output_dir = Path(args.output_dir)
    sparse_output_dir = base_output_dir / "sparse_validation"
    corrected_output_dir = base_output_dir / "corrected_ground_truth"
    
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Process selected frames if provided
    selected_frames = None
    if args.selected_frames:
        selected_frames = [int(f.strip()) for f in args.selected_frames.split(',')]
        print(f"Using user-selected frames: {selected_frames}")
    
    # Run sparse validation if requested
    if args.mode in ["sparse", "both"]:
        os.makedirs(sparse_output_dir, exist_ok=True)
        
        if selected_frames:
            print(f"\n=== Running Sparse Validation with {len(selected_frames)} user-selected frames ===")
        else:
            print(f"\n=== Running Sparse Validation with {args.num_frames} frames ===")
            
        results = create_sparse_validation_set(
            args.gt_dir,
            args.algo_dir,
            args.num_frames,
            sparse_output_dir,
            selected_frames
        )
        
        # Print summary metrics
        if "summary" in results:
            print("\nSummary Metrics (Sparse Validation Set):")
            print(f"Mean IoU: {results['summary']['mean_iou']:.4f}")
            print(f"Median IoU: {results['summary']['median_iou']:.4f}")
            print(f"Mean Dice: {results['summary']['mean_dice']:.4f}")
            print(f"Frames with IoU > 0.7: {results['summary']['iou_over_07']*100:.1f}%")
            
            # Save a readable summary to the output directory
            with open(os.path.join(sparse_output_dir, "summary.txt"), 'w') as f:
                f.write("Sparse Validation Summary\n")
                f.write("=========================\n\n")
                f.write(f"Video ID: {args.video_id}\n")
                
                if selected_frames:
                    f.write(f"Frames evaluated: {len(results['frame_ious'])} (user-selected)\n")
                    f.write(f"Selected frames: {', '.join(map(str, sorted(selected_frames)))}\n\n")
                else:
                    f.write(f"Frames evaluated: {len(results['frame_ious'])} (auto-selected)\n\n")
                
                f.write(f"Mean IoU: {results['summary']['mean_iou']:.4f}\n")
                f.write(f"Median IoU: {results['summary']['median_iou']:.4f}\n")
                f.write(f"Mean Dice: {results['summary']['mean_dice']:.4f}\n")
                f.write(f"Frames with IoU > 0.7: {results['summary']['iou_over_07']*100:.1f}%\n\n")
                f.write("Note: This evaluation uses a selected subset of frames\n")
                f.write("and may be more representative of algorithm performance\n")
                f.write("than full evaluation against potentially flawed ground truth.\n")
    
    # Run ground truth correction if requested
    if args.mode in ["correct", "both"]:
        os.makedirs(corrected_output_dir, exist_ok=True)
        print("\n=== Running Ground Truth Correction Tool ===")
        print("This will open an interactive interface.")
        print("For each frame, you can:")
        print("  - Keep the original ground truth")
        print("  - Use the algorithm prediction as the new ground truth")
        print("  - Skip to the next frame")
        print("\nFrames are sorted by discrepancy between ground truth and algorithm predictions.")
        
        corrector = GroundTruthCorrector(
            args.video_id,
            args.gt_dir,
            args.algo_dir,
            args.frames_dir,
            corrected_output_dir
        )
        
        corrector.start()
    
    print("\nValidation complete!")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 