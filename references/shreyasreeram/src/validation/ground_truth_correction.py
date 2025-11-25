import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import shutil
from pathlib import Path

class GroundTruthCorrector:
    def __init__(self, video_id, ground_truth_dir, algorithm_dir, frames_dir, output_dir):
        """
        Tool for semi-automated ground truth correction based on algorithm predictions
        
        Args:
            video_id: ID of the video being processed
            ground_truth_dir: Directory containing ground truth masks
            algorithm_dir: Directory containing algorithm predicted masks
            frames_dir: Directory containing original video frames
            output_dir: Directory to save corrected ground truth
        """
        self.video_id = video_id
        self.ground_truth_dir = Path(ground_truth_dir)
        self.algorithm_dir = Path(algorithm_dir)
        self.frames_dir = Path(frames_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get list of frames
        self.gt_files = sorted([f for f in os.listdir(self.ground_truth_dir) if f.endswith('.png')])
        self.algo_files = sorted([f for f in os.listdir(self.algorithm_dir) if f.endswith('.png')])
        
        # Find frames that exist in both ground truth and algorithm dirs
        self.common_frames = []
        for f in self.gt_files:
            if f in self.algo_files:
                self.common_frames.append(f)
        
        # Calculate discrepancy scores to prioritize frames
        self.discrepancy_scores = self._calculate_discrepancies()
        
        # Sort frames by discrepancy score (highest first)
        self.review_queue = [f for _, f in sorted(zip(self.discrepancy_scores, self.common_frames), reverse=True)]
        
        # Initialize tracking variables
        self.current_index = 0
        self.corrections_made = 0
        self.frames_reviewed = 0
        
        # Set up the UI
        self.fig = None
        self.current_frame = None
        self.current_gt = None
        self.current_algo = None
    
    def _calculate_discrepancies(self):
        """Calculate discrepancy scores between ground truth and algorithm predictions"""
        scores = []
        
        for frame_file in self.common_frames:
            gt_path = self.ground_truth_dir / frame_file
            algo_path = self.algorithm_dir / frame_file
            
            gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0
            algo_mask = cv2.imread(str(algo_path), cv2.IMREAD_GRAYSCALE) > 0
            
            # Calculate IoU
            intersection = np.logical_and(gt_mask, algo_mask).sum()
            union = np.logical_or(gt_mask, algo_mask).sum()
            
            if union > 0:
                iou = intersection / union
            else:
                iou = 1.0  # Both empty
            
            # Calculate symmetric difference (disagreement)
            diff = np.logical_xor(gt_mask, algo_mask).sum()
            
            # Score is a combination of low IoU and high difference
            scores.append((1.0 - iou) * (diff + 1))
        
        return scores
    
    def _load_current_frame(self):
        """Load the current frame and masks"""
        if self.current_index >= len(self.review_queue):
            return False
        
        frame_file = self.review_queue[self.current_index]
        frame_num = int(frame_file.split('.')[0])
        
        # Load original frame
        frame_path = self.frames_dir / f"{frame_num}.png"
        if not frame_path.exists():
            print(f"Warning: Original frame {frame_path} not found")
            self.current_frame = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            self.current_frame = cv2.imread(str(frame_path))
            self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        
        # Load ground truth
        gt_path = self.ground_truth_dir / frame_file
        self.current_gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) > 0
        
        # Load algorithm prediction
        algo_path = self.algorithm_dir / frame_file
        self.current_algo = cv2.imread(str(algo_path), cv2.IMREAD_GRAYSCALE) > 0
        
        return True
    
    def _update_display(self):
        """Update the display with current frame information"""
        if self.fig is None:
            plt.ion()
            self.fig, self.axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.subplots_adjust(bottom=0.2)
            
            # Add buttons
            self.ax_keep_gt = plt.axes([0.2, 0.05, 0.2, 0.075])
            self.ax_use_algo = plt.axes([0.45, 0.05, 0.2, 0.075])
            self.ax_skip = plt.axes([0.7, 0.05, 0.1, 0.075])
            
            self.btn_keep_gt = Button(self.ax_keep_gt, 'Keep Ground Truth')
            self.btn_use_algo = Button(self.ax_use_algo, 'Use Algorithm')
            self.btn_skip = Button(self.ax_skip, 'Skip')
            
            self.btn_keep_gt.on_clicked(self._on_keep_gt)
            self.btn_use_algo.on_clicked(self._on_use_algo)
            self.btn_skip.on_clicked(self._on_skip)
        
        frame_file = self.review_queue[self.current_index]
        frame_num = int(frame_file.split('.')[0])
        
        # Calculate metrics
        intersection = np.logical_and(self.current_gt, self.current_algo).sum()
        union = np.logical_or(self.current_gt, self.current_algo).sum()
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 1.0
        
        # Clear axes and update with new content
        for ax in self.axes:
            ax.clear()
        
        # Original frame
        self.axes[0].imshow(self.current_frame)
        self.axes[0].set_title(f"Frame {frame_num}")
        
        # Ground truth overlay
        gt_overlay = self.current_frame.copy()
        gt_overlay[self.current_gt, 0] = 0
        gt_overlay[self.current_gt, 1] = 255
        gt_overlay[self.current_gt, 2] = 0
        self.axes[1].imshow(gt_overlay)
        self.axes[1].set_title("Ground Truth")
        
        # Algorithm overlay
        algo_overlay = self.current_frame.copy()
        algo_overlay[self.current_algo, 0] = 255
        algo_overlay[self.current_algo, 1] = 0
        algo_overlay[self.current_algo, 2] = 0
        self.axes[2].imshow(algo_overlay)
        self.axes[2].set_title(f"Algorithm (IoU: {iou:.3f})")
        
        # Add progress information
        plt.suptitle(f"Video: {self.video_id} - Progress: {self.frames_reviewed}/{len(self.review_queue)} - Corrections: {self.corrections_made}")
        
        self.fig.canvas.draw_idle()
    
    def _on_keep_gt(self, event):
        """Handle keeping ground truth mask"""
        frame_file = self.review_queue[self.current_index]
        
        # Copy original ground truth to output directory
        src_path = self.ground_truth_dir / frame_file
        dst_path = self.output_dir / frame_file
        shutil.copy(src_path, dst_path)
        
        print(f"Kept original ground truth for frame {frame_file}")
        
        self.frames_reviewed += 1
        self.current_index += 1
        self._next_frame()
    
    def _on_use_algo(self, event):
        """Handle using algorithm prediction as corrected ground truth"""
        frame_file = self.review_queue[self.current_index]
        
        # Copy algorithm mask to output directory
        src_path = self.algorithm_dir / frame_file
        dst_path = self.output_dir / frame_file
        shutil.copy(src_path, dst_path)
        
        print(f"Used algorithm prediction for frame {frame_file}")
        
        self.corrections_made += 1
        self.frames_reviewed += 1
        self.current_index += 1
        self._next_frame()
    
    def _on_skip(self, event):
        """Handle skipping the current frame"""
        self.current_index += 1
        self._next_frame()
    
    def _next_frame(self):
        """Move to the next frame in the queue"""
        if not self._load_current_frame():
            # No more frames to review
            plt.close(self.fig)
            print("\nReview completed!")
            print(f"Frames reviewed: {self.frames_reviewed}/{len(self.review_queue)}")
            print(f"Corrections made: {self.corrections_made}")
            return
        
        self._update_display()
    
    def start(self):
        """Start the review process"""
        if not self._load_current_frame():
            print("No frames to review!")
            return
        
        self._update_display()
        plt.show(block=True)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Semi-automated ground truth correction tool")
    parser.add_argument("--video_id", required=True, help="ID of the video being processed")
    parser.add_argument("--gt_dir", required=True, help="Directory containing ground truth masks")
    parser.add_argument("--algo_dir", required=True, help="Directory containing algorithm predicted masks")
    parser.add_argument("--frames_dir", required=True, help="Directory containing original video frames")
    parser.add_argument("--output_dir", required=True, help="Directory to save corrected ground truth")
    
    args = parser.parse_args()
    
    corrector = GroundTruthCorrector(
        args.video_id,
        args.gt_dir,
        args.algo_dir,
        args.frames_dir,
        args.output_dir
    )
    
    corrector.start()

if __name__ == "__main__":
    main() 