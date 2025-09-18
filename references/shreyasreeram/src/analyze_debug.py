import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def analyze_debug_output(debug_dir):
    """
    Analyze debug output from the tracking system.
    
    Args:
        debug_dir: Path to debug directory
    """
    print("\n=== Analyzing Debug Output ===")
    
    # 1. Analyze tracking parameters
    params_files = [f for f in os.listdir(debug_dir) if f.endswith('tracking_params.txt')]
    if params_files:
        print("\nTracking Parameters Used:")
        with open(os.path.join(debug_dir, params_files[0]), 'r') as f:
            print(f.read())
    
    # 2. Analyze frame anomalies
    anomaly_dirs = [d for d in os.listdir(debug_dir) if 'anomaly' in d]
    if anomaly_dirs:
        print(f"\nDetected {len(anomaly_dirs)} frame anomalies")
        
        # Create anomaly visualization
        plt.figure(figsize=(15, 5))
        anomaly_frames = [int(d.split('_')[1]) for d in anomaly_dirs]
        plt.plot(sorted(anomaly_frames), [1]*len(anomaly_frames), 'ro', label='Anomalies')
        plt.title('Frame Anomaly Distribution')
        plt.xlabel('Frame Number')
        plt.ylabel('Occurrence')
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(debug_dir, f'anomaly_distribution_{timestamp}.png'))
        plt.close()
    
    # 3. Analyze tracking quality
    quality_data = []
    for d in os.listdir(debug_dir):
        if d.startswith('track_'):
            analysis_file = os.path.join(debug_dir, d, 'analysis.txt')
            if os.path.exists(analysis_file):
                with open(analysis_file, 'r') as f:
                    quality_data.append(f.read())
    
    if quality_data:
        print("\nTracking Quality Summary:")
        print(f"Total tracking segments analyzed: {len(quality_data)}")
        
    # 4. Create tracking quality visualization
    if os.path.exists(os.path.join(debug_dir, 'metrics.json')):
        with open(os.path.join(debug_dir, 'metrics.json'), 'r') as f:
            metrics = json.load(f)
            
        plt.figure(figsize=(15, 5))
        
        # Plot IoU scores if available
        if 'iou_scores' in metrics:
            frames = sorted(list(map(int, metrics['iou_scores'].keys())))
            scores = [metrics['iou_scores'][str(f)] for f in frames]
            plt.plot(frames, scores, label='IoU Score')
            plt.axhline(y=metrics.get('mean_iou', 0), color='r', linestyle='--', 
                       label=f"Mean IoU: {metrics.get('mean_iou', 0):.3f}")
        
        plt.title('Tracking Quality Over Time')
        plt.xlabel('Frame Number')
        plt.ylabel('IoU Score')
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(debug_dir, f'tracking_quality_{timestamp}.png'))
        plt.close()
    
    print("\nVisualization files created:")
    print(f"1. Anomaly distribution: anomaly_distribution_{timestamp}.png")
    print(f"2. Tracking quality: tracking_quality_{timestamp}.png")

def main():
    """
    Example usage of the analysis function.
    """
    debug_dir = "path/to/debug/directory"  # Replace with actual debug directory
    if os.path.exists(debug_dir):
        analyze_debug_output(debug_dir)
    else:
        print(f"Debug directory not found: {debug_dir}")

if __name__ == "__main__":
    main() 