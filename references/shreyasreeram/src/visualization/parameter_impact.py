import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def visualize_parameter_impact(feedback_loop_dir):
    """
    Create visualizations showing how parameter changes affect IoU and Dice scores.
    
    Args:
        feedback_loop_dir: Directory containing feedback loop results
    """
    # Find all iteration directories
    iteration_dirs = sorted([d for d in os.listdir(feedback_loop_dir) 
                           if d.startswith('iteration_')])
    
    # Collect data from each iteration
    iterations = []
    mean_ious = []
    mean_dices = []
    params = []
    
    for iter_dir in iteration_dirs:
        # Load evaluation results
        eval_dir = os.path.join(feedback_loop_dir, iter_dir, 'evaluation')
        for subdir in os.listdir(eval_dir):
            if 'evaluation' in subdir:
                results_file = os.path.join(eval_dir, subdir, 'genuine_evaluation', 'sparse_validation_results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        iterations.append(int(iter_dir.split('_')[1]))
                        mean_ious.append(results.get('mean_iou', 0))
                        mean_dices.append(results.get('mean_dice', 0))
        
        # Load parameters
        param_files = [f for f in os.listdir(os.path.join(feedback_loop_dir, iter_dir))
                      if f.startswith('tracking_params')]
        if param_files:
            with open(os.path.join(feedback_loop_dir, iter_dir, param_files[0]), 'r') as f:
                params.append(json.load(f))
    
    # Create visualization directory
    vis_dir = os.path.join(feedback_loop_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Plot 1: Performance Metrics Over Iterations
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, mean_ious, 'b-o', label='Mean IoU')
    plt.plot(iterations, mean_dices, 'g-o', label='Mean Dice')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('Performance Metrics Across Iterations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'performance_metrics.png'))
    plt.close()
    
    # Plot 2: Parameter Changes
    if params:
        param_keys = ['flow_quality_threshold', 'tracking_strategy_weight', 
                     'expert_feedback_weight', 'border_constraint_weight']
        plt.figure(figsize=(12, 6))
        
        for key in param_keys:
            values = [p.get('tracking_params', {}).get(key, 0) for p in params]
            plt.plot(iterations, values, '-o', label=key)
        
        plt.xlabel('Iteration')
        plt.ylabel('Parameter Value')
        plt.title('Parameter Values Across Iterations')
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'parameter_changes.png'))
        plt.close()
    
    # Plot 3: Frame-by-Frame Performance
    plt.figure(figsize=(15, 6))
    for i, iter_dir in enumerate(iteration_dirs):
        eval_dir = os.path.join(feedback_loop_dir, iter_dir, 'evaluation')
        for subdir in os.listdir(eval_dir):
            if 'evaluation' in subdir:
                results_file = os.path.join(eval_dir, subdir, 'genuine_evaluation', 'sparse_validation_results.json')
                if os.path.exists(results_file):
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        frame_ious = results.get('frame_ious', {})
                        frames = sorted([int(k) for k in frame_ious.keys()])
                        ious = [frame_ious[str(f)] for f in frames]
                        plt.plot(frames, ious, '-', alpha=0.7, 
                               label=f'Iteration {iterations[i]}')
    
    plt.xlabel('Frame Number')
    plt.ylabel('IoU Score')
    plt.title('Frame-by-Frame IoU Scores Across Iterations')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'frame_performance.png'))
    plt.close()
    
    # Create summary report
    report_path = os.path.join(vis_dir, 'parameter_impact_report.md')
    with open(report_path, 'w') as f:
        f.write('# Parameter Impact Analysis\n\n')
        f.write(f'Analysis generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## Performance Summary\n\n')
        f.write('| Iteration | Mean IoU | Mean Dice |\n')
        f.write('|-----------|-----------|------------|\n')
        for i, (iou, dice) in enumerate(zip(mean_ious, mean_dices)):
            f.write(f'| {iterations[i]} | {iou:.4f} | {dice:.4f} |\n')
        
        if params:
            f.write('\n## Parameter Evolution\n\n')
            f.write('| Iteration | ' + ' | '.join(param_keys) + ' |\n')
            f.write('|-----------|' + '|'.join(['--------' for _ in param_keys]) + '|\n')
            for i, p in enumerate(params):
                values = [str(p.get('tracking_params', {}).get(k, 'N/A')) for k in param_keys]
                f.write(f"| {iterations[i]} | {' | '.join(values)} |\n")
    
    print(f"\nVisualization results saved to {vis_dir}")
    print(f"Generated visualizations:")
    print("1. performance_metrics.png - IoU and Dice scores across iterations")
    print("2. parameter_changes.png - Parameter value changes across iterations")
    print("3. frame_performance.png - Frame-by-frame IoU scores")
    print(f"4. parameter_impact_report.md - Detailed analysis report") 