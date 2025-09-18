import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

class ParameterTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.tracking_file = os.path.join(output_dir, 'parameter_tracking.json')
        self.runs = self._load_or_create_tracking()
        
    def _load_or_create_tracking(self):
        if os.path.exists(self.tracking_file):
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return {
            'runs': [],
            'best_iou': 0,
            'best_params': None,
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
    
    def add_run(self, params, results):
        """Add a new parameter combination and its results"""
        run_data = {
            'parameters': {
                'flow_quality_threshold': float(params.get('FLOW_QUALITY_THRESHOLD', 0)),
                'tracking_strategy_weight': float(params.get('TRACKING_STRATEGY_WEIGHT', 0)),
                'expert_feedback_weight': float(params.get('EXPERT_FEEDBACK_WEIGHT', 0)),
                'min_tracking_frames': int(params.get('MIN_TRACKING_FRAMES', 0)),
                'max_tracking_frames': int(params.get('MAX_TRACKING_FRAMES', 0)),
                'force_tracking': int(params.get('FORCE_TRACKING', 0)),
                'border_constraint_weight': float(params.get('BORDER_CONSTRAINT_WEIGHT', 0)),
                'flow_noise_threshold': float(params.get('FLOW_NOISE_THRESHOLD', 0))
            },
            'results': {
                'mean_iou': float(results.get('mean_iou', 0)),
                'mean_dice': float(results.get('mean_dice', 0)),
                'frames_over_70_iou': float(results.get('frames_over_70_iou', 0))
            },
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
        }
        
        self.runs['runs'].append(run_data)
        
        # Update best parameters if this run is better
        if run_data['results']['mean_iou'] > self.runs['best_iou']:
            self.runs['best_iou'] = run_data['results']['mean_iou']
            self.runs['best_params'] = run_data['parameters']
        
        self._save_tracking()
        
    def _save_tracking(self):
        """Save tracking data to file"""
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.tracking_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
            
    def visualize_parameter_impact(self):
        """Create visualizations of parameter impact on performance"""
        if not self.runs['runs']:
            print("No runs to visualize")
            return
            
        vis_dir = os.path.join(self.output_dir, 'parameter_analysis')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Extract data
        runs = self.runs['runs']
        timestamps = [run['timestamp'] for run in runs]
        mean_ious = [run['results']['mean_iou'] for run in runs]
        mean_dices = [run['results']['mean_dice'] for run in runs]
        
        # Plot 1: Performance Over Time
        plt.figure(figsize=(12, 6))
        x = range(len(runs))
        plt.plot(x, mean_ious, 'b-o', label='Mean IoU')
        plt.plot(x, mean_dices, 'g-o', label='Mean Dice')
        plt.xlabel('Run Number')
        plt.ylabel('Score')
        plt.title('Performance Metrics Across Parameter Combinations')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'performance_over_time.png'))
        plt.close()
        
        # Plot 2: Parameter Impact on IoU
        param_keys = ['flow_quality_threshold', 'tracking_strategy_weight', 
                     'expert_feedback_weight', 'border_constraint_weight']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(param_keys):
            values = [run['parameters'][param] for run in runs]
            ax = axes[i]
            ax.scatter(values, mean_ious, alpha=0.6)
            ax.set_xlabel(param)
            ax.set_ylabel('Mean IoU')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(values, mean_ious, 1)
            p = np.poly1d(z)
            ax.plot(values, p(values), "r--", alpha=0.8)
            
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'parameter_impact.png'))
        plt.close()
        
        # Plot 3: Parameter Correlation Matrix
        param_values = {param: [run['parameters'][param] for run in runs] for param in param_keys}
        param_values['mean_iou'] = mean_ious
        
        correlations = np.zeros((len(param_keys) + 1, len(param_keys) + 1))
        all_keys = param_keys + ['mean_iou']
        
        for i, key1 in enumerate(all_keys):
            for j, key2 in enumerate(all_keys):
                correlations[i, j] = np.corrcoef(param_values[key1], param_values[key2])[0, 1]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(correlations, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(all_keys)), all_keys, rotation=45, ha='right')
        plt.yticks(range(len(all_keys)), all_keys)
        plt.title('Parameter Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, 'correlation_matrix.png'))
        plt.close()
        
        # Create summary report
        report_path = os.path.join(vis_dir, 'parameter_analysis_report.md')
        with open(report_path, 'w') as f:
            f.write('# Parameter Analysis Report\n\n')
            f.write(f'Analysis generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            f.write('## Best Performance\n\n')
            f.write(f'Best Mean IoU: {self.runs["best_iou"]:.4f}\n\n')
            f.write('Best Parameters:\n```\n')
            for param, value in self.runs['best_params'].items():
                f.write(f'{param}: {value}\n')
            f.write('```\n\n')
            
            f.write('## All Runs\n\n')
            f.write('| Run | ' + ' | '.join(param_keys) + ' | Mean IoU | Mean Dice |\n')
            f.write('|-----|' + '|'.join(['--------' for _ in range(len(param_keys) + 2)]) + '|\n')
            
            for i, run in enumerate(runs):
                params = [f"{run['parameters'][k]:.3f}" for k in param_keys]
                f.write(f"| {i} | {' | '.join(params)} | {run['results']['mean_iou']:.4f} | {run['results']['mean_dice']:.4f} |\n")
        
        print(f"\nVisualization results saved to {vis_dir}")
        print(f"Generated visualizations:")
        print("1. performance_over_time.png - IoU and Dice scores across runs")
        print("2. parameter_impact.png - Impact of each parameter on IoU")
        print("3. correlation_matrix.png - Parameter correlation analysis")
        print(f"4. parameter_analysis_report.md - Detailed analysis report") 