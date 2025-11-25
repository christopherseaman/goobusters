#!/usr/bin/env python3
import argparse
from src.visualization.parameter_tracking import ParameterTracker
from src.utils.parameter_utils import parse_env_params, parse_results_file

def main():
    parser = argparse.ArgumentParser(description='Track and analyze parameter combinations')
    parser.add_argument('--params', required=True, help='Parameter string (e.g., "FLOW_QUALITY_THRESHOLD=0.5 TRACKING_STRATEGY_WEIGHT=0.8")')
    parser.add_argument('--results-file', required=True, help='Path to feedback loop results file')
    parser.add_argument('--output-dir', default='src/output/parameter_tracking', help='Directory to save tracking results')
    args = parser.parse_args()
    
    # Parse parameters and results
    params = parse_env_params(args.params)
    results = parse_results_file(args.results_file)
    
    # Initialize tracker and add run
    tracker = ParameterTracker(args.output_dir)
    tracker.add_run(params, results)
    
    # Generate visualizations
    tracker.visualize_parameter_impact()

if __name__ == '__main__':
    main() 