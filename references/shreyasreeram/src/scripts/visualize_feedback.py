#!/usr/bin/env python3
import os
import argparse
from src.visualization.parameter_impact import visualize_parameter_impact

def main():
    parser = argparse.ArgumentParser(description='Visualize parameter impact on tracking performance')
    parser.add_argument('feedback_dir', help='Directory containing feedback loop results')
    args = parser.parse_args()
    
    if not os.path.exists(args.feedback_dir):
        print(f"Error: Directory {args.feedback_dir} does not exist")
        return
        
    visualize_parameter_impact(args.feedback_dir)

if __name__ == '__main__':
    main() 