#!/usr/bin/env python3
import json

def parse_env_params(param_string):
    """Parse parameter string into a dictionary"""
    params = {}
    parts = param_string.split()
    for part in parts:
        if '=' in part:
            key, value = part.split('=')
            params[key] = value
    return params

def parse_results_file(results_file):
    """Parse results from the feedback loop results file"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    return {
        'mean_iou': results.get('iterations', [{}])[0].get('evaluation_results', {}).get('summary', {}).get('overall_mean_iou', 0),
        'mean_dice': results.get('iterations', [{}])[0].get('evaluation_results', {}).get('summary', {}).get('overall_mean_dice', 0),
        'frames_over_70_iou': results.get('iterations', [{}])[0].get('evaluation_results', {}).get('summary', {}).get('iou_over_0.7', 0)
    } 