# Feedback Loop Optimization Guide

## Overview
The feedback loop system in Goobusters is designed to optimise tracking parameters through iterative evaluation and adjustment. This guide explains the parameters that can be optimsed, their effects, and best practices for achieving optimal results.

## Optimisable Parameters

### Flow Quality Parameters

#### `FLOW_QUALITY_THRESHOLD` (Default: 0.7)
Controls how strict the system is in accepting optical flow results.
- Range: 0.0 to 1.0
- Higher values (>0.8): More strict, fewer but more reliable tracks
- Lower values (<0.6): More permissive, more tracks but potentially less accurate
- Best practices:
  - Start with 0.7 and adjust based on results
  - For clear videos: Try 0.8-0.9
  - For noisy videos: Try 0.5-0.6

#### `FLOW_MAGNITUDE_THRESHOLD` (Default: 2.0)
Minimum movement required to consider flow significant.
- Range: 0.5 to 5.0
- Higher values: Only track significant movement
- Lower values: Track subtle movements
- Best practices:
  - Increase for shaky videos
  - Decrease for very stable videos

### Tracking Strategy Parameters

#### `TRACKING_STRATEGY_WEIGHT` (Default: 0.6)
Balance between optical flow and feature matching.
- Range: 0.0 to 1.0
- Higher values: More emphasis on optical flow
- Lower values: More emphasis on feature matching
- Best practices:
  - Use higher values (0.7-0.8) for continuous movement
  - Use lower values (0.4-0.5) for discontinuous movement

#### `CONFIDENCE_THRESHOLD` (Default: 0.75)
Minimum confidence required for tracking results.
- Range: 0.0 to 1.0
- Effects:
  - Higher values: More conservative tracking
  - Lower values: More aggressive tracking
- Trade-offs:
  - Higher values reduce false positives but may miss true fluid
  - Lower values catch more fluid but may include artifacts

### Motion Detection Parameters

#### `MOTION_DETECTION_SENSITIVITY` (Default: 30)
Sensitivity to frame-to-frame changes.
- Range: 10 to 50
- Higher values: More sensitive to small changes
- Lower values: Only detect significant changes
- Use cases:
  - High (40-50): Subtle fluid movement
  - Low (10-20): Obvious fluid movement

## Optimisation Strategies

### 1. Basic Optimization
```bash
python src/consolidated_tracking.py --feedback-loop \
    --learning-mode \
    --iterations 5 \
    --exam-id 186
```
Best for: Initial parameter tuning

### 2. Fine-Tuning
```bash
python src/consolidated_tracking.py --feedback-loop \
    --learning-mode \
    --genuine-evaluation \
    --sampling-rate 5 \
    --iterations 10
```
Best for: Precise parameter optimization

### 3. Production Optimisation
```bash
python src/consolidated_tracking.py --feedback-loop \
    --learning-mode \
    --genuine-evaluation \
    --sampling-rate 10 \
    --iterations 15 \
    --params-file base_params.json
```
Best for: Final parameter tuning before deployment

## Parameter Configuration File

Example `params.json`:
```json
{
    "flow_quality_threshold": 0.7,
    "flow_magnitude_threshold": 2.0,
    "tracking_strategy_weight": 0.6,
    "confidence_threshold": 0.75,
    "motion_detection_sensitivity": 30
}
```

## Best Practices

### 1. Iterative Optimisation
1. Start with default parameters
2. Run basic optimisation (5 iterations)
3. Identify problematic cases
4. Fine-tune specific parameters
5. Validate with genuine evaluation

### 2. Issue-Specific Tuning

#### For Disappearing/Reappearing Fluid
```json
{
    "flow_quality_threshold": 0.6,
    "tracking_strategy_weight": 0.4,
    "confidence_threshold": 0.7
}
```

#### For Continuous Fluid Movement
```json
{
    "flow_quality_threshold": 0.8,
    "tracking_strategy_weight": 0.7,
    "confidence_threshold": 0.8
}
```

#### For Multiple Distinct Regions
```json
{
    "flow_quality_threshold": 0.75,
    "tracking_strategy_weight": 0.5,
    "confidence_threshold": 0.8
}
```

### 3. Performance Monitoring

Monitor these metrics during optimisation:
- IoU (Intersection over Union)
- Dice coefficient
- False positive rate
- False negative rate
- Processing time

## Troubleshooting

### Common Issues and Solutions

1. **Over-tracking**
   - Symptoms: Too many regions tracked
   - Solution: Increase flow_quality_threshold and confidence_threshold

2. **Under-tracking**
   - Symptoms: Missing obvious fluid regions
   - Solution: Decrease flow_quality_threshold and confidence_threshold

3. **Unstable Tracking**
   - Symptoms: Regions appear/disappear frequently
   - Solution: Adjust tracking_strategy_weight and motion_detection_sensitivity

4. **Slow Processing**
   - Symptoms: Long processing times
   - Solution: Increase sampling_rate, reduce iterations

## Advanced Usage

### Custom Parameter Ranges
```python
parameter_ranges = {
    'flow_quality_threshold': (0.5, 0.9),
    'tracking_strategy_weight': (0.3, 0.8),
    'confidence_threshold': (0.6, 0.9)
}
```

### Learning Rate Adjustment
```bash
python src/consolidated_tracking.py --feedback-loop \
    --learning-mode \
    --learning-rate 0.1 \
    --iterations 10
```

## Evaluation Metrics

Understanding the output metrics:
- IoU > 0.7: Excellent tracking
- IoU 0.5-0.7: Good tracking
- IoU < 0.5: Poor tracking, needs optimization

## Version History

Track your parameter configurations:
```json
{
    "version": "1.0",
    "date": "2025-05-28",
    "params": {
        "flow_quality_threshold": 0.75,
        "tracking_strategy_weight": 0.65
    },
    "performance": {
        "mean_iou": 0.72,
        "mean_dice": 0.68
    }
}
``` 