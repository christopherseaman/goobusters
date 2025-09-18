# Ground Truth Creation and Feedback Loop Documentation

## Overview

This document explains the ground truth creation process and feedback loop functionality in the Goobusters project. These tools are designed to improve the accuracy of fluid detection in ultrasound videos through iterative learning and expert feedback.

## Ground Truth Creation

### Purpose
The ground truth creation process generates a dataset of verified fluid annotations that can be used to:
- Train and validate the fluid detection algorithm
- Evaluate algorithm performance
- Create benchmarks for testing improvements

### Running Ground Truth Creation

Basic command:
```bash
python src/consolidated_tracking.py --create-ground-truth
```

#### Options

- `--ground-truth-single-exam EXAM_NUMBER`: Process only a specific exam number
  ```bash
  python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186
  ```

- `--upload`: Upload the generated annotations to MD.ai
  ```bash
  python src/consolidated_tracking.py --create-ground-truth --upload
  ```

- `--output-dir PATH`: Specify custom output directory (default: src/output/ground_truth)
  ```bash
  python src/consolidated_tracking.py --create-ground-truth --output-dir /path/to/output
  ```

## Feedback Loop

### Purpose
The feedback loop is an iterative process that:
1. Runs the fluid detection algorithm
2. Compares results with ground truth
3. Adjusts parameters based on performance
4. Repeats to improve accuracy

### Running the Feedback Loop

Basic command:
```bash
python src/consolidated_tracking.py --feedback-loop
```

#### Options

- `--iterations N`: Number of feedback loop iterations (default: 3)
  ```bash
  python src/consolidated_tracking.py --feedback-loop --iterations 5
  ```

- `--exam-id EXAM_NUMBER`: Run feedback loop on specific exam
  ```bash
  python src/consolidated_tracking.py --feedback-loop --exam-id 186
  ```

- `--learning-mode`: Enable learning mode for parameter optimization
  ```bash
  python src/consolidated_tracking.py --feedback-loop --learning-mode
  ```

- `--sampling-rate N`: Set frame sampling rate (default: 10)
  ```bash
  python src/consolidated_tracking.py --feedback-loop --sampling-rate 5
  ```

- `--use-genuine-evaluation`: Use genuine evaluation mode
  ```bash
  python src/consolidated_tracking.py --feedback-loop --use-genuine-evaluation
  ```

### Additional Parameters

- `--label-id-fluid ID`: Specify fluid label ID
- `--label-id-no-fluid ID`: Specify no-fluid label ID
- `--label-id-machine ID`: Specify machine label ID
- `--project-id ID`: MD.ai project ID
- `--dataset-id ID`: MD.ai dataset ID

## Output and Results

### Ground Truth Output
- Mask files for each processed frame
- JSON file with processing results
- Visualization files (if enabled)
- MD.ai annotations (if upload enabled)

### Feedback Loop Output
- Performance metrics for each iteration
- Parameter adjustments log
- Comparison visualizations
- Final evaluation report

## Common Use Cases

1. **Initial Ground Truth Creation**
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --upload
   ```

2. **Single Exam Processing**
   ```bash
   python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186 --upload
   ```

3. **Feedback Loop with Learning**
   ```bash
   python src/consolidated_tracking.py --feedback-loop --learning-mode --iterations 5 --exam-id 186
   ```

4. **Quick Evaluation**
   ```bash
   python src/consolidated_tracking.py --feedback-loop --use-genuine-evaluation --sampling-rate 20
   ```

## Environment Setup

Required environment variables:
- `DATA_DIR`: Path to data directory
- `LABEL_ID_FREE_FLUID`: MD.ai label ID for free fluid
- `LABEL_ID_NO_FLUID`: MD.ai label ID for no fluid
- `LABEL_ID_MACHINE_GROUP`: MD.ai label ID for machine annotations

## Troubleshooting

Common issues and solutions:

1. **Missing Data Directory**
   - Ensure `DATA_DIR` environment variable is set
   - Verify data directory contains required files

2. **MD.ai Authentication**
   - Check MD.ai API key is set
   - Verify project and dataset IDs

3. **Memory Issues**
   - Reduce batch size
   - Process single exams instead of full dataset
   - Clear cache between runs

4. **Performance Issues**
   - Adjust sampling rate
   - Reduce number of iterations
   - Use single exam processing

## Best Practices

1. Always start with a single exam test before processing full dataset
2. Use `--upload` flag only after verifying output quality
3. Keep sampling rate balanced between speed and accuracy
4. Monitor system resources during processing
5. Save and version control parameter configurations that work well 