# Goobusters: Optical Flow-Based Free Fluid Tracking in Pediatric FAST Examinations

## Overview
Goobusters is a semi-automated pipeline that transforms sparse expert annotations into densely annotated ultrasound examinations using Dense Inverse Search (DIS) optical flow tracking. This research addresses the critical challenge of annotation scarcity in medical ultrasound by investigating how sparse input sampling affects tracking performance in pediatric Focused Assessment with Sonography for Trauma (FAST) examinations.

## Primary Research Question
**How does input annotation density affect optical flow tracking performance in medical ultrasound, and what is the minimum viable annotation input for clinically acceptable tracking accuracy?**

### Specific Sub-Questions:
- **Minimum Viable Input**: What is the sparsest annotation sampling rate that maintains clinically acceptable tracking performance?
- **Performance Degradation**: How does tracking accuracy degrade as input annotations become increasingly sparse?
- **Parameter Compensation**: Can adaptive parameter learning offset performance losses from sparse sampling?
- **Clinical Translation**: What annotation density requirements are practical for emergency clinical settings?

The system addresses these by:
- Testing multiple sampling rates (1:5, 1:15, 1:20, 1:30, 1:50 frame sampling)
- Using bidirectional optical flow propagation across video sequences (as described in MultiFrameTracker)
- Evaluating performance with IoU and Dice coefficients
- Investigating adaptive parameter learning for compensation

## Key Innovation: Clinical FAST Annotation Pipeline

### The Clinical Problem
- Intra-abdominal injuries (IAI) are a leading cause of pediatric trauma mortality
- CT scans expose children to harmful ionizing radiation
- FAST examinations offer safer alternatives but suffer from operator variability (35-80% sensitivity range)
- Limited availability of annotated datasets hinders AI development for FAST interpretation

### Our Solution: Multi-Frame Tracking Architecture
1. **Sparse Input**: Emergency physicians annotate key frames (1:5 to 1:50 sampling rates)
2. **DIS Optical Flow**: Dense Inverse Search algorithm tracks fluid motion between frames
3. **Bidirectional Propagation**: Forward and backward tracking with temporal weighting
4. **Quality Validation**: IoU and Dice coefficient evaluation against expert ground truth

### Research Findings
Based on 466 experimental runs across 14 pediatric FAST examinations:

| Sampling Rate | Input Density | Mean IoU (Baseline) | Annotation Burden Reduction | Clinical Viability |
|---------------|---------------|---------------------|---------------------------|-------------------|
| 1:5 | 20% | 0.651 | 79% | Good |
| 1:15 | 6.7% | 0.615 | 93.6% | Acceptable |
| 1:20 | 5.0% | 0.611 | 95% | Acceptable |
| 1:30 | 3.3% | 0.487 | 97% | Poor |
| 1:50 | 2.0% | 0.466 | 98% | Poor |

**Key Finding**: Performance remains stable between 1:15 and 1:20 sampling, with a significant drop occurring at 1:30 sampling (3.3% input density).

## Features

### Core Capabilities
- **Dense Inverse Search (DIS) Optical Flow**: Superior performance in ultrasound speckle environments
- **Bidirectional Tracking**: Forward and backward propagation with adaptive temporal weighting
- **Multi-Frame Architecture**: Specialized processing for different annotation segment types
- **Quality Control**: Area ratio monitoring, morphological operations, and boundary consistency
- **PECARN Dataset Integration**: Validated on pediatric emergency care research network data

### Clinical Validation & Limitations
- **Best Performance**: Complex mixed presentations (IoU: 0.644), uncomplicated (IoU: 0.609), branching fluid (IoU: 0.611)
- **Challenging Cases**: Multiple distinct fluid patterns (IoU: 0.440), disappear-reappear patterns (IoU: 0.449)
- **Clinical Success Rate**: 40.0% of uncomplicated cases achieve IoU > 0.7, while multiple distinct patterns achieve 0.0%
- **Parameter Learning**: Consistently counterproductive (5.2% overall performance decrease)

### Validation & Quality Control
- **Ground Truth Creation**: Establish verified annotation datasets
- **Feedback Loop System**: Continuous improvement through expert verification
- **Genuine Evaluation**: Prevent data leakage in quality assessment
- **Comprehensive Metrics**: IoU, Dice coefficients, and clinical relevance scores

### Integration & Workflow
- **MD.ai Integration**: Seamless annotation management and storage
- **Batch Processing**: Handle multiple exams simultaneously
- **Visualization Tools**: Debug and verify tracking results
- **Export Capabilities**: Multiple output formats for downstream use

## Setup

### Prerequisites
- Python 3.8+
- OpenCV 4.5+
- NumPy, Pandas, SciPy
- MD.ai Python client
- Sufficient storage for video processing

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ShreyaSreeram/goobusters.git
   cd goobusters
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your MD.ai credentials and paths
   ```

### Environment Variables
Essential configuration:
- `MDAI_TOKEN`: MD.ai API access token
- `DATA_DIR`: Path to PECARN ultrasound data directory
- `LABEL_ID_FREE_FLUID`: MD.ai label ID for free fluid annotations
- `LABEL_ID_NO_FLUID`: MD.ai label ID for clear frames
- `LABEL_ID_GROUND_TRUTH`: MD.ai label ID for verified annotations
- `PROJECT_ID`: MD.ai project identifier (default: x9N2LJBZ)
- `DATASET_ID`: MD.ai dataset identifier (default: D_V688LQ)

## Usage Guide

### 1. Initial Ground Truth Creation
Create verified dense annotations from expert sparse input using PECARN dataset:

```bash
# Create ground truth for pediatric FAST examination
python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186

# Test at optimal sampling rate (1:15)
python src/consolidated_tracking.py --create-ground-truth \
    --ground-truth-single-exam 186 \
    --input-sampling-rate 15
```

### 2. Clinical Sampling Rate Analysis
Test different sparse annotation strategies based on research findings:

```bash
# Test clinically acceptable range (1:5 to 1:20)
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 15 \
    --eval-sampling-rate 1 \
    --exam-id 186

# Compare across quality cliff (1:15 vs 1:20)
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 20 \
    --genuine-evaluation \
    --exam-id 186
```

### 3. Parameter Learning Investigation
Test whether adaptive parameters compensate for sparse input:

```bash
# Test parameter compensation hypothesis
python src/consolidated_tracking.py --feedback-loop \
    --learning-mode \ #learning mode enables parameter tuning across iterations
    --input-sampling-rate 15 \
    --iterations 4 \
    --exam-id 186

# Compare baseline vs learning mode
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 30 \
    --iterations 4 \
    --exam-id 186  #Test without learning mode
```

### 4. Fluid Complexity Analysis
Analyse performance by fluid presentation type:

```bash
# Test on multiple distinct fluid patterns (challenging)
python src/consolidated_tracking.py --feedback-loop \
    --issue multiple_distinct \
    --input-sampling-rate 15

# Test on uncomplicated presentations (optimal)
python src/consolidated_tracking.py --feedback-loop \
    --issue branching_fluid \
    --input-sampling-rate 15
```

## Key Parameters Explained

### Clinical Sampling Rates (Based on Research Findings)
- `--input-sampling-rate 5`: 20% density, good quality (IoU: 0.651), 79% time saving
- `--input-sampling-rate 15`: 6.7% density, acceptable quality (IoU: 0.615), 93.6% time saving
- `--input-sampling-rate 20`: 5.0% density, acceptable quality (IoU: 0.611), 95% time saving
- `--input-sampling-rate 30`: 3.3% density, poor quality (IoU: 0.487), not recommended
- `--input-sampling-rate 50`: 2.0% density, poor quality (IoU: 0.466), not recommended

### Clinical Translation Guidelines

| Annotation Density | Clinical Use Case | Time Investment | Quality Expectation |
|-------------------|------------------|-----------------|-------------------|
| 1:5 (20%) | High-precision research | 13.2 min/exam | Good (IoU: 0.651) |
| 1:15 (6.7%) | **Optimal clinical workflow** | 4.1 min/exam | Acceptable (IoU: 0.615) |
| 1:20 (5.0%) | **Alternative workflow** | 3.2 min/exam | Acceptable (IoU: 0.611) |
| 1:30+ (<3.3%) | Not recommended | <2 min/exam | Poor (IoU ≤ 0.487) |

### Algorithm-Specific Parameters
- `--learning-mode`: **NOT RECOMMENDED** for sparse input (counterproductive)
- `--genuine-evaluation`: Prevents data leakage, provides realistic metrics
- `--compare-methods`: Evaluates DIS vs other optical flow methods

## Research Applications

### Fluid Morphology Studies
```bash
# Study challenging cases (multiple distinct patterns)
python src/consolidated_tracking.py --create-ground-truth \
    --issue multiple_distinct \
    --input-sampling-rate 15

# Study optimal cases (uncomplicated presentations)
python src/consolidated_tracking.py --create-ground-truth \
    --issue branching_fluid \
    --input-sampling-rate 15

# Comprehensive morphology analysis
python src/consolidated_tracking.py --create-ground-truth \
    --all-issues \
    --input-sampling-rate 15 \
    --ground-truth-videos 20
```

### FAST Workflow Optimisation
```bash
# Emergency department simulation (minimal time)
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 15 \
    --eval-sampling-rate 1 \
    --iterations 1

# Research-grade validation
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 5 \
    --genuine-evaluation \
    --iterations 4
```

## Output and Results

### Performance Metrics
The system provides comprehensive evaluation based on computer vision standards:
- **IoU (Intersection over Union)**: Primary metric for boundary accuracy
- **Dice Coefficient**: Balanced overlap evaluation for segmentation quality
- **Clinical Threshold**: IoU > 0.7 considered clinically acceptable
- **Temporal Consistency**: Frame-to-frame tracking smoothness assessment

### Generated Outputs
1. **Dense Annotation Files**: Complete frame-by-frame binary masks
2. **Performance Reports**: IoU/Dice metrics across sampling rates and modes
3. **Tracking Visualizations**: Side-by-side comparison videos
4. **Quality Analysis**: Heatmaps showing performance by examination and sampling rate
5. **Parameter Logs**: Learning mode optimization history (when enabled)

### Research-Validated Results
Based on 466 experimental runs across 14 pediatric FAST examinations:

**Optimal Clinical Performance:**
- **1:15 sampling (6.7% input)**: Mean IoU 0.615, 93.6% time reduction
- **1:20 sampling (5.0% input)**: Mean IoU 0.611, 95% time reduction  
- **Clinical success rate**: 40.0% of uncomplicated cases achieve IoU > 0.7
- **Best case**: Exam 132, 1:5 sampling, IoU 0.975 (near-perfect)
- **Worst case**: Exam 97, 1:30 sampling, IoU 0.075 (failure mode)

**Morphology-Specific Performance:**
- **Complex mixed**: IoU 0.644 (30.0% clinical success rate)
- **Branching patterns**: IoU 0.611 (28.8% clinical success rate)  
- **Uncomplicated fluid**: IoU 0.609 (40.0% clinical success rate)
- **Multiple distinct**: IoU 0.440 (0.0% clinical success rate)
- **Disappear-reappear**: IoU 0.449 (0.0% clinical success rate)

**Parameter Learning Analysis:**
- **Overall impact**: 5.2% performance decrease across all conditions
- **Statistical significance**: Significant degradation at sparse sampling rates (p<0.01 for 1:20, 1:30, 1:50)
- **Clinical recommendation**: Use baseline mode; avoid learning mode

## Command Reference

### Essential Commands

```bash
# Quick quality test on single exam
python src/consolidated_tracking.py --create-ground-truth --ground-truth-single-exam 186 --debug

# Production sparse-to-dense pipeline
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 10 \
    --genuine-evaluation \
    --learning-mode \
    --upload

# Comprehensive method comparison
python src/consolidated_tracking.py --feedback-loop \
    --compare-methods \
    --input-sampling-rate 10 \
    --eval-sampling-rate 1 \
    --iterations 5
```

### Research-Specific Commands

```bash
# Annotation burden analysis
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 20 \
    --genuine-evaluation \
    --iterations 3

# Quality threshold determination
python src/consolidated_tracking.py --feedback-loop \
    --input-sampling-rate 5 \
    --eval-sampling-rate 1 \
    --learning-mode \
    --iterations 10

# Large-scale validation
python src/consolidated_tracking.py --create-ground-truth \
    --all-issues \
    --ground-truth-videos 100 \
    --input-sampling-rate 15
```

## Project Structure
```
goobusters/
├── src/
│   ├── consolidated_tracking.py           # Main sparse-to-dense pipeline
│   ├── multi_frame_tracking/              # Optical flow tracking modules
│   │   ├── multi_frame_tracker.py        # Core tracking algorithms
│   │   ├── opticalflowprocessor.py       # Flow computation
│   │   └── utils.py                       # Tracking utilities
│   ├── validation/                        # Quality assessment tools
│   │   └── sparse_validation.py          # Sparse annotation validation
│   └── utils/                             # General utilities
├── docs/                                  # Documentation and research notes
├── data/                                  # Ultrasound data (gitignored)
├── output/                                # Generated annotations and reports
└── tests/                                 # Test suite
```

## Research Impact

### Clinical Workflow Transformation
- **Before**: 64 minutes per exam for complete annotation
- **After**: 4.1 minutes per exam with acceptable quality (1:15 sampling, IoU: 0.615)
- **Time Savings**: 93.6% reduction in expert annotation burden
- **Quality Maintained**: Mean IoU 0.611-0.651 at practical sampling rates (1:15 to 1:5)
- **Statistical Validation**: 466 experiments across 14 examinations, 1114 total iterations

### Key Clinical Findings
- **Performance Plateau**: Stable performance between 1:15 (IoU: 0.615) and 1:20 (IoU: 0.611) sampling
- **Quality Threshold**: Sharp performance drop at 1:30 sampling (IoU: 0.487)
- **Morphology Dependency**: Algorithm performance varies dramatically by fluid complexity
- **Parameter Learning Ineffectiveness**: Statistically significant performance degradation (5.2% overall)
- **Clinical Success Rates**: Only uncomplicated cases achieve reliable IoU > 0.7 (40% success rate)

### AI Training Dataset Impact
For pediatric FAST AI model development:
- **Traditional approach**: 1067 hours for 1000-video dataset
- **Optimized approach**: 68 hours at 1:15 sampling (94% time reduction)
- **Quality assurance**: Maintains clinical viability for training data

## Limitations and Considerations

### Known Algorithm Limitations
1. **Speckle Noise Sensitivity**: DIS optical flow affected by ultrasound speckle artifacts
2. **Temporal Discontinuities**: Mask "jumping" between non-contiguous regions in complex cases
3. **Morphology Dependence**: Poor performance on multiple distinct fluid patterns
4. **Parameter Learning**: Counterproductive under extreme sparsity conditions

### Clinical Deployment Considerations
- **Video Length**: Validated on short-duration FAST videos (≤150 frames)
- **Institution Specificity**: Single-institution PECARN dataset
- **Pediatric Focus**: May not generalize to adult FAST examinations
- **Expert Validation**: Requires quality assurance review for clinical use

### Quality Assurance Requirements
- Moderate IoU reduction (6-16%) at optimal sampling rates
- Need for expert review of generated annotations
- Institution-specific validation recommended
- Adaptive algorithms required for different fluid complexities


## Acknowledgments

- Pediatric Emergency Care Applied Research Network (PECARN) for the data
- Dr. Aaron Kornblith, Newton Addo, and Christopher Seaman from UCSF.ß
- MD.ai platform for annotation management
- OpenCV community for Dense Inverse Search implementation