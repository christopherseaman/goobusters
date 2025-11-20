# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goobusters is an optical flow tracking system for free fluid detection in ultrasound videos. The name comes from tracking ("busting") free fluid ("goo") in medical imaging. Free fluid is particularly challenging to track because it often appears as featureless dark space in ultrasounds.

**Key Challenge**: Tracking amorphous, featureless regions across frames where traditional feature-based tracking fails.

**Approach**: Dense optical flow algorithms (Farneback, DIS, RAFT) applied to mask propagation with bidirectional temporal weighting.

## Development Philosophy

This project strictly adheres to the principles in `.cursor/rules/goobusters.mdc`:
- **KISS/DRY**: Simplicity and code reuse are paramount
- **Configuration > Hardcoding**: All parameters in `dot.env`, never hardcoded
- **Test Before Success**: ALWAYS run `track.py` on actual videos to validate changes
- **NO MOCKS**: Only test with real data and actual execution
- **User Validation Required**: Check with user for output quality before declaring success

## Core Commands

### Running Tracking

```bash
# Main entrypoint - Run tracking on videos
uv run python3 track.py

# Debug mode - Process only 5 random videos
DEBUG=True uv run python3 track.py

# Test specific video (specify in dot.env)
# Set TEST_STUDY_UID and TEST_SERIES_UID in dot.env first
uv run python3 track.py
```

### Testing

```bash
# Validation test suite (must pass 5/5 tests)
uv run python3 test_implementation.py
```

### Configuration

All configuration is in `dot.env` (copy from `dot.env.example`). Critical variables:
- `MDAI_TOKEN`: MD.ai API access token
- `PROJECT_ID`, `DATASET_ID`: MD.ai project identifiers
- `LABEL_ID`: Free fluid label identifier
- `EMPTY_ID`: No-fluid frame label identifier
- `FLOW_METHOD`: Comma-separated optical flow methods (e.g., "farneback,dis,raft")
- `DEBUG`: Set to "True" to process subset of videos
- `TEST_STUDY_UID`, `TEST_SERIES_UID`: Specific video to debug

## Architecture

### Data Flow

```
MD.ai Annotations → track.py → MultiFrameTracker → OpticalFlowProcessor → Output
                                      ↓
                              Video + Annotations
                                      ↓
                         Bidirectional Tracking Strategy
                                      ↓
                    Forward + Backward + Temporal Weighting
                                      ↓
                         Per-frame masks + metadata
```

### Key Components

#### 1. **track.py** (Entrypoint)
- Loads configuration from `dot.env`
- Downloads/caches MD.ai dataset
- Groups annotations by video
- Orchestrates multi-method optical flow tracking
- Creates output directories per method and video

#### 2. **lib/multi_frame_tracker.py** (Core Tracking Logic)
- `MultiFrameTracker`: Main tracking class
- Implements bidirectional tracking strategy from `multiple_annotation_strategy.md`
- Handles four annotation scenarios:
  - **F→F**: Bidirectional tracking with temporal weighting
  - **F→C**: Forward-only tracking (fluid to clear)
  - **C→F**: Backward-only tracking (clear to fluid)
  - **C→C**: No tracking (maintain clear state)
- `SharedParams`: Adaptive parameter management
- Processes video segments between annotations

#### 3. **lib/opticalflowprocessor.py** (Flow Algorithms)
- `OpticalFlowProcessor`: Unified interface for multiple flow methods
- Supports: Farneback (OpenCV), DIS (OpenCV), DeepFlow (OpenCV contrib), RAFT (PyTorch)
- `calculate_flow()`: Main method to compute optical flow
- Device management (CPU/MPS/CUDA) via `performance_config.py`

#### 4. **lib/optical.py** (Utilities)
- `create_identity_file()`: Generate YAML metadata for each video
- `copy_annotations_to_output()`: Save input annotations (excluding track_id)
- Helper functions for MD.ai integration

#### 5. **lib/performance_config.py** (Hardware Optimization)
- `PerformanceOptimizer`: Auto-detects hardware capabilities
- Configures OpenCV, PyTorch, and multiprocessing settings
- Apple Silicon (MPS) acceleration support
- Memory and thread management scaled to available resources

### Output Structure

Each video/method combination produces:

```
output/{method}/{study_uid}_{series_uid}/
├── identity.yaml                    # Video metadata
├── input_annotations.json           # Original annotations (no track_id)
├── tracked_annotations.json         # Generated annotations (with track_id)
├── mask_data.json                   # Frame-by-frame metadata
├── tracked_video.mp4                # Visualization (green=annotation, orange=tracked)
└── masks/                           # Individual frame masks
    ├── frame_000001_mask.png
    └── ...
```

## Bidirectional Tracking Strategy

See `multiple_annotation_strategy.md` for complete specification.

### Key Concepts

- **Annotations**: Human-verified masks at specific frames
  - `LABEL_ID`: Frames with free fluid
  - `EMPTY_ID`: Frames explicitly marked as clear
- **Predictions**: Tracked masks generated between annotations
- **Temporal Weighting**: Combine forward/backward predictions based on distance from annotations
  - Weight = distance from opposite annotation / total distance
  - Closer to annotation A → higher weight from A→B tracking

### Implementation Notes

The current implementation in `lib/multi_frame_tracker.py`:
1. Classifies annotations as 'fluid' or 'empty' based on label IDs
2. Identifies segments between consecutive annotations
3. Determines tracking strategy per segment (bidirectional/forward/backward/none)
4. Applies optical flow in appropriate direction(s)
5. Combines predictions using distance-weighted averaging
6. Applies morphological operations to clean masks

**Known Complexity Issues** (per TODO.md #1):
- Possible over-engineering with both temporal weighting AND temporal smoothing
- Vestigial variables from reference implementation
- Need to validate if all complexity provides measurable value

## Reference Implementation Notes

The `references/shreyasreeram/` directory contains a student implementation with known issues:
- **Delayed change tracking**: Masks lag behind actual fluid movement
- **Over-complex code**: Multiple abstraction layers that obscure logic
- **Bugs in multi-frame tracking**: Temporal consistency issues

**Do NOT copy patterns from reference implementation**. Use as cautionary examples only. The main implementation attempts to address these issues but may still contain over-engineered solutions.

## Critical Development Rules

### Before Making Changes

1. **Read the code**: Use `Read` tool to examine files before modification
2. **Understand the architecture**: Review this file and `multiple_annotation_strategy.md`
3. **Check configuration**: Ensure parameters are in `dot.env`, not hardcoded

### When Making Changes

1. **Maintain simplicity**: Remove complexity, don't add it
2. **Reuse functions**: Check `lib/` for existing functionality
3. **Configuration driven**: All magic numbers must become config values
4. **No optimizations without measurement**: Profile before optimizing

### After Making Changes

1. **Test with real data**: Run `track.py` on at least one video
2. **Validate output**: Check generated masks and videos for quality
3. **User confirmation**: Show results to user for quality assessment
4. **Never declare success without testing**: This is non-negotiable

### Specific Constraints

- **NEVER use mocks or fake functions**: All testing must use real data
- **NEVER hardcode values**: Use `dot.env` or function parameters
- **NEVER skip validation**: Always run on actual videos
- **ALWAYS check with user**: Get explicit confirmation on output quality

## Common Tasks

### Debugging Tracking Issues

1. Set `TEST_STUDY_UID` and `TEST_SERIES_UID` in `dot.env` for specific video
2. Run `track.py` to process that video only
3. Examine output in `output/{method}/{study_uid}_{series_uid}/`
4. Check `tracked_video.mp4` for visual quality
5. Review `mask_data.json` for per-frame metadata

### Testing Changes

```bash
# Always test on real video after code changes
uv run python3 track.py

# Run validation suite
uv run python3 test_implementation.py
```

### Understanding Optical Flow Methods

- **Farneback**: Fast, CPU-based, good baseline (OpenCV)
- **DIS**: Dense inverse search, faster than Farneback (OpenCV)
- **DeepFlow**: Slower, removed from main pipeline per TODO.md #1
- **RAFT**: Deep learning, most accurate but slowest (PyTorch)

Configure via `FLOW_METHOD` in `dot.env` (comma-separated for multiple methods).

## MD.ai Integration

### Current Implementation (TODO.md #2 - NEEDS FIX)

Track.py currently downloads entire project:
```python
project = mdai_client.project(project_id=PROJECT_ID, path=DATA_DIR)
dataset = project.get_dataset_by_id(DATASET_ID)
```

**Should be**:
```python
# Only pull PROJECT_ID and DATASET_ID from dot.env
mdai_client.project(project_id=PROJECT_ID, dataset_id=DATASET_ID, path=DATA_DIR)
```

### Annotation Format

MD.ai annotations have:
- `StudyInstanceUID`: Identifies patient study
- `SeriesInstanceUID`: Identifies video sequence
- `frameNumber`: 0-indexed frame number
- `labelId`: Matches `LABEL_ID` (fluid) or `EMPTY_ID` (clear)
- `data.foreground`: Polygon coordinates for mask

## Performance Considerations (TODO.md #5)

The `lib/performance_config.py` module handles:
- Automatic hardware detection (Apple Silicon, Intel, NVIDIA)
- OpenCV thread optimization
- PyTorch device selection (CPU/MPS/CUDA)
- Memory management based on available RAM

**Open question**: Are current settings optimal? Profile to validate.

## Known Issues & TODO

See `TODO.md` for full list. Critical items:

1. **Code cleanup**: Remove unused/overcomplexed bits (e.g., deepflow, temporal smoothing vs weighting redundancy)
2. **MD.ai integration**: Fix to only pull PROJECT_ID/DATASET_ID
3. **Empty frame annotations**: Properly handle EMPTY_ID in bidirectional tracking
4. **Jerky tracking**: Frame-to-frame jumps need investigation
5. **Performance tuning**: CPU vs GPU acceleration, parallelization

## Development Workflow

1. Make changes following KISS/DRY principles
2. Update `dot.env` if new configuration needed
3. Test on single video: `TEST_STUDY_UID=... TEST_SERIES_UID=... uv run python3 track.py`
4. Validate output quality visually and with user
5. Run test suite: `uv run python3 test_implementation.py`
6. Only after explicit user approval, declare success

## Anti-Patterns to Avoid

Based on `.cursor/rules/goobusters.mdc`:

- ❌ Hardcoding parameters in code
- ❌ Using mocks or fake data in tests
- ❌ Declaring success without running real tests
- ❌ Adding complexity without proven value
- ❌ Duplicating code instead of creating shared functions
- ❌ Changing working directory mid-script
- ❌ Testing on cached/old output

## Key Files Reference

- `track.py`: Main entrypoint (151 lines)
- `lib/multi_frame_tracker.py`: Core tracking logic (993 lines)
- `lib/opticalflowprocessor.py`: Flow algorithm wrapper (155 lines)
- `lib/optical.py`: MD.ai utilities (1999 lines, includes helper classes)
- `lib/performance_config.py`: Hardware optimization (476 lines)
- `dot.env`: Configuration (all runtime parameters)
- `TODO.md`: Current work items
- `multiple_annotation_strategy.md`: Bidirectional tracking specification
- `.cursor/rules/goobusters.mdc`: Development rules and best practices

## Questions or Uncertainties?

When in doubt:
1. Check `multiple_annotation_strategy.md` for tracking behavior
2. Check `.cursor/rules/goobusters.mdc` for development practices
3. Ask user for clarification rather than making assumptions
4. Test with real data to validate behavior
5. Review reference implementation as anti-pattern examples
