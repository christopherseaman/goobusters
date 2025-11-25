# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Goobusters is an optical flow tracking system for free fluid detection in ultrasound videos. The name comes from tracking ("busting") free fluid ("goo") in medical imaging. Free fluid is particularly challenging to track because it often appears as featureless dark space in ultrasounds.

**Key Challenge**: Tracking amorphous, featureless regions across frames where traditional feature-based tracking fails.

**Approach**: Dense optical flow algorithms (Farneback, DIS, RAFT) applied to mask propagation with bidirectional temporal weighting.

**Current Work**: See `TODO.md` for active issues and improvements. When encountering unused or overly complex code, simplify/remove it, but ALWAYS validate that changes don't break `track.py` by running it on real videos.

## Core Development Principles

### Configuration-Driven Development
- **NEVER hardcode variables** - use `dot.env` for all parameters
- Store thresholds, settings, and constants in configuration files
- Scripts read from config, not embedded data definitions
- Magic numbers become named configuration values

### DRY (Don't Repeat Yourself)
- **Single source of truth** for each piece of knowledge
- **Reuse shared functions** from `lib/` modules before writing new code
- Extract common operations into centralized utilities
- Never duplicate functionality across scripts
- Organize shared code logically in `lib/` directory

### Separation of Concerns
- **One responsibility per module** - focused, clear purpose
- Keep business logic separate from orchestration
- Maintain distinct layers: data processing, analysis, presentation
- Coordinators orchestrate; they don't perform business logic

### Simplicity Over Complexity (KISS)
- **Clear, concise code** beats defensive programming
- Choose simplest solution that works
- Avoid unnecessary abstraction layers
- Prefer explicit over implicit behavior
- Minimize boilerplate code
- **As simple as possible, and no simpler**

### Continuous Simplification
- **Remove complexity as encountered** during normal work
- Refactor overly complex code when touching it
- Delete unused code, vestigial variables, dead branches
- Question whether each layer of abstraction earns its keep
- **Always validate** simplifications don't break `track.py`

### Test Before Success
- **NEVER declare completion without validation**
- Run actual code in target environment with real data
- Avoid mocks or simulated outputs for validation
- User confirmation required for output quality
- This is non-negotiable

## Critical Testing Rules

### NO MOCKS OR FAKE FUNCTIONS

**This cannot be emphasized enough**: NEVER utilize mocks or fake functions. ALWAYS test with real data and run the actual `track.py` on at least a single video. Validate the outputs and check with user for output quality.

### Validation Protocol

1. **Run with real data**: Execute `track.py` on actual videos
2. **Inspect outputs**: Check generated masks, videos, metadata
3. **Validate quality**: Review tracking accuracy visually
4. **User confirmation**: Get explicit approval before declaring success

### Pre-Commit Checklist

- [ ] Code runs without errors on real video
- [ ] Test suite passes: `uv run python3 test_implementation.py`
- [ ] Output quality validated visually
- [ ] User has confirmed results are acceptable
- [ ] No hardcoded values introduced
- [ ] No code duplication introduced

## Core Commands

### Running Tracking

```bash
# Main entrypoint - Run tracking on videos
uv run python3 track.py

# Debug mode - Process only 5 random videos
DEBUG=True uv run python3 track.py

# Test specific video (set TEST_STUDY_UID and TEST_SERIES_UID in dot.env first)
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

**Note**: The implementation may contain over-engineering (e.g., temporal weighting AND temporal smoothing, vestigial variables). When working in this code, simplify complexity that doesn't provide proven value, but always validate changes don't break tracking.

## Reference Implementation

The `references/shreyasreeram/` directory contains a student implementation with known issues:
- **Delayed change tracking**: Masks lag behind actual fluid movement
- **Over-complex code**: Multiple abstraction layers that obscure logic
- **Bugs in multi-frame tracking**: Temporal consistency issues

**Do NOT copy patterns from reference implementation**. Use as cautionary examples only.

## Code Quality Standards

### Naming Conventions
- Use descriptive names: `calculate_optical_flow()`, `save_tracking_results()`
- Avoid generic names: `process()`, `do_stuff()`, `handle()`
- Variables: `video_annotations`, `study_instance_uid`
- Not: `data`, `stuff`, `temp`

### Commenting Strategy
- Focus on **context, purpose, and interpretation** when logic isn't self-evident
- Never reference previous implementations or changes
- Describe current state only, avoiding change logs
- No "Fixed bug", "Updated logic", "Replaced X with Y"

### Error Handling
- Implement simple try/except with clear error messages
- Avoid complex error hierarchies
- Log failures loudly, successes quietly

### External Mappings
- Use lookup tables and dictionaries
- No hardcoded mappings or transformations

## Development Workflow

### Before Making Changes

1. **Read the code**: Use `Read` tool to examine files before modification
2. **Understand the architecture**: Review this file and `multiple_annotation_strategy.md`
3. **Check configuration**: Ensure parameters are in `dot.env`, not hardcoded
4. **Check existing utilities**: Review `lib/` for reusable functions

### When Making Changes

1. **Maintain simplicity**: Remove complexity, don't add it
2. **Reuse functions**: Check `lib/` for existing functionality before writing new
3. **Configuration driven**: All magic numbers must become config values
4. **No optimizations without measurement**: Profile before optimizing
5. **Remove as you go**: Delete unused code, simplify complex sections
6. **Preserve functionality**: Keep changes focused, avoid side effects

### After Making Changes

1. **Test with real data**: Run `track.py` on at least one video
2. **Validate output**: Check generated masks and videos for quality
3. **User confirmation**: Show results to user for quality assessment
4. **Never declare success without testing**: This is absolutely non-negotiable

## Session Work Tracking

### Purpose of WORK_SUMMARY.md

Track high-level work progress during development sessions between major commits and functionality releases. This provides a running log of what's being worked on, decisions made, and incremental progress.

### When to Log

- **Start of session**: Note what you're working on from TODO.md
- **After significant changes**: Document what was modified and why
- **Before major commits**: Summarize completed work
- **When switching tasks**: Record current state before pivoting
- **End of session**: Final summary of progress and next steps

### What to Log

Keep entries high-level and concise:

```markdown
## [Date] - [Brief Description]

**Working On**: [TODO.md item or task description]

**Changes Made**:
- [File/module]: [What changed and why]
- [File/module]: [What changed and why]

**Testing**: [Results from track.py validation]

**Status**: [Complete/In Progress/Blocked]

**Next Steps**: [What needs to happen next]
```

### Example Entry

```markdown
## 2025-01-15 - Simplify Multi-Frame Tracker

**Working On**: TODO.md #1 - Remove complexity from multi_frame_tracker.py

**Changes Made**:
- lib/multi_frame_tracker.py: Removed redundant temporal_smoothing (already have temporal_weighting)
- lib/multi_frame_tracker.py: Deleted unused SharedParams fields (window_size, distance_decay_factor)

**Testing**: Ran track.py on test video 1.2.840.113... - tracking quality unchanged, masks identical

**Status**: In Progress

**Next Steps**: Review opticalflowprocessor.py for similar complexity
```

### Guidelines

- **Be factual**: Document what happened, not opinions
- **Be concise**: High-level only, not line-by-line changes
- **Link to validation**: Reference test results
- **Note blockers**: If stuck, document why
- **Don't duplicate git**: This is working notes, not commit messages

## Environment & Dependency Management

### Package Management
- Use `uv` for dependency management
- Prefer direct execution: `uv run python3 track.py`
- Keep dependencies minimal and well-documented
- Use inline dependency definitions for one-off scripts when supported

### Working Directory Practices
- Maintain consistent working directory (project root)
- Use relative paths consistently
- Avoid changing directories mid-script
- Organize temporary files in dedicated subdirectories (`tmp/`, `temp/`)
- Create temporary test scripts instead of complex one-liners

### Sensible Defaults
- Scripts should run without arguments when possible
- Use conventional paths (`data/`, `config/`, `output/`)
- Allow overrides via command line or environment variables

## Long-Running Script Protocol

### Duration Guidelines
- Scripts under 30 seconds: run directly with normal output
- Scripts over 30 seconds: implement logging and completion hooks

### Implementation for Extended Processes
- Redirect output to execution logs (stdout and stderr)
- Add completion markers with timestamps
- Monitor with `tail -f` or grep for completion/error patterns
- Verify script status before proceeding

## Git Commit Standards

### Required Format
- Use conventional commit format: `feat:`, `fix:`, `docs:`, `refactor:`
- Focus on technical changes and measurable impact
- Maintain professional, technical tone
- Keep subject line under 72 characters

### Prohibited Practices
- No "Co-Authored-By: Claude" or similar attribution
- No "via Happy" or credit references
- No casual or cutesy language
- Focus on what changed, not who changed it

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

### Simplifying Complex Code

When encountering overly complex code:
1. Understand what it does (read thoroughly)
2. Identify unnecessary abstractions or steps
3. Simplify while preserving functionality
4. **Test immediately** on real video to validate
5. Document what was simplified and why
6. Get user confirmation on results

### Understanding Optical Flow Methods

- **Farneback**: Fast, CPU-based, good baseline (OpenCV)
- **DIS**: Dense inverse search, faster than Farneback (OpenCV)
- **DeepFlow**: Slower, being removed from main pipeline (TODO.md #1)
- **RAFT**: Deep learning, most accurate but slowest (PyTorch)

Configure via `FLOW_METHOD` in `dot.env` (comma-separated for multiple methods).

## MD.ai Integration

### Annotation Format

MD.ai annotations have:
- `StudyInstanceUID`: Identifies patient study
- `SeriesInstanceUID`: Identifies video sequence
- `frameNumber`: 0-indexed frame number
- `labelId`: Matches `LABEL_ID` (fluid) or `EMPTY_ID` (clear)
- `data.foreground`: Polygon coordinates for mask

## Performance Considerations

The `lib/performance_config.py` module handles:
- Automatic hardware detection (Apple Silicon, Intel, NVIDIA)
- OpenCV thread optimization
- PyTorch device selection (CPU/MPS/CUDA)
- Memory management based on available RAM

Profile before making performance claims. Measure, don't assume.

## Anti-Patterns to Avoid

### Code Smells
- ❌ Hardcoding parameters in code
- ❌ Using mocks or fake data in tests
- ❌ Declaring success without running real tests
- ❌ Adding complexity without proven value
- ❌ Duplicating code instead of creating shared functions
- ❌ Changing working directory mid-script
- ❌ Testing on cached/old output
- ❌ Magic numbers instead of named constants
- ❌ Functions over 50 lines (consider breaking down)
- ❌ Deep nesting (more than 3 levels)
- ❌ Silent failures (catching exceptions without handling)

### Testing Anti-Patterns
- ❌ Testing old output instead of fresh generation
- ❌ Declaring success without validation
- ❌ Ignoring test failures
- ❌ Relying only on automated tests (need visual validation too)

### Performance Anti-Patterns
- ❌ Premature optimization (optimize before measuring)
- ❌ Optimization without testing (adding complexity without validation)
- ❌ Ignoring overhead (not measuring cost of "optimizations")

## Output & Communication Standards

### Quality Requirements
- Present factual, measurable information and numerical results
- Avoid subjective interpretation; stick to observable outcomes
- Use neutral, objective language when describing patterns
- Clearly document methodology, tests, and assumptions
- Limit conclusions to what data directly supports
- Maintain standardized formatting and terminology

### What to Include
- What's being tested
- Why it matters
- What results mean
- Methodology used
- Underlying assumptions

## Key Files Reference

- `track.py`: Main entrypoint (196 lines)
- `lib/multi_frame_tracker.py`: Core tracking logic (993 lines)
- `lib/opticalflowprocessor.py`: Flow algorithm wrapper (155 lines)
- `lib/optical.py`: MD.ai utilities (1999 lines)
- `lib/performance_config.py`: Hardware optimization (476 lines)
- `dot.env`: Configuration (all runtime parameters)
- **`TODO.md`**: Current work items and known issues
- **`WORK_SUMMARY.md`**: Session work log (high-level progress tracking)
- `multiple_annotation_strategy.md`: Bidirectional tracking specification
- `.cursor/rules/goobusters.mdc`: Detailed development rules

## Questions or Uncertainties?

When in doubt:
1. Check `TODO.md` for known issues and current work
2. Check `multiple_annotation_strategy.md` for tracking behavior
3. Check `.cursor/rules/goobusters.mdc` for development practices
4. Ask user for clarification rather than making assumptions
5. Test with real data to validate behavior
6. Review reference implementation as anti-pattern examples
