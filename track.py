#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "opencv-contrib-python",  # IMPORTANT: Only contrib, not regular opencv-python
#     "numpy>=1.21.0,<3.0.0",  # Support both NumPy 1.x and 2.x
#     "pandas",
#     "python-dotenv",
#     "mdai==0.16.0",
#     "pydicom>=3.0.0",
#     "tqdm",
#     "scikit-image",
#     "scipy",
#     "pillow",
#     "pyyaml",
#     "torch>=2.0.0",  # Use newer PyTorch that supports NumPy 2.x
#     "torchvision>=0.15.0",  # Compatible with newer PyTorch
#     "psutil"  # For performance monitoring
# ]
# ///

"""
Goobusters Multi-Frame Optical Flow Tracker

Entrypoint for Goobusters tracking tools.
Loads settings from dot.env and performs tracking inline.
"""

import os
import sys
from dotenv import load_dotenv
import mdai
import pandas as pd
from tqdm import tqdm

# Import from lib modules
from lib.optical import create_identity_file, copy_annotations_to_output
from lib.multi_frame_tracker import process_video_with_multi_frame_tracking
from lib.opticalflowprocessor import OpticalFlowProcessor


def find_annotations_file(data_dir: str, project_id: str, dataset_id: str) -> str:
    """
    Find the most recent annotations JSON file for the given project and dataset.

    Args:
        data_dir: Directory containing MD.ai data
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID

    Returns:
        Path to the annotations JSON file

    Raises:
        FileNotFoundError: If no matching annotations file is found
    """
    import glob

    # Pattern: mdai_{domain}_project_{project_id}_annotations_dataset_{dataset_id}_*.json
    pattern = os.path.join(
        data_dir, f"mdai_*_project_{project_id}_annotations_dataset_{dataset_id}_*.json"
    )
    matches = glob.glob(pattern)

    if not matches:
        raise FileNotFoundError(
            f"No annotations file found: {project_id}, dataset {dataset_id} in {data_dir}. "
            f"Expected pattern: mdai_*_project_{project_id}_annotations_dataset_{dataset_id}_*.json"
        )

    # Return the most recent file (sorted by filename which includes timestamp)
    return sorted(matches)[-1]


def find_images_dir(data_dir: str, project_id: str, dataset_id: str) -> str:
    """
    Find the most recent images directory for the given project and dataset.

    Args:
        data_dir: Directory containing MD.ai data
        project_id: MD.ai project ID
        dataset_id: MD.ai dataset ID

    Returns:
        Path to the images directory

    Raises:
        FileNotFoundError: If no matching images directory is found
    """
    import glob

    # Pattern: mdai_{domain}_project_{project_id}_images_dataset_{dataset_id}_*
    pattern = os.path.join(
        data_dir, f"mdai_*_project_{project_id}_images_dataset_{dataset_id}_*"
    )
    matches = [d for d in glob.glob(pattern) if os.path.isdir(d)]

    if not matches:
        raise FileNotFoundError(
            f"No images directory found: {project_id}, dataset {dataset_id} in {data_dir}. "
            f"Expected pattern: mdai_*_project_{project_id}_images_dataset_{dataset_id}_*"
        )

    # Return the most recent directory (sorted by dirname which includes timestamp)
    return sorted(matches)[-1]


def main():
    """Main execution function for the tracking pipeline."""
    load_dotenv("dot.env")

    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

    # Load optical flow methods from environment
    FLOW_METHOD = [
        method.strip() for method in os.getenv("FLOW_METHOD", "farneback").split(",")
    ]

    ACCESS_TOKEN = os.getenv("MDAI_TOKEN")
    DATA_DIR = os.getenv("DATA_DIR")
    DOMAIN = os.getenv("DOMAIN")
    PROJECT_ID = os.getenv("PROJECT_ID")
    DATASET_ID = os.getenv("DATASET_ID")
    LABEL_ID = os.getenv("LABEL_ID")

    if ACCESS_TOKEN is None:
        print("ACCESS_TOKEN is not set, please set ACCESS_TOKEN in dot.env")
        sys.exit(1)

    # Start MD.ai client (skip connection test if we have cached data)
    try:
        mdai_client = mdai.Client(domain=DOMAIN, access_token=ACCESS_TOKEN)
    except Exception as e:
        print(f"Warning: Could not connect to MD.ai ({e})")
        print("Attempting to use cached data...")
        mdai_client = None

    # Download the dataset from MD.ai (or use cached version)
    if mdai_client:
        project = mdai_client.project(
            project_id=PROJECT_ID, dataset_id=DATASET_ID, path=DATA_DIR
        )
        BASE = project.images_dir
    else:
        # Use cached data - auto-detect images directory
        BASE = find_images_dir(DATA_DIR, PROJECT_ID, DATASET_ID)

    # After download, automatically find the annotations file
    ANNOTATIONS = find_annotations_file(DATA_DIR, PROJECT_ID, DATASET_ID)
    print(f"Using annotations file: {os.path.basename(ANNOTATIONS)}")

    # Load the annotations
    annotations_data = mdai.common_utils.json_to_dataframe(ANNOTATIONS)
    annotations_df = pd.DataFrame(annotations_data["annotations"])
    labels = annotations_df["labelId"].unique()

    # Create the label map, LABEL_ID => 1, others in labels => 0
    labels_dict = {LABEL_ID: 1}
    if mdai_client:
        project.set_labels_dict(labels_dict)

    # Get EMPTY_ID from environment
    EMPTY_ID = os.getenv("EMPTY_ID", "")

    # Filter annotations for the free fluid label AND empty frames (both needed for tracking)
    free_fluid_annotations = annotations_df[
        (annotations_df["labelId"] == LABEL_ID)
        | (annotations_df["labelId"] == EMPTY_ID)
    ].copy()

    # Function to construct the video path
    def construct_video_path(base_dir, study_uid, series_uid):
        return os.path.join(base_dir, study_uid, f"{series_uid}.mp4")

    # Add video paths to the dataframe
    free_fluid_annotations["video_path"] = free_fluid_annotations.apply(
        lambda row: construct_video_path(
            BASE, row["StudyInstanceUID"], row["SeriesInstanceUID"]
        ),
        axis=1,
    )

    # Check if video files exist
    free_fluid_annotations["file_exists"] = free_fluid_annotations["video_path"].apply(
        os.path.exists
    )

    # Count the number of annotations with and without corresponding video files
    num_with_files = free_fluid_annotations["file_exists"].sum()
    num_without_files = len(free_fluid_annotations) - num_with_files

    # Select annotations for processing
    TEST_STUDY_UID = os.getenv("TEST_STUDY_UID")
    TEST_SERIES_UID = os.getenv("TEST_SERIES_UID")

    if TEST_STUDY_UID and TEST_SERIES_UID:
        # Use specific test study if provided
        matched_annotations = free_fluid_annotations[
            (free_fluid_annotations["StudyInstanceUID"] == TEST_STUDY_UID)
            & (free_fluid_annotations["SeriesInstanceUID"] == TEST_SERIES_UID)
            & (free_fluid_annotations["file_exists"])
        ]
    elif DEBUG:
        matched_annotations = free_fluid_annotations[
            free_fluid_annotations["file_exists"]
        ].sample(n=5, random_state=42)
    else:
        matched_annotations = free_fluid_annotations[
            free_fluid_annotations["file_exists"]
        ]

    # Group annotations by video for multi-frame processing
    video_groups = matched_annotations.groupby(
        ["StudyInstanceUID", "SeriesInstanceUID"]
    )

    # Main processing loop using true multi-frame tracking
    total_videos = len(video_groups)
    total_methods = len(FLOW_METHOD)
    print(
        f"\nFound {total_videos} videos with {total_methods} optical flow method(s): {', '.join(FLOW_METHOD)}"
    )

    for method in FLOW_METHOD:
        print(f"\n{'=' * 60}")
        print(f"Running {method} optical flow method")
        print(f"{'=' * 60}")
        output_base_dir = os.path.join("output", method)
        os.makedirs(output_base_dir, exist_ok=True)

        # Process each video with all its annotations
        video_count = 0
        total_videos = len(video_groups)
        for (study_uid, series_uid), video_annotations in tqdm(
            video_groups, desc=f"{method}", position=0, leave=True
        ):
            video_count += 1
            try:
                # Get video path (should be the same for all annotations in this group)
                video_path = video_annotations.iloc[0]["video_path"]

                # Create output directory for this video
                video_output_dir = os.path.join(
                    output_base_dir, f"{study_uid}_{series_uid}"
                )
                os.makedirs(video_output_dir, exist_ok=True)

                # Generate identity file for this video output folder
                create_identity_file(
                    video_output_dir,
                    study_uid,
                    series_uid,
                    video_annotations,
                    annotations_data["studies"],
                )

                # Copy original annotations JSON to video output directory
                copy_annotations_to_output(
                    video_output_dir, video_annotations, annotations_data
                )

                # Initialize optical flow processor
                flow_processor = OpticalFlowProcessor(method)

                # Process the video with multi-frame tracking
                result = process_video_with_multi_frame_tracking(
                    video_path=video_path,
                    annotations_df=video_annotations,
                    study_uid=study_uid,
                    series_uid=series_uid,
                    flow_processor=flow_processor,
                    output_dir=video_output_dir,
                    mdai_client=mdai_client,
                    label_id_fluid=LABEL_ID,
                    label_id_machine=os.getenv("LABEL_ID_MACHINE_GROUP", "G_RJY6Qn"),
                    upload_to_mdai=False,  # Set to True if you want to upload results
                    project_id=PROJECT_ID,
                    dataset_id=DATASET_ID,
                )

                # Clean up GPU memory after processing
                flow_processor.cleanup_memory()

            except Exception as e:
                print(
                    f"Error processing video {study_uid}/{series_uid} with {method}: {str(e)}"
                )
                import traceback

                traceback.print_exc()
                # Clean up GPU memory even on error
                flow_processor.cleanup_memory()
                continue

    print("\n" + "=" * 60)
    print("✅ All optical flow methods completed successfully")
    print("=" * 60)

    # Ensure clean exit
    sys.stdout.flush()
    sys.stderr.flush()


if __name__ == "__main__":
    main()

