#!/usr/bin/env python3
import os
import sys
import numpy as np
import cv2
import mdai
from dotenv import load_dotenv
import traceback
import logging
import time
import subprocess
import glob
import argparse
import shutil

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum information
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("save_masks.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Set environment variables for debugging
os.environ['DEBUG_UPLOADS'] = '1'
os.environ['DEBUG_TRACKING'] = '1'
os.environ['DEBUG'] = '1'

# Load environment variables
load_dotenv('.env')

# Exam #64 specific paths
VIDEO_PATH = 'data/mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-05-19-171632/1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661/1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392.mp4'
STUDY_UID = '1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661'
SERIES_UID = '1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392'

def print_header(message):
    """Print a nicely formatted header"""
    logging.info("=" * 70)
    logging.info(message)
    logging.info("=" * 70)

def create_masks_directory(video_name, exam_id="64"):
    """Create a directory to store masks"""
    # Get timestamp for unique directory name
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Use the specified exam ID or extract from environment
    exam_id_from_env = os.environ.get('EXAM_ID', exam_id)
    if not exam_id_from_env:
        # Extract the exam ID from the video path if possible
        if SERIES_UID in video_name:
            exam_id_from_env = f"{SERIES_UID[-10:]}"
        else:
            exam_id_from_env = "unknown"
    
    # Create masks directory
    masks_dir = os.getenv('SAVE_MASKS_DIR')
    if not masks_dir:
        masks_dir = os.path.join("output", f"masks_exam_{exam_id_from_env}_{timestamp}")
    
    os.makedirs(masks_dir, exist_ok=True)
    logging.info(f"Created masks directory: {masks_dir}")
    
    # Create metadata file with video path and UIDs
    with open(os.path.join(masks_dir, "metadata.txt"), "w") as f:
        f.write(f"Video path: {video_name}\n")
        f.write(f"Study UID: {STUDY_UID}\n")
        f.write(f"Series UID: {SERIES_UID}\n")
        f.write(f"Exam ID: {exam_id_from_env}\n")  
        f.write(f"Generated: {timestamp}\n")
    
    return masks_dir

def extract_masks_from_feedback_loop(masks_dir, video_path=VIDEO_PATH):
    """Extract masks from feedback loop output directories"""
    print_header("EXTRACTING MASKS FROM FEEDBACK LOOP OUTPUT")
    
    # Find the most recent feedback loop output directory
    feedback_loop_dirs = sorted(glob.glob("output/feedback_loop_*"), reverse=True)
    
    if not feedback_loop_dirs:
        logging.error("No feedback loop output directories found")
        return False
    
    latest_feedback_dir = feedback_loop_dirs[0]
    logging.info(f"Using latest feedback loop directory: {latest_feedback_dir}")
    
    # Find the exam directory for the target video
    exam_number = "64"  # Hardcoded for this specific exam 
    exam_dirs = glob.glob(f"{latest_feedback_dir}/iteration_*/exam_{exam_number}*")
    
    if not exam_dirs:
        logging.error(f"No exam directory found for exam #{exam_number}")
        return False
    
    exam_dir = exam_dirs[0]
    logging.info(f"Found exam directory: {exam_dir}")
    
    # Look for mask images in various subdirectories
    mask_files = []
    
    # Search patterns to find masks
    search_patterns = [
        f"{exam_dir}/**/mask_*.png",
        f"{exam_dir}/**/binary_mask_*.png",
        f"{exam_dir}/**/predicted_mask_*.png",
        f"{exam_dir}/debug/**/mask_*.png",
        f"{exam_dir}/**/masks/mask_*.png",
    ]
    
    for pattern in search_patterns:
        found_files = glob.glob(pattern, recursive=True)
        if found_files:
            logging.info(f"Found {len(found_files)} mask files with pattern: {pattern}")
            mask_files.extend(found_files)
    
    if not mask_files:
        logging.error(f"No mask files found in {exam_dir}")
        return False
    
    # Sort mask files by frame number if possible
    frame_sorted_masks = []
    for mask_file in mask_files:
        try:
            # Extract frame number from filename (mask_00001.png -> 1)
            frame_num = int(os.path.basename(mask_file).split("_")[1].split(".")[0])
            frame_sorted_masks.append((frame_num, mask_file))
        except Exception as e:
            logging.error(f"Could not extract frame number from {mask_file}: {str(e)}")
    
    # Sort by frame number
    frame_sorted_masks.sort()
    
    # Copy masks to output directory
    for frame_num, mask_file in frame_sorted_masks:
        try:
            # Read mask
            mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
            if mask_img is None:
                logging.error(f"Could not read mask file: {mask_file}")
                continue
            
            # Save as PNG
            out_png_path = os.path.join(masks_dir, f"mask_{frame_num:05d}.png")
            cv2.imwrite(out_png_path, mask_img)
            
            # Also save as NPY
            out_npy_path = os.path.join(masks_dir, f"mask_{frame_num:05d}.npy")
            # Convert to binary (0 or 1) format
            binary_mask = (mask_img > 127).astype(np.uint8)
            np.save(out_npy_path, binary_mask)
            
            logging.info(f"Saved mask for frame {frame_num} to {masks_dir}")
        except Exception as e:
            logging.error(f"Error saving mask {mask_file}: {str(e)}")
    
    return len(frame_sorted_masks) > 0

def extract_masks_from_debug(masks_dir, video_path=VIDEO_PATH):
    """Search for any debug output containing masks"""
    print_header("SEARCHING FOR DEBUG MASKS")
    
    # Find any debug directories in output
    debug_dirs = glob.glob("output/**/debug", recursive=True)
    
    if not debug_dirs:
        logging.error("No debug directories found")
        return False
    
    logging.info(f"Found {len(debug_dirs)} debug directories to search")
    
    # Look for mask images in debug directories
    mask_files = []
    
    # Search patterns to find masks in debug dirs
    for debug_dir in debug_dirs:
        # Try to find all possible mask files
        found_files = glob.glob(f"{debug_dir}/**/*mask*.png", recursive=True)
        if found_files:
            logging.info(f"Found {len(found_files)} mask files in {debug_dir}")
            mask_files.extend(found_files)
    
    if not mask_files:
        logging.error("No mask files found in debug directories")
        return False
    
    # Try to extract frame numbers and sort
    frame_sorted_masks = []
    for mask_file in mask_files:
        try:
            # Look for number patterns in filename
            basename = os.path.basename(mask_file)
            # Try different patterns
            if 'frame_' in basename:
                # frame_0001.png pattern
                frame_num = int(basename.split('frame_')[1].split('.')[0])
            elif '_frame_' in basename:
                # something_frame_0001.png pattern
                frame_num = int(basename.split('_frame_')[1].split('_')[0])
            else:
                # Try to extract any number from the filename
                import re
                numbers = re.findall(r'\d+', basename)
                if numbers:
                    frame_num = int(numbers[0])
                else:
                    # No frame number found, skip
                    continue
                
            frame_sorted_masks.append((frame_num, mask_file))
        except Exception as e:
            logging.warning(f"Could not extract frame number from {mask_file}: {str(e)}")
    
    # Sort by frame number
    frame_sorted_masks.sort()
    
    if not frame_sorted_masks:
        logging.error("Could not sort mask files by frame number")
        return False
        
    # Copy masks to output directory
    for frame_num, mask_file in frame_sorted_masks:
        try:
            # Read mask
            mask_img = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            
            if mask_img is None:
                logging.error(f"Could not read mask file: {mask_file}")
                continue
            
            # Save as PNG
            out_png_path = os.path.join(masks_dir, f"mask_{frame_num:05d}.png")
            cv2.imwrite(out_png_path, mask_img)
            
            # Also save as NPY
            out_npy_path = os.path.join(masks_dir, f"mask_{frame_num:05d}.npy")
            # Convert to binary (0 or 1) format for NPY
            binary_mask = (mask_img > 127).astype(np.uint8)
            np.save(out_npy_path, binary_mask)
            
            logging.info(f"Saved mask for frame {frame_num} from debug to {masks_dir}")
        except Exception as e:
            logging.error(f"Error saving mask {mask_file}: {str(e)}")
    
    return len(frame_sorted_masks) > 0

def run_algorithm_with_mask_extraction(output_dir, video_path=VIDEO_PATH):
    """Run the algorithm and extract/save masks for later upload"""
    print_header("Running algorithm to generate masks")
    
    # Create a special environment variable to tell the algorithm to save masks
    os.environ['SAVE_MASKS_DIR'] = output_dir
    # Add additional debug flags for maximum information
    os.environ['DEBUG_MASKS'] = '1'
    os.environ['DEBUG_TRACKING'] = '1'
    os.environ['DEBUG_FLOW'] = '1'
    os.environ['DEBUG'] = '1'
    
    logging.info(f"Set SAVE_MASKS_DIR={output_dir}")
    
    # Run algorithm with --debug flag for maximum debugging
    cmd = [
        "python", 
        "src/consolidated_tracking.py", 
        "--video-path", video_path,
        "--debug",
        "--feedback-loop",
        "--skip-mdai",  # Skip the built-in upload
        "--timeout", "300"  # Set a reasonable timeout
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and pipe output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=dict(os.environ)  # Explicitly pass environment variables
        )
        
        # Process output line by line
        for line in process.stdout:
            # Print output to see progress
            print(line.strip())
            
            # Log when masks are saved
            if "Saving mask" in line or "saved mask" in line.lower() or "mask" in line.lower():
                logging.info(line.strip())
        
        process.wait()
        return process.returncode == 0
    
    except Exception as e:
        logging.error(f"Error running algorithm: {str(e)}")
        traceback.print_exc()
        return False

def validate_saved_masks(masks_dir):
    """Validate that masks were successfully saved"""
    # Check for PNG files
    png_files = glob.glob(os.path.join(masks_dir, "*.png"))
    # Check for numpy files
    npy_files = glob.glob(os.path.join(masks_dir, "*.npy"))
    
    logging.info(f"Found {len(png_files)} PNG masks and {len(npy_files)} numpy masks")
    
    if len(png_files) == 0 and len(npy_files) == 0:
        logging.error("No masks were saved!")
        return False
    
    # Log details of a few masks
    sample_files = sorted(png_files)[:5] if png_files else sorted(npy_files)[:5]
    for mask_file in sample_files:
        if mask_file.endswith(".png"):
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            logging.info(f"Mask {os.path.basename(mask_file)}: shape={mask.shape}, sum={np.sum(mask)}")
        elif mask_file.endswith(".npy"):
            mask = np.load(mask_file)
            logging.info(f"Mask {os.path.basename(mask_file)}: shape={mask.shape}, sum={np.sum(mask)}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the tracking algorithm and save masks to disk")
    parser.add_argument("--video-path", type=str, default=VIDEO_PATH, help="Path to the video file")
    args = parser.parse_args()
    
    print_header("STARTING MASK GENERATION AND SAVE PROCESS")
    
    # Create directory for masks
    masks_dir = create_masks_directory(os.path.basename(args.video_path))
    
    # First attempt: run the algorithm
    success = run_algorithm_with_mask_extraction(masks_dir, args.video_path)
    
    # Check if masks were generated
    if validate_saved_masks(masks_dir):
        print_header("MASK GENERATION SUCCESSFUL")
        logging.info(f"Masks saved to: {masks_dir}")
        logging.info("You can now use upload_masks.py to upload these masks to MD.ai")
    else:
        logging.info("No masks found after running algorithm directly")
        
        # Second attempt: try to extract masks from feedback loop output
        if extract_masks_from_feedback_loop(masks_dir, args.video_path):
            print_header("MASK EXTRACTION FROM FEEDBACK LOOP SUCCESSFUL")
            logging.info(f"Masks extracted to: {masks_dir}")
            logging.info("You can now use upload_masks.py to upload these masks to MD.ai")
        else:
            logging.info("Could not find masks in feedback loop output")
            
            # Third attempt: try to extract from debug directories
            if extract_masks_from_debug(masks_dir, args.video_path):
                print_header("MASK EXTRACTION FROM DEBUG DIRECTORIES SUCCESSFUL")
                logging.info(f"Masks extracted to: {masks_dir}")
                logging.info("You can now use upload_masks.py to upload these masks to MD.ai") 
            else:
                print_header("MASK GENERATION AND EXTRACTION FAILED")
                logging.error("Could not generate or find any mask files")
    
    # Final status report
    png_files = glob.glob(os.path.join(masks_dir, "*.png"))
    npy_files = glob.glob(os.path.join(masks_dir, "*.npy"))
    
    print_header("PROCESS COMPLETED")
    print(f"\nMasks directory: {masks_dir}")
    print(f"Total masks: PNG={len(png_files)}, NPY={len(npy_files)}")

if __name__ == "__main__":
    main() 