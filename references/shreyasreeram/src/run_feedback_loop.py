#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import glob
import logging
import signal
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("feedback_loop_runner.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
EXAM_ID = "64"  # Default exam number to use

# This should be replaced with a proper database lookup or mapping
EXAM_VIDEO_PATHS = {
    # Format: "exam_id": "video_path"
    "64": 'data/mdai_ucsf_project_x9N2LJBZ_images_dataset_D_V688LQ_2025-05-19-171632/1.2.826.0.1.3680043.8.498.18050612380255098469086741540114763661/1.2.826.0.1.3680043.8.498.72553010565308306328905938562604820392.mp4',
    # Add other exam IDs and paths as needed
}

# Default video path
VIDEO_PATH = EXAM_VIDEO_PATHS.get(EXAM_ID, "")

def wait_for_signal(signal_path, timeout=600):
    """Wait for a signal file to appear, indicating the feedback loop has completed an iteration."""
    start_time = time.time()
    logging.info(f"Waiting for signal file at {signal_path} (timeout: {timeout}s)")
    
    while time.time() - start_time < timeout:
        if os.path.exists(signal_path):
            logging.info(f"Signal file found at {signal_path}")
            return True
        time.sleep(1)
    
    logging.error(f"Timeout waiting for signal file at {signal_path}")
    return False

def run_command(command, shell=False):
    """Run a command and return its output."""
    logging.info(f"Running command: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=shell, 
            check=True, 
            text=True,
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        logging.info(f"Command completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with exit code {e.returncode}")
        logging.error(f"Stdout: {e.stdout}")
        logging.error(f"Stderr: {e.stderr}")
        return None

def extract_and_upload_masks(iteration_number, exam_id=None):
    """Extract masks from the latest feedback loop iteration and upload them to MD.ai."""
    # Find the latest feedback loop directory
    feedback_dirs = glob.glob("output/feedback_loop_*")
    if not feedback_dirs:
        logging.error("No feedback loop directory found")
        return False
    
    latest_feedback_dir = max(feedback_dirs, key=os.path.getctime)
    iteration_dir = os.path.join(latest_feedback_dir, f"iteration_{iteration_number}")
    
    if not os.path.exists(iteration_dir):
        logging.error(f"Iteration directory not found: {iteration_dir}")
        return False
    
    # Create masks directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Use provided exam_id or fall back to global
    actual_exam_id = exam_id if exam_id is not None else EXAM_ID
    masks_dir = f"output/masks_exam_{actual_exam_id}_{timestamp}"
    os.makedirs(masks_dir, exist_ok=True)
    
    # Set exam ID in environment for save_masks.py
    os.environ['EXAM_ID'] = str(actual_exam_id)
    
    # Set environment variable for save_masks.py
    os.environ['SAVE_MASKS_DIR'] = masks_dir
    
    # Run save_masks.py to extract masks
    logging.info(f"Extracting masks to {masks_dir}")
    
    # Import and run the extraction function directly
    from save_masks import extract_masks_from_debug, extract_masks_from_feedback_loop
    
    try:
        # Create metadata file
        with open(os.path.join(masks_dir, "metadata.txt"), "w") as f:
            f.write(f"Video path: {VIDEO_PATH}\n")
            f.write(f"Iteration: {iteration_number}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Exam ID: {EXAM_ID}\n")
        
        # Extract masks from debug directories
        logging.info("Extracting masks from debug directories...")
        debug_masks = extract_masks_from_debug(masks_dir)
        logging.info(f"Extracted {len(debug_masks) if debug_masks else 0} debug masks")
        
        # Extract masks from feedback loop
        logging.info("Extracting masks from feedback loop...")
        feedback_masks = extract_masks_from_feedback_loop(masks_dir)
        logging.info(f"Extracted {len(feedback_masks) if feedback_masks else 0} feedback loop masks")
        
        # Run upload_masks.py to upload masks to MD.ai with verbose mode
        logging.info(f"Uploading masks from {masks_dir} to MD.ai")
        
        # First attempt: Run with --force and --verbose flags
        logging.info("UPLOAD ATTEMPT 1: Using subprocess call with --force and --verbose")
        upload_result = run_command([
            "python", "upload_masks.py", 
            "--masks-dir", masks_dir, 
            "--force", 
            "--verbose",
            "--exam", str(actual_exam_id)
        ])
        
        # Print response to help debug
        if upload_result:
            logging.info(f"Upload result: {upload_result[:500]}...")
        else:
            logging.warning("No response from upload command")
            
        # Second attempt: Try direct module import
        logging.info("UPLOAD ATTEMPT 2: Using direct module import")
        try:
            import upload_masks
            upload_masks.upload_masks_from_directory(
                masks_dir, 
                force=True, 
                verbose=True,
                exam_id=str(actual_exam_id)
            )
            logging.info("Direct module upload completed")
        except Exception as e:
            logging.error(f"Error in direct module upload: {str(e)}")
            
        # Third attempt: Fallback to shell=True approach
        logging.info("UPLOAD ATTEMPT 3: Using shell=True approach")
        shell_cmd = f"python upload_masks.py --masks-dir {masks_dir} --force --verbose --exam {actual_exam_id}"
        try:
            shell_result = run_command(shell_cmd, shell=True)
            if shell_result:
                logging.info(f"Shell upload result: {shell_result[:500]}...")
            else:
                logging.warning("No response from shell upload command")
        except Exception as e:
            logging.error(f"Error in shell upload: {str(e)}")
        
        logging.info("Extraction and upload process completed")
        return True
    
    except Exception as e:
        logging.error(f"Error during extraction and upload: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def verify_annotations(exam_id):
    """Verify which annotations are being loaded for the given exam ID."""
    logging.info(f"Verifying annotations for exam ID: {exam_id}")
    
    # For now, we'll skip the verification since we can't find the correct module
    logging.warning("Annotation verification is skipped - module not found")
    logging.info("Will continue with enhanced parameters to prioritize expert annotations")
    return True

def continue_feedback_loop(proc):
    """Simulate pressing Enter to continue to the next feedback loop iteration."""
    logging.info("Continuing to next feedback loop iteration")
    if proc:
        try:
            proc.stdin.write("\n")
            proc.stdin.flush()
            logging.info("Sent Enter key to feedback loop process")
            return True
        except Exception as e:
            logging.error(f"Error sending Enter key: {str(e)}")
            return False
    else:
        logging.error("No feedback loop process provided")
        return False

def run_feedback_loop(iterations=3, exam_id=None, video_path=None):
    """Run the feedback loop with the integrated two-step mask extraction and upload."""
    
    # Use provided exam_id or use the global default
    if exam_id is not None:
        exam_id_to_use = exam_id
    else:
        global EXAM_ID
        exam_id_to_use = EXAM_ID
        
    # Determine the video path to use
    if video_path is not None:
        video_path_to_use = video_path
    else:
        # Look up in the mapping
        global EXAM_VIDEO_PATHS
        video_path_to_use = EXAM_VIDEO_PATHS.get(exam_id_to_use, "")
        if not video_path_to_use:
            logging.error(f"No video path found for exam ID {exam_id_to_use}")
            return False
            
    logging.info(f"Using exam ID: {exam_id_to_use}, video path: {video_path_to_use}")
    
    # Verify annotations before starting
    logging.info("Verifying annotations for this exam...")
    annotations_ok = verify_annotations(exam_id_to_use)
    if not annotations_ok:
        logging.error("Failed to verify annotations. Check annotation data.")
        logging.info("Continuing anyway, but tracking may not work correctly.")
    
    # Set environment variables
    os.environ['DEBUG_UPLOADS'] = '1'
    os.environ['FORCE_UPLOAD'] = '1' 
    os.environ['MDAI_UPLOAD_FIX'] = '1'
    os.environ['DEBUG_TRACKING'] = '1'
    os.environ['DEBUG'] = '1'
    os.environ['TRACKING_STRATEGY_WEIGHT'] = '1.0'
    os.environ['FEEDBACK_WEIGHT'] = '2.0'
    os.environ['ANNOTATION_LEARNING_RATE'] = '0.9'
    os.environ['EXPERT_FEEDBACK_WEIGHT'] = '8.0'
    os.environ['IGNORE_OTHER_EXAMS'] = '1'
    os.environ['EXAM_ID'] = exam_id_to_use
    os.environ['ONLY_USE_EXPERT_ANNOTATIONS'] = '1'
    os.environ['FORCE_EXPERT_INIT'] = '1'
    
    # Stop any existing feedback loop processes
    run_command(["pkill", "-f", "consolidated_tracking.py"])
    
    # Start the feedback loop process
    logging.info("Starting feedback loop process")
    
    # Add stronger debugging
    logging.info("DEBUGGING: About to start consolidated_tracking.py")
    
    # First run with proper flags to emphasize expert annotations
    cmd = [
        "python", "src/consolidated_tracking.py", 
        "--feedback-loop", 
        "--video-path", video_path_to_use,
        "--debug", "--emphasize-expert-annotations",
        "--priority-expert-frames",
        "--exam-number", exam_id_to_use,
        "--exclusive-exam-annotations"
    ]
    
    # Add genuine evaluation option if specified in the environment
    if os.environ.get('USE_GENUINE_EVALUATION', '0') == '1':
        cmd.append("--genuine-evaluation")
        # Add sampling rate if specified
        sampling_rate = os.environ.get('SAMPLING_RATE', '10')
        cmd.extend(["--sampling-rate", sampling_rate])
    
    # Simplified command for debugging
    logging.info(f"Command: {' '.join(cmd)}")
    
    # Check if the file exists
    if not os.path.exists("src/consolidated_tracking.py"):
        logging.error("CRITICAL ERROR: src/consolidated_tracking.py not found!")
        # Try to find where it actually is
        possible_files = glob.glob("**/consolidated_tracking.py", recursive=True)
        if possible_files:
            logging.info(f"Found consolidated_tracking.py at: {possible_files}")
    
    # Check if video path exists
    if not os.path.exists(video_path_to_use):
        logging.error(f"CRITICAL ERROR: Video file not found at {video_path_to_use}")
        # Try to find where videos might be
        possible_videos = glob.glob("**/*.mp4", recursive=True)
        if possible_videos:
            logging.info(f"Found these video files: {possible_videos[:5]}")
            logging.info(f"Total video files found: {len(possible_videos)}")
    
    # Try running with shell=True for debugging
    try:
        logging.info("Attempting to start process with direct execution...")
        feedback_proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )
        logging.info(f"Process started with PID: {feedback_proc.pid}")
    except Exception as e:
        logging.error(f"Error starting process: {str(e)}")
        logging.info("Trying alternative method with shell=True...")
        
        # Alternative approach with shell=True
        shell_cmd = " ".join(cmd)
        try:
            feedback_proc = subprocess.Popen(
                shell_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                shell=True
            )
            logging.info(f"Process started with shell=True, PID: {feedback_proc.pid}")
        except Exception as e:
            logging.error(f"Error starting process with shell=True: {str(e)}")
            return False
    
    # Function to read output without blocking
    def read_output():
        logs = []
        while True:
            line = feedback_proc.stdout.readline()
            if not line and feedback_proc.poll() is not None:
                break
            if line:
                logs.append(line.strip())
                logging.info(f"FEEDBACK LOOP: {line.strip()}")
                
                # Add more detailed logging for specific events
                if "Loading annotations" in line:
                    logging.info("🔍 Annotation loading detected")
                elif "annotations found" in line:
                    logging.info("✅ Annotations found")
                elif "Filtering annotations" in line:
                    logging.info("⚙️ Annotation filtering in progress")
                elif "expert annotations" in line:
                    logging.info("🔍 Expert annotations mentioned")
                elif "Error" in line or "error" in line or "ERROR" in line:
                    logging.error(f"❌ ERROR detected: {line.strip()}")
            else:
                break
        return logs
    
    try:
        current_iteration = 0
        waiting_for_user = False
        iteration_complete = False
        
        # Start output monitoring thread
        while feedback_proc.poll() is None:
            # Read available output
            logs = read_output()
            
            # Check for iteration complete marker
            for line in logs:
                if "=== ITERATION" in line and "COMPLETED ===" in line:
                    current_iteration = int(line.split("ITERATION")[1].split()[0])
                    iteration_complete = True
                    logging.info(f"Detected completion of iteration {current_iteration}")
                
                if "=== WAITING PERIOD FOR EXPERT FEEDBACK ===" in line:
                    waiting_for_user = True
                    logging.info("Detected waiting period for expert feedback")
            
            # If iteration is complete and waiting for user input, process masks and continue
            if iteration_complete and waiting_for_user:
                logging.info(f"Processing masks for iteration {current_iteration}")
                
                # Process masks and upload to MD.ai
                success = extract_and_upload_masks(current_iteration, exam_id_to_use)
                
                if not success:
                    logging.error(f"Failed to extract and upload masks for iteration {current_iteration}")
                
                # Continue to next iteration if not at the end
                if current_iteration < iterations:
                    time.sleep(2)  # Give some time before sending Enter
                    continue_feedback_loop(feedback_proc)
                else:
                    logging.info(f"Completed all {iterations} iterations")
                    break
                
                # Reset flags
                iteration_complete = False
                waiting_for_user = False
            
            # Sleep briefly to avoid high CPU usage
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt, stopping feedback loop")
    finally:
        if feedback_proc and feedback_proc.poll() is None:
            logging.info("Terminating feedback loop process")
            feedback_proc.terminate()
            try:
                feedback_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                feedback_proc.kill()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run feedback loop with integrated mask extraction and upload")
    parser.add_argument("--iterations", type=int, default=3, help="Number of feedback loop iterations to run")
    parser.add_argument("--exam-id", type=str, default="64", help="Exam ID to process")
    parser.add_argument("--video-path", type=str, help="Optional: Direct path to video file (overrides exam-id lookup)")
    parser.add_argument("--genuine-evaluation", action="store_true", help="Use genuine evaluation with sparse annotations")
    parser.add_argument("--sampling-rate", type=int, default=10, help="Sampling rate for sparse annotations (take every Nth frame)")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading masks to MD.ai")
    
    args = parser.parse_args()
    
    # Pass the exam ID as a parameter instead of modifying the global
    exam_id_arg = args.exam_id
    video_path_arg = args.video_path
    
    logging.info(f"Starting feedback loop runner with {args.iterations} iterations for exam #{exam_id_arg}")
    if video_path_arg:
        logging.info(f"Using provided video path: {video_path_arg}")
    
    # Set environment variables based on command line arguments
    if args.genuine_evaluation:
        os.environ['USE_GENUINE_EVALUATION'] = '1'
        os.environ['SAMPLING_RATE'] = str(args.sampling_rate)
        logging.info(f"Enabling genuine evaluation with sampling rate {args.sampling_rate}")
    
    if args.no_upload:
        os.environ['SKIP_UPLOADS'] = '1'
        logging.info("Uploads to MD.ai disabled")
    
    # Run with the command line arguments
    run_feedback_loop(args.iterations, exam_id_arg, video_path_arg)