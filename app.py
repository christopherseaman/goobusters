#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "flask",
#     "opencv-contrib-python",
#     "numpy>=1.21.0,<3.0.0",
#     "pillow",
#     "python-dotenv",
#     "pyyaml",
#     "torch>=2.0.0",
#     "torchvision>=0.15.0",
#     "mdai==0.16.0",
#     "pandas",
#     "tqdm",
#     "scikit-image",
#     "scipy",
#     "psutil"
# ]
# ///

"""
Goobusters Annotation Review App

Local web interface for reviewing and modifying optical flow tracking results.
Allows frame-by-frame navigation, mask editing, and EMPTY_ID annotations.
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import cv2
import numpy as np
from pathlib import Path
import base64
import io
import shutil
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
import threading

# Load environment variables from dot.env (project uses dot.env, not .env)
load_dotenv('dot.env')

app = Flask(__name__)

# Disable caching for static files during development
@app.after_request
def add_header(response):
    """Add headers to prevent caching of static files."""
    if 'static' in request.path:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Configuration
OUTPUT_DIR = Path('output')
DATA_DIR = Path('data')
ANNOTATIONS_DIR = Path('annotations')  # Store modified annotations outside output/

def get_annotations_file(study_uid, series_uid):
    """Get path to modified annotations JSON file."""
    return ANNOTATIONS_DIR / f"{study_uid}_{series_uid}" / "modified_annotations.json"

def get_annotations_dir(study_uid, series_uid):
    """Get directory for storing modified annotations."""
    return ANNOTATIONS_DIR / f"{study_uid}_{series_uid}"

def load_modified_annotations(study_uid, series_uid):
    """Load modified annotations from JSON file."""
    annotations_file = get_annotations_file(study_uid, series_uid)
    if annotations_file.exists():
        with open(annotations_file) as f:
            return json.load(f)
    return {}

def save_modified_annotations(study_uid, series_uid, annotations):
    """Save modified annotations to JSON file."""
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = get_annotations_file(study_uid, series_uid)
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)


def get_available_videos():
    """
    Get list of available videos from output directory.

    Returns:
        List of dicts with video metadata
    """
    videos = []

    # Check all methods in output directory
    if not OUTPUT_DIR.exists():
        return videos

    for method_dir in OUTPUT_DIR.iterdir():
        if not method_dir.is_dir():
            continue

        method = method_dir.name

        # Each video has format: {study_uid}_{series_uid}
        for video_dir in method_dir.iterdir():
            if not video_dir.is_dir():
                continue

            # Parse study and series UIDs from directory name
            dir_name = video_dir.name
            parts = dir_name.split('_')
            if len(parts) != 2:
                continue

            study_uid, series_uid = parts

            # Check for required files
            identity_file = video_dir / 'identity.yaml'
            tracked_video = video_dir / 'tracked_video.mp4'
            if not all([identity_file.exists(), tracked_video.exists()]):
                continue

            # Load identity metadata
            import yaml
            with open(identity_file) as f:
                identity = yaml.safe_load(f)

            videos.append({
                'method': method,
                'study_uid': study_uid,
                'series_uid': series_uid,
                'exam_number': identity.get('exam_number', 'Unknown'),
                'video_path': str(tracked_video.relative_to(OUTPUT_DIR)),
                'masks_dir': str((video_dir / 'masks').relative_to(OUTPUT_DIR)),
                'labels': identity.get('labels', [])
            })

    return sorted(videos, key=lambda x: (x['method'], x['exam_number']))

@app.route('/')
def index():
    """Main page - video selector."""
    videos = get_available_videos()

    # If no videos, show message
    if not videos:
        return render_template('no_videos.html')

    # Get selected video from query params or default to first
    selected_method = request.args.get('method', videos[0]['method'])
    selected_study = request.args.get('study')
    selected_series = request.args.get('series')

    # Find selected video or default to first
    selected_video = None
    if selected_study and selected_series:
        selected_video = next((v for v in videos if
                             v['study_uid'] == selected_study and
                             v['series_uid'] == selected_series and
                             v['method'] == selected_method), None)

    if not selected_video:
        selected_video = videos[0]

    return render_template('viewer.html',
                         videos=videos,
                         selected_video=selected_video)

@app.route('/api/videos')
def api_videos():
    """API endpoint to get list of available videos."""
    videos = get_available_videos()
    return jsonify(videos)

@app.route('/api/video/<method>/<study_uid>/<series_uid>')
def api_video_data(method, study_uid, series_uid):
    """
    API endpoint to get video data including masks and annotations.

    Args:
        method: Optical flow method (dis, farneback, raft)
        study_uid: Study Instance UID
        series_uid: Series Instance UID

    Returns:
        JSON with video metadata, masks, and annotations
    """
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"

    if not video_dir.exists():
        return jsonify({'error': 'Video not found'}), 404

    # Check for annotations/ masks.json first (from save or retrack), else use output/
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    annotations_masks_path = annotations_dir / 'masks.json'
    output_masks_path = video_dir / 'masks.json'

    masks_annotations = []
    if annotations_masks_path.exists():
        with open(annotations_masks_path) as f:
            masks_annotations = json.load(f)
    elif output_masks_path.exists():
        with open(output_masks_path) as f:
            masks_annotations = json.load(f)

    # modified_frames is now deprecated - we use masks.json directly
    modified_frames = {}

    # Get total frames by counting frames in frames directory
    frames_dir = video_dir / 'frames'
    if frames_dir.exists():
        total_frames = len(list(frames_dir.glob('frame_*.webp')))
    else:
        # Fallback: try video file
        video_path = video_dir / 'tracked_video.mp4'
        if not video_path.exists():
            for images_dir in DATA_DIR.glob('mdai_*_images_dataset_*'):
                candidate = images_dir / study_uid / f"{series_uid}.mp4"
                if candidate.exists():
                    video_path = candidate
                    break
        
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            return jsonify({'error': 'Video and frames not found'}), 404
    
    # Load identity file for video metadata
    identity_file = video_dir / 'identity.yaml'
    exam_number = 'Unknown'
    labels = []
    if identity_file.exists():
        import yaml
        with open(identity_file) as f:
            identity = yaml.safe_load(f)
            exam_number = identity.get('exam_number', 'Unknown')
            labels = identity.get('labels', [])
    
    return jsonify({
        'masks_annotations': masks_annotations,
        'modified_frames': modified_frames,
        'total_frames': total_frames,
        'method': method,
        'study_uid': study_uid,
        'series_uid': series_uid,
        'exam_number': exam_number,
        'labels': labels
    })

@app.route('/api/frames/<method>/<study_uid>/<series_uid>')
def api_all_frames(method, study_uid, series_uid):
    """
    Return frames archive URL and masks JSON.
    Frames are pre-extracted as webp and archived in tar.gz.
    Masks are stored as RLE-encoded JSON.
    """
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    
    # Get total frames by counting frames in frames directory (most reliable)
    frames_dir = video_dir / 'frames'
    if frames_dir.exists():
        total_frames = len(list(frames_dir.glob('frame_*.webp')))
    else:
        # Fallback: try to get from video file
        video_path = video_dir / 'tracked_video.mp4'
        if not video_path.exists():
            for images_dir in DATA_DIR.glob('mdai_*_images_dataset_*'):
                candidate = images_dir / study_uid / f"{series_uid}.mp4"
                if candidate.exists():
                    video_path = candidate
                    break
        
        if video_path.exists():
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            return jsonify({'error': 'Frames and video not found'}), 404
    
    frames_archive_url = f'/api/frames_archive/{method}/{study_uid}/{series_uid}'
    # masks_archive endpoint already serves from annotations/ when available
    masks_archive_url = f'/api/masks_archive/{method}/{study_uid}/{series_uid}'

    return jsonify({
        'frames_archive_url': frames_archive_url,
        'masks_archive_url': masks_archive_url,
        'total_frames': total_frames
    })

@app.route('/api/frames_archive/<method>/<study_uid>/<series_uid>')
def api_frames_archive(method, study_uid, series_uid):
    """Serve frames archive (tar.gz)."""
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    frames_archive = video_dir / 'frames.tar.gz'
    
    if not frames_archive.exists():
        return jsonify({'error': 'Frames archive not found'}), 404
    
    return send_file(str(frames_archive), mimetype='application/gzip', as_attachment=False)

@app.route('/api/masks_archive/<method>/<study_uid>/<series_uid>')
def api_masks_archive(method, study_uid, series_uid):
    """Serve masks archive - from annotations/ if exists, otherwise from output/."""
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    annotations_masks_archive = annotations_dir / 'masks.tar.gz'

    # Prefer annotations/ (from save or retrack)
    if annotations_masks_archive.exists():
        return send_file(str(annotations_masks_archive), mimetype='application/gzip', as_attachment=False)

    # Fall back to output/
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    masks_archive = video_dir / 'masks.tar.gz'

    if not masks_archive.exists():
        return jsonify({'error': 'Masks archive not found'}), 404

    return send_file(str(masks_archive), mimetype='application/gzip', as_attachment=False)

@app.route('/api/masks_archive_modified/<study_uid>/<series_uid>')
def api_masks_archive_modified(study_uid, series_uid):
    """Serve modified masks archive from annotations/ (tar.gz)."""
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    masks_archive = annotations_dir / 'masks.tar.gz'
    
    if not masks_archive.exists():
        return jsonify({'error': 'Modified masks archive not found'}), 404
    
    return send_file(str(masks_archive), mimetype='application/gzip', as_attachment=False)


@app.route('/api/save_mask', methods=['POST'])
def api_save_mask():
    """
    Save a single modified mask to annotations directory (outside output/).
    
    NOTE: This is a legacy endpoint. Use /api/save_changes for saving all annotations.

    Expected JSON:
        {
            'method': str,
            'study_uid': str,
            'series_uid': str,
            'frame_number': int,
            'mask_data': str (base64),
            'label_id': str
        }
    """
    data = request.json
    method = data['method']
    study_uid = data['study_uid']
    series_uid = data['series_uid']
    frame_num = data['frame_number']
    mask_b64 = data['mask_data']
    label_id = data.get('label_id', os.getenv('LABEL_ID', ''))

    # Decode mask
    mask_bytes = base64.b64decode(mask_b64.split(',')[1] if ',' in mask_b64 else mask_b64)
    mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
    mask_array = np.array(mask_img)

    # Save mask as webp in annotations directory
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f'frame_{frame_num:06d}_mask.webp'
    cv2.imwrite(str(mask_path), mask_array, [cv2.IMWRITE_WEBP_QUALITY, 85])

    # Save metadata to JSON
    modified_annotations = load_modified_annotations(study_uid, series_uid)
    frame_key = f'frame_{frame_num}'
    modified_annotations[frame_key] = {
        'label_id': label_id,
        'is_empty': False,
        'modified_at': datetime.now().isoformat()
    }
    save_modified_annotations(study_uid, series_uid, modified_annotations)

    return jsonify({'success': True})

@app.route('/api/save_changes', methods=['POST'])
def api_save_changes():
    """
    Save user modifications on top of existing tracking results.

    Copies ALL frames from output/ (or existing annotations/), then overlays user modifications.
    Outputs in same format as track.py: masks.json + masks/ + masks.tar.gz

    Expected JSON:
        {
            'method': str,
            'study_uid': str,
            'series_uid': str,
            'modified_frames': dict of {frame_num: {mask_data: base64, is_empty: bool}} - user-edited frames
        }
    """
    import tarfile

    data = request.json
    method = data['method']
    study_uid = data['study_uid']
    series_uid = data['series_uid']
    modified_frames_data = data.get('modified_frames', {})
    label_id = os.getenv('LABEL_ID', '')
    empty_id = os.getenv('EMPTY_ID', '')

    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Get video dimensions
    video_path = video_dir / 'tracked_video.mp4'
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    height, width = (first_frame.shape[:2]) if ret else (0, 0)
    cap.release()

    # Setup masks directory
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Start with existing masks.json (from annotations/ if exists, else output/)
    existing_masks_json = []
    source_masks_dir = None

    existing_annotations_masks = annotations_dir / 'masks.json'
    output_masks_json = video_dir / 'masks.json'

    if existing_annotations_masks.exists():
        # Use existing annotations as base
        with open(existing_annotations_masks) as f:
            existing_masks_json = json.load(f)
        source_masks_dir = annotations_dir / 'masks'
    elif output_masks_json.exists():
        # Use output as base
        with open(output_masks_json) as f:
            existing_masks_json = json.load(f)
        source_masks_dir = video_dir / 'masks'

    # Build lookup by frame number
    masks_by_frame = {entry['frameNumber']: entry for entry in existing_masks_json}
    modified_frame_nums = set(int(k) for k in modified_frames_data.keys())

    # Copy all existing masks that aren't being modified
    if source_masks_dir and source_masks_dir.exists():
        for entry in existing_masks_json:
            frame_num = entry['frameNumber']
            if frame_num not in modified_frame_nums:
                mask_file = entry.get('mask_file', f'frame_{frame_num:06d}_mask.webp')
                src_mask = source_masks_dir / mask_file
                dst_mask = masks_dir / mask_file
                if src_mask.exists() and src_mask != dst_mask:
                    shutil.copy2(src_mask, dst_mask)

    # Apply user modifications
    for frame_num_str, frame_data in modified_frames_data.items():
        frame_num = int(frame_num_str)
        is_empty = frame_data.get('is_empty', False)
        frame_label_id = empty_id if is_empty else label_id
        mask_file = f'frame_{frame_num:06d}_mask.webp'
        mask_path = masks_dir / mask_file

        # Decode and save mask
        mask_b64 = frame_data.get('mask_data', '')
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64.split(',')[1] if ',' in mask_b64 else mask_b64)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask_array = np.array(mask_img)
            cv2.imwrite(str(mask_path), mask_array, [cv2.IMWRITE_WEBP_QUALITY, 85])
        elif is_empty:
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.imwrite(str(mask_path), empty_mask, [cv2.IMWRITE_WEBP_QUALITY, 85])

        # Update or create entry
        masks_by_frame[frame_num] = {
            'id': f'annotation_{frame_num}',
            'labelId': frame_label_id,
            'StudyInstanceUID': study_uid,
            'SeriesInstanceUID': series_uid,
            'frameNumber': frame_num,
            'type': 'empty' if is_empty else 'fluid',
            'is_annotation': True,
            'track_id': f'annotation_{frame_num}',
            'label_id': frame_label_id,
            'mask_file': mask_file
        }

    # Build final masks_json sorted by frame
    masks_json = [masks_by_frame[fn] for fn in sorted(masks_by_frame.keys())]

    # Write masks.json
    with open(annotations_dir / 'masks.json', 'w') as f:
        json.dump(masks_json, f, indent=2)

    # Create masks.tar.gz archive
    masks_archive = annotations_dir / 'masks.tar.gz'
    with tarfile.open(masks_archive, 'w:gz') as tar:
        tar.add(masks_dir, arcname='masks')

    annotation_count = sum(1 for m in masks_json if m.get('is_annotation', False))
    return jsonify({
        'success': True,
        'saved_count': len(masks_json),
        'annotation_count': annotation_count,
        'total_frames': total_frames
    })

@app.route('/api/mark_empty', methods=['POST'])
def api_mark_empty():
    """
    Mark a single frame as EMPTY_ID (no fluid).
    
    NOTE: This endpoint is not currently used by the frontend.
    The frontend marks frames as empty in memory, then saves all changes via /api/save_changes.
    This endpoint is kept for potential future use (e.g., auto-save on mark).

    Expected JSON:
        {
            'method': str,
            'study_uid': str,
            'series_uid': str,
            'frame_number': int
        }
    """
    data = request.json
    method = data['method']
    study_uid = data['study_uid']
    series_uid = data['series_uid']
    frame_num = data['frame_number']
    empty_id = os.getenv('EMPTY_ID', '')

    # Create empty mask
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"

    # Get frame dimensions from video
    video_path = video_dir / 'tracked_video.mp4'
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Frame not found'}), 404

    height, width = frame.shape[:2]
    empty_mask = np.zeros((height, width), dtype=np.uint8)

    # Save empty mask to file in annotations directory (outside output/)
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f'frame_{frame_num:06d}_mask.webp'
    cv2.imwrite(str(mask_path), empty_mask, [cv2.IMWRITE_WEBP_QUALITY, 85])

    # Save metadata to JSON
    modified_annotations = load_modified_annotations(study_uid, series_uid)
    frame_key = f'frame_{frame_num}'
    modified_annotations[frame_key] = {
        'label_id': empty_id,
        'is_empty': True,
        'modified_at': datetime.now().isoformat()
    }
    save_modified_annotations(study_uid, series_uid, modified_annotations)

    return jsonify({'success': True})

# Retrack status tracking (in-memory for simplicity)
retrack_status = {}

@app.route('/api/retrack/<study_uid>/<series_uid>', methods=['POST'])
def api_retrack(study_uid, series_uid):
    """
    Re-run optical flow tracking using ONLY human-verified annotations (label_id and empty_id).

    This endpoint triggers retracking for a single video using the modified annotations
    stored in annotations/{study_uid}_{series_uid}/. It uses ONLY human-verified
    annotations (label_id for fluid, empty_id for no fluid) as ground truth,
    and regenerates track_id predictions for frames in between.

    Expected JSON (optional):
        {
            'method': str (optional, defaults to current method from request)
        }

    Returns:
        JSON with status and task_id for polling progress
    """
    data = request.json or {}
    method = data.get('method', 'farneback')  # Default to farneback if not specified

    # Generate task ID
    task_id = f"{study_uid}_{series_uid}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Check if annotations exist (masks.json from save_changes)
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    masks_json_file = annotations_dir / 'masks.json'
    if not masks_json_file.exists():
        return jsonify({
            'success': False,
            'error': 'No saved annotations found. Save your edits first.'
        }), 400

    # Check if video exists
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    if not video_dir.exists():
        return jsonify({
            'success': False,
            'error': f'Video directory not found: {video_dir}'
        }), 404

    # Find the source video
    video_path = None
    # First try tracked_video.mp4 in output dir
    tracked_video = video_dir / 'tracked_video.mp4'
    if tracked_video.exists():
        # We need the original video, not the tracked one
        # Look in data directory
        for images_dir in DATA_DIR.glob('mdai_*_images_dataset_*'):
            candidate = images_dir / study_uid / f"{series_uid}.mp4"
            if candidate.exists():
                video_path = candidate
                break

    if not video_path:
        return jsonify({
            'success': False,
            'error': 'Original video file not found'
        }), 404

    # Initialize status
    retrack_status[task_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing retrack...',
        'error': None
    }

    # Output to annotations/ directory (not output/)
    annotations_output_dir = get_annotations_dir(study_uid, series_uid)
    annotations_output_dir.mkdir(parents=True, exist_ok=True)

    # Run retracking in background thread
    thread = threading.Thread(
        target=run_retrack,
        args=(task_id, study_uid, series_uid, method, str(video_path), str(annotations_output_dir))
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'task_id': task_id,
        'message': 'Retracking started'
    })

def run_retrack(task_id, study_uid, series_uid, method, video_path, output_dir):
    """
    Background task to run retracking.

    Outputs in same format as track.py: masks.json + masks/ + masks.tar.gz
    Uses lib/ functions directly (not track.py) as per RETRACK_DESIGN.md.
    """
    try:
        retrack_status[task_id]['status'] = 'loading'
        retrack_status[task_id]['message'] = 'Loading annotations...'

        # Import lib functions
        from lib.local_annotations import convert_local_to_annotations_df
        from lib.opticalflowprocessor import OpticalFlowProcessor
        from lib.multi_frame_tracker import process_video_with_multi_frame_tracking
        import tarfile

        # Convert local annotations to DataFrame
        annotations_df = convert_local_to_annotations_df(study_uid, series_uid)

        if annotations_df.empty:
            retrack_status[task_id]['status'] = 'error'
            retrack_status[task_id]['error'] = 'No valid annotations found'
            return

        retrack_status[task_id]['message'] = f'Found {len(annotations_df)} annotations'
        retrack_status[task_id]['progress'] = 10

        # Create optical flow processor
        retrack_status[task_id]['status'] = 'processing'
        retrack_status[task_id]['message'] = f'Creating {method} optical flow processor...'

        flow_processor = OpticalFlowProcessor(method)

        retrack_status[task_id]['progress'] = 20
        retrack_status[task_id]['message'] = 'Running optical flow tracking...'

        # Get label IDs from environment
        label_id_fluid = os.getenv('LABEL_ID', '')
        label_id_empty = os.getenv('EMPTY_ID', '')
        label_id_track = os.getenv('TRACK_ID', '')

        # Run tracking - use a temp dir since we'll reformat the output
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            result = process_video_with_multi_frame_tracking(
                video_path=video_path,
                annotations_df=annotations_df,
                study_uid=study_uid,
                series_uid=series_uid,
                flow_processor=flow_processor,
                output_dir=temp_dir,
                mdai_client=None,
                label_id_fluid=label_id_fluid,
                label_id_machine=label_id_track,
                upload_to_mdai=False
            )

            retrack_status[task_id]['progress'] = 80
            retrack_status[task_id]['message'] = 'Saving results...'

            # Convert all_masks to masks.json format
            all_masks = result.get('all_masks', {})
            masks_json = []
            masks_dir = Path(output_dir) / 'masks'
            # Clear existing masks before writing new ones
            if masks_dir.exists():
                shutil.rmtree(masks_dir)
            masks_dir.mkdir(parents=True, exist_ok=True)

            for frame_num, frame_info in all_masks.items():
                if not isinstance(frame_info, dict):
                    continue

                mask = frame_info.get('mask')
                if mask is None:
                    continue

                is_annotation = frame_info.get('is_annotation', False)
                frame_type = frame_info.get('type', 'unknown')
                is_empty = frame_type == 'empty'

                # Determine label_id based on type
                if is_annotation:
                    frame_label_id = label_id_empty if is_empty else label_id_fluid
                else:
                    frame_label_id = label_id_track

                # Save mask
                mask_file = f'frame_{frame_num:06d}_mask.webp'
                mask_path = masks_dir / mask_file
                cv2.imwrite(str(mask_path), mask, [cv2.IMWRITE_WEBP_QUALITY, 85])

                # Build entry
                entry = {
                    'id': frame_info.get('track_id', f'track_{frame_num}'),
                    'labelId': frame_label_id,
                    'StudyInstanceUID': study_uid,
                    'SeriesInstanceUID': series_uid,
                    'frameNumber': frame_num,
                    'type': frame_type,
                    'is_annotation': is_annotation,
                    'track_id': frame_info.get('track_id', f'track_{frame_num}'),
                    'label_id': frame_label_id,
                    'mask_file': mask_file
                }
                if is_empty:
                    entry['data'] = None

                masks_json.append(entry)

            # Sort by frame number
            masks_json.sort(key=lambda x: x['frameNumber'])

            # Write masks.json
            with open(Path(output_dir) / 'masks.json', 'w') as f:
                json.dump(masks_json, f, indent=2)

            # Create masks.tar.gz archive
            masks_archive = Path(output_dir) / 'masks.tar.gz'
            with tarfile.open(masks_archive, 'w:gz') as tar:
                tar.add(masks_dir, arcname='masks')

        retrack_status[task_id]['progress'] = 90
        retrack_status[task_id]['message'] = 'Finalizing...'

        # Clean up flow processor memory
        flow_processor.cleanup_memory()

        annotated_count = sum(1 for m in masks_json if m.get('is_annotation', False))
        predicted_count = len(masks_json) - annotated_count

        retrack_status[task_id]['status'] = 'complete'
        retrack_status[task_id]['progress'] = 100
        retrack_status[task_id]['message'] = f"Retrack complete: {annotated_count} annotations, {predicted_count} predictions"
        retrack_status[task_id]['result'] = {
            'annotated_frames': annotated_count,
            'predicted_frames': predicted_count,
            'total_frames': len(masks_json)
        }

    except Exception as e:
        import traceback
        retrack_status[task_id]['status'] = 'error'
        retrack_status[task_id]['error'] = str(e)
        retrack_status[task_id]['message'] = f'Error: {str(e)}'
        print(f"Retrack error: {traceback.format_exc()}")

@app.route('/api/retrack/status/<task_id>')
def api_retrack_status(task_id):
    """Get status of a retrack task."""
    if task_id not in retrack_status:
        return jsonify({'error': 'Task not found'}), 404

    return jsonify(retrack_status[task_id])

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from output directory."""
    return send_file(OUTPUT_DIR / filename)

if __name__ == '__main__':
    # Check if output directory exists
    if not OUTPUT_DIR.exists():
        print(f"⚠️  Output directory not found: {OUTPUT_DIR}")
        print("   Run 'track.py' first to generate tracking output")
    else:
        videos = get_available_videos()
        print(f"✅ Found {len(videos)} videos available for review")

    print("\n🚀 Starting Goobusters Annotation Review App")
    print("   Navigate to: http://localhost:5000")

    app.run(host='0.0.0.0', port=5000, debug=True)
