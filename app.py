#!/usr/bin/env python3
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
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

def get_annotations_file(method, study_uid, series_uid):
    """Get path to modified annotations JSON file."""
    return ANNOTATIONS_DIR / method / f"{study_uid}_{series_uid}" / "modified_annotations.json"

def get_annotations_dir(method, study_uid, series_uid):
    """Get directory for storing modified annotations."""
    return ANNOTATIONS_DIR / method / f"{study_uid}_{series_uid}"

def load_modified_annotations(method, study_uid, series_uid):
    """Load modified annotations from JSON file."""
    annotations_file = get_annotations_file(method, study_uid, series_uid)
    if annotations_file.exists():
        with open(annotations_file) as f:
            return json.load(f)
    return {}

def save_modified_annotations(method, study_uid, series_uid, annotations):
    """Save modified annotations to JSON file."""
    annotations_dir = get_annotations_dir(method, study_uid, series_uid)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    annotations_file = get_annotations_file(method, study_uid, series_uid)
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
            mask_data = video_dir / 'mask_data.json'

            if not all([identity_file.exists(), tracked_video.exists(), mask_data.exists()]):
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
                'mask_data_path': str(mask_data.relative_to(OUTPUT_DIR)),
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

    # Load mask data
    mask_data_path = video_dir / 'mask_data.json'
    with open(mask_data_path) as f:
        mask_data = json.load(f)

    # Load tracked annotations
    tracked_annotations_path = video_dir / 'tracked_annotations.json'
    with open(tracked_annotations_path) as f:
        tracked_annotations = json.load(f)

    # Check for local modifications in JSON
    modified_annotations = load_modified_annotations(method, study_uid, series_uid)
    modified_frames = {}
    for frame_key, frame_data in modified_annotations.items():
        frame_num = int(frame_key.replace('frame_', ''))
        modified_frames[frame_num] = {
            'label_id': frame_data.get('label_id', ''),
            'is_empty': frame_data.get('is_empty', False),
            'modified_at': frame_data.get('modified_at', '')
        }

    return jsonify({
        'mask_data': mask_data,
        'tracked_annotations': tracked_annotations,
        'modified_frames': modified_frames,
        'total_frames': len(mask_data)
    })

@app.route('/api/frame/<method>/<study_uid>/<series_uid>/<int:frame_num>')
def api_frame(method, study_uid, series_uid, frame_num):
    """
    Get a specific frame and its mask.

    Returns:
        JSON with frame image (base64) and mask image (base64)
    """
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"

    # Find original ultrasound video in data directory
    # Pattern: data/mdai_*_images_dataset_*/{study_uid}/{series_uid}.mp4
    video_path = None
    for images_dir in DATA_DIR.glob('mdai_*_images_dataset_*'):
        candidate = images_dir / study_uid / f"{series_uid}.mp4"
        if candidate.exists():
            video_path = candidate
            break

    if not video_path:
        # Fallback to tracked video if original not found
        video_path = video_dir / 'tracked_video.mp4'
        if not video_path.exists():
            return jsonify({'error': 'Video not found'}), 404

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({'error': 'Frame not found'}), 404

    # Encode frame as base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_b64 = base64.b64encode(buffer).decode('utf-8')

    # Get mask - check annotations directory first for modified annotations, then filesystem
    mask_b64 = None
    annotations_dir = get_annotations_dir(method, study_uid, series_uid)
    modified_mask_path = annotations_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
    
    if modified_mask_path.exists():
        # Use modified mask from annotations directory
        mask = cv2.imread(str(modified_mask_path), cv2.IMREAD_GRAYSCALE)
        _, buffer = cv2.imencode('.png', mask)
        mask_b64 = base64.b64encode(buffer).decode('utf-8')
    else:
        # Fall back to filesystem mask in output directory
        mask_path = video_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            _, buffer = cv2.imencode('.png', mask)
            mask_b64 = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'frame': frame_b64,
        'mask': mask_b64,
        'frame_number': frame_num
    })

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

    # Save mask to file in annotations directory (outside output/)
    annotations_dir = get_annotations_dir(method, study_uid, series_uid)
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f'frame_{frame_num:06d}_mask.png'
    cv2.imwrite(str(mask_path), mask_array)

    # Save metadata to JSON
    modified_annotations = load_modified_annotations(method, study_uid, series_uid)
    frame_key = f'frame_{frame_num}'
    modified_annotations[frame_key] = {
        'label_id': label_id,
        'is_empty': False,
        'modified_at': datetime.now().isoformat()
    }
    save_modified_annotations(method, study_uid, series_uid, modified_annotations)

    return jsonify({'success': True})

@app.route('/api/save_changes', methods=['POST'])
def api_save_changes():
    """
    Save all label_id and empty_id annotations for a video.
    
    Expected JSON:
        {
            'method': str,
            'study_uid': str,
            'series_uid': str,
            'modified_frames': optional dict of {frame_num: {mask_data: base64, is_empty: bool}}
        }
    """
    data = request.json
    method = data['method']
    study_uid = data['study_uid']
    series_uid = data['series_uid']
    modified_frames_data = data.get('modified_frames', {})  # In-memory frames from frontend
    label_id = os.getenv('LABEL_ID', '')
    empty_id = os.getenv('EMPTY_ID', '')
    
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    annotations_dir = get_annotations_dir(method, study_uid, series_uid)
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing mask data to find all annotated frames
    mask_data_path = video_dir / 'mask_data.json'
    with open(mask_data_path) as f:
        mask_data = json.load(f)
    
    # Load existing modified annotations
    modified_annotations = load_modified_annotations(method, study_uid, series_uid)
    
    # Get video dimensions
    video_path = video_dir / 'tracked_video.mp4'
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if ret:
        height, width = first_frame.shape[:2]
    cap.release()
    
    saved_count = 0
    
    # First, save any in-memory modified frames from the frontend
    for frame_num_str, frame_data in modified_frames_data.items():
        frame_num = int(frame_num_str)
        is_empty = frame_data.get('is_empty', False)
        
        # Decode and save mask
        mask_b64 = frame_data.get('mask_data', '')
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64.split(',')[1] if ',' in mask_b64 else mask_b64)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask_array = np.array(mask_img)
            
            # Save mask to annotations directory
            mask_path = masks_dir / f'frame_{frame_num:06d}_mask.png'
            cv2.imwrite(str(mask_path), mask_array)
            
            # Update metadata
            frame_key = f'frame_{frame_num}'
            modified_annotations[frame_key] = {
                'label_id': empty_id if is_empty else label_id,
                'is_empty': is_empty,
                'modified_at': datetime.now().isoformat()
            }
    
    # Process all frames (including ones just saved above)
    for frame_num in range(total_frames):
        frame_key = f'frame_{frame_num}'
        frame_str = str(frame_num)
        
        # Check if frame has annotation (label_id or empty_id)
        has_annotation = False
        is_empty = False
        frame_label_id = None
        
        # Check in modified_annotations first (takes precedence - these are definitely label_id or empty_id)
        if frame_key in modified_annotations:
            frame_data = modified_annotations[frame_key]
            has_annotation = True
            is_empty = frame_data.get('is_empty', False)
            frame_label_id = frame_data.get('label_id', empty_id if is_empty else label_id)
        # Check in mask_data for original annotations
        elif frame_str in mask_data:
            frame_info = mask_data[frame_str]
            frame_label_id_val = frame_info.get('label_id', '')
            frame_type = frame_info.get('type', '')
            
            # Check if it's a human annotation (label_id) - not a tracked annotation
            if frame_info.get('is_annotation', False):
                has_annotation = True
                frame_label_id = frame_label_id_val if frame_label_id_val else label_id
                # Check if mask is empty
                mask_path = video_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
                if mask_path.exists():
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is not None and np.sum(mask) == 0:
                        is_empty = True
                        frame_label_id = empty_id
            # Also check if label_id explicitly matches our label_id or empty_id
            elif frame_label_id_val in [label_id, empty_id]:
                has_annotation = True
                is_empty = (frame_label_id_val == empty_id)
                frame_label_id = frame_label_id_val
        
        if not has_annotation:
            continue
        
        # Load or create mask
        mask = None
        
        # Try annotations directory first
        annotation_mask_path = masks_dir / f'frame_{frame_num:06d}_mask.png'
        if annotation_mask_path.exists():
            mask = cv2.imread(str(annotation_mask_path), cv2.IMREAD_GRAYSCALE)
        else:
            # Try output directory
            output_mask_path = video_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
            if output_mask_path.exists():
                mask = cv2.imread(str(output_mask_path), cv2.IMREAD_GRAYSCALE)
            elif is_empty:
                # Create empty mask
                mask = np.zeros((height, width), dtype=np.uint8)
        
        if mask is None:
            continue
        
        # Save mask to annotations directory
        cv2.imwrite(str(annotation_mask_path), mask)
        
        # Update metadata
        modified_annotations[frame_key] = {
            'label_id': frame_label_id or (empty_id if is_empty else label_id),
            'is_empty': is_empty,
            'modified_at': datetime.now().isoformat()
        }
        
        saved_count += 1
    
    # Save all metadata at once
    save_modified_annotations(method, study_uid, series_uid, modified_annotations)
    
    return jsonify({
        'success': True,
        'saved_count': saved_count,
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
    annotations_dir = get_annotations_dir(method, study_uid, series_uid)
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    mask_path = masks_dir / f'frame_{frame_num:06d}_mask.png'
    cv2.imwrite(str(mask_path), empty_mask)

    # Save metadata to JSON
    modified_annotations = load_modified_annotations(method, study_uid, series_uid)
    frame_key = f'frame_{frame_num}'
    modified_annotations[frame_key] = {
        'label_id': empty_id,
        'is_empty': True,
        'modified_at': datetime.now().isoformat()
    }
    save_modified_annotations(method, study_uid, series_uid, modified_annotations)

    return jsonify({'success': True})

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
