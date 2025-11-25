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

    # Load masks annotations
    masks_annotations_path = video_dir / 'masks.json'
    masks_annotations = []
    if masks_annotations_path.exists():
        with open(masks_annotations_path) as f:
            masks_annotations = json.load(f)

    # Check for local modifications in JSON
    modified_annotations = load_modified_annotations(study_uid, series_uid)
    modified_frames = {}
    for frame_key, frame_data in modified_annotations.items():
        frame_num = int(frame_key.replace('frame_', ''))
        modified_frames[frame_num] = {
            'label_id': frame_data.get('label_id', ''),
            'is_empty': frame_data.get('is_empty', False),
            'modified_at': frame_data.get('modified_at', '')
        }

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
    masks_archive_url = f'/api/masks_archive/{method}/{study_uid}/{series_uid}'
    
    # Check if modified masks exist - we'll serve both, frontend will merge
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    modified_masks_archive = annotations_dir / 'masks.tar.gz'
    modified_masks_archive_url = None
    if modified_masks_archive.exists():
        modified_masks_archive_url = f'/api/masks_archive_modified/{study_uid}/{series_uid}'
    
    return jsonify({
        'frames_archive_url': frames_archive_url,
        'masks_archive_url': masks_archive_url,
        'modified_masks_archive_url': modified_masks_archive_url,  # Optional, for merging
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
    """Serve masks archive from output/ (tar.gz)."""
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
    annotations_dir = get_annotations_dir(study_uid, series_uid)
    annotations_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing modified annotations
    modified_annotations = load_modified_annotations(study_uid, series_uid)
    
    # Get video dimensions
    video_path = video_dir / 'tracked_video.mp4'
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, first_frame = cap.read()
    if ret:
        height, width = first_frame.shape[:2]
    cap.release()
    
    saved_count = 0
    
    # Save masks directory
    masks_dir = annotations_dir / 'masks'
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Only save modified frames from the frontend (user modifications)
    # Do NOT copy all frames from masks.json - only save what the user actually modified
    for frame_num_str, frame_data in modified_frames_data.items():
        frame_num = int(frame_num_str)
        is_empty = frame_data.get('is_empty', False)
        
        # Decode and save mask as webp
        mask_b64 = frame_data.get('mask_data', '')
        if mask_b64:
            mask_bytes = base64.b64decode(mask_b64.split(',')[1] if ',' in mask_b64 else mask_b64)
            mask_img = Image.open(io.BytesIO(mask_bytes)).convert('L')
            mask_array = np.array(mask_img)
            
            mask_path = masks_dir / f'frame_{frame_num:06d}_mask.webp'
            cv2.imwrite(str(mask_path), mask_array, [cv2.IMWRITE_WEBP_QUALITY, 85])
            
            # Update metadata
            frame_key = f'frame_{frame_num}'
            modified_annotations[frame_key] = {
                'label_id': empty_id if is_empty else label_id,
                'is_empty': is_empty,
                'modified_at': datetime.now().isoformat()
            }
            saved_count += 1
        elif is_empty:
            # Empty frame - create empty mask
            empty_mask = np.zeros((height, width), dtype=np.uint8)
            mask_path = masks_dir / f'frame_{frame_num:06d}_mask.webp'
            cv2.imwrite(str(mask_path), empty_mask, [cv2.IMWRITE_WEBP_QUALITY, 85])
            
            frame_key = f'frame_{frame_num}'
            modified_annotations[frame_key] = {
                'label_id': empty_id,
                'is_empty': True,
                'modified_at': datetime.now().isoformat()
            }
            saved_count += 1
    
    # Create masks.tar.gz archive
    import tarfile
    masks_archive = annotations_dir / 'masks.tar.gz'
    with tarfile.open(masks_archive, 'w:gz') as tar:
        tar.add(masks_dir, arcname='masks')
    
    # Save all metadata at once
    save_modified_annotations(study_uid, series_uid, modified_annotations)
    
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
