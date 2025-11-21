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
import duckdb
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
DB_PATH = Path('annotations.duckdb')

# Initialize DuckDB for local annotation storage
def init_db():
    """Initialize DuckDB database for storing modified annotations."""
    conn = duckdb.connect(str(DB_PATH))

    # Create table for modified annotations if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS modified_annotations (
            id VARCHAR PRIMARY KEY,
            study_uid VARCHAR NOT NULL,
            series_uid VARCHAR NOT NULL,
            frame_number INTEGER NOT NULL,
            label_id VARCHAR NOT NULL,
            mask_data BLOB,
            is_empty BOOLEAN DEFAULT FALSE,
            modified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(study_uid, series_uid, frame_number)
        )
    """)

    conn.close()

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

    # Check for local modifications in DuckDB
    conn = duckdb.connect(str(DB_PATH))
    modifications = conn.execute("""
        SELECT frame_number, label_id, is_empty, modified_at
        FROM modified_annotations
        WHERE study_uid = ? AND series_uid = ?
    """, [study_uid, series_uid]).fetchall()
    conn.close()

    modified_frames = {row[0]: {'label_id': row[1], 'is_empty': row[2], 'modified_at': str(row[3])}
                      for row in modifications}

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

    # Get mask if it exists
    mask_path = video_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
    mask_b64 = None
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
    Save a modified mask to local database and update mask file.

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

    # Save mask to file
    video_dir = OUTPUT_DIR / method / f"{study_uid}_{series_uid}"
    mask_path = video_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
    cv2.imwrite(str(mask_path), mask_array)

    # Save to DuckDB
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("""
        INSERT OR REPLACE INTO modified_annotations
        (id, study_uid, series_uid, frame_number, label_id, mask_data, is_empty)
        VALUES (?, ?, ?, ?, ?, ?, FALSE)
    """, [
        f"{study_uid}_{series_uid}_{frame_num}",
        study_uid,
        series_uid,
        frame_num,
        label_id,
        mask_array.tobytes()
    ])
    conn.close()

    return jsonify({'success': True})

@app.route('/api/mark_empty', methods=['POST'])
def api_mark_empty():
    """
    Mark a frame as EMPTY_ID (no fluid).

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

    # Save empty mask
    mask_path = video_dir / 'masks' / f'frame_{frame_num:06d}_mask.png'
    cv2.imwrite(str(mask_path), empty_mask)

    # Save to DuckDB with EMPTY_ID
    conn = duckdb.connect(str(DB_PATH))
    conn.execute("""
        INSERT OR REPLACE INTO modified_annotations
        (id, study_uid, series_uid, frame_number, label_id, mask_data, is_empty)
        VALUES (?, ?, ?, ?, ?, ?, TRUE)
    """, [
        f"{study_uid}_{series_uid}_{frame_num}",
        study_uid,
        series_uid,
        frame_num,
        empty_id,
        empty_mask.tobytes()
    ])
    conn.close()

    return jsonify({'success': True})

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from output directory."""
    return send_file(OUTPUT_DIR / filename)

if __name__ == '__main__':
    # Initialize database
    init_db()

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
