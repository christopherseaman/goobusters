// Goobusters Annotation Viewer - Simplified Based on Teef Reference

class AnnotationViewer {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d', { willReadFrequently: true });

        // Overlay canvas for mask (like teef)
        this.overlayCanvas = document.getElementById('overlayCanvas');
        this.overlayCtx = this.overlayCanvas.getContext('2d', { willReadFrequently: true });

        // Set canvases to full viewport size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Create hidden canvas for mask manipulation (like teef's maskCanvas)
        this.maskCanvas = document.createElement('canvas');
        this.maskCtx = this.maskCanvas.getContext('2d', { willReadFrequently: true });

        // State
        this.currentVideo = INITIAL_VIDEO;
        this.currentFrame = 0;
        this.totalFrames = 0;
        this.isPlaying = false;
        this.playInterval = null;
        this.playbackSpeed = 30;

        this.frameImage = null;
        this.maskImageData = null; // Grayscale ImageData (like singleChannelMask in teef)
        this.originalMaskImageData = null; // For reset (unmodified mask from output/)
        this.savedMaskImageData = null; // Saved mask from annotations/ (if exists)
        this.maskType = null; // 'LABEL_ID' or 'TRACK_ID'
        this.originalMaskType = null;

        this.drawMode = 'draw';
        this.brushSize = 15;
        this.maskOpacity = 0.3;
        this.maskVisible = true; // Mask visibility toggle

        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;

        this.videoData = null;
        this.hasUnsavedChanges = false;
        this.renderRect = null; // For coordinate conversion
        
        // Track all modified frames across the session, per video
        // Structure: Map<videoKey, Map<frame_num, { maskData, maskType, is_empty }>>
        this.modifiedFrames = new Map(); // videoKey -> Map(frame_num -> data)
        
        
        this.frameCache = new Map(); // frameNum -> {frameImage, maskImageData, maskType, originalMaskImageData, originalMaskType, canvasWidth, canvasHeight}
        this.framesArchive = null; // Extracted frames from tar.gz
        this.masksArchive = {}; // Masks from tar.gz archive (webp images)

        this.toolsVisible = true;

        this.init();
    }

    hasUnsavedChangesForCurrentVideo() {
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        return videoModifiedFrames.size > 0 || this.hasUnsavedChanges;
    }

    updateSaveButtonState() {
        const saveBtn = document.getElementById('saveChanges');
        if (this.hasUnsavedChangesForCurrentVideo()) {
            saveBtn.classList.add('unsaved');
            saveBtn.classList.remove('success');
        } else {
            saveBtn.classList.remove('unsaved');
            saveBtn.classList.remove('success');
        }
    }

    resizeCanvas() {
        // Main canvas sized to viewport for rendering
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        // Overlay canvas will be sized to image dimensions in goToFrame()
        if (this.frameImage) {
            this.render();
        }
    }

    async init() {
        this.setupEventListeners();
        await this.loadVideoData();
        await this.loadFramesArchive();
        await this.goToFrame(0);
    }

    setupEventListeners() {
        // Modal controls
        document.getElementById('infoBtn').addEventListener('click', () => this.showModal('infoModal'));
        document.getElementById('navBtn').addEventListener('click', () => this.showModal('navModal'));
        document.getElementById('closeInfo').addEventListener('click', () => this.hideModal('infoModal'));
        document.getElementById('closeNav').addEventListener('click', () => this.hideModal('navModal'));

        // Close modals on background click
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) this.hideModal(modal.id);
            });
        });

        // Video selection
        document.getElementById('videoSelect').addEventListener('change', (e) => {
            const [method, studyUid, seriesUid] = e.target.value.split('|');
            this.loadVideo(method, studyUid, seriesUid);
        });

        document.getElementById('prevVideo').addEventListener('click', () => this.navigateVideo(-1));
        document.getElementById('nextVideo').addEventListener('click', () => this.navigateVideo(1));

        // Playback controls
        document.getElementById('prevFrame').addEventListener('click', () => this.navigateFrame(-1));
        document.getElementById('playPause').addEventListener('click', () => this.togglePlayPause());
        document.getElementById('nextFrame').addEventListener('click', () => this.navigateFrame(1));
        document.getElementById('toggleMask').addEventListener('click', () => this.toggleMaskVisibility());

        // Frame slider
        document.getElementById('frameSlider').addEventListener('input', (e) => {
            this.goToFrame(parseInt(e.target.value));
        });

        // Edit controls
        document.getElementById('drawMode').addEventListener('click', () => this.setDrawMode('draw'));
        document.getElementById('eraseMode').addEventListener('click', () => this.setDrawMode('erase'));
        document.getElementById('markEmpty').addEventListener('click', () => this.markEmpty());
        document.getElementById('saveChanges').addEventListener('click', () => this.saveChanges());
        document.getElementById('resetMask').addEventListener('click', () => this.resetMask());

        // Brush size sliders (both modal and inline)
        const brushSizeModal = document.getElementById('brushSize');
        const brushSizeInline = document.getElementById('brushSizeInline');

        brushSizeModal.addEventListener('input', (e) => {
            this.brushSize = parseInt(e.target.value);
            brushSizeInline.value = this.brushSize;
            document.getElementById('brushSizeValue').textContent = this.brushSize;
            this.updateBrushPreview();
        });

        brushSizeInline.addEventListener('input', (e) => {
            this.brushSize = parseInt(e.target.value);
            brushSizeModal.value = this.brushSize;
            document.getElementById('brushSizeValue').textContent = this.brushSize;
            this.updateBrushPreview();
        });

        // Mask opacity
        document.getElementById('maskOpacity').addEventListener('input', (e) => {
            this.maskOpacity = parseInt(e.target.value) / 100;
            document.getElementById('maskOpacityValue').textContent = e.target.value;
            this.render();
        });

        // Drawing events (mouse)
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => {
            if (this.isDrawing) this.draw(e);
            this.updateBrushPreview(e);
        });
        this.canvas.addEventListener('mouseup', () => this.stopDrawing());
        this.canvas.addEventListener('mouseleave', () => {
            this.stopDrawing();
            this.hideBrushPreview();
        });

        // Drawing events (touch)
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e);
        });
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            if (this.isDrawing) this.draw(e);
        });
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.stopDrawing();
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT') return;

            switch(e.key) {
                case ' ':
                    e.preventDefault();
                    this.togglePlayPause();
                    break;
                case 'ArrowLeft':
                    e.preventDefault();
                    this.navigateFrame(-1);
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    this.navigateFrame(1);
                    break;
                case 'd':
                    this.setDrawMode('draw');
                    break;
                case 'e':
                    this.setDrawMode('erase');
                    break;
                case 's':
                    if (e.metaKey || e.ctrlKey) {
                        e.preventDefault();
                        this.saveChanges();
                    }
                    break;
            }
        });
    }

    updateBrushPreview(e) {
        const preview = document.getElementById('brushPreview');

        // Calculate display scale (mask pixels to screen pixels)
        const scale = this.renderRect ? (this.renderRect.width / this.maskCanvas.width) : 1;
        const displayRadius = this.brushSize * scale;
        const displayDiameter = displayRadius * 2;

        if (e) {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
            const y = (e.clientY || (e.touches && e.touches[0].clientY)) - rect.top;
            // Center the preview on cursor by offsetting by scaled brush radius
            preview.style.left = `${x - displayRadius}px`;
            preview.style.top = `${y - displayRadius}px`;
            preview.style.width = `${displayDiameter}px`;
            preview.style.height = `${displayDiameter}px`;
            preview.style.display = 'block';
        } else {
            preview.style.width = `${displayDiameter}px`;
            preview.style.height = `${displayDiameter}px`;
            preview.style.display = 'block';
            setTimeout(() => preview.style.display = 'none', 1000);
        }
    }



    hideBrushPreview() {
        document.getElementById('brushPreview').style.display = 'none';
    }

    async loadVideoData() {
        const { method, studyUid, seriesUid } = this.currentVideo;
        const response = await fetch(`/api/video/${method}/${studyUid}/${seriesUid}`);
        const data = await response.json();
        this.totalFrames = data.total_frames;
        
        // Build mask_data structure from masks_annotations and modified_frames
        this.videoData = {
            total_frames: data.total_frames,
            mask_data: {},
            method: data.method || method,
            study_uid: data.study_uid || studyUid,
            series_uid: data.series_uid || seriesUid,
            exam_number: data.exam_number || 'Unknown',
            labels: data.labels || []
        };
        
        // Add annotations from masks_annotations
        if (data.masks_annotations) {
            for (const ann of data.masks_annotations) {
                const frameNum = ann.frameNumber;
                this.videoData.mask_data[frameNum] = {
                    type: ann.type || 'unknown',
                    label_id: ann.labelId || ann.label_id || '',
                    is_annotation: ann.is_annotation || false,
                    modified: false
                };
            }
        }
        
        // Override with modified_frames (user modifications take precedence)
        // Modified masks are always human-verified annotations (green), unless they're empty
        if (data.modified_frames) {
            for (const [frameNum, frameData] of Object.entries(data.modified_frames)) {
                const num = parseInt(frameNum);
                if (this.videoData.mask_data[num]) {
                    // Modified frames are always annotations (human-verified)
                    this.videoData.mask_data[num].label_id = frameData.label_id;
                    this.videoData.mask_data[num].modified = true;
                    // Mark as annotation unless it's an empty frame
                    this.videoData.mask_data[num].is_annotation = !frameData.is_empty;
                } else {
                    // New frame (not in original annotations) - mark as annotation since user created it
                    this.videoData.mask_data[num] = {
                        type: frameData.is_empty ? 'empty' : 'fluid',
                        label_id: frameData.label_id,
                        is_annotation: !frameData.is_empty, // User-created frames are annotations (unless empty)
                        modified: true
                    };
                }
            }
        }

        // Set slider range from 0 to totalFrames - 1
        document.getElementById('sliderMin').textContent = '0';
        document.getElementById('sliderMax').textContent = this.totalFrames - 1;
        document.getElementById('frameSlider').min = 0;
        document.getElementById('frameSlider').max = this.totalFrames - 1;
        this.updateFrameCounter();
    }


    async goToFrame(frameNum) {
        const targetFrame = Math.max(0, Math.min(frameNum, this.totalFrames - 1));
        if (targetFrame === this.currentFrame && this.frameImage) return;
        
        // Preserve unsaved changes
        if (this.hasUnsavedChanges && this.maskImageData && this.currentFrame !== targetFrame) {
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            videoModifiedFrames.set(this.currentFrame, {
                maskData: this.cloneImageData(this.maskImageData),
                maskType: this.maskType,
                is_empty: this.maskType === 'empty'
            });
        }
        
        this.currentFrame = targetFrame;
        document.getElementById('frameSlider').value = this.currentFrame;
        this.updateFrameCounter();
        this.updateSaveButtonState();

        // Get from cache or load from archive
        let cached = this.frameCache.get(targetFrame);
        if (!cached) {
            const frameData = this.framesArchive[`frames/frame_${targetFrame.toString().padStart(6, '0')}.webp`];
            if (!frameData) {
                console.error(`Frame ${targetFrame} not found in archive`);
                return;
            }
            const blob = new Blob([frameData], { type: 'image/webp' });
            const frameUrl = URL.createObjectURL(blob);
            const frameImage = await this.loadImage(frameUrl);
            
            // Load mask from JSON (RLE encoded)
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = frameImage.width;
            tempCanvas.height = frameImage.height;
            const tempCtx = tempCanvas.getContext('2d');
            
            // Load current mask (may be modified or original)
            let maskImageData;
            const maskFileName = `masks/frame_${String(targetFrame).padStart(6, '0')}_mask.webp`;
            if (this.masksArchive && this.masksArchive[maskFileName]) {
                // Load mask from archive (webp image) - may be modified or original
                const maskData = this.masksArchive[maskFileName];
                const blob = new Blob([maskData], { type: 'image/webp' });
                const maskUrl = URL.createObjectURL(blob);
                const maskImg = new Image();
                await new Promise((resolve, reject) => {
                    maskImg.onload = () => {
                        tempCtx.drawImage(maskImg, 0, 0);
                        maskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                        URL.revokeObjectURL(maskUrl);
                        resolve();
                    };
                    maskImg.onerror = reject;
                    maskImg.src = maskUrl;
                });
            } else {
                // No mask - create empty
                tempCtx.fillStyle = 'black';
                tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                maskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            }
            
            // Load original unmodified mask from output/ (for reset functionality)
            let originalMaskImageData;
            if (this.originalMasksArchive && this.originalMasksArchive[maskFileName]) {
                const originalMaskData = this.originalMasksArchive[maskFileName];
                const blob = new Blob([originalMaskData], { type: 'image/webp' });
                const maskUrl = URL.createObjectURL(blob);
                const maskImg = new Image();
                await new Promise((resolve, reject) => {
                    maskImg.onload = () => {
                        tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
                        tempCtx.drawImage(maskImg, 0, 0);
                        originalMaskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                        URL.revokeObjectURL(maskUrl);
                        resolve();
                    };
                    maskImg.onerror = reject;
                    maskImg.src = maskUrl;
                });
            } else {
                // No original mask - use empty
                tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
                tempCtx.fillStyle = 'black';
                tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
                originalMaskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
            }
            
            const maskType = this.videoData.mask_data[targetFrame]?.is_annotation ? 'human' : 'tracked';
            // Original mask type is always 'tracked' (from output/) unless it was an annotation
            const originalMaskType = 'tracked';
            
            cached = {
                frameImage,
                maskImageData: this.cloneImageData(maskImageData),
                originalMaskImageData: this.cloneImageData(originalMaskImageData), // Original unmodified mask from output/
                maskType,
                originalMaskType: originalMaskType,
                canvasWidth: frameImage.width,
                canvasHeight: frameImage.height
            };
            this.frameCache.set(targetFrame, cached);
        }

        // Use preprocessed data (instant)
        this.frameImage = cached.frameImage;
        this.maskCanvas.width = cached.canvasWidth;
        this.maskCanvas.height = cached.canvasHeight;
        this.overlayCanvas.width = cached.canvasWidth;
        this.overlayCanvas.height = cached.canvasHeight;
        
        this.maskImageData = this.cloneImageData(cached.maskImageData);
        this.originalMaskImageData = this.cloneImageData(cached.originalMaskImageData);
        this.maskType = cached.maskType;
        this.originalMaskType = cached.originalMaskType;
        
        this.maskCtx.putImageData(this.maskImageData, 0, 0);
        
        // Check if this frame was already modified in this session
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        if (videoModifiedFrames.has(this.currentFrame)) {
            const saved = videoModifiedFrames.get(this.currentFrame);
            this.maskImageData = saved.maskData;
            this.maskType = saved.maskType;
            this.hasUnsavedChanges = true;
        } else {
            this.hasUnsavedChanges = false;
        }

        this.updateSaveButtonState();
        this.render();
        this.updateInfoPanel();
    }

    cloneImageData(imageData) {
        const cloned = this.maskCtx.createImageData(imageData.width, imageData.height);
        cloned.data.set(imageData.data);
        return cloned;
    }

    loadImage(src) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = src;
        });
    }

    render() {
        // Clear both canvases
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);

        if (!this.frameImage) return;

        // Calculate aspect-fit dimensions
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const imageAspect = this.frameImage.width / this.frameImage.height;
        const viewportAspect = viewportWidth / viewportHeight;

        let displayWidth, displayHeight;
        if (imageAspect > viewportAspect) {
            displayWidth = viewportWidth;
            displayHeight = viewportWidth / imageAspect;
        } else {
            displayHeight = viewportHeight;
            displayWidth = viewportHeight * imageAspect;
        }

        const displayX = (viewportWidth - displayWidth) / 2;
        const displayY = (viewportHeight - displayHeight) / 2;

        // Store dimensions for coordinate conversion
        this.renderRect = { x: displayX, y: displayY, width: displayWidth, height: displayHeight };

        // Draw frame on main canvas (scaled to viewport)
        this.ctx.drawImage(this.frameImage, displayX, displayY, displayWidth, displayHeight);

        // Position and size overlay canvas to match display dimensions (like teef adjustCanvasSize)
        this.overlayCanvas.style.width = `${displayWidth}px`;
        this.overlayCanvas.style.height = `${displayHeight}px`;
        this.overlayCanvas.style.left = `${displayX}px`;
        this.overlayCanvas.style.top = `${displayY}px`;

        // Draw mask overlay directly at image resolution (like teef applyMaskToOverlay)
        if (this.maskImageData && this.maskVisible) {
            const isHuman = this.maskType === 'human' || this.maskType === 'empty';
            const maskColor = isHuman ? { r: 0, g: 255, b: 0 } : { r: 255, g: 165, b: 0 };

            // Create overlay imageData at image resolution
            const overlayImageData = this.overlayCtx.createImageData(this.overlayCanvas.width, this.overlayCanvas.height);

            for (let i = 0; i < this.maskImageData.data.length; i += 4) {
                const gray = this.maskImageData.data[i];
                overlayImageData.data[i] = maskColor.r;
                overlayImageData.data[i + 1] = maskColor.g;
                overlayImageData.data[i + 2] = maskColor.b;
                overlayImageData.data[i + 3] = gray > 0 ? Math.round(gray * this.maskOpacity) : 0;
            }

            // Put directly on overlay canvas (no scaling, CSS handles that)
            this.overlayCtx.putImageData(overlayImageData, 0, 0);
        }
    }

    getCanvasCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
        const y = (e.clientY || (e.touches && e.touches[0].clientY)) - rect.top;

        // Convert to mask coordinates
        if (!this.renderRect) return { x: 0, y: 0 };

        const maskX = ((x - this.renderRect.x) / this.renderRect.width) * this.maskCanvas.width;
        const maskY = ((y - this.renderRect.y) / this.renderRect.height) * this.maskCanvas.height;

        return { x: Math.floor(maskX), y: Math.floor(maskY) };
    }

    startDrawing(e) {
        this.isDrawing = true;
        const coords = this.getCanvasCoordinates(e);
        this.lastX = coords.x;
        this.lastY = coords.y;

        // Convert tracked masks to human annotation on first edit
        if (this.maskType === 'tracked') {
            console.log('CONVERTING tracked to human on edit');
            this.maskType = 'human';
            this.render(); // Show color change immediately
        }

        // Remove empty marker when drawing
        if (this.maskType === 'empty') {
            console.log('REMOVING empty marker on draw');
            this.maskType = 'human';
            this.render(); // Show color change immediately
        }

        console.log('Start drawing - Mode:', this.drawMode, 'Type:', this.maskType);

        this.draw(e);
    }

    draw(e) {
        if (!this.isDrawing) return;

        const coords = this.getCanvasCoordinates(e);
        this.drawLine(this.lastX, this.lastY, coords.x, coords.y);
        this.lastX = coords.x;
        this.lastY = coords.y;

        this.hasUnsavedChanges = true;
        
        // Store in modified frames map (in memory only, no disk save)
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        videoModifiedFrames.set(this.currentFrame, {
            maskData: this.cloneImageData(this.maskImageData),
            maskType: this.maskType,
            is_empty: false
        });
        
        this.updateSaveButtonState();
        this.render();
    }

    // Bresenham's line algorithm (like teef)
    drawLine(x0, y0, x1, y1) {
        const dx = Math.abs(x1 - x0);
        const dy = Math.abs(y1 - y0);
        const sx = x0 < x1 ? 1 : -1;
        const sy = y0 < y1 ? 1 : -1;
        let err = dx - dy;

        while (true) {
            this.drawPoint(x0, y0);
            if (x0 === x1 && y0 === y1) break;
            const e2 = 2 * err;
            if (e2 > -dy) { err -= dy; x0 += sx; }
            if (e2 < dx) { err += dx; y0 += sy; }
        }
    }

    // Draw circular brush (like teef)
    drawPoint(x, y) {
        const value = this.drawMode === 'draw' ? 255 : 0;

        for (let dx = -this.brushSize; dx <= this.brushSize; dx++) {
            for (let dy = -this.brushSize; dy <= this.brushSize; dy++) {
                if (dx*dx + dy*dy <= this.brushSize*this.brushSize) {
                    const cx = x + dx;
                    const cy = y + dy;
                    if (cx >= 0 && cx < this.maskCanvas.width && cy >= 0 && cy < this.maskCanvas.height) {
                        const i = (cy * this.maskCanvas.width + cx) * 4;
                        this.maskImageData.data[i] = value;
                        this.maskImageData.data[i + 1] = value;
                        this.maskImageData.data[i + 2] = value;
                        this.maskImageData.data[i + 3] = 255;
                    }
                }
            }
        }
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    setDrawMode(mode) {
        this.drawMode = mode;
        document.getElementById('drawMode').classList.toggle('active', mode === 'draw');
        document.getElementById('eraseMode').classList.toggle('active', mode === 'erase');
    }

    async saveChanges() {
        // Save current frame if it has unsaved changes
        if (this.hasUnsavedChanges) {
            // Store in modified frames map
            this.maskCtx.putImageData(this.maskImageData, 0, 0);
            const maskData = this.maskCanvas.toDataURL('image/png');
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            videoModifiedFrames.set(this.currentFrame, {
                maskData: this.cloneImageData(this.maskImageData),
                maskType: this.maskType,
                is_empty: this.maskType === 'empty'
            });
        }

        const { method, studyUid, seriesUid } = this.currentVideo;

        try {
            // Prepare modified frames data for the API (only for current video)
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            const modifiedFramesData = {};
            for (const [frameNum, frameData] of videoModifiedFrames.entries()) {
                this.maskCtx.putImageData(frameData.maskData, 0, 0);
                const maskData = this.maskCanvas.toDataURL('image/png');
                modifiedFramesData[frameNum] = {
                    mask_data: maskData,
                    is_empty: frameData.is_empty
                };
            }

            // Save ALL label_id and empty_id annotations for the entire video
            // (including in-memory modified frames)
            const allResponse = await fetch('/api/save_changes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    method,
                    study_uid: studyUid,
                    series_uid: seriesUid,
                    modified_frames: modifiedFramesData
                })
            });

            if (allResponse.ok) {
                const result = await allResponse.json();
                console.log(`✅ Saved all annotations: ${result.saved_count} frame(s) total`);
                
                // Clear modified frames for current video and reset current frame state
                const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
                videoModifiedFrames.clear();
                this.hasUnsavedChanges = false;
                this.originalMaskImageData = this.cloneImageData(this.maskImageData);
                
                // Show success state briefly, then return to default (no fill)
                const saveBtn = document.getElementById('saveChanges');
                saveBtn.classList.remove('unsaved');
                saveBtn.classList.add('success');
                setTimeout(() => {
                    saveBtn.classList.remove('success');
                    this.updateSaveButtonState(); // Ensure correct state after timeout
                }, 2000);
                
                // Reload video data to get updated annotations
                await this.loadVideoData();
                
                // Reload masks archive to get updated saved masks
                await this.loadFramesArchive();
                
                // Update cache for saved frames with the newly saved masks
                for (const frameNumStr of Object.keys(modifiedFramesData)) {
                    const frameNum = parseInt(frameNumStr);
                    const cached = this.frameCache.get(frameNum);
                    if (cached) {
                        // Update cache with saved mask from archive
                        const maskFileName = `masks/frame_${String(frameNum).padStart(6, '0')}_mask.webp`;
                        if (this.masksArchive && this.masksArchive[maskFileName]) {
                            const tempCanvas = document.createElement('canvas');
                            tempCanvas.width = cached.canvasWidth;
                            tempCanvas.height = cached.canvasHeight;
                            const tempCtx = tempCanvas.getContext('2d');
                            
                            const maskData = this.masksArchive[maskFileName];
                            const blob = new Blob([maskData], { type: 'image/webp' });
                            const maskUrl = URL.createObjectURL(blob);
                            const maskImg = new Image();
                            await new Promise((resolve, reject) => {
                                maskImg.onload = () => {
                                    tempCtx.drawImage(maskImg, 0, 0);
                                    const savedMaskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
                                    URL.revokeObjectURL(maskUrl);
                                    
                                    // Update cache with saved mask
                                    cached.maskImageData = this.cloneImageData(savedMaskImageData);
                                    cached.maskType = this.videoData.mask_data[frameNum]?.is_annotation ? 'human' : 'tracked';
                                    
                                    resolve();
                                };
                                maskImg.onerror = reject;
                                maskImg.src = maskUrl;
                            });
                        }
                    }
                }
                
                // Update current frame display if it was saved
                if (modifiedFramesData[this.currentFrame.toString()]) {
                    const cached = this.frameCache.get(this.currentFrame);
                    if (cached) {
                        this.maskImageData = this.cloneImageData(cached.maskImageData);
                        this.maskType = cached.maskType;
                        this.render();
                    }
                }
            } else {
                console.error('❌ Failed to save all annotations');
            }
        } catch (error) {
            console.error('Error saving:', error);
        }
    }

    resetMask() {
        // Reset to the original unmodified mask from output/ (not the saved modified mask)
        this.maskImageData = this.cloneImageData(this.originalMaskImageData);
        this.maskType = this.originalMaskType;
        
        // Check if there was a saved modified mask for this frame
        const frameData = this.videoData?.mask_data?.[this.currentFrame];
        const hadSavedModification = frameData?.modified === true;
        
        // If there was a saved modification, resetting is a new modification (unsaved)
        // If there was no saved modification, resetting clears any unsaved changes
        if (hadSavedModification) {
            // Resetting to original is a new modification (different from saved state)
            this.hasUnsavedChanges = true;
            // Store in modified frames so it will be saved
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            videoModifiedFrames.set(this.currentFrame, {
                maskData: this.cloneImageData(this.maskImageData),
                maskType: this.maskType,
                is_empty: this.maskType === 'empty'
            });
        } else {
            // No saved modification, so resetting clears unsaved changes
            this.hasUnsavedChanges = false;
            // Remove from modified frames if it was there
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            videoModifiedFrames.delete(this.currentFrame);
        }
        
        this.updateSaveButtonState();
        this.render();
    }

    markEmpty() {
        // Clear the mask (set all pixels to black)
        for (let i = 0; i < this.maskImageData.data.length; i += 4) {
            this.maskImageData.data[i] = 0;
            this.maskImageData.data[i + 1] = 0;
            this.maskImageData.data[i + 2] = 0;
            this.maskImageData.data[i + 3] = 255;
        }

        // Mark as empty type (clears label_id & track_id, adds empty_id annotation)
        this.maskType = 'empty';
        this.hasUnsavedChanges = true;
        
        // Store in modified frames map (in memory only, will be saved when user clicks Save)
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        videoModifiedFrames.set(this.currentFrame, {
            maskData: this.cloneImageData(this.maskImageData),
            maskType: 'empty',
            is_empty: true
        });
        
        this.updateSaveButtonState();
        this.render();
    }

    navigateFrame(delta) {
        const newFrame = this.currentFrame + delta;
        if (newFrame >= 0 && newFrame < this.totalFrames) {
            this.goToFrame(newFrame);
        }
    }

    togglePlayPause() {
        this.isPlaying = !this.isPlaying;
        document.getElementById('playPause').textContent = this.isPlaying ? '⏸️' : '▶️';
        if (this.isPlaying) {
            this.playInterval = setInterval(() => {
                if (this.currentFrame < this.totalFrames - 1) {
                    this.goToFrame(this.currentFrame + 1);
                } else {
                    this.togglePlayPause();
                }
            }, 1000 / this.playbackSpeed);
        } else {
            if (this.playInterval) {
                clearInterval(this.playInterval);
                this.playInterval = null;
            }
        }
    }

    toggleMaskVisibility() {
        this.maskVisible = !this.maskVisible;
        const btn = document.getElementById('toggleMask');
        // Toggle between disguised face (visible) and sunglasses (hidden) icons
        btn.textContent = this.maskVisible ? '🥸' : '😎';
        btn.classList.toggle('active', this.maskVisible);
        btn.title = this.maskVisible ? 'Hide Mask' : 'Show Mask';
        this.render(); // Re-render to show/hide mask
    }

    navigateVideo(delta) {
        const select = document.getElementById('videoSelect');
        const newIndex = select.selectedIndex + delta;
        if (newIndex >= 0 && newIndex < select.options.length) {
            select.selectedIndex = newIndex;
            const [method, studyUid, seriesUid] = select.value.split('|');
            this.loadVideo(method, studyUid, seriesUid);
        }
    }

    getVideoKey(method, studyUid, seriesUid) {
        return `${method}|${studyUid}|${seriesUid}`;
    }

    getModifiedFramesForCurrentVideo() {
        const videoKey = this.getVideoKey(this.currentVideo.method, this.currentVideo.studyUid, this.currentVideo.seriesUid);
        if (!this.modifiedFrames.has(videoKey)) {
            this.modifiedFrames.set(videoKey, new Map());
        }
        return this.modifiedFrames.get(videoKey);
    }

    async loadVideo(method, studyUid, seriesUid) {
        this.frameImage = null;
        this.frameCache.clear();
        this.framesArchive = null;
        this.masksArchive = {};
        this.currentVideo = { method, studyUid, seriesUid };
        // Load video data first so we know which frames are modified
        await this.loadVideoData();
        await this.loadFramesArchive();
        this.currentFrame = -1;
        await this.goToFrame(0);
        this.updateSaveButtonState();
        this.updateVideoInfo();
    }
    
    async loadFramesArchive() {
        const loadingDiv = document.getElementById('loadingIndicator') || (() => {
            const d = document.createElement('div');
            d.id = 'loadingIndicator';
            d.className = 'loading-indicator';
            d.innerHTML = '<div class="loading-content"><div class="loading-spinner">⏳</div></div>';
            document.body.appendChild(d);
            return d;
        })();
        loadingDiv.style.display = 'flex';
        
        try {
            const { method, studyUid, seriesUid } = this.currentVideo;
            const response = await fetch(`/api/frames/${method}/${studyUid}/${seriesUid}`);
            const { frames_archive_url, masks_archive_url, modified_masks_archive_url } = await response.json();
            
            // Load frames archive
            const framesArchiveResponse = await fetch(frames_archive_url);
            const framesArrayBuffer = await framesArchiveResponse.arrayBuffer();
            const framesDecompressed = fflate.gunzipSync(new Uint8Array(framesArrayBuffer));
            
            // Parse tar and extract frames
            this.framesArchive = {};
            let offset = 0;
            while (offset < framesDecompressed.length) {
                if (offset + 512 > framesDecompressed.length) break;
                
                const name = new TextDecoder().decode(framesDecompressed.slice(offset, offset + 100)).replace(/\0/g, '');
                if (!name) break;
                
                const size = parseInt(new TextDecoder().decode(framesDecompressed.slice(offset + 124, offset + 136)), 8);
                offset += 512;
                
                // Skip directory entries (size == 0) and only process frame files
                if (name.endsWith('.webp') && name.includes('frame_') && !name.includes('_mask') && size > 0) {
                    const data = framesDecompressed.slice(offset, offset + size);
                    // Store with full path from tar (e.g., "frames/frame_000000.webp")
                    this.framesArchive[name] = data;
                }
                
                offset += Math.ceil(size / 512) * 512;
            }
            
            // Load masks archive from output/ (all frames - original unmodified masks)
            this.masksArchive = {}; // Current masks (may include modified)
            this.originalMasksArchive = {}; // Original unmodified masks from output/
            if (masks_archive_url) {
                const masksArchiveResponse = await fetch(masks_archive_url);
                const masksArrayBuffer = await masksArchiveResponse.arrayBuffer();
                const masksDecompressed = fflate.gunzipSync(new Uint8Array(masksArrayBuffer));
                
                // Parse tar and extract masks
                offset = 0;
                while (offset < masksDecompressed.length) {
                    if (offset + 512 > masksDecompressed.length) break;
                    
                    const name = new TextDecoder().decode(masksDecompressed.slice(offset, offset + 100)).replace(/\0/g, '');
                    if (!name) break;
                    
                    const size = parseInt(new TextDecoder().decode(masksDecompressed.slice(offset + 124, offset + 136)), 8);
                    offset += 512;
                    
                    // Skip directory entries (size == 0) and only process mask files
                    if (name.endsWith('.webp') && name.includes('_mask') && size > 0) {
                        const data = masksDecompressed.slice(offset, offset + size);
                        // Store original unmodified masks
                        this.originalMasksArchive[name] = data;
                        // Also store in current masks (will be overridden by modified if exists)
                        this.masksArchive[name] = data;
                    }
                    
                    offset += Math.ceil(size / 512) * 512;
                }
            }
            
            // Load modified masks archive from annotations/ and override only modified frames
            // Use modified_frames from videoData to know which frames to override
            if (modified_masks_archive_url && this.videoData && this.videoData.mask_data) {
                // Find which frames are modified
                const modifiedFrameNumbers = [];
                for (const [frameNum, frameData] of Object.entries(this.videoData.mask_data)) {
                    if (frameData.modified) {
                        modifiedFrameNumbers.push(parseInt(frameNum));
                    }
                }
                
                if (modifiedFrameNumbers.length > 0) {
                    const modifiedMasksArchiveResponse = await fetch(modified_masks_archive_url);
                    const modifiedMasksArrayBuffer = await modifiedMasksArchiveResponse.arrayBuffer();
                    const modifiedMasksDecompressed = fflate.gunzipSync(new Uint8Array(modifiedMasksArrayBuffer));
                    
                    // Parse tar and extract only modified frame masks (frame-level override)
                    offset = 0;
                    while (offset < modifiedMasksDecompressed.length) {
                        if (offset + 512 > modifiedMasksDecompressed.length) break;
                        
                        const name = new TextDecoder().decode(modifiedMasksDecompressed.slice(offset, offset + 100)).replace(/\0/g, '');
                        if (!name) break;
                        
                        const size = parseInt(new TextDecoder().decode(modifiedMasksDecompressed.slice(offset + 124, offset + 136)), 8);
                        offset += 512;
                        
                        // Extract frame number from filename (e.g., "masks/frame_000002_mask.webp" -> 2)
                        const frameMatch = name.match(/frame_(\d+)_mask\.webp/);
                        if (frameMatch) {
                            const frameNum = parseInt(frameMatch[1]);
                            // Only override if this frame is in the modified frames list
                            if (modifiedFrameNumbers.includes(frameNum) && size > 0) {
                                const data = modifiedMasksDecompressed.slice(offset, offset + size);
                                // Override this specific frame's mask
                                this.masksArchive[name] = data;
                            }
                        }
                        
                        offset += Math.ceil(size / 512) * 512;
                    }
                }
            }
        } finally {
            loadingDiv.style.display = 'none';
        }
    }

    showModal(modalId) {
        document.getElementById(modalId).classList.add('active');
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }

    updateVideoInfo() {
        // Update video-level info (study, series, method, exam)
        if (this.videoData) {
            const infoStudy = document.getElementById('infoStudy');
            const infoSeries = document.getElementById('infoSeries');
            const infoMethod = document.getElementById('infoMethod');
            const infoExam = document.getElementById('infoExam');
            
            if (infoStudy) infoStudy.textContent = this.videoData.study_uid || this.currentVideo.studyUid;
            if (infoSeries) infoSeries.textContent = this.videoData.series_uid || this.currentVideo.seriesUid;
            if (infoMethod) infoMethod.textContent = this.videoData.method || this.currentVideo.method;
            if (infoExam) infoExam.textContent = this.videoData.exam_number || 'Unknown';
        }
    }
    
    updateFrameCounter() {
        const frameCounter = document.getElementById('frameCounter');
        if (frameCounter && this.totalFrames) {
            frameCounter.textContent = `${this.currentFrame} / ${this.totalFrames - 1}`;
        }
    }
    
    updateInfoPanel() {
        const frameMetadata = this.videoData?.mask_data?.[this.currentFrame];
        document.getElementById('frameInfo').textContent = `${this.currentFrame} / ${this.totalFrames - 1}`;
        document.getElementById('frameType').textContent = frameMetadata?.type || '-';
        document.getElementById('frameLabelId').textContent = frameMetadata?.label_id || '-';
        document.getElementById('frameModified').textContent = frameMetadata?.modified ? 'Yes ✓' : 'No';
    }
}

// Initialize viewer when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new AnnotationViewer();
});
