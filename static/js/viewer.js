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
        // currentVideo will be set by loadNextSeries() on init
        this.currentVideo = null;
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
        
        // Navigation history for Prev button (server-driven selection)
        this.videoHistory = []; // Stack of {method, studyUid, seriesUid} for prev navigation
        
        this.frameCache = new Map(); // frameNum -> {frameImage, maskImageData, maskType, originalMaskImageData, originalMaskType, canvasWidth, canvasHeight}
        this.framesArchive = null; // Extracted frames from tar
        this.masksArchive = {}; // Masks from tar archive (webp images)
        this.currentVersionId = null; // Track version ID from server for optimistic locking
        this.activityPingInterval = null; // Interval ID for activity pings (30s)

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
        // Load initial video from server's "next" selection
        try {
            await this.loadNextSeries();
        } catch (error) {
            console.error('Failed to load initial series from server:', error);
            // Fallback to INITIAL_VIDEO if defined (legacy support)
            if (typeof INITIAL_VIDEO !== 'undefined' && INITIAL_VIDEO && INITIAL_VIDEO.studyUid) {
                console.warn('Falling back to INITIAL_VIDEO from template');
                await this.loadVideo(INITIAL_VIDEO.method, INITIAL_VIDEO.studyUid, INITIAL_VIDEO.seriesUid);
            } else {
                alert('Failed to load series. Please check server connection.');
            }
        }
        this.checkDatasetSyncStatus().catch(() => {});
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

        // Completion modal
        const closeComplete = document.getElementById('closeComplete');
        const cancelComplete = document.getElementById('cancelComplete');
        const confirmComplete = document.getElementById('confirmComplete');
        if (closeComplete) closeComplete.addEventListener('click', () => this.hideModal('completeModal'));
        if (cancelComplete) cancelComplete.addEventListener('click', () => this.hideModal('completeModal'));
        if (confirmComplete) confirmComplete.addEventListener('click', () => this.confirmCompleteSeries());

        // Conflict modal
        const closeConflict = document.getElementById('closeConflict');
        const cancelConflict = document.getElementById('cancelConflict');
        const resetAndReload = document.getElementById('resetAndReload');
        if (closeConflict) closeConflict.addEventListener('click', () => this.hideModal('conflictModal'));
        if (cancelConflict) cancelConflict.addEventListener('click', () => this.hideModal('conflictModal'));
        if (resetAndReload) resetAndReload.addEventListener('click', () => this.handleResetAndReload());

        // Video selection (server-driven)
        // Keep videoSelect for manual selection if needed, but wire Next/Prev to server
        const videoSelect = document.getElementById('videoSelect');
        if (videoSelect) {
            videoSelect.addEventListener('change', (e) => {
                const [method, studyUid, seriesUid] = e.target.value.split('|');
                this.loadVideo(method, studyUid, seriesUid);
            });
        }

        document.getElementById('prevVideo').addEventListener('click', () => this.navigateVideoPrev());
        document.getElementById('nextVideo').addEventListener('click', () => this.navigateVideoNext());

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
        document.getElementById('retrackBtn').addEventListener('click', () => this.retrackVideo());
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
        if (!this.currentVideo) {
            throw new Error('No current video set');
        }
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
        // Modified masks are always human-verified annotations (both fluid AND empty)
        if (data.modified_frames) {
            for (const [frameNum, frameData] of Object.entries(data.modified_frames)) {
                const num = parseInt(frameNum);
                if (this.videoData.mask_data[num]) {
                    // Modified frames are always annotations (human-verified)
                    this.videoData.mask_data[num].label_id = frameData.label_id;
                    this.videoData.mask_data[num].modified = true;
                    this.videoData.mask_data[num].is_annotation = true;  // All user edits are annotations
                    this.videoData.mask_data[num].type = frameData.is_empty ? 'empty' : 'fluid';
                } else {
                    // New frame (not in original annotations) - mark as annotation since user created it
                    this.videoData.mask_data[num] = {
                        type: frameData.is_empty ? 'empty' : 'fluid',
                        label_id: frameData.label_id,
                        is_annotation: true,  // All user edits are annotations
                        modified: true
                    };
                }
            }
        }

        // If metadata from masks archive later provides richer per-frame info,
        // it will call updateMaskDataFromMetadata() to refine mask_data.

        // Set slider range from 0 to totalFrames - 1
        document.getElementById('sliderMin').textContent = '0';
        document.getElementById('sliderMax').textContent = this.totalFrames - 1;
        document.getElementById('frameSlider').min = 0;
        document.getElementById('frameSlider').max = this.totalFrames - 1;
        this.updateFrameCounter();
    }

    updateMaskDataFromMetadata(metadata) {
        if (!metadata || !Array.isArray(metadata.frames)) {
            return;
        }

        if (!this.videoData) {
            this.videoData = {
                total_frames: metadata.frame_count || 0,
                mask_data: {},
                method: metadata.flow_method || this.currentVideo.method,
                study_uid: metadata.study_uid,
                series_uid: metadata.series_uid,
                exam_number: 'Unknown',
                labels: this.videoData?.labels || [],
            };
        }

        for (const entry of metadata.frames) {
            const frameNum = entry.frame_number;
            if (frameNum === undefined || frameNum === null) continue;

            const hasMask = !!entry.has_mask;
            const labelId = entry.label_id;
            const isAnnotation = !!entry.is_annotation;

            // Derive a simple type from is_annotation + has_mask:
            // - annotation + mask   -> fluid (human)
            // - annotation + no mask -> empty
            // - non-annotation + mask -> tracked
            // - non-annotation + no mask -> empty (should be rare)
            let type = 'tracked';
            if (isAnnotation && hasMask) {
                type = 'fluid';
            } else if (isAnnotation && !hasMask) {
                type = 'empty';
            } else if (!hasMask) {
                type = 'empty';
            } else {
                type = 'tracked';
            }

            this.videoData.mask_data[frameNum] = {
                type,
                label_id: labelId,
                is_annotation: isAnnotation,
                has_mask: hasMask,
                modified: false,
            };
        }
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
            // Server mask archives store files as "frame_XXXXXX_mask.webp" at root
            const maskFileName = `frame_${String(targetFrame).padStart(6, '0')}_mask.webp`;
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
            
            // Derive mask type from per-frame metadata where available
            const originalMaskImageData = this.cloneImageData(maskImageData);
            const frameMeta = this.videoData?.mask_data?.[targetFrame];
            let maskType = 'tracked';
            if (frameMeta) {
                if (frameMeta.type === 'empty') {
                    maskType = 'empty';
                } else if (frameMeta.type === 'fluid') {
                    maskType = 'human';
                } else if (frameMeta.type === 'tracked') {
                    maskType = 'tracked';
                }
            } else {
                const hasMask = !!(this.masksArchive && this.masksArchive[maskFileName]);
                maskType = hasMask ? 'tracked' : 'empty';
            }
            const originalMaskType = maskType;
            
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

    // Helper: Convert canvas ImageData to WebP blob
    async imageDataToWebP(imageData, width, height) {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
        return new Promise((resolve) => {
            canvas.toBlob(resolve, 'image/webp', 0.95);
        });
    }

    // Helper: Build .tar archive from masks and metadata
    async buildMaskArchive(modifiedFrames, studyUid, seriesUid, flowMethod) {
        const frames = [];
        const maskBlobs = [];
        
        // Get label IDs from video data
        const labelId = this.videoData?.labels?.find(l => l.labelName === 'Fluid')?.labelId || '';
        const emptyId = this.videoData?.labels?.find(l => l.labelName === 'Empty')?.labelId || '';
        
        for (const [frameNum, frameData] of modifiedFrames.entries()) {
            const frameNumber = parseInt(frameNum);
            const isEmpty = frameData.is_empty;
            const filename = `frame_${String(frameNumber).padStart(6, '0')}_mask.webp`;
            
            // Convert mask ImageData to WebP blob
            const webpBlob = await this.imageDataToWebP(
                frameData.maskData,
                this.maskCanvas.width,
                this.maskCanvas.height
            );
            maskBlobs.push({ filename, blob: webpBlob });
            
            frames.push({
                frame_number: frameNumber,
                has_mask: !isEmpty,
                is_annotation: true, // All user edits are annotations
                label_id: isEmpty ? emptyId : labelId,
                filename: isEmpty ? null : filename
            });
        }
        
        // Build metadata.json
        const metadata = {
            study_uid: studyUid,
            series_uid: seriesUid,
            version_id: null, // Will be set by server
            flow_method: flowMethod,
            generated_at: new Date().toISOString(),
            frame_count: frames.length,
            mask_count: frames.filter(f => f.has_mask).length,
            frames: frames
        };
        
        // Build .tar archive (no gzip)
        const tarData = new Uint8Array(1024 * 1024); // 1MB buffer, grow if needed
        let offset = 0;
        
        // Helper: Add file to tar
        const addFile = (name, data) => {
            const nameBytes = new TextEncoder().encode(name);
            const nameLen = Math.min(nameBytes.length, 100);
            tarData.set(nameBytes.slice(0, nameLen), offset);
            offset += 100;
            
            // File mode, uid, gid, size, mtime (simplified)
            const size = data.length;
            const sizeStr = size.toString(8).padStart(11, '0');
            const sizeBytes = new TextEncoder().encode(sizeStr);
            tarData.set(sizeBytes, offset + 124);
            offset += 512;
            
            // File data
            if (size > 0) {
                tarData.set(data, offset);
                offset += Math.ceil(size / 512) * 512;
            }
        };
        
        // Add metadata.json
        const metadataJson = JSON.stringify(metadata, null, 2);
        const metadataBytes = new TextEncoder().encode(metadataJson);
        addFile('metadata.json', metadataBytes);
        
        // Add mask files
        for (const { filename, blob } of maskBlobs) {
            const arrayBuffer = await blob.arrayBuffer();
            addFile(filename, new Uint8Array(arrayBuffer));
        }
        
        // Pad to 512-byte boundary
        const remainder = offset % 512;
        if (remainder > 0) {
            offset += 512 - remainder;
        }
        
        return tarData.slice(0, offset);
    }

    async saveChanges() {
        // Save current frame if it has unsaved changes
        if (this.hasUnsavedChanges) {
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            videoModifiedFrames.set(this.currentFrame, {
                maskData: this.cloneImageData(this.maskImageData),
                maskType: this.maskType,
                is_empty: this.maskType === 'empty'
            });
        }

        const { method, studyUid, seriesUid } = this.currentVideo;
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        
        if (videoModifiedFrames.size === 0) {
            console.log('No changes to save');
            return;
        }

        try {
            // Prepare modified_frames payload with PNG data URLs
            const modifiedFramesPayload = {};
            for (const [frameNum, frameData] of videoModifiedFrames.entries()) {
                const frameNumber = parseInt(frameNum);
                // Render mask to offscreen canvas to get PNG data URL
                const offscreen = document.createElement('canvas');
                offscreen.width = this.maskCanvas.width;
                offscreen.height = this.maskCanvas.height;
                const offCtx = offscreen.getContext('2d');
                offCtx.putImageData(frameData.maskData, 0, 0);
                const pngDataUrl = offscreen.toDataURL('image/png');

                modifiedFramesPayload[frameNumber] = {
                    mask_data: pngDataUrl,
                    is_empty: frameData.is_empty || frameData.maskType === 'empty'
                };
            }

            const payload = {
                method,
                study_uid: studyUid,
                series_uid: seriesUid,
                previous_version_id: this.currentVersionId || '',
                modified_frames: modifiedFramesPayload
            };

            const allResponse = await fetch('/api/save_changes', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (allResponse.ok) {
                const result = await allResponse.json();
                console.log(`✅ Saved and queued retrack: ${result.version_id}`);

                // Update version ID for next save
                this.currentVersionId = result.version_id;

                // Clear modified frames
                videoModifiedFrames.clear();
                this.hasUnsavedChanges = false;

                // Show success state
                const saveBtn = document.getElementById('saveChanges');
                saveBtn.classList.remove('unsaved');
                saveBtn.classList.add('success');
                setTimeout(() => {
                    saveBtn.classList.remove('success');
                    this.updateSaveButtonState();
                }, 2000);

                // Poll retrack status and reload when complete
                if (result.retrack_queued) {
                    await this.pollRetrackStatus(studyUid, seriesUid);
                }
            } else {
                const error = await allResponse.json().catch(() => ({ error: 'Save failed' }));
                console.error('❌ Failed to save:', error);
                
                // Handle 409 conflicts with specific error codes
                if (allResponse.status === 409) {
                    if (error.error_code === 'VERSION_MISMATCH') {
                        this.showConflictModal(
                            'Version Mismatch',
                            error.message || 'Someone else edited this series. Your changes conflict with the server version.',
                            `Your version: ${error.your_version || 'unknown'}\nServer version: ${error.current_version || 'unknown'}`
                        );
                    } else if (error.error_code === 'RETRACK_IN_PROGRESS') {
                        // Retrack is already in progress - just show message and start polling
                        alert(`Retrack already in progress: ${error.message || 'Please wait for retrack to complete.'}`);
                        // Start polling retrack status
                        await this.pollRetrackStatus(studyUid, seriesUid);
                    } else {
                        // Other 409 errors
                        alert(`Conflict: ${error.error_code || error.error || 'Unknown conflict'}`);
                    }
                } else {
                    // Other errors (400, 500, etc.)
                    alert(`Save failed: ${error.error_code || error.error || 'Unknown error'}`);
                }
            }
        } catch (error) {
            console.error('Error saving:', error);
            alert(`Error saving: ${error.message}`);
        }
    }

    async pollRetrackStatus(studyUid, seriesUid) {
        const maxAttempts = 180; // 3 minutes at 1s intervals
        let attempts = 0;
        
        const poll = async () => {
            attempts++;
            const response = await fetch(`/proxy/api/retrack/status/${studyUid}/${seriesUid}`);
            const status = await response.json();
            
            if (status.status === 'completed') {
                console.log('Retrack complete, reloading...');
                this.frameCache.clear();
                this.framesArchive = null;
                this.masksArchive = {};
                await this.loadVideoData();
                await this.loadFramesArchive();
                await this.goToFrame(this.currentFrame);
            } else if (status.status === 'failed') {
                console.error('Retrack failed:', status.error);
                alert(`Retrack failed: ${status.error || 'Unknown error'}`);
            } else if (attempts < maxAttempts) {
                // Still processing - poll again
                setTimeout(poll, 1000);
            } else {
                alert('Retrack timeout - check server status');
            }
        };
        
        poll();
    }

    async retrackVideo() {
        // Retrack is now triggered by save - this is legacy, just call save
        if (this.hasUnsavedChangesForCurrentVideo()) {
            await this.saveChanges();
        } else {
            alert('No changes to save. Retrack is triggered automatically when you save edits.');
        }
    }

    resetMask() {
        // Reset to the last saved state (from annotations/ if exists, else output/)
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

    async confirmCompleteSeries() {
        // Hide modal immediately
        this.hideModal('completeModal');

        const { studyUid, seriesUid } = this.currentVideo;
        try {
            const resp = await fetch(`/proxy/api/series/${studyUid}/${seriesUid}/complete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                alert(`Failed to mark complete: ${data.error || resp.status}`);
                return;
            }
            // Optional: simple toast via alert for now
            alert('Series marked as done on server.');
        } catch (e) {
            alert(`Failed to mark complete: ${e.message}`);
        }
    }

    showConflictModal(title, message, details) {
        document.getElementById('conflictTitle').textContent = title;
        document.getElementById('conflictMessage').textContent = message;
        document.getElementById('conflictDetails').textContent = details || '';
        this.showModal('conflictModal');
    }

    async handleResetAndReload() {
        this.hideModal('conflictModal');
        
        const { studyUid, seriesUid } = this.currentVideo;
        const videoKey = `${studyUid}__${seriesUid}`;
        
        // Clear all local edits for this video
        this.modifiedFrames.delete(videoKey);
        this.hasUnsavedChanges = false;
        this.updateSaveButtonState();
        
        // Clear frame cache
        this.frameCache.clear();
        this.framesArchive = null;
        this.masksArchive = {};
        
        // Reload from server
        console.log('Resetting and reloading from server...');
        try {
            await this.loadVideoData();
            await this.loadFramesArchive();
            await this.goToFrame(this.currentFrame);
            console.log('✅ Reloaded from server');
        } catch (error) {
            console.error('Error reloading from server:', error);
            alert(`Error reloading: ${error.message}`);
        }
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

    /**
     * Load the next series from server's selection logic.
     * This is the primary navigation method - server decides what to work on next.
     */
    async loadNextSeries() {
        try {
            const response = await fetch('/proxy/api/series/next', {
                headers: {
                    'X-User-Email': this.getUserEmail() || ''
                }
            });
            
            if (!response.ok) {
                throw new Error(`Failed to fetch next series: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.no_available_series) {
                alert('No available series to work on. All series may be completed or in progress.');
                return;
            }
            
            // Server returns SeriesMetadata with study_uid, series_uid, exam_number, etc.
            // We need to determine the method - for now, use flow_method from config or default
            // The server doesn't return method, so we'll need to infer it or use a default
            // For now, assume method is 'dis' (the flow method)
            const method = 'dis'; // TODO: Get from server response or config
            
            // Add current video to history before loading new one (if exists)
            if (this.currentVideo && this.currentVideo.studyUid) {
                this.videoHistory.push({
                    method: this.currentVideo.method,
                    studyUid: this.currentVideo.studyUid,
                    seriesUid: this.currentVideo.seriesUid
                });
            }
            
            // Load the series from server
            // Note: loadVideo() will start activity pings automatically
            await this.loadVideo(method, data.study_uid, data.series_uid);
        } catch (error) {
            console.error('Error loading next series:', error);
            alert(`Failed to load next series: ${error.message}`);
        }
    }
    
    /**
     * Navigate to previous video from history (client-side stack).
     */
    async navigateVideoPrev() {
        if (this.videoHistory.length === 0) {
            // No history - try to load next series instead
            await this.loadNextSeries();
            return;
        }
        
        const prevVideo = this.videoHistory.pop();
        await this.loadVideo(prevVideo.method, prevVideo.studyUid, prevVideo.seriesUid);
    }
    
    /**
     * Navigate to next video using server's selection logic.
     */
    async navigateVideoNext() {
        await this.loadNextSeries();
    }
    
    /**
     * Get user email from config or prompt (for identification header).
     * The client backend (client.py) automatically adds X-User-Email header
     * from config, so we don't need to set it here. This is a placeholder
     * for future client-side identification if needed.
     */
    getUserEmail() {
        // Client backend handles X-User-Email header automatically via proxy_to_server
        // Return empty string - header will be added by backend if config.user_email is set
        return '';
    }
    
    /**
     * Start sending activity pings to server every 30 seconds.
     * This marks the series as "actively being viewed" to prevent conflicts.
     */
    startActivityPings() {
        // Clear any existing interval
        this.stopActivityPings();
        
        if (!this.currentVideo || !this.currentVideo.studyUid) {
            return;
        }
        
        // Send initial ping immediately
        this.sendActivityPing();
        
        // Then ping every 30 seconds
        this.activityPingInterval = setInterval(() => {
            this.sendActivityPing();
        }, 30000); // 30 seconds
    }
    
    /**
     * Stop sending activity pings (when navigating away or unloading).
     */
    stopActivityPings() {
        if (this.activityPingInterval !== null) {
            clearInterval(this.activityPingInterval);
            this.activityPingInterval = null;
        }
    }
    
    /**
     * Send a single activity ping to the server.
     * This marks the current series as "actively being viewed".
     */
    async sendActivityPing() {
        if (!this.currentVideo || !this.currentVideo.studyUid) {
            return;
        }
        
        try {
            const response = await fetch(
                `/proxy/api/series/${this.currentVideo.studyUid}/${this.currentVideo.seriesUid}/activity`,
                {
                    method: 'POST',
                    headers: {
                        'X-User-Email': this.getUserEmail() || ''
                    }
                }
            );
            
            if (!response.ok) {
                console.warn(`Activity ping failed: ${response.status} ${response.statusText}`);
            } else {
                console.debug('Activity ping sent successfully');
            }
        } catch (error) {
            console.warn('Failed to send activity ping:', error);
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
        // Stop activity pings for previous video (if any)
        this.stopActivityPings();
        
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
        
        // Start activity pings for this video (every 30s)
        this.startActivityPings();
    }
    
    /**
     * Cleanup when viewer is destroyed or page unloads.
     */
    cleanup() {
        this.stopActivityPings();
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
            const { frames_archive_url, masks_archive_url } = await response.json();

            // Load frames archive (.tar format, no gzip)
            const framesArchiveResponse = await fetch(frames_archive_url);
            const framesArrayBuffer = await framesArchiveResponse.arrayBuffer();
            const framesTarData = new Uint8Array(framesArrayBuffer);

            // Parse tar and extract frames
            this.framesArchive = {};
            let offset = 0;
            while (offset < framesTarData.length) {
                if (offset + 512 > framesTarData.length) break;

                const name = new TextDecoder().decode(framesTarData.slice(offset, offset + 100)).replace(/\0/g, '');
                if (!name) break;

                const size = parseInt(new TextDecoder().decode(framesTarData.slice(offset + 124, offset + 136)), 8);
                offset += 512;

                // Skip directory entries (size == 0) and only process frame files
                if (name.endsWith('.webp') && name.includes('frame_') && !name.includes('_mask') && size > 0) {
                    const data = framesTarData.slice(offset, offset + size);
                    // Store with full path from tar (e.g., "frames/frame_000000.webp")
                    this.framesArchive[name] = data;
                }

                offset += Math.ceil(size / 512) * 512;
            }

            // Load masks archive (.tar format, no gzip) - get version ID from headers
            this.masksArchive = {};
            if (masks_archive_url) {
                const masksArchiveResponse = await fetch(masks_archive_url);
                const masksArrayBuffer = await masksArchiveResponse.arrayBuffer();
                const masksTarData = new Uint8Array(masksArrayBuffer);
                
                // Store version ID from response headers for optimistic locking
                this.currentVersionId = masksArchiveResponse.headers.get('X-Version-ID') || null;

                let metadataBytes = null;

                // Parse tar: extract masks and metadata.json
                offset = 0;
                while (offset < masksTarData.length) {
                    if (offset + 512 > masksTarData.length) break;

                    const name = new TextDecoder().decode(masksTarData.slice(offset, offset + 100)).replace(/\0/g, '');
                    if (!name) break;

                    const size = parseInt(new TextDecoder().decode(masksTarData.slice(offset + 124, offset + 136)), 8);
                    offset += 512;

                    if (size > 0) {
                        const fileData = masksTarData.slice(offset, offset + size);

                        if (name === 'metadata.json') {
                            metadataBytes = fileData;
                        } else if (name.endsWith('.webp') && name.includes('_mask')) {
                            // Mask files live at root of archive
                            this.masksArchive[name] = fileData;
                        }
                    }

                    offset += Math.ceil(size / 512) * 512;
                }

                // Apply per-frame metadata for coloring and info panel
                if (metadataBytes) {
                    try {
                        const metadataText = new TextDecoder().encode
                            ? new TextDecoder().decode(metadataBytes)
                            : new TextDecoder('utf-8').decode(metadataBytes);
                        const metadata = JSON.parse(metadataText);
                        this.updateMaskDataFromMetadata(metadata);
                    } catch (e) {
                        console.error('Failed to parse metadata.json from masks archive', e);
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

AnnotationViewer.prototype.checkDatasetSyncStatus = async function () {
    try {
        const resp = await fetch('/api/dataset/version_status');
        if (!resp.ok) return;
        const data = await resp.json();
        const banner = document.getElementById('datasetWarning');
        if (!banner) return;
        if (data.in_sync) {
            banner.classList.add('hidden');
        } else {
            banner.classList.remove('hidden');
        }
    } catch (e) {
        const banner = document.getElementById('datasetWarning');
        if (banner) banner.classList.add('hidden');
    }
};

// Initialize viewer when DOM is ready
let viewer;
document.addEventListener('DOMContentLoaded', () => {
    viewer = new AnnotationViewer();
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (viewer) {
            viewer.cleanup();
        }
    });
});
