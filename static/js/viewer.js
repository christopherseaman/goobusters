// Goobusters Annotation Viewer - Simplified Based on Teef Reference

class AnnotationViewer {
    constructor() {
        this.canvas = document.getElementById('canvas');
        this.ctx = this.canvas.getContext('2d');

        // Set canvas to full viewport size
        this.resizeCanvas();
        window.addEventListener('resize', () => this.resizeCanvas());

        // Create hidden canvas for mask manipulation (like teef's maskCanvas)
        this.maskCanvas = document.createElement('canvas');
        this.maskCtx = this.maskCanvas.getContext('2d');

        // State
        this.currentVideo = INITIAL_VIDEO;
        this.currentFrame = 0;
        this.totalFrames = 0;
        this.isPlaying = false;
        this.playInterval = null;
        this.playbackSpeed = 30;

        this.frameImage = null;
        this.maskImageData = null; // Grayscale ImageData (like singleChannelMask in teef)
        this.originalMaskImageData = null; // For reset
        this.maskType = null; // 'LABEL_ID' or 'TRACK_ID'
        this.originalMaskType = null;

        this.drawMode = 'draw';
        this.brushSize = 15;
        this.maskOpacity = 0.5;

        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;

        this.videoData = null;
        this.hasUnsavedChanges = false;
        this.renderRect = null; // For coordinate conversion

        this.toolsVisible = true;

        this.init();
    }

    resizeCanvas() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
        if (this.frameImage) {
            this.render();
        }
    }

    async init() {
        this.setupEventListeners();
        await this.loadVideoData();
        this.goToFrame(1); // Start from frame 1 (frame 0 has no mask data)
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
        if (e) {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX || (e.touches && e.touches[0].clientX)) - rect.left;
            const y = (e.clientY || (e.touches && e.touches[0].clientY)) - rect.top;
            preview.style.left = `${x}px`;
            preview.style.top = `${y}px`;
            preview.style.width = `${this.brushSize * 2}px`;
            preview.style.height = `${this.brushSize * 2}px`;
            preview.style.display = 'block';
        } else {
            preview.style.width = `${this.brushSize * 2}px`;
            preview.style.height = `${this.brushSize * 2}px`;
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
        this.videoData = await response.json();
        this.totalFrames = this.videoData.total_frames;

        // Frame 0 has no mask data, start from frame 1
        document.getElementById('sliderMin').textContent = '1';
        document.getElementById('sliderMax').textContent = this.totalFrames - 1;
        document.getElementById('frameSlider').min = 1;
        document.getElementById('frameSlider').max = this.totalFrames - 1;
    }

    async goToFrame(frameNum) {
        // Frame 0 has no mask data, enforce minimum of 1
        this.currentFrame = Math.max(1, Math.min(frameNum, this.totalFrames - 1));
        document.getElementById('frameSlider').value = this.currentFrame;

        const { method, studyUid, seriesUid } = this.currentVideo;
        const response = await fetch(`/api/frame/${method}/${studyUid}/${seriesUid}/${this.currentFrame}`);
        const data = await response.json();

        // Load frame image
        this.frameImage = await this.loadImage(`data:image/jpeg;base64,${data.frame}`);

        // Set mask canvas size to match frame
        this.maskCanvas.width = this.frameImage.width;
        this.maskCanvas.height = this.frameImage.height;

        // Load mask if exists
        if (data.mask) {
            const maskImg = await this.loadImage(`data:image/png;base64,${data.mask}`);
            this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
            this.maskCtx.drawImage(maskImg, 0, 0);

            // Server returns grayscale PNG (cv2.IMREAD_GRAYSCALE)
            // Extract R channel as grayscale value (R=G=B in grayscale)
            const rawImageData = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
            this.maskImageData = this.maskCtx.createImageData(this.maskCanvas.width, this.maskCanvas.height);

            for (let i = 0; i < rawImageData.data.length; i += 4) {
                // Grayscale PNG has R=G=B, use R channel
                const gray = rawImageData.data[i];
                this.maskImageData.data[i] = gray;
                this.maskImageData.data[i + 1] = gray;
                this.maskImageData.data[i + 2] = gray;
                this.maskImageData.data[i + 3] = 255;
            }
        } else {
            // No mask - create empty
            this.maskCtx.fillStyle = 'black';
            this.maskCtx.fillRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
            this.maskImageData = this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        }

        // Store original for reset
        this.originalMaskImageData = this.cloneImageData(this.maskImageData);

        // Determine mask type based on is_annotation flag
        const frameMetadata = this.videoData.mask_data[this.currentFrame];
        this.maskType = frameMetadata?.is_annotation ? 'human' : 'tracked';
        this.originalMaskType = this.maskType;

        this.hasUnsavedChanges = false;
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
        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.frameImage) return;

        // Calculate aspect-fit dimensions
        const canvasAspect = this.canvas.width / this.canvas.height;
        const imageAspect = this.frameImage.width / this.frameImage.height;

        let drawWidth, drawHeight, drawX, drawY;

        if (imageAspect > canvasAspect) {
            drawWidth = this.canvas.width;
            drawHeight = this.canvas.width / imageAspect;
            drawX = 0;
            drawY = (this.canvas.height - drawHeight) / 2;
        } else {
            drawHeight = this.canvas.height;
            drawWidth = this.canvas.height * imageAspect;
            drawX = (this.canvas.width - drawWidth) / 2;
            drawY = 0;
        }

        // Store dimensions for coordinate conversion
        this.renderRect = { x: drawX, y: drawY, width: drawWidth, height: drawHeight };

        // Draw frame
        this.ctx.drawImage(this.frameImage, drawX, drawY, drawWidth, drawHeight);

        // Draw mask overlay (like applyMaskToOverlay in teef)
        if (this.maskImageData) {
            const overlayImageData = this.ctx.createImageData(this.maskImageData.width, this.maskImageData.height);

            // Determine color based on mask type
            // Human annotations are green, tracked predictions (fluid_*) are orange
            const isHuman = this.maskType === 'human' || this.maskType === 'empty';
            const maskColor = isHuman ? { r: 0, g: 255, b: 0 } : { r: 255, g: 165, b: 0 };
            console.log('Rendering mask - Type:', this.maskType, 'Human:', isHuman, 'Color:', maskColor);

            // Convert grayscale mask to colored overlay
            for (let i = 0; i < this.maskImageData.data.length; i += 4) {
                const gray = this.maskImageData.data[i]; // Grayscale value (0 = no mask, 255 = mask)

                if (gray > 0) {
                    // Mask exists - show colored overlay at 50% opacity
                    overlayImageData.data[i] = maskColor.r;
                    overlayImageData.data[i + 1] = maskColor.g;
                    overlayImageData.data[i + 2] = maskColor.b;
                    overlayImageData.data[i + 3] = 128; // 50% opacity (128/255)
                } else {
                    // No mask - fully transparent
                    overlayImageData.data[i] = 0;
                    overlayImageData.data[i + 1] = 0;
                    overlayImageData.data[i + 2] = 0;
                    overlayImageData.data[i + 3] = 0;
                }
            }

            // Draw overlay on temp canvas then to main canvas
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = this.maskImageData.width;
            tempCanvas.height = this.maskImageData.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.putImageData(overlayImageData, 0, 0);

            this.ctx.drawImage(tempCanvas, drawX, drawY, drawWidth, drawHeight);
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
        // Tracked types: fluid_forward, fluid_backward, fluid_bidirectional
        if (this.maskType && this.maskType.startsWith('fluid_')) {
            console.log('CONVERTING', this.maskType, 'to human!');
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
        if (!this.hasUnsavedChanges) {
            console.log('No changes to save');
            return;
        }

        try {
            // Put mask data back to mask canvas and convert to base64
            this.maskCtx.putImageData(this.maskImageData, 0, 0);
            const maskData = this.maskCanvas.toDataURL('image/png');

            const { method, studyUid, seriesUid } = this.currentVideo;
            const response = await fetch('/api/save_mask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    method,
                    study_uid: studyUid,
                    series_uid: seriesUid,
                    frame_number: this.currentFrame,
                    mask_data: maskData
                })
            });

            if (response.ok) {
                this.hasUnsavedChanges = false;
                this.originalMaskImageData = this.cloneImageData(this.maskImageData);
                document.getElementById('saveChanges').classList.add('success');
                setTimeout(() => document.getElementById('saveChanges').classList.remove('success'), 1000);
                console.log('✅ Changes saved');
            } else {
                console.error('❌ Failed to save');
            }
        } catch (error) {
            console.error('Error saving:', error);
        }
    }

    resetMask() {
        this.maskImageData = this.cloneImageData(this.originalMaskImageData);
        this.maskType = this.originalMaskType;
        this.hasUnsavedChanges = false;
        this.render();
    }

    async markEmpty() {
        const { method, studyUid, seriesUid } = this.currentVideo;
        const response = await fetch('/api/mark_empty', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                method,
                study_uid: studyUid,
                series_uid: seriesUid,
                frame_number: this.currentFrame
            })
        });

        if (response.ok) {
            // Clear mask and reload frame
            await this.goToFrame(this.currentFrame);
        }
    }

    navigateFrame(delta) {
        const newFrame = this.currentFrame + delta;
        if (newFrame >= 0 && newFrame < this.totalFrames) {
            this.goToFrame(newFrame);
        }
    }

    togglePlayPause() {
        this.isPlaying = !this.isPlaying;
        const btn = document.getElementById('playPause');
        btn.textContent = this.isPlaying ? '⏸️' : '▶️';

        if (this.isPlaying) {
            this.playInterval = setInterval(() => {
                if (this.currentFrame < this.totalFrames - 1) {
                    this.goToFrame(this.currentFrame + 1);
                } else {
                    this.togglePlayPause();
                }
            }, 1000 / this.playbackSpeed);
        } else {
            clearInterval(this.playInterval);
        }
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

    async loadVideo(method, studyUid, seriesUid) {
        this.currentVideo = { method, studyUid, seriesUid };
        await this.loadVideoData();
        this.goToFrame(0);
    }

    showModal(modalId) {
        document.getElementById(modalId).classList.add('active');
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }

    updateInfoPanel() {
        const frameMetadata = this.videoData.mask_data[this.currentFrame];
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
