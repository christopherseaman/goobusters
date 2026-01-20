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
        window.addEventListener('resize', () => {
            this.resizeCanvas();
            this.updateSliderTypeBar();
        });

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
        
        
        this.frameCache = new Map(); // frameNum -> {frameImage, maskImageData, maskType, originalMaskImageData, originalMaskType, canvasWidth, canvasHeight}
        this.framesArchive = null; // Extracted frames from tar
        this.masksArchive = {}; // Masks from tar archive (webp images)
        this.currentVersionId = null; // Track version ID from server for optimistic locking
        this.activityPingInterval = null; // Interval ID for activity pings (30s)
        this.userEmail = null; // User name for identification (from localStorage or prompt)

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
        const settings = await this.loadSettings();
        
        // Check if token is configured - show blocking setup modal if not
        if (!settings || !settings.mdai_token_present) {
            this.showSetupModal();
            return; // Don't proceed until token is set
        }
        
        // Check if token is set but dataset is not ready - show blocking sync modal
        await this.checkAndSyncDatasetIfNeeded();
        
        // Ensure user name is set (prompt if missing)
        await this.ensureUserEmail();
        
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
                alert('Failed to load series. Please check connection to tracking server.');
            }
        }
        this.checkDatasetSyncStatus().catch(() => {});
    }
    
    async checkAndSyncDatasetIfNeeded() {
        // Check if dataset is ready
        const seriesResp = await fetch('/api/local/series').catch(() => null);
        if (seriesResp && seriesResp.ok) {
            const series = await seriesResp.json().catch(() => []);
            if (series && series.length > 0) {
                return; // Dataset is ready, no sync needed
            }
        }
        
        // Token is set but dataset is not ready - show blocking sync modal
        console.log('[Init] Token set but dataset not ready - showing sync modal');
        await this.showSyncModal();
    }
    
    async showSyncModal() {
        // Create blocking modal for sync
        const modal = document.createElement('div');
        modal.id = 'syncModal';
        modal.style.cssText = 'position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.9); z-index: 10000; display: flex; align-items: center; justify-content: center;';
        modal.innerHTML = `
            <div style="background: white; padding: 30px; border-radius: 12px; text-align: center; max-width: 400px;">
                <h2 style="margin-top: 0;">🔄 Syncing Dataset</h2>
                <p>Downloading data from MD.ai...</p>
                <div id="syncProgress" style="margin: 20px 0; color: #666;">Starting sync...</div>
                <div style="width: 100%; height: 4px; background: #eee; border-radius: 2px; overflow: hidden;">
                    <div id="syncProgressBar" style="width: 0%; height: 100%; background: #4CAF50; transition: width 0.3s;"></div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Start sync (blocking)
        await this.performBlockingSync(modal);
    }
    
    async performBlockingSync(modal) {
        const progressDiv = document.getElementById('syncProgress');
        const progressBar = document.getElementById('syncProgressBar');
        
        try {
            progressDiv.textContent = 'Syncing dataset from MD.ai...';
            progressBar.style.width = '30%';
            
            const syncResp = await fetch('/api/dataset/sync', {
                method: 'POST',
            });
            
            if (!syncResp.ok) {
                const errorData = await syncResp.json().catch(() => ({}));
                throw new Error(errorData.error || 'Sync failed');
            }
            
            progressBar.style.width = '100%';
            progressDiv.textContent = 'Sync complete!';
            
            const data = await syncResp.json();
            setTimeout(() => {
                modal.remove();
                // Reload page to show synced data
                window.location.reload();
            }, 1000);
        } catch (error) {
            progressDiv.textContent = `❌ Error: ${error.message}`;
            progressDiv.style.color = '#f44336';
            progressBar.style.background = '#f44336';
            // Keep modal visible so user can see the error
        }
    }
    
    showSetupModal() {
        const setupModal = document.getElementById('setupModal');
        if (!setupModal) return;
        
        setupModal.style.display = 'flex';
        
        // Pre-fill name if available
        const setupEmail = document.getElementById('setupEmail');
        if (setupEmail) {
            setupEmail.value = this.getUserEmail();
        }
        
        // Set up save button handler
        const saveSetupBtn = document.getElementById('saveSetup');
        if (saveSetupBtn) {
            saveSetupBtn.onclick = () => {
                console.log('[Setup] Save button clicked');
                this.saveSetupAndContinue();
            };
        } else {
            console.error('[Setup] Save button not found!');
        }
        
        // Update token status
        this.updateSetupTokenStatus();
    }
    
    hideSetupModal() {
        const setupModal = document.getElementById('setupModal');
        if (setupModal) {
            setupModal.style.display = 'none';
        }
    }
    
    async updateSetupTokenStatus() {
        try {
            const resp = await fetch('/api/settings');
            if (!resp.ok) return;
            const data = await resp.json();
            const tokenStatus = document.getElementById('setupTokenStatus');
            if (tokenStatus) {
                tokenStatus.textContent = data.mdai_token_present ? 'Token: set ✓' : 'Token: not set';
                tokenStatus.style.color = data.mdai_token_present ? '#4CAF50' : '#f44336';
            }
        } catch (err) {
            console.warn('Failed to update setup token status', err);
        }
    }
    
    async saveSetupAndContinue() {
        console.log('[Setup] saveSetupAndContinue called');
        const emailInput = document.getElementById('setupEmail');
        const tokenInput = document.getElementById('setupToken');
        
        const user_email = emailInput ? emailInput.value.trim() : '';
        const token_value = tokenInput ? tokenInput.value.trim() : '';
        
        console.log('[Setup] Token length:', token_value.length);
        
        if (!token_value) {
            alert('MD.ai API token is required. Please enter your token.');
            return;
        }
        
        const payload = { user_email };
        if (token_value) {
            payload.mdai_token = token_value;
        }
        
        console.log('[Setup] Sending POST /api/settings with token');
        try {
            const resp = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            
            console.log('[Setup] Response status:', resp.status);
            
            if (!resp.ok) {
                const errorData = await resp.json().catch(() => ({}));
                alert(`Failed to save settings: ${errorData.error || resp.statusText}`);
                return;
            }
            
            const data = await resp.json();
            if (data.user_email) {
                this.userEmail = data.user_email;
                localStorage.setItem('userEmail', data.user_email);
            }
            
            // Verify token is now set
            console.log('[Setup] Token present:', data.mdai_token_present);
            if (data.mdai_token_present) {
                console.log('[Setup] Token valid, hiding modal and starting sync');
                this.hideSetupModal();
                // Continue with initialization
                await this.ensureUserEmail();
                
                // Trigger dataset sync now that token is set
                try {
                    console.log('[Setup] Token set, syncing dataset from MD.ai...');
                    // Show loading indicator
                    const loadingMsg = document.createElement('div');
                    loadingMsg.id = 'syncLoadingMsg';
                    loadingMsg.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(0,0,0,0.9); color: white; padding: 20px; border-radius: 8px; z-index: 10002; text-align: center;';
                    loadingMsg.innerHTML = '🔄 Syncing dataset from MD.ai...<br><small>This may take a moment</small>';
                    document.body.appendChild(loadingMsg);
                    
                    const syncResp = await fetch('/api/dataset/sync', {
                        method: 'POST',
                    });
                    
                    // Remove loading indicator
                    const msg = document.getElementById('syncLoadingMsg');
                    if (msg) msg.remove();
                    
                    if (!syncResp.ok) {
                        const errorData = await syncResp.json().catch(() => ({}));
                        const errorMsg = errorData.error || syncResp.statusText || 'Unknown error';
                        console.error('Dataset sync failed:', errorMsg);
                        alert(`Dataset sync failed: ${errorMsg}\n\nYou can try syncing manually from Settings.`);
                    } else {
                        const syncData = await syncResp.json();
                        console.log('Dataset sync completed:', syncData);
                        // Show success briefly
                        const successMsg = document.createElement('div');
                        successMsg.style.cssText = 'position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); background: rgba(76,175,80,0.9); color: white; padding: 20px; border-radius: 8px; z-index: 10002; text-align: center;';
                        successMsg.innerHTML = `✅ Dataset synced!<br><small>Found ${syncData.series_count || 0} series</small>`;
                        document.body.appendChild(successMsg);
                        setTimeout(() => successMsg.remove(), 3000);
                    }
                } catch (syncError) {
                    const msg = document.getElementById('syncLoadingMsg');
                    if (msg) msg.remove();
                    console.error('Failed to sync dataset:', syncError);
                    alert(`Failed to sync dataset: ${syncError.message}\n\nYou can try syncing manually from Settings.`);
                }
                
                // Load initial video from server's "next" selection
                try {
                    await this.loadNextSeries();
                } catch (error) {
                    console.error('Failed to load initial series from tracking server:', error);
                    if (typeof INITIAL_VIDEO !== 'undefined' && INITIAL_VIDEO && INITIAL_VIDEO.studyUid) {
                        console.warn('Falling back to INITIAL_VIDEO from template');
                        await this.loadVideo(INITIAL_VIDEO.method, INITIAL_VIDEO.studyUid, INITIAL_VIDEO.seriesUid);
                    } else {
                        alert('Failed to load series. Please check server connection.');
                    }
                }
                this.checkDatasetSyncStatus().catch(() => {});
            } else {
                alert('Token was not saved. Please try again.');
                this.updateSetupTokenStatus();
            }
        } catch (err) {
            alert(`Failed to save settings: ${err.message}`);
        }
    }
    
    async ensureUserEmail() {
        // Check already loaded from settings/localStorage
        let userEmail = this.getUserEmail();
        
        if (!userEmail) {
            // Prompt user for name
            userEmail = prompt(
                'Please enter your name for identification:\n\n' +
                'This is used to track which series you\'ve worked on and coordinate with other users.',
                ''
            );
            
            if (!userEmail || !userEmail.trim()) {
                // User cancelled or entered empty - use a default
                userEmail = `user_${Date.now()}@local`;
                console.warn('No name provided, using temporary identifier:', userEmail);
            } else {
                userEmail = userEmail.trim();
            }
            
            // Store in localStorage
            localStorage.setItem('userEmail', userEmail);

            // Persist to backend (best-effort)
            try {
                await fetch('/api/settings', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ user_email: userEmail }),
                });
            } catch (e) {
                console.warn('Failed to persist user email to backend', e);
            }
        }
        
        this.userEmail = userEmail;

        // Keep settings modal in sync
        const emailInput = document.getElementById('settingsEmail');
        if (emailInput && !emailInput.value) {
            emailInput.value = userEmail;
        }

        return userEmail;
    }

    async loadSettings() {
        try {
            const resp = await fetch('/api/settings');
            if (!resp.ok) return;
            const data = await resp.json();
            if (data.user_email) {
                this.userEmail = data.user_email;
                localStorage.setItem('userEmail', data.user_email);
                const emailInput = document.getElementById('settingsEmail');
                if (emailInput) {
                    emailInput.value = data.user_email;
                }
            }
            const tokenStatus = document.getElementById('tokenStatus');
            if (tokenStatus) {
                tokenStatus.textContent = data.mdai_token_present ? 'Token: set' : 'Token: not set';
            }
            return data;
        } catch (err) {
            console.warn('Failed to load settings', err);
            return null;
        }
    }

    async syncDataset() {
        const syncBtn = document.getElementById('syncDatasetBtn');
        const syncStatus = document.getElementById('syncStatus');
        
        if (!syncBtn || !syncStatus) return;
        
        // Disable button and show status
        syncBtn.disabled = true;
        const originalText = syncBtn.textContent;
        syncStatus.textContent = 'Downloading dataset from MD.ai...';
        syncStatus.style.color = '#666';
        
        try {
            const resp = await fetch('/api/dataset/sync', {
                method: 'POST',
            });
            
            if (!resp.ok) {
                const errorData = await resp.json().catch(() => ({}));
                const errorMsg = errorData.error || resp.statusText || 'Unknown error';
                syncStatus.textContent = `❌ Error: ${errorMsg}`;
                syncStatus.style.color = '#f44336';
                throw new Error(errorMsg);
            }
            
            const data = await resp.json();
            syncStatus.textContent = `✅ Synced! Found ${data.series_count || 0} series`;
            syncStatus.style.color = '#4CAF50';
            
            // Reload page to show new series
            setTimeout(() => {
                window.location.reload();
            }, 1500);
        } catch (error) {
            console.error('Dataset sync failed:', error);
            syncStatus.textContent = `❌ Failed: ${error.message}`;
            syncStatus.style.color = '#f44336';
        } finally {
            syncBtn.disabled = false;
            syncBtn.textContent = originalText;
        }
    }

    async saveSettings() {
        const emailInput = document.getElementById('settingsEmail');
        const tokenInput = document.getElementById('settingsToken');
        const tokenStatus = document.getElementById('tokenStatus');

        const user_email = emailInput ? emailInput.value.trim() : '';
        const token_value = tokenInput ? tokenInput.value.trim() : '';

        const payload = { user_email };
        if (tokenInput && token_value !== '') {
            payload.mdai_token = token_value;
        }

        try {
            const resp = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            if (!resp.ok) {
                const errorData = await resp.json().catch(() => ({}));
                alert(`Failed to save settings: ${errorData.error || resp.statusText}`);
                return;
            }
            const data = await resp.json();
            if (data.user_email) {
                this.userEmail = data.user_email;
                localStorage.setItem('userEmail', data.user_email);
            }
            if (tokenStatus) {
                tokenStatus.textContent = data.mdai_token_present ? 'Token: set' : 'Token: not set';
            }
            if (tokenInput) {
                tokenInput.value = '';
            }
            this.hideModal('settingsModal');
        } catch (err) {
            alert(`Failed to save settings: ${err.message}`);
        }
    }

    openSettings() {
        const emailInput = document.getElementById('settingsEmail');
        if (emailInput && !emailInput.value) {
            emailInput.value = this.getUserEmail();
        }
        this.showModal('settingsModal');
    }

    setupEventListeners() {
        // Modal controls
        document.getElementById('infoBtn').addEventListener('click', () => this.showModal('infoModal'));
        document.getElementById('examBadge').addEventListener('click', () => this.showModal('seriesSelectModal'));
        document.getElementById('closeInfo').addEventListener('click', () => this.hideModal('infoModal'));
        document.getElementById('closeSeriesSelect').addEventListener('click', () => this.hideModal('seriesSelectModal'));
        document.getElementById('closeKeyboardShortcuts').addEventListener('click', () => this.hideModal('keyboardShortcutsModal'));
        document.getElementById('settingsBtn').addEventListener('click', () => this.openSettings());
        document.getElementById('closeSettings').addEventListener('click', () => this.hideModal('settingsModal'));
        const saveSettingsBtn = document.getElementById('saveSettings');
        if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        const syncDatasetBtn = document.getElementById('syncDatasetBtn');
        if (syncDatasetBtn) syncDatasetBtn.addEventListener('click', () => this.syncDataset());

        // Close modals on background click (except retrack loading modal which is blocking)
        document.querySelectorAll('.modal').forEach(modal => {
            modal.addEventListener('click', (e) => {
                if (e.target === modal && modal.id !== 'retrackLoadingModal') {
                    this.hideModal(modal.id);
                }
            });
        });

        // Completion modal
        const closeComplete = document.getElementById('closeComplete');
        const cancelComplete = document.getElementById('cancelComplete');
        const confirmComplete = document.getElementById('confirmComplete');
        if (closeComplete) closeComplete.addEventListener('click', () => this.hideModal('completeModal'));
        if (cancelComplete) cancelComplete.addEventListener('click', () => this.hideModal('completeModal'));
        if (confirmComplete) confirmComplete.addEventListener('click', () => this.confirmCompleteSeries(true));

        // Conflict modal
        const closeConflict = document.getElementById('closeConflict');
        const cancelConflict = document.getElementById('cancelConflict');
        const resetAndReload = document.getElementById('resetAndReload');
        if (closeConflict) closeConflict.addEventListener('click', () => this.hideModal('conflictModal'));
        if (cancelConflict) cancelConflict.addEventListener('click', () => this.hideModal('conflictModal'));
        if (resetAndReload) resetAndReload.addEventListener('click', () => this.handleResetAndReload());

        // Reset retrack buttons (moved to info modal)
        const resetRetrackBtn = document.getElementById('resetRetrackBtn');
        if (resetRetrackBtn) resetRetrackBtn.addEventListener('click', () => {
            this.hideModal('infoModal');
            this.confirmResetRetrack();
        });

        const resetRetrackAllBtn = document.getElementById('resetRetrackAllBtn');
        if (resetRetrackAllBtn) resetRetrackAllBtn.addEventListener('click', () => {
            this.hideModal('infoModal');
            this.confirmResetRetrackAll();
        });

        // Video selection (server-driven)
        // Keep videoSelect for manual selection if needed, but wire Next/Prev to server
        const videoSelect = document.getElementById('videoSelect');
        if (videoSelect) {
            videoSelect.addEventListener('change', async (e) => {
                const [method, studyUid, seriesUid] = e.target.value.split('|');
                // Mark activity immediately when selecting from dropdown
                const activityResponse = await this.markSeriesActivity(studyUid, seriesUid);
                // Update warnings from activity response
                if (activityResponse && activityResponse.ok) {
                    const activityData = await activityResponse.json().catch(() => null);
                    if (activityData) {
                        await this.updateMultiplayerWarning(activityData);
                    }
                }
                await this.loadVideo(method, studyUid, seriesUid);
            });
        }

        document.getElementById('markComplete').addEventListener('click', () => this.showCompleteModal());

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
        document.getElementById('resetMask').addEventListener('click', () => this.handleResetAndReload());

        // Brush size slider (inline only - modal slider removed)
        const brushSizeInline = document.getElementById('brushSizeInline');
        if (brushSizeInline) {
            brushSizeInline.addEventListener('input', (e) => {
                this.brushSize = parseInt(e.target.value);
                this.updateBrushPreview();
            });
        }

        // Mask opacity (optional - was in nav modal which was removed)
        const maskOpacityEl = document.getElementById('maskOpacity');
        if (maskOpacityEl) {
            maskOpacityEl.addEventListener('input', (e) => {
                this.maskOpacity = parseInt(e.target.value) / 100;
                const maskOpacityValueEl = document.getElementById('maskOpacityValue');
                if (maskOpacityValueEl) {
                    maskOpacityValueEl.textContent = e.target.value;
                }
                this.render();
            });
        }

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
                case '?':
                    e.preventDefault();
                    this.showModal('keyboardShortcutsModal');
                    break;
                case ' ':
                    e.preventDefault();
                    // If at end of video, restart from beginning; otherwise toggle play/pause
                    if (this.currentFrame >= this.totalFrames - 1) {
                        this.goToFrame(0);
                    this.togglePlayPause();
                    } else {
                        this.togglePlayPause();
                    }
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
        if (!preview) return;

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
        const preview = document.getElementById('brushPreview');
        if (preview) {
            preview.style.display = 'none';
        }
    }

    async loadVideoData(skipModifiedFrames = false) {
        if (!this.currentVideo) {
            throw new Error('No current video set');
        }
        const { method, studyUid, seriesUid } = this.currentVideo;
        // Client backend proxies to server (no local caching)
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
            status: data.status || 'pending',  // "pending" or "completed"
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
        // Skip if resetting (skipModifiedFrames = true)
        if (data.modified_frames && !skipModifiedFrames) {
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
        // Don't update scrubline here - it will be updated after masks archive metadata is loaded
    }
    
    updateSliderTypeBar() {
        const typeBar = document.getElementById('sliderTypeBar');
        if (!typeBar || !this.videoData || !this.totalFrames) return;
        
        const slider = document.getElementById('frameSlider');
        if (!slider) return;
        
        // Wait for next frame to ensure layout is complete
        requestAnimationFrame(() => {
            const sliderRect = slider.getBoundingClientRect();
            
            // Set canvas size to match slider width
            typeBar.width = sliderRect.width;
            typeBar.height = 24; // Match thumb height
            
            const ctx = typeBar.getContext('2d');
            const width = typeBar.width;
            const height = typeBar.height;
            const frameWidth = width / this.totalFrames;
            
            // Clear canvas
            ctx.clearRect(0, 0, width, height);
            
            // Get unsaved edits to check for modified frames
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            
            // Draw colored sections for each frame
            for (let frameNum = 0; frameNum < this.totalFrames; frameNum++) {
                const x = frameNum * frameWidth;
                const w = frameWidth;
                
                let color = 'transparent';
                
                // Check unsaved edits first (takes precedence)
                if (videoModifiedFrames.has(frameNum)) {
                    const modified = videoModifiedFrames.get(frameNum);
                    if (modified.maskType === 'empty') {
                        color = 'rgba(255, 0, 0, 0.25)';  // Red for empty
                    } else if (modified.maskType === 'human' || modified.maskType === 'fluid') {
                        color = 'rgba(0, 255, 0, 0.25)';  // Green for fluid
                    }
                } else {
                    // Check saved frame data
                    const frameData = this.videoData.mask_data?.[frameNum];
                    if (frameData) {
                        if (frameData.type === 'empty') {
                            color = 'rgba(255, 0, 0, 0.25)';  // Red for empty_id
                        } else if (frameData.type === 'fluid') {
                            color = 'rgba(0, 255, 0, 0.25)';  // Green for label_id
                        }
                    }
                }
                
                if (color !== 'transparent') {
                    ctx.fillStyle = color;
                    ctx.fillRect(x, 0, w, height);
                }
            }
            
            // Draw scrubline (thin line) across entire width
            // Scrubline is always green - empty_id frames are indicated by the red glow around the image
            const scrublineColor = 'rgba(76, 175, 80, 0.9)';
            
            // Draw scrubline across entire width (draw on top of colored sections)
            const scrublineY = height / 2;
            const scrublineHeight = 3;
            ctx.fillStyle = scrublineColor;
            ctx.fillRect(0, scrublineY - scrublineHeight / 2, width, scrublineHeight);
        });
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
            
            // Use type from metadata if provided (server is source of truth)
            // Otherwise derive from is_annotation + has_mask
            let type = entry.type;
            if (!type) {
                // Fallback: derive type from is_annotation + has_mask
                if (isAnnotation && hasMask) {
                    type = 'fluid';
                } else if (isAnnotation && !hasMask) {
                    type = 'empty';
                } else if (!hasMask) {
                    type = 'empty';
                } else {
                    type = 'tracked';
                }
            }

            this.videoData.mask_data[frameNum] = {
                type,
                label_id: labelId,
                is_annotation: isAnnotation,
                has_mask: hasMask,
                modified: false,
            };
        }
        
        // Update slider type bar after metadata is loaded
        this.updateSliderTypeBar();
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
        // Only restore if hasUnsavedChanges is true (prevents restoring after reset)
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        if (this.hasUnsavedChanges && videoModifiedFrames.has(this.currentFrame)) {
            const saved = videoModifiedFrames.get(this.currentFrame);
            this.maskImageData = saved.maskData;
            this.maskType = saved.maskType;
        } else {
            this.hasUnsavedChanges = false;
        }

        this.updateSaveButtonState();
        this.render();
        this.updateInfoPanel();
        this.updateSliderTypeBar();
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

        // Check if current frame is empty_id for visual indicator
        // Check both stored frameData and current maskType (for in-progress edits)
        const frameData = this.videoData?.mask_data?.[this.currentFrame];
        const isEmptyFrame = (frameData && frameData.type === 'empty') || this.maskType === 'empty';
        
        // Position and show/hide empty frame glow indicator to match image area
        const glowElement = document.getElementById('emptyFrameGlow');
        if (glowElement && this.renderRect) {
            if (isEmptyFrame) {
                glowElement.style.display = 'block';
                glowElement.style.left = `${this.renderRect.x}px`;
                glowElement.style.top = `${this.renderRect.y}px`;
                glowElement.style.width = `${this.renderRect.width}px`;
                glowElement.style.height = `${this.renderRect.height}px`;
            } else {
                glowElement.style.display = 'none';
            }
        }
        
        // Scrubline is now drawn on type bar canvas, so update it
        this.updateSliderTypeBar();

        // Draw frame on main canvas (scaled to viewport)
        this.ctx.drawImage(this.frameImage, displayX, displayY, displayWidth, displayHeight);
        
        // Note: Red outline/border for empty frames is now handled by CSS (.empty-frame-indicator class)

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
        
        if (!this.maskImageData) {
            console.error('Cannot start drawing: maskImageData not initialized');
            this.isDrawing = false;
            return;
        }

        this.draw(e);
    }

    draw(e) {
        if (!this.isDrawing || !this.maskImageData) return;

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
        if (!this.maskImageData) return;
        
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

    // Helper: Build .tar archive from all annotation frames and metadata
    // Uses simple tar format - no gzip (WebP already compressed)
    async buildMaskArchive(annotationFrames, studyUid, seriesUid, flowMethod) {
        const frames = [];
        const files = new Map(); // filename -> Uint8Array
        
        // Get label IDs from video data
        const labelId = this.videoData?.labels?.find(l => l.labelName === 'Fluid')?.labelId || '';
        const emptyId = this.videoData?.labels?.find(l => l.labelName === 'Empty')?.labelId || '';
        
        for (const [frameNumStr, frameData] of Object.entries(annotationFrames)) {
            const frameNumber = parseInt(frameNumStr);
            const isEmpty = frameData.is_empty;
            
            frames.push({
                frame_number: frameNumber,
                has_mask: !isEmpty,
                is_annotation: true,
                label_id: isEmpty ? emptyId : labelId,
                filename: isEmpty ? null : `frame_${String(frameNumber).padStart(6, '0')}_mask.webp`
            });
            
            // Only add mask file for non-empty frames
            if (!isEmpty) {
                const filename = `frame_${String(frameNumber).padStart(6, '0')}_mask.webp`;
                // Convert mask ImageData to WebP blob
                const webpBlob = await this.imageDataToWebP(
                    frameData.maskData,
                    frameData.maskData.width,
                    frameData.maskData.height
                );
                const arrayBuffer = await webpBlob.arrayBuffer();
                files.set(filename, new Uint8Array(arrayBuffer));
            }
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
        const metadataJson = JSON.stringify(metadata, null, 2);
        files.set('metadata.json', new TextEncoder().encode(metadataJson));
        
        // Build tar using simple format
        return this._createSimpleTar(files);
    }
    
    // Simple tar creator (no gzip - WebP already compressed)
    _createSimpleTar(files) {
        const chunks = [];
        
        for (const [filename, data] of files) {
            // Tar header (512 bytes)
            const header = new Uint8Array(512);
            const encoder = new TextEncoder();
            
            // Filename (100 bytes)
            const nameBytes = encoder.encode(filename);
            header.set(nameBytes.slice(0, 100), 0);
            
            // Mode (8 bytes) - 0644
            encoder.encodeInto('0000644\0', header.subarray(100, 108));
            
            // UID/GID (8 bytes each) - 0
            encoder.encodeInto('0000000\0', header.subarray(108, 116));
            encoder.encodeInto('0000000\0', header.subarray(116, 124));
            
            // Size (12 bytes) - octal
            const sizeStr = data.length.toString(8).padStart(11, '0') + '\0';
            encoder.encodeInto(sizeStr, header.subarray(124, 136));
            
            // mtime (12 bytes) - current time in octal
            const mtime = Math.floor(Date.now() / 1000);
            const mtimeStr = mtime.toString(8).padStart(11, '0') + '\0';
            encoder.encodeInto(mtimeStr, header.subarray(136, 148));
            
            // Checksum placeholder (8 bytes) - spaces
            encoder.encodeInto('        ', header.subarray(148, 156));
            
            // Type flag (1 byte) - 0 = normal file
            header[156] = 0;
            
            // Link name (100 bytes) - empty
            // Magic "ustar\0" (6 bytes)
            encoder.encodeInto('ustar\0', header.subarray(257, 263));
            // Version "00" (2 bytes)
            encoder.encodeInto('00', header.subarray(263, 265));
            
            // Calculate checksum
            let checksum = 0;
            for (let i = 0; i < 512; i++) {
                checksum += header[i];
            }
            const checksumStr = checksum.toString(8).padStart(6, '0') + '\0 ';
            encoder.encodeInto(checksumStr, header.subarray(148, 156));
            
            chunks.push(header);
            
            // File data (padded to 512-byte boundary)
            chunks.push(data);
            const padding = (512 - (data.length % 512)) % 512;
            if (padding > 0) {
                chunks.push(new Uint8Array(padding));
            }
        }
        
        // Two empty blocks at end
        chunks.push(new Uint8Array(512));
        chunks.push(new Uint8Array(512));
        
        // Concatenate all chunks
        const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
        const result = new Uint8Array(totalLength);
        let offset = 0;
        for (const chunk of chunks) {
            result.set(chunk, offset);
            offset += chunk.length;
        }
        
        return result;
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
        
        // Collect ALL annotation frames (label_id and empty_id) from in-memory videoData
        // This ensures previously saved empty_id frames are preserved
        const allAnnotationFrames = {};
        
        // Helper to get mask ImageData for a frame (from cache, modified, or archive)
        const getMaskImageDataForFrame = async (frameNum) => {
            // Check if frame is in modified frames (has current edits)
            if (videoModifiedFrames.has(frameNum)) {
                return videoModifiedFrames.get(frameNum).maskData;
            }
            
            // Check frame cache
            const cached = this.frameCache.get(frameNum);
            if (cached && cached.maskImageData) {
                return cached.maskImageData;
            }
            
            // Load from archive if available
            const maskFileName = `frame_${frameNum.toString().padStart(6, '0')}_mask.webp`;
            if (this.masksArchive && this.masksArchive[maskFileName]) {
                const maskData = this.masksArchive[maskFileName];
                const blob = new Blob([maskData], { type: 'image/webp' });
                const maskUrl = URL.createObjectURL(blob);
                const maskImg = new Image();
                const canvas = document.createElement('canvas');
                await new Promise((resolve, reject) => {
                    maskImg.onload = () => {
                        canvas.width = maskImg.width;
                        canvas.height = maskImg.height;
                        const ctx = canvas.getContext('2d');
                        ctx.drawImage(maskImg, 0, 0);
                        URL.revokeObjectURL(maskUrl);
                        resolve();
                    };
                    maskImg.onerror = reject;
                    maskImg.src = maskUrl;
                });
                return canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height);
            }
            
            // No mask - return empty black mask
            const emptyCanvas = document.createElement('canvas');
            emptyCanvas.width = this.maskCanvas.width;
            emptyCanvas.height = this.maskCanvas.height;
            const ctx = emptyCanvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, emptyCanvas.width, emptyCanvas.height);
            return ctx.getImageData(0, 0, emptyCanvas.width, emptyCanvas.height);
        };
        
        // First, collect all saved annotation frames from videoData
        if (this.videoData && this.videoData.mask_data) {
            for (const [frameNumStr, frameData] of Object.entries(this.videoData.mask_data)) {
                const frameNum = parseInt(frameNumStr);
                // Include frames that are annotations (fluid or empty)
                if (frameData && (frameData.type === 'fluid' || frameData.type === 'empty')) {
                    // Skip if this frame was modified (will be added below from videoModifiedFrames)
                    if (!videoModifiedFrames.has(frameNum)) {
                        const maskImageData = await getMaskImageDataForFrame(frameNum);
                        allAnnotationFrames[frameNum] = {
                            maskData: maskImageData,
                            is_empty: frameData.type === 'empty'
                        };
                    }
                }
            }
        }
        
        // Then, add/override with ALL modified frames (user edits take precedence)
        for (const [frameNum, frameData] of videoModifiedFrames.entries()) {
            const frameNumber = parseInt(frameNum);
            allAnnotationFrames[frameNumber] = {
                maskData: frameData.maskData,
                is_empty: frameData.is_empty || frameData.maskType === 'empty'
            };
        }
        
        if (Object.keys(allAnnotationFrames).length === 0) {
            console.log('No annotation frames to save');
            return;
        }

        if (this.videoData && this.videoData.status === 'completed') {
            const shouldReopen = confirm(
                'This series is marked as completed. Saving will reopen it and clear the completed status. Continue?'
            );
            if (!shouldReopen) {
                return;
            }
            await this.reopenSeries();
        }

        console.log(`Saving ${Object.keys(allAnnotationFrames).length} annotation frames`);
        try {
            // Build tar archive with all annotation frames
            const archiveData = await this.buildMaskArchive(
                allAnnotationFrames,
                studyUid,
                seriesUid,
                method
            );
            console.log(`Built archive: ${archiveData.length} bytes`);

            const userEmail = this.getUserEmail();
            if (!userEmail) {
                // Prompt for email if missing
                await this.ensureUserEmail();
                const retryEmail = this.getUserEmail();
                if (!retryEmail) {
                    alert('User email is required to save. Please refresh and enter your email.');
                    return;
                }
            }

            // Show blocking loading overlay
            this.showRetrackLoading('Saving and retracking...');

            const allResponse = await fetch(`/proxy/api/masks/${studyUid}/${seriesUid}`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/x-tar',
                    'X-User-Email': this.getUserEmail(),
                    'X-Previous-Version-ID': this.currentVersionId || ''
                },
                body: archiveData
            });

            if (allResponse.ok) {
                const result = await allResponse.json();
                console.log(`✅ Saved and queued retrack: ${result.version_id}`);

                // Update version ID for next save
                this.currentVersionId = result.version_id;

                // DO NOT clear modified frames yet - wait for retrack to complete
                // Edits are only committed when retrack succeeds (atomic operation)
                // Keep edits visible so user can see them until retrack completes
                // Save button stays in unsaved state until retrack completes

                // Poll retrack status and reload when complete
                if (result.retrack_queued) {
                    // Update loading message and start polling - edits will be cleared only after retrack completes successfully
                    this.showRetrackLoading('Retracking in progress...');
                    this.pollRetrackStatus(studyUid, seriesUid);
                } else {
                    // No retrack queued (shouldn't happen, but handle gracefully)
                    this.hideRetrackLoading();
                videoModifiedFrames.clear();
                this.hasUnsavedChanges = false;
                    this.updateSaveButtonState();
                }
            } else {
                this.hideRetrackLoading();
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
                        // Retrack is already in progress - show loading and start polling
                        this.showRetrackLoading('Retrack already in progress...');
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
            this.hideRetrackLoading();
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
                // Keep loading visible during reload
                this.showRetrackLoading('Reloading retracked masks...');
                
                // Clear edits only after retrack completes successfully (atomic operation)
                const { studyUid, seriesUid, method } = this.currentVideo;
                const videoKey = this.getVideoKey(method, studyUid, seriesUid);
                const videoModifiedFrames = this.modifiedFrames.get(videoKey);
                if (videoModifiedFrames) {
                    videoModifiedFrames.clear();
                }
                this.hasUnsavedChanges = false;
                
                // Reload from server to ensure fresh masks
                this.frameCache.clear();
                this.framesArchive = null;
                this.masksArchive = {};
                await this.loadVideoData();
                await this.loadFramesArchive();
                // Force reload current frame to ensure fresh masks are displayed
                const currentFrameNum = this.currentFrame;
                this.currentFrame = -1; // Force reload
                await this.goToFrame(currentFrameNum);
                
                // Update save button state after reload
                this.updateSaveButtonState();
                this.hideRetrackLoading();
            } else if (status.status === 'failed') {
                console.error('Retrack failed:', status.error);
                this.hideRetrackLoading();
                alert(`Retrack failed: ${status.error || 'Unknown error'}`);
                // On failure, edits remain in modifiedFrames - user can try again or reset
                // Keep unsaved state so user knows edits are still pending
                this.hasUnsavedChanges = true;
                this.updateSaveButtonState();
            } else if (attempts < maxAttempts) {
                // Still processing - poll again
                setTimeout(poll, 1000);
            } else {
                this.hideRetrackLoading();
                alert('Retrack timeout - check server status');
            }
        };
        
        poll();
    }



    /**
     * Show completion confirmation modal.
     */
    showCompleteModal() {
        if (!this.currentVideo) {
            return;
        }
        // If already completed, show reopen option instead
        if (this.videoData && this.videoData.status === 'completed') {
            if (confirm('This series is already marked as complete. Reopen it for editing?')) {
                this.reopenSeries();
            }
            return;
        }
        this.showModal('completeModal');
    }

    /**
     * Confirm completion and optionally load next series.
     */
    async confirmCompleteSeries(loadNext = true) {
        // Hide modal immediately
        this.hideModal('completeModal');

        const { studyUid, seriesUid } = this.currentVideo;
        try {
            const userEmail = this.getUserEmail();
            const resp = await fetch(`/proxy/api/series/${studyUid}/${seriesUid}/complete`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-User-Email': userEmail || ''
                },
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                alert(`Failed to mark complete: ${data.error || resp.status}`);
                return;
            }
            
            // Refresh status from server (no client caching)
            await this.refreshCompletionStatus();
            this.updateCompletionIndicator();
            // Refresh all series statuses to update dropdown
            await this.refreshAllSeriesStatuses();
            
            if (loadNext) {
                // Get next series (server will exclude this completed series)
                await this.loadNextSeries();
            } else {
                alert('Series marked as complete.');
            }
        } catch (e) {
            alert(`Failed to mark complete: ${e.message}`);
        }
    }

    /**
     * Reopen a completed series for editing.
     */
    async reopenSeries() {
        if (!this.currentVideo) {
            return;
        }
        
        const { studyUid, seriesUid } = this.currentVideo;
        try {
            const resp = await fetch(`/proxy/api/series/${studyUid}/${seriesUid}/reopen`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
            });
            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                alert(`Failed to reopen series: ${data.error || resp.status}`);
                return;
            }
            
            // Refresh status from server (no client caching)
            await this.refreshCompletionStatus();
            this.updateCompletionIndicator();
            // Refresh all series statuses to update dropdown
            await this.refreshAllSeriesStatuses();
        } catch (e) {
            alert(`Failed to reopen series: ${e.message}`);
        }
    }

    /**
     * Update completion status indicator in UI.
     */
    updateCompletionIndicator() {
        const markCompleteBtn = document.getElementById('markComplete');
        const examBadge = document.getElementById('examBadge');
        
        if (!this.videoData) {
            return;
        }
        
        const isCompleted = this.videoData.status === 'completed';
        
        // Update button appearance and tooltip
        if (markCompleteBtn) {
            if (isCompleted) {
                markCompleteBtn.classList.add('completed');
                markCompleteBtn.title = 'Series Complete - Click to Reopen';
            } else {
                markCompleteBtn.classList.remove('completed');
                markCompleteBtn.title = 'Mark Complete & Get Next Series';
            }
        }
        
        // Update exam badge to show completion status
        if (examBadge) {
            const examText = examBadge.textContent;
            if (isCompleted) {
                if (!examText.includes('✓')) {
                    examBadge.textContent = examText.replace(/#/, '#✓ ');
                }
            } else {
                examBadge.textContent = examText.replace('#✓ ', '#');
            }
        }
    }

    /**
     * Update video select dropdown to show current series as selected.
     */
    updateVideoSelect() {
        const videoSelect = document.getElementById('videoSelect');
        if (!videoSelect || !this.currentVideo) {
            return;
        }
        
        const { method, studyUid, seriesUid } = this.currentVideo;
        const optionValue = `${method}|${studyUid}|${seriesUid}`;
        
        // Find and select the matching option
        for (let i = 0; i < videoSelect.options.length; i++) {
            if (videoSelect.options[i].value === optionValue) {
                videoSelect.selectedIndex = i;
                break;
            }
        }
    }

    /**
     * Refresh completion status from server (no client caching).
     * Updates videoData.status to reflect server state.
     */
    async refreshCompletionStatus() {
        if (!this.currentVideo) {
            return;
        }
        
        const { studyUid, seriesUid } = this.currentVideo;
        
        try {
            // Fetch fresh status from server
            const resp = await fetch(`/proxy/api/series/${studyUid}/${seriesUid}`);
            if (resp.ok) {
                const data = await resp.json();
                const serverStatus = data.status || 'pending';
                
                // Update local videoData (this is the only place we store status, no caching)
                if (this.videoData) {
                    this.videoData.status = serverStatus;
                }
            }
        } catch (e) {
            console.warn('Failed to refresh completion status from server:', e);
        }
    }

    /**
     * Refresh all series statuses from server and update dropdown.
     * This ensures dropdown reflects server state after operations like reset all.
     */
    async refreshAllSeriesStatuses() {
        const videoSelect = document.getElementById('videoSelect');
        if (!videoSelect) {
            return;
        }
        
        try {
            // Fetch all series with statuses and activity from server
            const resp = await fetch('/proxy/api/series');
            if (!resp.ok) {
                console.warn('Failed to fetch series list:', resp.status);
                return;
            }
            
            const allSeries = await resp.json();
            const currentUserEmail = this.getUserEmail();
            const now = new Date();
            const twentyFourHoursAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
            
            // Create a map of (study_uid, series_uid) -> {status, activity}
            const seriesMap = new Map();
            for (const series of allSeries) {
                const key = `${series.study_uid}|${series.series_uid}`;
                seriesMap.set(key, {
                    status: series.status || 'pending',
                    activity: series.activity || {}
                });
            }
            
            // Helper to determine indicator emoji based on priority
            const getIndicator = (status, activity) => {
                // Priority 1: Completed
                if (status === 'completed') {
                    return '✅';
                }
                
                // Priority 2: Other user active in last 24h (trumps current user activity)
                if (activity && Object.keys(activity).length > 0) {
                    let hasOtherUserActivity = false;
                    let hasCurrentUserActivity = false;
                    
                    for (const [userEmail, timestamp] of Object.entries(activity)) {
                        const activityTime = new Date(timestamp);
                        if (activityTime >= twentyFourHoursAgo) {
                            if (userEmail === currentUserEmail) {
                                hasCurrentUserActivity = true;
                            } else {
                                hasOtherUserActivity = true;
                            }
                        }
                    }
                    
                    // Priority 2: Other user active in last 24h (warning trumps current user)
                    if (hasOtherUserActivity) {
                        return '⚠️';
                    }
                    
                    // Priority 3: Only current user active in last 24h
                    if (hasCurrentUserActivity) {
                        return '🖌️';
                    }
                }
                
                // Priority 4: Inactive (no recent activity)
                return '◻️';
            };
            
            // Update all dropdown options
            for (let i = 0; i < videoSelect.options.length; i++) {
                const option = videoSelect.options[i];
                const [method, studyUid, seriesUid] = option.value.split('|');
                const key = `${studyUid}|${seriesUid}`;
                const seriesData = seriesMap.get(key);
                
                if (!seriesData) {
                    // Series not found in API response, use default
                    const match = option.text.match(/Exam (\d+)/);
                    if (match) {
                        option.text = `◻️ Exam ${match[1]}`;
                    }
                    continue;
                }
                
                // Extract exam number from current text
                const match = option.text.match(/Exam (\d+)/);
                if (match) {
                    const examNum = match[1];
                    const indicator = getIndicator(seriesData.status, seriesData.activity);
                    option.text = `${indicator} Exam ${examNum}`;
                }
            }
        } catch (e) {
            console.warn('Failed to refresh all series statuses:', e);
        }
    }

    /**
     * Update video select dropdown option text to reflect current completion status.
     * @deprecated Use refreshAllSeriesStatuses() instead for better consistency.
     */
    updateVideoSelectDropdown() {
        // For now, just refresh all statuses to ensure consistency
        this.refreshAllSeriesStatuses();
    }

    async confirmResetRetrack() {
        const { studyUid, seriesUid } = this.currentVideo;
        if (!studyUid || !seriesUid) {
            alert('No series selected');
            return;
        }

        // Show blocking loading overlay
        this.showRetrackLoading('Resetting retrack data...');

        try {
            const resp = await fetch(`/proxy/api/reset-retrack/${studyUid}/${seriesUid}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                alert(`Failed to reset retrack data: ${data.error || resp.status}`);
                this.hideRetrackLoading();
                return;
            }

            const result = await resp.json();
            // Keep loading visible during reload
            this.showRetrackLoading('Reloading reset masks...');

            // Clear all local edits for this video (atomic operation - edits only persist on retrack success)
            const videoKey = `${studyUid}__${seriesUid}`;
            this.modifiedFrames.delete(videoKey);
            this.hasUnsavedChanges = false;
            this.updateSaveButtonState();
            
            // Reload current series with reset masks (initial tracking)
            // Clear ALL caches first
            this.frameCache.clear();
            this.framesArchive = null;
            this.masksArchive = {};
            // Clear videoData to force fresh fetch from server (no client caching)
            this.videoData = null;

            // Reload current series directly (don't call loadNextSeries - stay on current series)
            // The user was just active on this series, so it should be returned by /api/series/next
            // But to be safe, reload the current series directly first, then verify with /api/series/next
            const currentFrameNum = this.currentFrame;
            // Force fresh fetch by skipping modified_frames (they're gone after reset)
            await this.loadVideoData(true); // skipModifiedFrames = true
            await this.loadFramesArchive();
            this.currentFrame = -1;
            await this.goToFrame(currentFrameNum);
            
            // Refresh completion status and activity indicators from server (no client caching)
            await this.refreshCompletionStatus();
            await this.refreshAllSeriesStatuses();
            
            // Update UI to reflect new status
            this.updateCompletionIndicator();
            
            this.hideRetrackLoading();
        } catch (e) {
            this.hideRetrackLoading();
            console.error('Error resetting retrack data:', e);
            alert(`Failed to reset retrack data: ${e.message}`);
        }
    }

    async confirmResetRetrackAll() {
        // Show blocking loading overlay
        this.showRetrackLoading('Resetting retrack data for all series...');

        try {
            const resp = await fetch(`/proxy/api/reset-retrack-all`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!resp.ok) {
                const data = await resp.json().catch(() => ({}));
                alert(`Failed to reset all retrack data: ${data.error || resp.status}`);
                this.hideRetrackLoading();
                return;
            }

            const result = await resp.json();
            // Keep loading visible during reload
            this.showRetrackLoading('Reloading reset masks...');

            // Clear all local edits for current video
            if (this.currentVideo && this.currentVideo.studyUid && this.currentVideo.seriesUid) {
                const { studyUid, seriesUid } = this.currentVideo;
                const videoKey = `${studyUid}__${seriesUid}`;
                
                // Mark activity immediately after reset (before reloading)
                await this.markSeriesActivity(studyUid, seriesUid);
                
                // Clear all local edits
                this.modifiedFrames.delete(videoKey);
                this.hasUnsavedChanges = false;
                this.updateSaveButtonState();
                
                // Clear ALL caches first
                this.frameCache.clear();
                this.framesArchive = null;
                this.masksArchive = {};
                
                // Reload current series directly (don't call loadNextSeries - stay on current series)
                const currentFrameNum = this.currentFrame;
                await this.loadVideoData(); // This will fetch fresh status from server
                await this.loadFramesArchive();
                this.currentFrame = -1;
                await this.goToFrame(currentFrameNum);
                
                // Refresh completion status from server (no client caching)
                await this.refreshCompletionStatus();
                
                // Update UI to reflect new status
                this.updateCompletionIndicator();
                // Refresh all series statuses to update dropdown (after activity is marked)
                await this.refreshAllSeriesStatuses();
            }
            
            this.hideRetrackLoading();
        } catch (e) {
            this.hideRetrackLoading();
            console.error('Error resetting all retrack data:', e);
            alert(`Failed to reset all retrack data: ${e.message}`);
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
        
        if (!this.currentVideo) {
            console.error('Cannot reset: no current video');
            return;
        }
        
        const { studyUid, seriesUid, method } = this.currentVideo;
        const videoKey = this.getVideoKey(method, studyUid, seriesUid);
        
        // Clear unsaved edits - ensure Map is completely removed
        this.modifiedFrames.delete(videoKey);
        this.hasUnsavedChanges = false;
        
        // Reload fresh from server (skip modified_frames)
        // This clears all caches, rebuilds videoData.mask_data from server, and updates UI
        await this.loadVideo(method, studyUid, seriesUid, true);
        
        // Force UI updates after reload to ensure fresh state is displayed
        this.updateSaveButtonState();
        this.updateSliderTypeBar();
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
                const errorText = await response.text().catch(() => response.statusText);
                throw new Error(`Failed to fetch next series: ${response.status} ${errorText}`);
            }
            
            let data;
            try {
                data = await response.json();
            } catch (e) {
                const text = await response.text();
                throw new Error(`Invalid JSON response: ${text.substring(0, 200)}`);
            }
            
            if (data.no_available_series) {
                alert('No available series to work on. All series may be completed or in progress.');
                return;
            }
            
            // Server returns SeriesMetadata with study_uid, series_uid, exam_number, etc.
            // We need to determine the method - for now, use flow_method from config or default
            // The server doesn't return method, so we'll need to infer it or use a default
            // For now, assume method is 'dis' (the flow method)
            const method = 'dis'; // TODO: Get from server response or config
            
            // Mark activity IMMEDIATELY when server selects a series (before any local checks)
            // This ensures other clients see activity right away
            const activityResponse = await this.markSeriesActivity(data.study_uid, data.series_uid);
            
            // Update warnings from the activity response (includes updated activity data)
            if (activityResponse && activityResponse.ok) {
                const activityData = await activityResponse.json().catch(() => null);
                if (activityData) {
                    await this.updateMultiplayerWarning(activityData);
                }
            }
            
            // Check if series exists locally before trying to load
            // The client needs to have synced its dataset to include this series
            const checkResponse = await fetch(`/api/video/${method}/${data.study_uid}/${data.series_uid}`);
            if (!checkResponse.ok) {
                if (checkResponse.status === 404) {
                    alert(`Series ${data.study_uid}/${data.series_uid} not found locally. Please sync your dataset first.`);
                    return;
                }
                throw new Error(`Failed to verify series locally: ${checkResponse.statusText}`);
            }
            
            // Load the series from server
            // Note: loadVideo() will also mark activity (redundant but safe)
            await this.loadVideo(method, data.study_uid, data.series_uid);
        } catch (error) {
            console.error('Error loading next series:', error);
            // Re-throw so caller can handle fallback (e.g., to INITIAL_VIDEO)
            throw error;
        }
    }
    
    /**
     * Navigate to next video using server's selection logic.
     */
    async navigateVideoNext() {
        await this.loadNextSeries();
    }
    
    /**
     * Mark current series as complete (note user), then get next series.
     * This replaces the old "next series" button - now it marks complete first.
     */
    
    /**
     * Get user email from localStorage (set during init).
     * This is sent in X-User-Email header to identify the user.
     */
    getUserEmail() {
        return this.userEmail || localStorage.getItem('userEmail') || '';
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
     * Mark a series as active with the current user.
     * Can be called with specific study/series UIDs or uses current video.
     */
    async markSeriesActivity(studyUid, seriesUid) {
        if (!studyUid || !seriesUid) {
            return null;
        }
        
        try {
            const response = await fetch(
                `/proxy/api/series/${studyUid}/${seriesUid}/activity`,
                {
                    method: 'POST',
                    headers: {
                        'X-User-Email': this.getUserEmail() || ''
                    }
                }
            );
            
            if (!response.ok) {
                console.warn(`Activity mark failed: ${response.status} ${response.statusText}`);
                return null;
            } else {
                console.debug('Activity marked successfully');
                return response; // Return response so caller can get updated data
            }
        } catch (error) {
            console.warn('Activity mark error:', error);
            return null;
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
        
        // Use the shared method
        const response = await this.markSeriesActivity(this.currentVideo.studyUid, this.currentVideo.seriesUid);
        
        // Update warnings after activity ping (response includes updated activity data)
        if (response && response.ok) {
            const data = await response.json().catch(() => null);
            if (data) {
                await this.updateMultiplayerWarning(data);
            }
        }
    }
    
    /**
     * Fetch series detail and update warnings.
     */
    async fetchAndUpdateWarnings() {
        if (!this.currentVideo || !this.currentVideo.studyUid) {
            return;
        }
        
        try {
            const response = await fetch(
                `/proxy/api/series/${this.currentVideo.studyUid}/${this.currentVideo.seriesUid}`
            );
            if (response.ok) {
                const data = await response.json();
                await this.updateMultiplayerWarning(data);
            }
        } catch (error) {
            console.warn('Failed to fetch series detail for warnings:', error);
        }
    }
    
    /**
     * Check for multiplayer warnings and update the banner.
     * Warnings shown for:
     * - Series is completed (status === 'completed')
     * - Other users have been active recently (within 24 hours)
     */
    async updateMultiplayerWarning(seriesData) {
        const warningEl = document.getElementById('multiplayerWarning');
        if (!warningEl || !this.currentVideo) {
            return;
        }
        
        const currentUserEmail = this.getUserEmail();
        const warnings = [];
        
        // Check if series is completed
        if (seriesData.status === 'completed') {
            warnings.push('⚠️ This series is marked as completed');
        }
        
        // Check for recent activity from other users
        if (seriesData.activity) {
            const now = new Date();
            const twentyFourHoursAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
            let mostRecentOtherUser = null;
            let mostRecentOtherTime = null;
            
            for (const [userEmail, timestamp] of Object.entries(seriesData.activity)) {
                if (userEmail === currentUserEmail) {
                    continue; // Skip current user
                }
                const activityTime = new Date(timestamp);
                if (activityTime >= twentyFourHoursAgo) {
                    if (!mostRecentOtherTime || activityTime > mostRecentOtherTime) {
                        mostRecentOtherTime = activityTime;
                        mostRecentOtherUser = userEmail;
                    }
                }
            }
            
            if (mostRecentOtherUser) {
                const timeAgo = Math.round((now - mostRecentOtherTime) / (1000 * 60)); // minutes ago
                const timeStr = timeAgo < 60 ? `${timeAgo}m ago` : `${Math.round(timeAgo / 60)}h ago`;
                warnings.push(`⚠️ Another user (${mostRecentOtherUser}) was active ${timeStr}`);
            }
        }
        
        // Update banner
        if (warnings.length > 0) {
            warningEl.textContent = warnings.join(' • ');
            warningEl.classList.remove('hidden');
        } else {
            warningEl.classList.add('hidden');
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

    async loadVideo(method, studyUid, seriesUid, skipModifiedFrames = false) {
        // Stop activity pings for previous video (if any)
        this.stopActivityPings();
        
        // Mark activity immediately when viewing a series
        await this.markSeriesActivity(studyUid, seriesUid);
        
        this.frameImage = null;
        this.frameCache.clear();
        this.framesArchive = null;
        this.masksArchive = {};
        this.currentVersionId = null; // Reset version ID - will be set from server response
        this.currentVideo = { method, studyUid, seriesUid };
        // Load video data first so we know which frames are modified
        await this.loadVideoData(skipModifiedFrames);
        await this.loadFramesArchive();
        this.currentFrame = -1;
        await this.goToFrame(0);
        this.updateSaveButtonState();
        this.updateVideoInfo();
        this.updateCompletionIndicator(); // Update completion status indicator
        this.updateVideoSelect(); // Update dropdown to show current series
        await this.refreshAllSeriesStatuses(); // Update all indicators with activity data
        
        // Fetch series detail to get activity data and update warnings
        await this.fetchAndUpdateWarnings();
        
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
            // Client backend proxies to server (no local caching)
            const url = `/proxy/api/frames/${method}/${studyUid}/${seriesUid}`;
            const response = await fetch(url);
            
            if (!response.ok) {
                if (response.status === 404) {
                    throw new Error(`Series ${studyUid}/${seriesUid} not found locally. Please sync your dataset.`);
                }
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `Failed to load frames: ${response.statusText}`);
            }
            
            const { frames_archive_url, masks_archive_url } = await response.json();

            // Rewrite to proxy if server returned /api/... paths
            let framesUrl = frames_archive_url;
            if (framesUrl && framesUrl.startsWith("/api/")) {
                framesUrl = `/proxy${framesUrl}`;
            }
            let masksUrl = masks_archive_url;
            if (masksUrl && masksUrl.startsWith("/api/")) {
                masksUrl = `/proxy${masksUrl}`;
            }

            // Load frames archive (.tar format, no gzip)
            // Client backend proxies to server (no local caching)
            const framesArchiveResponse = await fetch(framesUrl);
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
            if (masksUrl) {
                // Client backend proxies to server (no local caching)
                const masksArchiveResponse = await fetch(masksUrl);
                
                if (!masksArchiveResponse.ok) {
                    console.warn(`Failed to load masks archive: ${masksArchiveResponse.status} ${masksArchiveResponse.statusText}`);
                    // Try to get error details
                    const errorText = await masksArchiveResponse.text().catch(() => '');
                    try {
                        const errorData = JSON.parse(errorText);
                        console.warn('Mask archive error:', errorData);
                        if (errorData.error_code === 'TRACK_FAILED' || errorData.error_code === 'TRACK_MISSING_ARCHIVE' || errorData.error_code === 'TRACK_PENDING') {
                            // Series not tracked yet or tracking failed - this is OK, just no masks
                            console.log(`Series tracking status: ${errorData.error_code} - continuing without masks`);
                        } else {
                            throw new Error(errorData.error || `Failed to load masks: ${masksArchiveResponse.statusText}`);
                        }
                    } catch (e) {
                        if (e instanceof Error && e.message.includes('Failed to load masks')) {
                            throw e;
                        }
                        // If we can't parse the error, just log and continue
                        console.warn('Could not parse mask archive error, continuing without masks');
                    }
                    // Continue without masks - viewer will show frames only
                } else {
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
            }
        } catch (error) {
            console.error('Error loading frames/masks archive:', error);
            // Don't throw - allow viewer to continue with frames only if masks fail
            if (error.message && !error.message.includes('not tracked yet') && !error.message.includes('TRACK_')) {
                alert(`Warning: ${error.message}. Viewer will continue without masks.`);
            }
        } finally {
            loadingDiv.style.display = 'none';
        }
    }

    async showModal(modalId) {
        // If showing series selection modal, refresh series statuses first to ensure fresh data
        if (modalId === 'seriesSelectModal') {
            await this.refreshAllSeriesStatuses();
            this.updateVideoSelect(); // Update dropdown to show current selection
        }
        
        // Show modal after statuses are refreshed (for seriesSelectModal) or immediately (for others)
        document.getElementById(modalId).classList.add('active');
    }

    hideModal(modalId) {
        document.getElementById(modalId).classList.remove('active');
    }

    showRetrackLoading(message = 'Processing retrack...') {
        const loadingModal = document.getElementById('retrackLoadingModal');
        const loadingText = document.getElementById('retrackLoadingText');
        if (loadingText) {
            loadingText.textContent = message;
        }
        if (loadingModal) {
            loadingModal.classList.add('active');
            // Prevent closing by clicking outside
            loadingModal.style.pointerEvents = 'auto';
        }
    }

    hideRetrackLoading() {
        const loadingModal = document.getElementById('retrackLoadingModal');
        if (loadingModal) {
            loadingModal.classList.remove('active');
        }
    }

    updateVideoInfo() {
        // Update video-level info (study, series, method, exam)
        const infoStudy = document.getElementById('infoStudy');
        const infoSeries = document.getElementById('infoSeries');
        const infoMethod = document.getElementById('infoMethod');
        const infoExam = document.getElementById('infoExam');
        const examBadge = document.getElementById('examBadge');
        
        // Use videoData if available, otherwise fall back to currentVideo
        const studyUid = this.videoData?.study_uid || this.currentVideo?.studyUid;
        const seriesUid = this.videoData?.series_uid || this.currentVideo?.seriesUid;
        const method = this.videoData?.method || this.currentVideo?.method;
        const examValue = this.videoData?.exam_number || 'Unknown';
        
        if (infoStudy && studyUid) infoStudy.textContent = studyUid;
        if (infoSeries && seriesUid) infoSeries.textContent = seriesUid;
        if (infoMethod && method) infoMethod.textContent = method;
        if (infoExam) infoExam.textContent = examValue;
        
        // Always update exam badge if we have any video info
        if (examBadge) {
            // Try to get exam number from videoData first, then from API response
            let finalExamValue = examValue;
            if ((!finalExamValue || finalExamValue === 'Unknown') && this.videoData) {
                // Check if exam_number was set in loadVideoData
                finalExamValue = this.videoData.exam_number || 'Unknown';
            }
            
            const badgeText = (finalExamValue && finalExamValue !== 'Unknown' && (typeof finalExamValue === 'number' || /^\d+$/.test(String(finalExamValue))))
                ? `# ${finalExamValue}`
                : '# --';
            examBadge.textContent = badgeText;
            examBadge.title = finalExamValue && finalExamValue !== 'Unknown' ? `Exam ${finalExamValue}` : 'Click to select series';
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
        const resp = await fetch('/api/dataset/version_status').catch(() => null);
        if (!resp || !resp.ok) {
            // Silently fail - dataset sync status is not critical
            return;
        }
        const data = await resp.json();
        const banner = document.getElementById('datasetWarning');
        if (!banner) return;
        if (data.in_sync) {
            banner.classList.add('hidden');
        } else {
            banner.classList.remove('hidden');
        }
    } catch (e) {
        // Silently fail - dataset sync status is not critical
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
