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

        // Retry state tracking for robust reconnect
        this.retryController = null;  // AbortController for current retry
        this.isRetrying = false;      // Guard against concurrent retries

        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;

        this.videoData = null;
        this.hasUnsavedChanges = false;
        this.renderRect = null; // For coordinate conversion
        
        // Track all modified frames across the session, per video
        // Structure: Map<videoKey, Map<frame_num, { maskData, maskType, is_empty }>>
        this.modifiedFrames = new Map(); // videoKey -> Map(frame_num -> data)
        
        
        this.frameCache = new Map(); // Kept for compatibility but no longer caches ImageData
        this.framesArchive = null; // Extracted frames from tar (compressed WebP)
        this.masksArchive = {}; // Masks from tar archive (compressed WebP)
        this.currentMaskWebP = null; // WebP bytes for current frame's mask
        this.versionIds = new Map(); // Map<videoKey, versionId> - per-series version tracking
        this.activityPingInterval = null; // Interval ID for activity pings (30s)
        this.userEmail = null; // User email for identification (from /api/settings)

        this.toolsVisible = true;
        this._renderGeneration = 0;
        
        // Server URL for direct API calls (injected from template)
        this.serverUrl = typeof SERVER_URL !== 'undefined' ? SERVER_URL : 'http://localhost:5000';
        this.clientUrl = typeof CLIENT_URL !== 'undefined' ? CLIENT_URL : 'http://localhost:8080';
        
        // Connection state tracking
        this.hasConnectedToServer = false; // Track if we've ever successfully connected
        this.serverConnectionRetryCount = 0;
        this.serverConnectionRetryTimeout = null;
        this.serverConnectionWarningVisible = false;
        this.reconnectPingInterval = null;
        this.needsRemoteServerConnection = false;

        this.init();
    }
    
    /**
     * Get full URL for server API endpoint (direct call, no proxy).
     */
    serverUrlFor(path) {
        // Remove leading slash if present (we'll add it)
        const cleanPath = path.startsWith('/') ? path.slice(1) : path;
        return `${this.serverUrl}/${cleanPath}`;
    }
    
    /**
     * Add Cloudflare headers to a headers object if configured.
     */
    _addCloudflareHeaders(headers = {}) {
        const result = { ...headers };
        
        // Add Cloudflare headers if configured
        if (typeof CF_ACCESS_CLIENT_ID !== 'undefined' && CF_ACCESS_CLIENT_ID) {
            result['CF-Access-Client-Id'] = CF_ACCESS_CLIENT_ID;
        }
        if (typeof CF_ACCESS_CLIENT_SECRET !== 'undefined' && CF_ACCESS_CLIENT_SECRET) {
            result['CF-Access-Client-Secret'] = CF_ACCESS_CLIENT_SECRET;
        }
        
        return result;
    }
    
    /**
     * Fetch from server API endpoint with Cloudflare headers if configured.
     * Accepts either a path (e.g., "api/status") or a full URL (e.g., "https://goo.badmath.org/api/status").
     */
    async fetchToServer(pathOrUrl, options = {}) {
        const headers = this._addCloudflareHeaders(options.headers);
        // If it's already a full URL, use it as-is; otherwise build from serverUrl
        const url = pathOrUrl.startsWith('http://') || pathOrUrl.startsWith('https://')
            ? pathOrUrl
            : this.serverUrlFor(pathOrUrl);

        // Add abort signal from retryController if retrying and no signal provided
        const signal = options.signal || (this.retryController?.signal);

        return fetch(url, { ...options, headers, signal });
    }
    
    /**
     * Get full URL for client backend endpoint (frames, settings, etc.).
     */
    clientUrlFor(path) {
        // Remove leading slash if present (we'll add it)
        const cleanPath = path.startsWith('/') ? path.slice(1) : path;
        return `${this.clientUrl}/${cleanPath}`;
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
        this.updateMarkCompleteButtonState();
    }

    updateMarkCompleteButtonState() {
        const btn = document.getElementById('markComplete');
        if (!btn) return;
        const hasUnsaved = this.hasUnsavedChangesForCurrentVideo();
        const isCompleted = this.videoData && this.videoData.status === 'completed';
        if (hasUnsaved || isCompleted) {
            btn.classList.add('disabled');
            btn.title = hasUnsaved
                ? 'Save changes before marking complete'
                : 'Series already complete';
        } else {
            btn.classList.remove('disabled');
            btn.title = 'Mark Complete & Get Next Series';
        }
    }

    resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const w = window.innerWidth;
        const h = window.innerHeight;

        // Set canvas backing store to physical pixels for crisp Retina rendering
        this.canvas.width = w * dpr;
        this.canvas.height = h * dpr;

        // Scale context so drawing code uses logical (CSS) coordinates
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        this.updateButtonSize();

        // Overlay canvas will be sized to image dimensions in goToFrame()
        if (this.frameImage) {
            this.render();
        }
    }

    updateButtonSize() {
        const vh = window.innerHeight;
        const sidebarTop = vh * 0.07;
        const bottomBar = vh * 0.15;
        const available = vh - sidebarTop - bottomBar;

        // Right bar (tallest): badge + 3 buttons + 3 gaps(0.1) + 2 paddings(0.1) = 4.5 * btn
        const computed = Math.floor(available / 4.5);
        // Phone/iPad mini (vh<800): smaller buttons; iPad Air+ (vh>=800): larger buttons
        const minSize = vh < 800 ? 36 : 44;
        const maxSize = vh < 800 ? 56 : 88;
        const btnSize = Math.max(minSize, Math.min(maxSize, computed));

        document.documentElement.style.setProperty('--btn-size', `${btnSize}px`);
    }

    async init() {
        this.showServerConnectionScreen('Connecting...');

        this.setupEventListeners();
        this.initBrushPreview();
        await this.loadSettings();
        await this.connectToServerWithRetry();
        await this.checkCredentialsAndSync();
        
        this.checkDatasetSyncStatus().catch(() => {});
    }
    
    /**
     * Check if credentials are set, and if so, trigger sync.
     * - Missing credentials → Setup page
     * - Invalid token → Setup page (to re-enter)
     * - Network/other error → Sync error overlay with retry button
     */
    async checkCredentialsAndSync() {
        try {
            // Check if credentials are set
            const settingsResp = await fetch('/api/settings');
            if (!settingsResp.ok) {
                this.showConfigModal();
                return;
            }
            
            const settings = await settingsResp.json();
            const hasToken = settings.mdai_token_present === true;
            const hasName = settings.user_email?.trim().length > 0;
            
            if (!hasToken || !hasName) {
                // Missing credentials - show blocking config modal
                this.showConfigModal();
                return;
            }
            
            // Credentials present - try sync
            this.showServerConnectionScreen('Syncing dataset...');
            const syncResp = await fetch('/api/dataset/sync', { method: 'POST' });
            
            if (!syncResp.ok) {
                const errorData = await syncResp.json().catch(() => ({}));
                console.error('Sync failed:', errorData);
                
                // Handle based on error type
                if (errorData.error_type === 'invalid_token') {
                    // Token invalid - show setup page to re-enter
                    this.showConfigModal();
                } else {
                    // Network/other error - show retry UI (don't lose credentials)
                    this.showSyncError(errorData.error_message || 'Sync failed');
                }
                return;
            }
            
            await new Promise(resolve => setTimeout(resolve, 500));

            await this.populateVideoSelectFromLocal();
            await this.refreshAllSeriesStatuses();

            // Server-dependent: loadNextSeries talks to remote server
            // If server is unreachable, show connection screen (not "Sync Failed")
            try {
                await this.loadNextSeries();
                this.hideServerConnectionScreen();
            } catch (serverError) {
                console.warn('Server unreachable after sync:', serverError.message);
                await this.connectToRemoteServerWithRetry();
            }
        } catch (error) {
            console.error('Error checking credentials/sync:', error);
            // Network errors during fetch - show retry UI
            this.showSyncError(error.message || 'Connection failed');
        }
    }
    
    /**
     * Show blocking config modal (no_videos.html setup page).
     */
    showConfigModal() {
        // Redirect to setup page
        window.location.href = '/no_videos';
    }
    
    /**
     * Show sync error overlay with retry button.
     * Used for network errors where credentials are valid but sync failed.
     */
    showSyncError(message) {
        this.hideServerConnectionScreen();
        const overlay = document.getElementById('syncErrorOverlay');
        const messageEl = document.getElementById('syncErrorMessage');
        if (overlay && messageEl) {
            messageEl.textContent = message;
            overlay.classList.remove('hidden');
        }
    }
    
    /**
     * Hide sync error overlay.
     */
    hideSyncError() {
        const overlay = document.getElementById('syncErrorOverlay');
        if (overlay) {
            overlay.classList.add('hidden');
        }
    }
    
    /**
     * Retry sync from error overlay.
     */
    async retrySyncFromError() {
        this.hideSyncError();
        await this.checkCredentialsAndSync();
    }
    
    /**
     * Resync dataset from settings modal.
     */
    async resyncDataset() {
        const btn = document.getElementById('resyncDataset');
        if (!btn) return;
        
        const originalText = btn.textContent;
        btn.disabled = true;
        btn.textContent = '🔄 Syncing...';
        
        try {
            this.showServerConnectionScreen('Syncing dataset...');
            const syncResp = await fetch('/api/dataset/sync', { method: 'POST' });
            
            if (!syncResp.ok) {
                const errorData = await syncResp.json().catch(() => ({}));
                throw new Error(errorData.error || 'Sync failed');
            }
            
            const syncData = await syncResp.json();
            
            // Wait a moment for dataset to be indexed
            await new Promise(resolve => setTimeout(resolve, 500));
            
            // Refresh UI
            await this.populateVideoSelectFromLocal();
            await this.refreshAllSeriesStatuses();
            
            // Hide connection screen and close settings modal
            this.hideServerConnectionScreen();
            this.hideModal('settingsModal');
            
            // Reload current series if available, otherwise load next
            if (this.currentVideo) {
                await this.loadVideo(
                    this.currentVideo.method,
                    this.currentVideo.study_uid,
                    this.currentVideo.series_uid
                );
            } else {
                await this.loadNextSeries();
            }
        } catch (error) {
            console.error('Resync failed:', error);
            alert(`Resync failed: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.textContent = originalText;
        }
    }
    
    /**
     * Connect to server with exponential backoff retry.
     * Shows blocking screen until first successful connection.
     */
    async connectToServerWithRetry(forceImmediate = false) {
        // If forcing immediate (from manual button), clear any pending timeout
        if (forceImmediate && this.serverConnectionRetryTimeout) {
            clearTimeout(this.serverConnectionRetryTimeout);
            this.serverConnectionRetryTimeout = null;
            this.serverConnectionRetryCount = 0;
        }
        
        this.showServerConnectionScreen('Connecting...');
        
        const maxRetries = 30;
        let retryDelay = 1000; // Start with 1 second
        
        const attemptConnection = async () => {
            try {
                // Check if backend is responding (don't try to load data - that requires sync)
                const healthResp = await fetch('/healthz', { 
                    method: 'GET',
                    signal: AbortSignal.timeout(5000)
                });
                
                if (!healthResp.ok) {
                    throw new Error(`Backend returned ${healthResp.status}`);
                }
                
                const healthData = await healthResp.json();
                // Health check just verifies LOCAL backend is running - don't check client_ready
                // (data sync status is checked separately)
                
                this.hasConnectedToServer = true;
                this.serverConnectionRetryCount = 0;
                return true;
            } catch (error) {
                this.serverConnectionRetryCount++;
                
                if (this.serverConnectionRetryCount >= maxRetries) {
                    this.showServerConnectionScreen('Backend Error');
                    return false;
                }
                
                retryDelay = Math.min(1000 * Math.pow(2, this.serverConnectionRetryCount - 1), 512000);
                this.showServerConnectionScreen('Connecting...');
                
                // Schedule next retry
                this.serverConnectionRetryTimeout = setTimeout(() => {
                    attemptConnection();
                }, retryDelay);
                
                return false;
            }
        };
        
        return await attemptConnection();
    }

    /**
     * Retry connecting to the REMOTE server indefinitely.
     * Pings api/status every 3s until success or abort.
     * On success, loads next series and hides connection screen.
     */
    async connectToRemoteServerWithRetry() {
        // Guard: prevent concurrent retries
        if (this.isRetrying) {
            console.log('Retry already in progress, skipping');
            return;
        }

        this.isRetrying = true;
        this.retryController = new AbortController();
        this.needsRemoteServerConnection = true;
        this.showServerConnectionScreen('Connecting to server...');

        const interval = 3000;

        try {
            while (true) {  // Retry forever until success or abort
                try {
                    const resp = await this.fetchToServer('api/status');
                    if (resp.ok) {
                        this.needsRemoteServerConnection = false;
                        await this.loadNextSeries();
                        this.hideServerConnectionScreen();
                        return;  // Success!
                    }
                } catch (e) {
                    if (e.name === 'AbortError') {
                        console.log('Retry aborted');
                        return;  // Aborted, exit cleanly
                    }
                    // Other errors - continue retrying
                }
                await new Promise(r => setTimeout(r, interval));
            }
        } finally {
            // Always clean up (runs even on return/throw)
            this.isRetrying = false;
            this.retryController = null;
        }
    }

    /**
     * Show blocking server connection screen.
     */
    showServerConnectionScreen(message) {
        const screen = document.getElementById('serverConnectionScreen');
        const headingEl = document.getElementById('serverConnectionHeading');
        const messageEl = document.getElementById('serverConnectionMessage');
        const detailsEl = document.getElementById('serverConnectionDetails');
        const spinnerEl = document.getElementById('serverConnectionSpinner');
        
        if (screen) {
            // Ensure screen is visible before updating content to prevent black flash
            screen.classList.remove('hidden');
            screen.style.display = 'flex';
            screen.style.opacity = '1';
            screen.style.pointerEvents = 'auto';
            screen.style.zIndex = '99999';
            
            // Update heading
            if (headingEl) {
                headingEl.textContent = message || 'Syncing dataset...';
            }
            
            // Set emoji based on message
            if (spinnerEl) {
                if (message === 'Connecting...' || message.startsWith('Connecting')) {
                    spinnerEl.textContent = '🔌';
                } else if (message === 'Syncing dataset...' || message.startsWith('Syncing')) {
                    spinnerEl.textContent = '📥';
                } else {
                    spinnerEl.textContent = '🔌';
                }
            }
            
            // Hide unused message and details elements
            if (messageEl) {
                messageEl.textContent = '';
                messageEl.style.display = 'none';
            }
            if (detailsEl) {
                detailsEl.textContent = '';
                detailsEl.style.display = 'none';
            }
        }
    }
    
    /**
     * Hide blocking server connection screen.
     */
    hideServerConnectionScreen() {
        const screen = document.getElementById('serverConnectionScreen');
        if (screen) {
            screen.style.opacity = '0';
            screen.style.pointerEvents = 'none';
            setTimeout(() => {
                screen.classList.add('hidden');
            }, 200);
        }
        if (this.serverConnectionRetryTimeout) {
            clearTimeout(this.serverConnectionRetryTimeout);
            this.serverConnectionRetryTimeout = null;
        }
    }
    
    /**
     * Show non-blocking server connection warning (after initial connection).
     */
    showServerConnectionWarning() {
        if (this.hasConnectedToServer && !this.serverConnectionWarningVisible) {
            const warning = document.getElementById('serverConnectionWarning');
            if (warning) {
                warning.classList.remove('hidden');
                this.serverConnectionWarningVisible = true;
            }
            this.startServerReconnectPings();
        }
    }

    /**
     * Hide server connection warning.
     */
    hideServerConnectionWarning() {
        const warning = document.getElementById('serverConnectionWarning');
        if (warning) {
            warning.classList.add('hidden');
            this.serverConnectionWarningVisible = false;
        }
        this.stopServerReconnectPings();
    }

    /**
     * Background pings while warning banner is visible.
     * Auto-dismisses the banner when server comes back.
     */
    startServerReconnectPings() {
        if (this.reconnectPingInterval) return;
        const started = Date.now();
        this.reconnectPingInterval = setInterval(async () => {
            try {
                const resp = await this.fetchToServer('api/status');
                if (resp.ok) {
                    this.hideServerConnectionWarning();
                    return;
                }
            } catch (e) { /* still down */ }
            // Escalate to blocking after 30s
            if (Date.now() - started >= 30000) {
                this.stopServerReconnectPings();
                this.hideServerConnectionWarning();
                this.showServerConnectionScreen('Server unavailable');
                this.needsRemoteServerConnection = true;
            }
        }, 10000);
    }

    stopServerReconnectPings() {
        if (this.reconnectPingInterval) {
            clearInterval(this.reconnectPingInterval);
            this.reconnectPingInterval = null;
        }
    }
    
    async ensureUserEmail() {
        // User email comes from /api/settings (loaded in loadSettings())
        let userEmail = this.getUserEmail();
        
        if (!userEmail) {
            // Prompt user for email if not set in backend
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

            // Persist to backend (this saves to credentials.json)
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
                const emailSelect = document.getElementById('settingsEmail');
                if (emailSelect) {
                    const matchingOption = Array.from(emailSelect.options).find(
                        opt => opt.value === data.user_email
                    );
                    emailSelect.value = matchingOption ? data.user_email : '';
                }
            }
            const tokenStatus = document.getElementById('tokenStatus');
            if (tokenStatus) {
                tokenStatus.textContent = data.mdai_token_present ? 'Token: set' : 'Token: not set';
            }
        } catch (err) {
            console.warn('Failed to load settings', err);
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
        const emailSelect = document.getElementById('settingsEmail');
        if (emailSelect) {
            const currentEmail = this.getUserEmail();
            const matchingOption = Array.from(emailSelect.options).find(
                opt => opt.value === currentEmail
            );
            emailSelect.value = matchingOption ? currentEmail : '';
        }
        this.updateVideoInfo();
        this.showModal('settingsModal');
    }

    setupEventListeners() {
        // Modal controls
        document.getElementById('examBadge').addEventListener('click', () => this.showModal('seriesSelectModal'));
        document.getElementById('closeSeriesSelect').addEventListener('click', () => this.hideModal('seriesSelectModal'));
        document.getElementById('closeKeyboardShortcuts').addEventListener('click', () => this.hideModal('keyboardShortcutsModal'));
        document.getElementById('settingsBtn').addEventListener('click', () => this.openSettings());
        document.getElementById('closeSettings').addEventListener('click', () => this.hideModal('settingsModal'));
        const saveSettingsBtn = document.getElementById('saveSettings');
        if (saveSettingsBtn) saveSettingsBtn.addEventListener('click', () => this.saveSettings());
        const resyncBtn = document.getElementById('resyncDataset');
        if (resyncBtn) resyncBtn.addEventListener('click', () => this.resyncDataset());

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

        // Reset retrack buttons — show confirmation modal instead of acting directly
        const resetRetrackBtn = document.getElementById('resetRetrackBtn');
        if (resetRetrackBtn) resetRetrackBtn.addEventListener('click', () => {
            this.hideModal('settingsModal');
            this.showModal('resetRetrackModal');
        });

        const resetRetrackAllBtn = document.getElementById('resetRetrackAllBtn');
        if (resetRetrackAllBtn) resetRetrackAllBtn.addEventListener('click', () => {
            this.hideModal('settingsModal');
            this.showModal('resetRetrackAllModal');
        });

        // Reset retrack confirmation modals
        const closeResetRetrack = document.getElementById('closeResetRetrack');
        const cancelResetRetrack = document.getElementById('cancelResetRetrack');
        const confirmResetRetrackBtn = document.getElementById('confirmResetRetrack');
        if (closeResetRetrack) closeResetRetrack.addEventListener('click', () => this.hideModal('resetRetrackModal'));
        if (cancelResetRetrack) cancelResetRetrack.addEventListener('click', () => this.hideModal('resetRetrackModal'));
        if (confirmResetRetrackBtn) confirmResetRetrackBtn.addEventListener('click', () => {
            this.hideModal('resetRetrackModal');
            this.confirmResetRetrack();
        });

        const closeResetRetrackAll = document.getElementById('closeResetRetrackAll');
        const cancelResetRetrackAll = document.getElementById('cancelResetRetrackAll');
        const confirmResetRetrackAllBtn = document.getElementById('confirmResetRetrackAll');
        if (closeResetRetrackAll) closeResetRetrackAll.addEventListener('click', () => this.hideModal('resetRetrackAllModal'));
        if (cancelResetRetrackAll) cancelResetRetrackAll.addEventListener('click', () => this.hideModal('resetRetrackAllModal'));
        if (confirmResetRetrackAllBtn) confirmResetRetrackAllBtn.addEventListener('click', () => {
            this.hideModal('resetRetrackAllModal');
            this.confirmResetRetrackAll();
        });

        // Video selection (server-driven)
        // Keep videoSelect for manual selection if needed, but wire Next/Prev to server
        const videoSelect = document.getElementById('videoSelect');
        if (videoSelect) {
            videoSelect.addEventListener('change', async (e) => {
                const [method, studyUid, seriesUid] = e.target.value.split('|');
                try {
                    // Fetch series detail from server to get version_id
                    const seriesResp = await this.fetchToServer(`api/series/${studyUid}/${seriesUid}`);
                    let serverVersionId = null;
                    if (seriesResp.ok) {
                        const seriesData = await seriesResp.json();
                        serverVersionId = seriesData.version_id;
                        // Update warnings from activity data in response
                        if (seriesData.activity) {
                            await this.updateMultiplayerWarning(seriesData);
                        }
                    }

                    // Mark activity when selecting from dropdown
                    await this.markSeriesActivity(studyUid, seriesUid);

                    await this.loadVideo(method, studyUid, seriesUid);

                    // Set version_id from server (server is definitive source)
                    if (serverVersionId !== undefined) {
                        this.setCurrentVersionId(serverVersionId);
                        console.log(`[Version] Loaded from server (dropdown): ${serverVersionId}`);
                    }

                    // Success - hide connection warning if shown
                    this.hideServerConnectionWarning();
                    // Close modal after selection
                    this.hideModal('seriesSelectModal');
                } catch (error) {
                    console.error('Failed to load series:', error);
                    // Show appropriate error based on connection state
                    if (this.hasConnectedToServer) {
                        // Connection lost after initial sync - show warning but don't block
                        this.showServerConnectionWarning();
                    } else {
                        this.showServerConnectionScreen('Failed to load series. Server unavailable.');
                    }
                }
            });
        }
        
        // Hourglass spinner is clickable - forces immediate connection attempt
        const serverConnectionSpinner = document.getElementById('serverConnectionSpinner');
        if (serverConnectionSpinner) {
            serverConnectionSpinner.addEventListener('click', async () => {
                if (this.needsRemoteServerConnection) {
                    await this.connectToRemoteServerWithRetry();
                } else {
                    await this.connectToServerWithRetry(true);
                }
            });
        }
        
        // Reconnect button (non-blocking warning)
        const reconnectBtn = document.getElementById('reconnectServerBtn');
        if (reconnectBtn) {
            reconnectBtn.addEventListener('click', async () => {
                try {
                    // Try to reconnect by checking server status
                    const response = await this.fetchToServer('api/status');
                    if (response.ok) {
                        // Server is back - hide warning and check version sync
                        this.hideServerConnectionWarning();
                        await this.checkDatasetSyncStatus();
                        // Try to reload current series to refresh status
                        if (this.currentVideo) {
                            await this.loadVideoData(true); // Skip modified frames
                            await this.refreshCompletionStatus();
                        }
                    } else {
                        throw new Error('Server not responding');
                    }
                } catch (error) {
                    console.error('Reconnection failed:', error);
                    // Keep warning visible
                }
            });
        }

        document.getElementById('markComplete').addEventListener('click', () => this.showCompleteModal());

        // Playback controls
        document.getElementById('playPause').addEventListener('click', () => this.togglePlayPause());
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

        // Drawing events (pointer — covers mouse, touch, and Apple Pencil)
        this.canvas.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            this.startDrawing(e);
        });
        this.canvas.addEventListener('pointermove', (e) => {
            if (this.isDrawing) {
                e.preventDefault();
                this.draw(e);
            }
        });
        this.canvas.addEventListener('pointerup', () => {
            this.stopDrawing();
        });
        this.canvas.addEventListener('pointerleave', () => {
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

        // Restart retry on wake if needed (but don't interfere with in-progress retry)
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && this.needsRemoteServerConnection && !this.isRetrying) {
                this.connectToRemoteServerWithRetry();
            }
        });
    }

    initBrushPreview() {
        const preview = document.getElementById('brushPreview');
        if (!preview) return;

        const getDisplayDiameter = (size) => {
            const scale = (this.renderRect && this.maskCanvas && this.maskCanvas.width)
                ? (this.renderRect.width / this.maskCanvas.width) : 1;
            return (size !== undefined ? size : this.brushSize) * scale * 2;
        };

        // Only offset the brush preview circle during slider drag
        let isSliderActive = false;
        const slider = document.getElementById('brushSizeInline');
        if (slider) {
            slider.addEventListener('pointerdown', () => { isSliderActive = true; });
            document.addEventListener('pointerup', () => { isSliderActive = false; });
        }

        document.addEventListener('pointermove', (e) => {
            const d = getDisplayDiameter();
            const offset = isSliderActive ? getDisplayDiameter(50) * 1.5 : 0;
            preview.style.left = `${e.clientX - d / 2 + offset}px`;
            preview.style.top = `${e.clientY - d / 2}px`;
            preview.style.width = `${d}px`;
            preview.style.height = `${d}px`;
            preview.style.display = 'block';
        });

        document.addEventListener('pointerleave', () => {
            preview.style.display = 'none';
        });
    }

    normalizeFrameType(type, hasMask) {
        if (type === 'empty') return 'empty';
        if (type === 'fluid') return 'fluid';
        if (type === 'tracked') return 'tracked';
        if (type?.startsWith('fluid_')) return 'tracked';
        return hasMask ? 'tracked' : 'empty';
    }

    async loadVideoData(skipModifiedFrames = false) {
        if (!this.currentVideo) {
            throw new Error('No current video set');
        }
        const { method, studyUid, seriesUid } = this.currentVideo;
        // Verify series exists locally first
        const verifyResponse = await fetch(`/api/video/${method}/${studyUid}/${seriesUid}`);
        if (!verifyResponse.ok) {
            if (verifyResponse.status === 404) {
                throw new Error(`Series ${studyUid}/${seriesUid} not found locally. Please sync your dataset.`);
            }
            throw new Error(`Failed to verify series: ${verifyResponse.statusText}`);
        }
        
        // Call server directly for video metadata
        const response = await this.fetchToServer(`api/video/${method}/${studyUid}/${seriesUid}`);
        
        if (!response.ok) {
            // Handle server unavailability
            if (response.status === 503 || response.status === 0) {
                if (this.hasConnectedToServer) {
                    // Connection lost after initial sync - show warning but don't throw
                    this.showServerConnectionWarning();
                    throw new Error('Server unavailable - connection lost');
                } else {
                    // Never connected - this will be handled by retry logic
                    throw new Error('Server unavailable');
                }
            }
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error_message || errorData.error || `HTTP ${response.status}`);
        }
        
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
                    type: this.normalizeFrameType(ann.type, ann.has_mask !== false),
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
            typeBar.height = typeBar.offsetHeight || 24;

            const ctx = typeBar.getContext('2d');
            const width = typeBar.width;
            const height = typeBar.height;

            // Account for thumb inset: range input thumb center at min is thumbHalf from left edge
            const btnSize = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--btn-size')) || 0;
            const thumbSize = btnSize * 0.3;
            const thumbHalf = thumbSize / 2;
            const trackWidth = width - thumbSize;
            const lastFrame = this.totalFrames - 1;
            // Slider has totalFrames values but (totalFrames-1) intervals between stops
            const step = lastFrame > 0 ? trackWidth / lastFrame : trackWidth;

            // Clear canvas
            ctx.clearRect(0, 0, width, height);

            // Get unsaved edits to check for modified frames
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();

            // Draw band centered on each thumb stop; first/last extend to canvas edges
            for (let frameNum = 0; frameNum < this.totalFrames; frameNum++) {
                const center = thumbHalf + frameNum * step;
                let x = center - step / 2;
                let w = step;
                if (frameNum === 0) { x = 0; w = center + step / 2; }
                else if (frameNum === lastFrame) { w = width - x; }
                
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
            
            // Draw scrubline across thumb-travel range (draw on top of colored sections)
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
            
            let type = this.normalizeFrameType(entry.type, hasMask);

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

        // Preserve unsaved changes when leaving current frame
        if (this.hasUnsavedChanges && this.maskImageData && this.currentFrame !== targetFrame) {
            const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
            videoModifiedFrames.set(this.currentFrame, {
                maskData: this.maskImageData, // Store directly, no clone needed
                maskType: this.maskType,
                is_empty: this.maskType === 'empty'
            });
        }

        this.currentFrame = targetFrame;
        document.getElementById('frameSlider').value = this.currentFrame;
        this.updateFrameCounter();
        this.updateSaveButtonState();

        // Load frame from archive (no caching)
        const frameData = this.framesArchive[`frames/frame_${targetFrame.toString().padStart(6, '0')}.webp`];
        if (!frameData) {
            console.error(`Frame ${targetFrame} not found in archive`);
            return;
        }

        // Load frame image
        const blob = new Blob([frameData], { type: 'image/webp' });
        const frameUrl = URL.createObjectURL(blob);
        this.frameImage = await this.loadImage(frameUrl);
        URL.revokeObjectURL(frameUrl);

        // Set canvas dimensions
        this.maskCanvas.width = this.frameImage.width;
        this.maskCanvas.height = this.frameImage.height;
        this.overlayCanvas.width = this.frameImage.width;
        this.overlayCanvas.height = this.frameImage.height;

        // Determine mask type from metadata
        const frameMeta = this.videoData?.mask_data?.[targetFrame];
        const maskFileName = `frame_${String(targetFrame).padStart(6, '0')}_mask.webp`;
        const hasMaskInArchive = !!(this.masksArchive && this.masksArchive[maskFileName]);

        if (frameMeta) {
            if (frameMeta.type === 'empty') {
                this.maskType = 'empty';
            } else if (frameMeta.type === 'fluid') {
                this.maskType = 'human';
            } else {
                this.maskType = 'tracked';
            }
        } else {
            this.maskType = hasMaskInArchive ? 'tracked' : 'empty';
        }
        this.originalMaskType = this.maskType;

        // Check if frame was modified this session
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        if (videoModifiedFrames.has(this.currentFrame)) {
            // Use modified version (already ImageData)
            const saved = videoModifiedFrames.get(this.currentFrame);
            this.maskImageData = saved.maskData;
            this.maskType = saved.maskType;
            this.hasUnsavedChanges = true;

            // Put mask data in maskCanvas for drawing operations
            this.maskCtx.putImageData(this.maskImageData, 0, 0);
        } else {
            // Frame not modified - will render mask directly from WebP in render()
            // Don't create ImageData unless user starts editing
            this.maskImageData = null;
            this.hasUnsavedChanges = false;
        }

        // Store mask WebP for lazy loading if needed
        this.currentMaskWebP = hasMaskInArchive ? this.masksArchive[maskFileName] : null;

        this.updateSaveButtonState();
        this.render();
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
        // Clear both canvases (main canvas uses logical dimensions due to DPR transform)
        this.ctx.clearRect(0, 0, window.innerWidth, window.innerHeight);
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

        // Draw mask overlay
        const generation = ++this._renderGeneration;
        if (this.maskVisible) {
            const isHuman = this.maskType === 'human' || this.maskType === 'empty';
            const maskColor = isHuman ? { r: 0, g: 255, b: 0 } : { r: 255, g: 165, b: 0 };

            if (this.maskImageData) {
                // Modified mask - render from ImageData
                const overlayImageData = this.overlayCtx.createImageData(this.overlayCanvas.width, this.overlayCanvas.height);

                for (let i = 0; i < this.maskImageData.data.length; i += 4) {
                    const gray = this.maskImageData.data[i];
                    overlayImageData.data[i] = maskColor.r;
                    overlayImageData.data[i + 1] = maskColor.g;
                    overlayImageData.data[i + 2] = maskColor.b;
                    overlayImageData.data[i + 3] = gray > 0 ? Math.round(gray * this.maskOpacity) : 0;
                }

                this.overlayCtx.putImageData(overlayImageData, 0, 0);
            } else if (this.currentMaskWebP && this.maskType !== 'empty') {
                // Unmodified mask - render directly from WebP without creating ImageData
                this.renderMaskFromWebP(this.currentMaskWebP, maskColor, generation).catch(err => {
                    console.error('Failed to render mask from WebP:', err);
                });
            }
        }
    }

    async loadMaskAsImageData() {
        // Create ImageData from current mask WebP
        if (this.currentMaskWebP) {
            const blob = new Blob([this.currentMaskWebP], { type: 'image/webp' });
            const maskUrl = URL.createObjectURL(blob);
            const maskImg = await this.loadImage(maskUrl);
            URL.revokeObjectURL(maskUrl);

            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = maskImg.width;
            tempCanvas.height = maskImg.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(maskImg, 0, 0);
            this.maskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        } else {
            // No mask - create blank
            this.maskImageData = this.maskCtx.createImageData(this.maskCanvas.width, this.maskCanvas.height);
            // Fill with black (empty mask)
            for (let i = 0; i < this.maskImageData.data.length; i += 4) {
                this.maskImageData.data[i] = 0;
                this.maskImageData.data[i + 1] = 0;
                this.maskImageData.data[i + 2] = 0;
                this.maskImageData.data[i + 3] = 255;
            }
        }

        // Put in maskCanvas for drawing operations
        this.maskCtx.putImageData(this.maskImageData, 0, 0);
    }

    async renderMaskFromWebP(maskWebPData, maskColor, generation) {
        // Load mask image from WebP bytes
        const blob = new Blob([maskWebPData], { type: 'image/webp' });
        const maskUrl = URL.createObjectURL(blob);
        const maskImg = await this.loadImage(maskUrl);
        URL.revokeObjectURL(maskUrl);

        // Bail if a newer render started while we were loading
        if (generation !== undefined && generation !== this._renderGeneration) return;

        // Create temp canvas to extract grayscale values
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = maskImg.width;
        tempCanvas.height = maskImg.height;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(maskImg, 0, 0);
        const maskImageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);

        // Create colored overlay
        const overlayImageData = this.overlayCtx.createImageData(this.overlayCanvas.width, this.overlayCanvas.height);
        for (let i = 0; i < maskImageData.data.length; i += 4) {
            const gray = maskImageData.data[i];
            overlayImageData.data[i] = maskColor.r;
            overlayImageData.data[i + 1] = maskColor.g;
            overlayImageData.data[i + 2] = maskColor.b;
            overlayImageData.data[i + 3] = gray > 0 ? Math.round(gray * this.maskOpacity) : 0;
        }

        this.overlayCtx.putImageData(overlayImageData, 0, 0);
    }

    getCanvasCoordinates(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX ?? e.touches?.[0]?.clientX ?? 0) - rect.left;
        const y = (e.clientY ?? e.touches?.[0]?.clientY ?? 0) - rect.top;

        // Convert to mask coordinates
        if (!this.renderRect) return { x: 0, y: 0 };

        const maskX = ((x - this.renderRect.x) / this.renderRect.width) * this.maskCanvas.width;
        const maskY = ((y - this.renderRect.y) / this.renderRect.height) * this.maskCanvas.height;

        return { x: Math.floor(maskX), y: Math.floor(maskY) };
    }

    async startDrawing(e) {
        this.isDrawing = true;
        const coords = this.getCanvasCoordinates(e);
        this.lastX = coords.x;
        this.lastY = coords.y;

        // Convert tracked masks to human annotation on first edit
        if (this.maskType === 'tracked') {
            console.log('CONVERTING tracked to human on edit');
            this.maskType = 'human';
        }

        // Remove empty marker when drawing
        if (this.maskType === 'empty') {
            console.log('REMOVING empty marker on draw');
            this.maskType = 'human';
        }

        console.log('Start drawing - Mode:', this.drawMode, 'Type:', this.maskType);

        // Lazy-create ImageData when user starts editing
        if (!this.maskImageData) {
            await this.loadMaskAsImageData();
        }

        this.render(); // Show color change immediately
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
        
        // Helper to get mask ImageData for a frame (from modified frames or archive)
        const getMaskImageDataForFrame = async (frameNum) => {
            // Check if frame is in modified frames (has current edits)
            if (videoModifiedFrames.has(frameNum)) {
                return videoModifiedFrames.get(frameNum).maskData;
            }

            // Load from archive if available (lazy conversion)
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

            const allResponse = await this.fetchToServer(`api/masks/${studyUid}/${seriesUid}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-tar',
                    'X-User-Email': this.getUserEmail(),
                    'X-Previous-Version-ID': this.getCurrentVersionId() || ''
                },
                body: archiveData
            });

            if (allResponse.ok) {
                const result = await allResponse.json();
                console.log(`✅ Saved and queued retrack: ${result.version_id}`);

                // Update version ID for next save
                this.setCurrentVersionId(result.version_id);

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
            try {
                const response = await this.fetchToServer(`api/retrack/status/${studyUid}/${seriesUid}`);
                const status = await response.json();

                if (status.status === 'completed') {
                    console.log('Retrack complete, reloading...');
                    this.showRetrackLoading('Reloading retracked masks...');

                    // Update version_id from retrack status (server is definitive source)
                    if (status.version_id) {
                        this.setCurrentVersionId(status.version_id);
                        console.log(`[Version] Updated after retrack: ${status.version_id}`);
                    }

                    // Clear edits only after retrack completes successfully (atomic operation)
                    const cv = this.currentVideo;
                    const videoKey = this.getVideoKey(cv.method, cv.studyUid, cv.seriesUid);
                    const videoModifiedFrames = this.modifiedFrames.get(videoKey);
                    if (videoModifiedFrames) {
                        videoModifiedFrames.clear();
                    }
                    this.hasUnsavedChanges = false;

                    // Reload from server to ensure fresh masks
                    try {
                        this.frameCache.clear();
                        this.framesArchive = null;
                        this.masksArchive = {};
                        await this.loadVideoData();
                        await this.loadFramesArchive();
                        // Force reload current frame to ensure fresh masks are displayed
                        const currentFrameNum = this.currentFrame;
                        this.currentFrame = -1; // Force reload
                        await this.goToFrame(currentFrameNum);
                    } catch (reloadErr) {
                        console.error('Error reloading after retrack:', reloadErr);
                    }

                    this.updateSaveButtonState();
                    this.hideRetrackLoading();
                } else if (status.status === 'failed') {
                    console.error('Retrack failed:', status.error);
                    this.hideRetrackLoading();
                    alert(`Retrack failed: ${status.error || 'Unknown error'}`);
                    this.hasUnsavedChanges = true;
                    this.updateSaveButtonState();
                } else if (attempts < maxAttempts) {
                    setTimeout(poll, 1000);
                } else {
                    this.hideRetrackLoading();
                    alert('Retrack timeout - check server status');
                }
            } catch (err) {
                console.warn(`Retrack poll error (attempt ${attempts}/${maxAttempts}):`, err);
                if (attempts < maxAttempts) {
                    setTimeout(poll, 1000);
                } else {
                    this.hideRetrackLoading();
                    alert('Retrack timeout - check server status');
                }
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
        // Block completion if there are unsaved changes
        if (this.hasUnsavedChangesForCurrentVideo()) {
            alert('Please save your changes before marking this series as complete.');
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
            const resp = await this.fetchToServer(`api/series/${studyUid}/${seriesUid}/complete`, {
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
            const resp = await this.fetchToServer(`api/series/${studyUid}/${seriesUid}/reopen`, {
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

        // Update button completed styling (green glow)
        if (markCompleteBtn) {
            if (isCompleted) {
                markCompleteBtn.classList.add('completed');
            } else {
                markCompleteBtn.classList.remove('completed');
            }
        }

        // Update disabled state and title
        this.updateMarkCompleteButtonState();
        
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
     * Populate video select dropdown from local series endpoint.
     * This is needed for iOS client which doesn't use server-side Jinja2 templates.
     */
    async populateVideoSelectFromLocal() {
        const videoSelect = document.getElementById('videoSelect');
        if (!videoSelect) {
            return;
        }
        
        try {
            // Use /api/videos which includes activity data merged from server
            const resp = await fetch('/api/videos');
            if (!resp.ok) {
                console.warn('Failed to fetch series list:', resp.status);
                return;
            }
            
            const allSeries = await resp.json();
            
            // Clear existing options
            videoSelect.innerHTML = '';
            
            // Sort by exam_number
            const sortedSeries = [...allSeries].sort((a, b) => {
                const aNum = a.exam_number || 0;
                const bNum = b.exam_number || 0;
                return aNum - bNum;
            });
            
            // Helper to determine indicator emoji (same logic as refreshAllSeriesStatuses)
            const currentUserEmail = this.getUserEmail();
            const now = new Date();
            const twentyFourHoursAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
            
            const getIndicator = (status, activity) => {
                if (status === 'completed') {
                    return '✅';
                }
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
                    if (hasOtherUserActivity) {
                        return '⚠️';
                    }
                    if (hasCurrentUserActivity) {
                        return '🖌️';
                    }
                }
                return '◻️';
            };
            
            // Populate dropdown with correct indicators
            for (const series of sortedSeries) {
                const option = document.createElement('option');
                const method = series.method || 'dis';
                option.value = `${method}|${series.study_uid}|${series.series_uid}`;
                const indicator = getIndicator(series.status || 'pending', series.activity || {});
                option.text = `${indicator} Exam ${series.exam_number || '-'}`;
                videoSelect.appendChild(option);
            }
        } catch (e) {
            console.warn('Failed to populate video select dropdown:', e);
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
            const resp = await this.fetchToServer(`api/series/${studyUid}/${seriesUid}`);
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
            // Use /api/videos which includes activity data merged from server
            // This works for both iOS client (merges server activity) and web client
            const resp = await fetch('/api/videos');
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
            const resp = await this.fetchToServer(`api/reset-retrack/${studyUid}/${seriesUid}`, {
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
            const resp = await this.fetchToServer('api/reset-retrack-all', {
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

    async markEmpty() {
        // Lazy-load ImageData if not already loaded
        if (!this.maskImageData) {
            await this.loadMaskAsImageData();
        }

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

        // Store in modified frames map (no clone needed)
        const videoModifiedFrames = this.getModifiedFramesForCurrentVideo();
        videoModifiedFrames.set(this.currentFrame, {
            maskData: this.maskImageData,
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
            const response = await this.fetchToServer('api/series/next', {
                headers: {
                    'X-User-Email': this.getUserEmail() || ''
                }
            });
            
            if (!response.ok) {
                if (response.status === 0 || response.status >= 500) {
                    // Network error or server error - connection issue
                    throw new Error('Server unavailable');
                }
                const errorText = await response.text().catch(() => response.statusText);
                throw new Error(`Failed to fetch next series: ${response.status} ${errorText}`);
            }
            
            // Successfully connected to remote server
            this.hasConnectedToServer = true;
            
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
            
            // Load the series from server
            // Note: loadVideo() will also mark activity (redundant but safe)
            // Connection screen stays visible during loadVideo - it will be hidden after video is loaded
            await this.loadVideo(method, data.study_uid, data.series_uid);

            // Set version_id from server response (server is definitive source for mask versions)
            if (data.version_id !== undefined) {
                this.setCurrentVersionId(data.version_id);
                console.log(`[Version] Loaded from server: ${data.version_id}`);
            }
            // Video is now loaded and rendered - connection screen will be hidden by caller
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
     * Get user email (loaded from /api/settings at init).
     * This is sent in X-User-Email header to identify the user.
     */
    getUserEmail() {
        return this.userEmail || '';
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
            const response = await this.fetchToServer(
                `api/series/${studyUid}/${seriesUid}/activity`,
                {
                    method: 'POST',
                    headers: {
                        'X-User-Email': this.getUserEmail() || ''
                    }
                }
            );
            
            if (!response.ok && response.status !== 0) {
                // Non-network error - log but don't show warning (might be 404, etc.)
                console.warn('Failed to mark activity:', response.status);
            } else if (response.status === 0) {
                // Network error - show warning if we've connected before
                if (this.hasConnectedToServer) {
                    this.showServerConnectionWarning();
                }
            }
            
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
            const response = await this.fetchToServer(
                `api/series/${this.currentVideo.studyUid}/${this.currentVideo.seriesUid}`
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

    getCurrentVersionId() {
        if (!this.currentVideo) {
            console.warn('getCurrentVersionId called with no currentVideo');
            return null;
        }
        const videoKey = this.getVideoKey(
            this.currentVideo.method,
            this.currentVideo.studyUid,
            this.currentVideo.seriesUid
        );
        const versionId = this.versionIds.get(videoKey) || null;
        console.log(`[Version] GET ${videoKey}: ${versionId}`);
        return versionId;
    }

    setCurrentVersionId(versionId) {
        if (!this.currentVideo) {
            console.warn('setCurrentVersionId called with no currentVideo');
            return;
        }
        const videoKey = this.getVideoKey(
            this.currentVideo.method,
            this.currentVideo.studyUid,
            this.currentVideo.seriesUid
        );

        if (versionId) {
            this.versionIds.set(videoKey, versionId);
            console.log(`[Version] SET ${videoKey}: ${versionId}`);
        } else {
            this.versionIds.delete(videoKey);
            console.log(`[Version] CLEAR ${videoKey}`);
        }
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
        this.currentVideo = { method, studyUid, seriesUid };
        // Load video data first so we know which frames are modified
        await this.loadVideoData(skipModifiedFrames);
        await this.loadFramesArchive();
        this.currentFrame = -1;
        await this.goToFrame(0); // This loads and renders the first frame
        this.updateSaveButtonState();
        this.updateVideoInfo();
        this.updateCompletionIndicator(); // Update completion status indicator
        this.updateVideoSelect(); // Update dropdown to show current series
        await this.refreshAllSeriesStatuses(); // Update all indicators with activity data
        
        // Fetch series detail to get activity data and update warnings
        await this.fetchAndUpdateWarnings();
        
        // Start activity pings for this video (every 30s)
        this.startActivityPings();
        
        // Video is fully loaded and rendered - connection screen can now be hidden
        // This will be called by checkCredentialsAndSync after loadNextSeries completes
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
            // Call client backend for frame archive URLs (frames are PHI, served locally)
            const url = `/api/frames/${method}/${studyUid}/${seriesUid}`;
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
            // Use fetchToServer to add headers (handles both paths and full URLs)
            const framesArchiveResponse = await this.fetchToServer(framesUrl);
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
                // Use fetchToServer to add headers (handles both paths and full URLs)
                const masksArchiveResponse = await this.fetchToServer(masksUrl);
                
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

                    // NOTE: version_id is intentionally NOT set from X-Version-ID header here.
                    // The definitive version_id comes from API responses:
                    // - /api/series/next (loadNextSeries)
                    // - /api/series/{study}/{series} (dropdown selection)
                    // - save response (saveChanges)
                    // - retrack status (pollRetrackStatus)
                    // Setting it from the masks archive could overwrite correct values with stale ones.

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
    
}

AnnotationViewer.prototype.checkDatasetSyncStatus = async function () {
    try {
        const resp = await fetch('/api/dataset/version_status').catch(() => null);
        // Note: Client route returns only client version, frontend calls server directly for server version
        if (!resp || !resp.ok) {
            // Silently fail - dataset sync status is not critical
            return;
        }
        const data = await resp.json();
        const banner = document.getElementById('datasetWarning');
        if (!banner) return;
        
        // Only show out-of-sync warning if:
        // 1. Server is reachable (server_version is not null)
        // 2. Versions are actually out of sync (in_sync === false)
        // If in_sync is null, server is unreachable - that's handled by server connection warning
        if (data.in_sync === false && data.server !== null) {
            // Server is reachable and versions differ - show warning
            banner.classList.remove('hidden');
        } else {
            // Either in sync, or server unreachable (in_sync === null) - hide warning
            banner.classList.add('hidden');
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
    const screen = document.getElementById('serverConnectionScreen');
    if (screen) {
        screen.classList.remove('hidden');
        screen.style.display = 'flex';
        screen.style.opacity = '1';
    }
    
    viewer = new AnnotationViewer();
    
    window.addEventListener('beforeunload', () => {
        if (viewer) {
            viewer.cleanup();
        }
    });
});
