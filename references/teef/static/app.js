const MAX_OPACITY = 0.4;
const imageCanvas = document.getElementById('imageCanvas');
const maskCanvas = document.getElementById('maskCanvas');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = imageCanvas.getContext('2d', { willReadFrequently: true });
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
const overlayCtx = overlayCanvas.getContext('2d', { willReadFrequently: true });
let BRUSH_SIZE = 10;
let BRUSH_INCREMENT = 5;
let MIN_BRUSH = 2;
let MAX_BRUSH = 50;

let currentImageUrl = '';
let currentMaskUrl = '';
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let tool = 'brush';
let singleChannelMask;
let brushPreview, brushSizeDisplay, brushSizeSlider, brushPreviewTimeout;

function adjustCanvasSize() {
    const container = document.querySelector('.canvas-container');
    const canvases = [imageCanvas, maskCanvas, overlayCanvas];
    
    canvases.forEach(canvas => {
        const scaleX = container.clientWidth / canvas.width;
        const scaleY = container.clientHeight / canvas.height;
        const scale = Math.min(scaleX, scaleY);
        
        canvas.style.width = `${canvas.width * scale}px`;
        canvas.style.height = `${canvas.height * scale}px`;
    });
}



function loadImagePair(img) {
    const imageUrl = `/image/images/${img}`;
    const maskUrl = `/image/masks/${img}`;
    currentImageUrl = imageUrl;
    currentMaskUrl = maskUrl;

    const image = new Image();
    image.onload = function() {
        // Set the actual canvas size to the image dimensions
        imageCanvas.width = maskCanvas.width = overlayCanvas.width = image.width;
        imageCanvas.height = maskCanvas.height = overlayCanvas.height = image.height;

        // Set the canvas container height based on the image aspect ratio
        const container = document.querySelector('.canvas-container');
        const containerWidth = container.clientWidth;
        const scale = containerWidth / image.width;
        container.style.height = `${image.height * scale}px`;

        ctx.drawImage(image, 0, 0, image.width, image.height);
        loadMask(maskUrl);

        // Adjust canvas size after loading
        adjustCanvasSize();
    };
    image.src = imageUrl;
}

function adjustCanvasSize() {
    const container = document.querySelector('.canvas-container');
    const canvases = [imageCanvas, maskCanvas, overlayCanvas];
    
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    
    canvases.forEach(canvas => {
        const scaleX = containerWidth / canvas.width;
        const scaleY = containerHeight / canvas.height;
        const scale = Math.min(scaleX, scaleY);
        
        canvas.style.width = `${canvas.width * scale}px`;
        canvas.style.height = `${canvas.height * scale}px`;
        
        // Center the canvas within the container
        canvas.style.left = `${(containerWidth - canvas.width * scale) / 2}px`;
        canvas.style.top = `${(containerHeight - canvas.height * scale) / 2}px`;
    });
}

function loadMask(maskUrl) {
    const maskImg = new Image();
    maskImg.onload = function() {
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        maskCtx.drawImage(maskImg, 0, 0, maskCanvas.width, maskCanvas.height);
        singleChannelMask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        applyMaskToOverlay();
    };
    maskImg.src = `${maskUrl}?t=${new Date().getTime()}`;
}

function applyMaskToOverlay() {
    const overlayImageData = overlayCtx.createImageData(overlayCanvas.width, overlayCanvas.height);
    
    for (let i = 0; i < singleChannelMask.data.length; i += 4) {
        const gray = singleChannelMask.data[i];
        overlayImageData.data[i] = 0; // Red
        overlayImageData.data[i + 1] = 255; // Green
        overlayImageData.data[i + 2] = 0; // Blue
        overlayImageData.data[i + 3] = Math.round(gray * MAX_OPACITY); // Alpha
    }

    overlayCtx.putImageData(overlayImageData, 0, 0);
}

function drawLine(x0, y0, x1, y1) {
    const dx = Math.abs(x1 - x0);
    const dy = Math.abs(y1 - y0);
    const sx = (x0 < x1) ? 1 : -1;
    const sy = (y0 < y1) ? 1 : -1;
    let err = dx - dy;

    while (true) {
        drawPoint(x0, y0);

        if (x0 === x1 && y0 === y1) break;
        const e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

function drawPoint(x, y) {
    for (let dx = -BRUSH_SIZE; dx <= BRUSH_SIZE; dx++) {
        for (let dy = -BRUSH_SIZE; dy <= BRUSH_SIZE; dy++) {
            if (dx*dx + dy*dy <= BRUSH_SIZE*BRUSH_SIZE) {
                const cx = x + dx;
                const cy = y + dy;
                if (cx >= 0 && cx < maskCanvas.width && cy >= 0 && cy < maskCanvas.height) {
                    const i = (cy * maskCanvas.width + cx) * 4;
                    const value = tool === 'brush' ? 255 : 0;
                    singleChannelMask.data[i] = value;
                    singleChannelMask.data[i+1] = value;
                    singleChannelMask.data[i+2] = value;
                    singleChannelMask.data[i+3] = 255;
                }
            }
        }
    }
}

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    let clientX, clientY;

    if (evt.touches && evt.touches[0]) {
        // Touch event
        clientX = evt.touches[0].clientX;
        clientY = evt.touches[0].clientY;
    } else {
        // Mouse event
        clientX = evt.clientX;
        clientY = evt.clientY;
    }

    // return {
    //     x: Math.floor((evt.clientX - rect.left) * scaleX),
    //     y: Math.floor((evt.clientY - rect.top) * scaleY)
    // };
    return {
        x: Math.floor((clientX - rect.left) * scaleX),
        y: Math.floor((clientY - rect.top) * scaleY)
    };
}

function startDrawing(e) {
    isDrawing = true;
    const pos = getMousePos(overlayCanvas, e);
    lastX = pos.x;
    lastY = pos.y;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    
    const pos = getMousePos(overlayCanvas, e);
    drawLine(lastX, lastY, pos.x, pos.y);
    lastX = pos.x;
    lastY = pos.y;
    
    maskCtx.putImageData(singleChannelMask, 0, 0);
    applyMaskToOverlay();
}

function stopDrawing() {
    isDrawing = false;
}

function clearMask() {
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    singleChannelMask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
    applyMaskToOverlay();
}

function saveMask() {
    return new Promise((resolve, reject) => {
        maskCtx.putImageData(singleChannelMask, 0, 0);
        const maskData = maskCanvas.toDataURL('image/jpeg', 0.98);
        fetch('/save_mask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: maskData,
                maskFilename: currentMaskUrl.split('/').pop()
            }),
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message);
            resolve();
        })
        .catch((error) => {
            console.error('Error:', error);
            reject(error);
        });
    });
}

function handleKeyNavigation(event) {
    if (event.key === 'ArrowLeft') {
        navigateImage('prev');
    } else if (event.key === 'ArrowRight') {
        navigateImage('next');
    } else if (event.key === 'l') {
        openTitleEdit(event);
    } else if (event.key === 's') {
        saveMask();
        showToast('Mask saved', 'success');
    } else if (event.key === ',') {
        console.log('Decrease brush size');
        updateBrushSize(BRUSH_SIZE - BRUSH_INCREMENT);
    } else if (event.key === '.') {
        console.log('Increase brush size');
        updateBrushSize(BRUSH_SIZE + BRUSH_INCREMENT);
    } else if (event.key === ' ') {
        event.preventDefault();  // Prevent scrolling
        toolToggle.click();
    } else if (event.key === 'd') {
        event.preventDefault();  // Prevent browser's default 'bookmark' action
        downloadAll();
    } else if (event.key === 'h') {
        overlayToggle.click();
    }
}

function updateBrushSize(newSize) {
    newSize = Math.max(MIN_BRUSH, Math.min(MAX_BRUSH, newSize));
    BRUSH_SIZE = newSize;
    brushSizeSlider.value = newSize;
    updateBrushPreview(newSize);
}

function updateBrushPreview(size) {
    brushPreview.style.width = `${size}px`;
    brushPreview.style.height = `${size}px`;
    brushPreview.style.display = 'block';
    brushPreview.style.backgroundColor = tool === 'brush' ? 'rgba(0, 255, 0, 0.3)' : 'rgba(255, 0, 0, 0.3)';
    brushPreview.style.borderColor = tool === 'brush' ? 'rgba(0, 255, 0, 0.6)' : 'rgba(255, 0, 0, 0.6)';
    
    clearTimeout(brushPreviewTimeout);
    brushPreviewTimeout = setTimeout(() => {
        brushPreview.style.display = 'none';
    }, 1000);
}

function hideBrushPreview() {
    clearTimeout(brushPreviewTimeout);
    brushPreview.style.display = 'none';
}


function downloadAll() {
    saveMask().then(() => {
        showToast('Downloading all masks...', 'info', 5000);
        fetch('/download_all', {
            method: 'POST'  // Changed from GET to POST
        })
        .then(response => response.blob())
        .then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            timestamp = new Date().toISOString().replace(/:/g, '-');
            a.download = `teef_${timestamp}.zip`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
        })
        .catch(error => console.error('Error downloading:', error));
    });
}

function showToast(message, type = 'info', duration = 300) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    const toastContainer = document.getElementById('toast');
    toastContainer.innerHTML = ''; // Clear any existing toasts
    toastContainer.appendChild(toast);
    
    // Trigger reflow
    toast.offsetHeight;
    
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 300);
    }, duration);
}

function navigateImage(direction) {
    saveMask(); // Save current mask before navigating
    fetch(`/get_image_pair?img=${currentImageUrl.split('/').pop()}&direction=${direction}`)
    .then(response => response.json())
    .then(data => {
        loadImagePair(data.filename);
        updateUrl(data.filename)
    });
}

function hideBrushOutline() {
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    applyMaskToOverlay();
}

function drawBrushOutline(event) {
    const pos = getMousePos(overlayCanvas, event);
    
    // Clear the previous overlay and redraw the mask
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    applyMaskToOverlay();
    
    // Draw brush outline
    overlayCtx.beginPath();
    overlayCtx.arc(pos.x, pos.y, BRUSH_SIZE, 0, Math.PI * 2);
    // Green outline for brush, red outline for eraser
    overlayCtx.strokeStyle = tool === 'brush' ? 'rgba(0, 255, 0, 0.6)' : 'rgba(255, 0, 0, 0.6)';
    
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();
}

function initializeTitleEdit() {
    const title = document.getElementById('title');
    const titleEditContainer = document.querySelector('.title-edit-container');
    const titleInput = document.getElementById('titleInput');
    const titleConfirm = document.getElementById('titleConfirm');
    const titleCancel = document.getElementById('titleCancel');

    title.addEventListener('click', openTitleEdit);

    titleConfirm.addEventListener('click', () => {
        const newImg = titleInput.value.trim();
        if (newImg) {
            loadImagePair(newImg);
            updateUrl(newImg);
        }
        document.addEventListener('keydown', handleKeyNavigation);
        closeTitleEdit();
    });

    titleCancel.addEventListener('click', closeTitleEdit);

    titleInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            titleConfirm.click();
        } else if (e.key === 'Escape') {
            titleCancel.click();
        }
    });
}

function openTitleEdit(e) {
    if (e) {
        e.preventDefault();
        e.stopPropagation();
    }
    const title = document.getElementById('title');
    const titleEditContainer = document.querySelector('.title-edit-container');
    const titleInput = document.getElementById('titleInput');
    title.style.display = 'none';
    titleEditContainer.style.display = 'flex';
    titleInput.value = title.textContent;
    titleInput.focus();
    document.removeEventListener('keydown', handleKeyNavigation);
}

function closeTitleEdit() {
    const title = document.getElementById('title');
    const titleEditContainer = document.querySelector('.title-edit-container');
    title.style.display = 'block';
    titleEditContainer.style.display = 'none';
    document.addEventListener('keydown', handleKeyNavigation);
}

function updateUrl(img) {
    history.pushState(null, '', `?img=${img}`);
    document.getElementById('title').textContent = img;
}

document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const img = urlParams.get('img');

    if (img) {
        loadImagePair(img);
        updateUrl(img);
    } else {
        fetch('/get_image_pair')
            .then(response => response.json())
            .then(data => {
                loadImagePair(data.filename);
                updateUrl(data.filename);
            });
    }

    window.addEventListener('resize', () => {
        const container = document.querySelector('.canvas-container');
        const containerWidth = container.clientWidth;
        const imageAspectRatio = imageCanvas.width / imageCanvas.height;
        container.style.height = `${containerWidth / imageAspectRatio}px`;
        adjustCanvasSize();
    });

    const toolToggle = document.getElementById('toolToggle');
    toolToggle.addEventListener('click', () => {
        // Swap Font Awesome brush and eraser tools
        // <button id="toolToggle"><i class="fas fa-paint-brush"></i></button>
        tool = tool === 'brush' ? 'eraser' : 'brush';
        const icon = tool === 'brush' ? 'fa-paint-brush' : 'fa-eraser';
        toolToggle.innerHTML = `<i class="fas ${icon}"></i>`;
    });

    const overlayToggle = document.getElementById('overlayToggle');
    overlayToggle.addEventListener('click', () => {
        overlayCanvas.style.display = overlayCanvas.style.display === 'none' ? 'block' : 'none';
        // Swap Font Awesome icon fa-eye and fa-eye-slash
        // <button id="overlayToggle"><i class="fa fa-eye"></i></button>
        const icon = overlayCanvas.style.display === 'none' ? 'fa-eye-slash' : 'fa-eye';
        overlayToggle.innerHTML = `<i class="fa ${icon}"></i>`;
    });

    // Keyboard event listeners
    document.addEventListener('keydown', handleKeyNavigation);
    initializeTitleEdit();

    // Mouse event listeners
    overlayCanvas.addEventListener('mousedown', startDrawing);
    overlayCanvas.addEventListener('mousemove', draw);
    overlayCanvas.addEventListener('mouseup', stopDrawing);
    overlayCanvas.addEventListener('mouseout', stopDrawing);

    // Touch event listeners
    overlayCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDrawing(e); });
    overlayCanvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });
    overlayCanvas.addEventListener('touchend', (e) => { e.preventDefault(); stopDrawing(); });
    overlayCanvas.addEventListener('touchcancel', (e) => { e.preventDefault(); stopDrawing(); });


    document.getElementById('clear').addEventListener('click', clearMask);
    document.getElementById('prev').addEventListener('click', () => navigateImage('prev'));
    document.getElementById('next').addEventListener('click', () => navigateImage('next'));
    document.getElementById('save').addEventListener('click', saveMask);
    document.getElementById('reload').addEventListener('click', location.reload.bind(location));


    // Add event listener for showing brush outline
    overlayCanvas.addEventListener('mousemove', drawBrushOutline);
    overlayCanvas.addEventListener('mouseleave', hideBrushOutline);
    overlayCanvas.addEventListener('touchmove', drawBrushOutline);
    overlayCanvas.addEventListener('touchend', hideBrushOutline);

    // Listen for brush changes
    brushSizeSlider = document.getElementById('brushSizeSlider');
    const decreaseBrush = document.getElementById('decreaseBrush');
    const increaseBrush = document.getElementById('increaseBrush');
    const brushControls = document.querySelector('.brush-controls');
    
    brushSizeSlider.min = MIN_BRUSH;
    brushSizeSlider.max = MAX_BRUSH;
    brushSizeSlider.value = BRUSH_SIZE;

    // Create brush preview element
    brushPreview = document.getElementById('brushPreview');
    // brushPreview = document.createElement('div');
    // brushPreview.className = 'brush-preview';
    // brushControls.appendChild(brushPreview);

    // Create brush size display element
    brushSizeDisplay = document.createElement('div');
    // brushSizeDisplay.className = 'brush-size-display';
    // brushControls.appendChild(brushSizeDisplay);

    brushSizeSlider.addEventListener('input', (event) => {
        updateBrushSize(parseInt(event.target.value));
    });

    decreaseBrush.addEventListener('click', () => {
        updateBrushSize(BRUSH_SIZE - BRUSH_INCREMENT);
    });

    increaseBrush.addEventListener('click', () => {
        updateBrushSize(BRUSH_SIZE + BRUSH_INCREMENT);
    });

    // Show preview while sliding
    brushSizeSlider.addEventListener('mousedown', () => {
        updateBrushPreview(BRUSH_SIZE);
    });

    brushSizeSlider.addEventListener('mouseup', () => {
        setTimeout(() => {
            brushPreview.style.display = 'none';
        }, 1000);
    });

    // For touch devices
    brushSizeSlider.addEventListener('touchstart', () => {
        updateBrushPreview(BRUSH_SIZE);
    });

    brushSizeSlider.addEventListener('touchend', () => {
        setTimeout(() => {
            brushPreview.style.display = 'none';
        }, 1000);
    });
});