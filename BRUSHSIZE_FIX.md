# Brush Size Preview Fix

## Problem
Brush size preview stopped showing after moving the slider from the left toolbar to a separate top container.

## Root Cause
The brush preview element had no positioning set when the slider changed (no event case). With `display: none` in CSS and no default position, the preview appeared at (0,0) or was never visible.

## Working Solution

### CSS Changes (static/css/viewer.css)

Add pointer-events to allow mouse/touch events to pass through toolbar containers to the canvas:

```css
/* Brush size slider (positioned right of left toolbar) */
.brush-slider-top {
    /* ... existing styles ... */
    pointer-events: none; /* Let mouse events pass through to canvas */
}

.brush-slider-top input,
.brush-slider-top .slider-group-vertical {
    pointer-events: auto; /* Slider itself stays interactive */
}

/* Left toolbar */
.toolbar-left {
    /* ... existing styles ... */
    pointer-events: none; /* Let mouse events pass through to canvas */
}

.toolbar-left button {
    pointer-events: auto; /* Buttons stay interactive */
}

/* Right toolbar */
.toolbar-right {
    /* ... existing styles ... */
    pointer-events: none; /* Let mouse events pass through to canvas */
}

.toolbar-right button,
.toolbar-right .control-badge {
    pointer-events: auto; /* Buttons and badges stay interactive */
}
```

### JavaScript Changes (static/js/viewer.js)

#### 1. Add touch event handlers for brush preview

In the touch event listeners section (~line 918):

```javascript
this.canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (this.isDrawing) this.draw(e);
    this.updateBrushPreview(e);  // ADD THIS LINE
});
this.canvas.addEventListener('touchend', (e) => {
    e.preventDefault();
    this.stopDrawing();
    this.hideBrushPreview();  // ADD THIS LINE
});
```

#### 2. Fix updateBrushPreview positioning logic

Replace the entire `updateBrushPreview(e)` function (~line 980):

```javascript
updateBrushPreview(e) {
    const preview = document.getElementById('brushPreview');
    if (!preview) return;

    // Calculate display scale (mask pixels to screen pixels)
    // Use scale=1 as fallback if renderRect or maskCanvas not initialized
    const scale = (this.renderRect && this.maskCanvas && this.maskCanvas.width)
        ? (this.renderRect.width / this.maskCanvas.width)
        : 1;
    const displayRadius = this.brushSize * scale;
    const displayDiameter = displayRadius * 2;

    if (e) {
        // For cursor tracking, we need maskCanvas to be initialized for proper scale
        if (!this.maskCanvas || !this.maskCanvas.width || !this.maskCanvas.height) return;

        const rect = this.canvas.getBoundingClientRect();
        const clientX = e.clientX || (e.touches && e.touches[0] && e.touches[0].clientX);
        const clientY = e.clientY || (e.touches && e.touches[0] && e.touches[0].clientY);

        if (clientX === undefined || clientY === undefined) return;

        const x = clientX - rect.left;
        const y = clientY - rect.top;

        // Center the preview on cursor by offsetting by scaled brush radius
        preview.style.transform = 'none'; // Clear any centering transform from slider mode
        preview.style.left = `${x - displayRadius}px`;
        preview.style.top = `${y - displayRadius}px`;
        preview.style.width = `${displayDiameter}px`;
        preview.style.height = `${displayDiameter}px`;
        preview.style.display = 'block';
    } else {
        // For slider adjustment, center preview on screen
        preview.style.left = '50%';
        preview.style.top = '50%';
        preview.style.transform = 'translate(-50%, -50%)';
        preview.style.width = `${displayDiameter}px`;
        preview.style.height = `${displayDiameter}px`;
        preview.style.display = 'block';
        setTimeout(() => preview.style.display = 'none', 1000);
    }
}
```

## Key Insights

1. **Transform toggle is critical**: The slider case uses `transform: translate(-50%, -50%)` for centering, but the cursor tracking case must clear this with `transform: 'none'` or the preview will be offset by -50%, -50% from where it should be.

2. **Two positioning modes**:
   - **Slider mode** (no event): Center on viewport using percentage positioning + transform
   - **Cursor mode** (with event): Pixel positioning following cursor, no transform

3. **Pointer-events CSS**: Toolbars need `pointer-events: none` to allow events to reach the canvas, while interactive elements within them need `pointer-events: auto` to remain clickable.

4. **Guard clauses**: Check if maskCanvas is initialized before attempting scale calculations in cursor mode. Use scale=1 as safe fallback for slider mode.

## Potential Issues / Further Investigation

- The pointer-events changes are broad - may need to verify all toolbar interactions still work correctly
- Scale calculation fallback (scale=1) in slider mode may not match actual scale when image is zoomed/fitted
- Touch coordinate extraction could be more robust (currently checks e.touches[0])
- May want to consider showing preview at cursor's last known position in slider mode instead of screen center

## Testing

- ✅ Slider adjustment shows centered preview that changes size
- ✅ Preview disappears 1 second after releasing slider
- Needs testing: Preview follows cursor when moving over canvas
- Needs testing: Preview works with touch events on iPad
- Needs testing: All toolbar buttons/badges remain clickable
