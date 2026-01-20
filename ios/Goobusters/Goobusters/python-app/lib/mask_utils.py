"""
Shared utilities for mask processing (base64 decoding, WebP conversion).

Used by app.py (legacy), client, and server to ensure consistent mask handling.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image


def decode_base64_mask_to_webp(
    mask_b64: str, output_path: Path, quality: int = 85
) -> bool:
    """
    Decode base64-encoded mask image (PNG) and save as WebP.

    Handles data URL format: "data:image/png;base64,..." or plain base64.

    Args:
        mask_b64: Base64-encoded image data (with or without data URL prefix)
        output_path: Path where WebP file will be saved
        quality: WebP quality (0-100, default 85)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Strip data URL prefix if present
        if "," in mask_b64:
            _, mask_b64 = mask_b64.split(",", 1)

        # Decode base64
        raw = base64.b64decode(mask_b64)

        # Load image and convert to grayscale
        img = Image.open(io.BytesIO(raw)).convert("L")

        # Save as WebP
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="WEBP", quality=quality)

        return True
    except Exception:
        return False

