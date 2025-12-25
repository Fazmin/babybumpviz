"""
Utility functions for Baby Kick Visualization App.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import tempfile
import os


def create_temp_directory() -> str:
    """Create and return a temporary directory for processing."""
    temp_dir = tempfile.mkdtemp(prefix="babykick_")
    return temp_dir


def cleanup_temp_files(temp_dir: str) -> None:
    """Clean up temporary processing files."""
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def resize_frame(frame: np.ndarray, max_width: int = 1280) -> Tuple[np.ndarray, float]:
    """
    Resize frame to max width while maintaining aspect ratio.
    
    Returns:
        Tuple of (resized_frame, scale_factor)
    """
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame, 1.0
    
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize frame for consistent processing across lighting conditions."""
    if len(frame.shape) == 3:
        # Convert to LAB and normalize L channel
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(frame)


def apply_roi_mask(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Apply ROI mask to frame.
    
    Args:
        frame: Input frame
        roi: (x, y, width, height) tuple
        
    Returns:
        Masked frame with only ROI visible
    """
    x, y, w, h = roi
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    
    if len(frame.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    return cv2.bitwise_and(frame, mask)


def extract_roi(frame: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract ROI region from frame."""
    x, y, w, h = roi
    return frame[y:y+h, x:x+w]


def overlay_on_frame(
    base_frame: np.ndarray,
    overlay: np.ndarray,
    roi: Optional[Tuple[int, int, int, int]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay visualization on base frame.
    
    Args:
        base_frame: Original video frame
        overlay: Visualization overlay (same size as ROI or full frame)
        roi: Optional ROI coordinates
        alpha: Overlay opacity (0-1)
        
    Returns:
        Composited frame
    """
    result = base_frame.copy()
    
    if roi is not None:
        x, y, w, h = roi
        # Resize overlay to match ROI if needed
        if overlay.shape[:2] != (h, w):
            overlay = cv2.resize(overlay, (w, h))
        
        # Blend in ROI area
        roi_region = result[y:y+h, x:x+w]
        blended = cv2.addWeighted(roi_region, 1 - alpha, overlay, alpha, 0)
        result[y:y+h, x:x+w] = blended
    else:
        # Full frame overlay
        if overlay.shape[:2] != base_frame.shape[:2]:
            overlay = cv2.resize(overlay, (base_frame.shape[1], base_frame.shape[0]))
        result = cv2.addWeighted(base_frame, 1 - alpha, overlay, alpha, 0)
    
    return result


def calculate_displacement_mm(
    pixel_displacement: float,
    reference_size_pixels: float,
    reference_size_mm: float = 100.0
) -> float:
    """
    Convert pixel displacement to millimeters using reference scaling.
    
    Args:
        pixel_displacement: Displacement in pixels
        reference_size_pixels: Known reference size in pixels
        reference_size_mm: Known reference size in mm (default assumes ~10cm ROI width)
        
    Returns:
        Displacement in millimeters
    """
    if reference_size_pixels == 0:
        return 0.0
    mm_per_pixel = reference_size_mm / reference_size_pixels
    return pixel_displacement * mm_per_pixel


def format_timestamp(frame_number: int, fps: float) -> str:
    """Convert frame number to timestamp string."""
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:05.2f}"


def smooth_signal(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply moving average smoothing to signal."""
    if len(signal) < window_size:
        return signal
    kernel = np.ones(window_size) / window_size
    return np.convolve(signal, kernel, mode='same')

