"""
Motion Detection Module for Baby Kick Visualization.
Implements optical flow analysis and motion vector computation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
from scipy import ndimage


@dataclass
class MotionData:
    """Container for motion analysis results."""
    magnitude: np.ndarray  # Motion magnitude map
    direction: np.ndarray  # Motion direction map (radians)
    flow_x: np.ndarray     # Horizontal flow component
    flow_y: np.ndarray     # Vertical flow component
    mean_magnitude: float  # Average motion magnitude
    max_magnitude: float   # Maximum motion magnitude
    

class MotionDetector:
    """
    Detects and analyzes motion between video frames using optical flow.
    """
    
    def __init__(
        self,
        method: str = "farneback",
        sensitivity: float = 1.0
    ):
        """
        Initialize motion detector.
        
        Args:
            method: Optical flow method ('farneback' or 'lucas_kanade')
            sensitivity: Motion sensitivity multiplier
        """
        self.method = method
        self.sensitivity = sensitivity
        self.prev_frame: Optional[np.ndarray] = None
        
        # Farneback parameters
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Feature detection for Lucas-Kanade
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
    
    def set_reference_frame(self, frame: np.ndarray) -> None:
        """Set reference frame for motion detection."""
        if len(frame.shape) == 3:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_frame = frame.copy()
    
    def compute_optical_flow(
        self,
        current_frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute optical flow between frames.
        
        Args:
            current_frame: Current frame (grayscale or BGR)
            prev_frame: Previous frame (optional, uses stored if None)
            
        Returns:
            Tuple of (flow_x, flow_y) arrays
        """
        # Convert to grayscale if needed
        if len(current_frame.shape) == 3:
            curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = current_frame
        
        if prev_frame is not None:
            if len(prev_frame.shape) == 3:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                prev_gray = prev_frame
        elif self.prev_frame is not None:
            prev_gray = self.prev_frame
        else:
            # First frame - return zero flow
            self.prev_frame = curr_gray.copy()
            zeros = np.zeros_like(curr_gray, dtype=np.float32)
            return zeros, zeros
        
        if self.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                **self.farneback_params
            )
            flow_x = flow[..., 0]
            flow_y = flow[..., 1]
        else:
            # Dense Lucas-Kanade approximation using grid
            flow_x, flow_y = self._compute_dense_lk(prev_gray, curr_gray)
        
        # Update previous frame
        self.prev_frame = curr_gray.copy()
        
        # Apply sensitivity
        flow_x *= self.sensitivity
        flow_y *= self.sensitivity
        
        return flow_x, flow_y
    
    def _compute_dense_lk(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dense optical flow using Lucas-Kanade on grid."""
        h, w = prev_gray.shape
        
        # Create grid of points
        step = 5
        y_coords = np.arange(0, h, step)
        x_coords = np.arange(0, w, step)
        
        points = []
        for y in y_coords:
            for x in x_coords:
                points.append([x, y])
        
        points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Calculate optical flow
        new_points, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            points,
            None,
            **self.lk_params
        )
        
        # Calculate flow vectors
        flow_vectors = new_points - points
        
        # Interpolate to dense flow
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        
        idx = 0
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                if status[idx]:
                    flow_x[y, x] = flow_vectors[idx, 0, 0]
                    flow_y[y, x] = flow_vectors[idx, 0, 1]
                idx += 1
        
        # Interpolate sparse flow to dense
        flow_x = ndimage.zoom(flow_x[::step, ::step], step, order=1)[:h, :w]
        flow_y = ndimage.zoom(flow_y[::step, ::step], step, order=1)[:h, :w]
        
        return flow_x, flow_y
    
    def analyze_motion(
        self,
        current_frame: np.ndarray,
        prev_frame: Optional[np.ndarray] = None
    ) -> MotionData:
        """
        Analyze motion between frames.
        
        Args:
            current_frame: Current frame
            prev_frame: Previous frame (optional)
            
        Returns:
            MotionData object with analysis results
        """
        flow_x, flow_y = self.compute_optical_flow(current_frame, prev_frame)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        direction = np.arctan2(flow_y, flow_x)
        
        return MotionData(
            magnitude=magnitude,
            direction=direction,
            flow_x=flow_x,
            flow_y=flow_y,
            mean_magnitude=float(np.mean(magnitude)),
            max_magnitude=float(np.max(magnitude))
        )
    
    def remove_global_motion(
        self,
        motion_data: MotionData,
        threshold_percentile: float = 75
    ) -> MotionData:
        """
        Remove global motion (camera shake) from motion data.
        
        Uses median flow subtraction to isolate local movements.
        """
        # Calculate median flow (global motion estimate)
        global_x = np.median(motion_data.flow_x)
        global_y = np.median(motion_data.flow_y)
        
        # Subtract global motion
        local_flow_x = motion_data.flow_x - global_x
        local_flow_y = motion_data.flow_y - global_y
        
        # Recalculate magnitude and direction
        magnitude = np.sqrt(local_flow_x**2 + local_flow_y**2)
        direction = np.arctan2(local_flow_y, local_flow_x)
        
        return MotionData(
            magnitude=magnitude,
            direction=direction,
            flow_x=local_flow_x,
            flow_y=local_flow_y,
            mean_magnitude=float(np.mean(magnitude)),
            max_magnitude=float(np.max(magnitude))
        )
    
    def detect_motion_regions(
        self,
        magnitude: np.ndarray,
        threshold: float = 1.0,
        min_area: int = 100
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect regions with significant motion.
        
        Args:
            magnitude: Motion magnitude map
            threshold: Minimum magnitude threshold
            min_area: Minimum region area in pixels
            
        Returns:
            List of bounding boxes (x, y, w, h) for motion regions
        """
        # Threshold magnitude
        motion_mask = (magnitude > threshold).astype(np.uint8) * 255
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            motion_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter by area and get bounding boxes
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append((x, y, w, h))
        
        return regions
    
    def calculate_motion_uniformity(
        self,
        motion_data: MotionData,
        grid_size: int = 4
    ) -> float:
        """
        Calculate motion uniformity across the frame.
        
        Low uniformity indicates localized motion (potential kick).
        High uniformity indicates global motion (breathing).
        
        Returns:
            Uniformity score (0-1, higher = more uniform)
        """
        h, w = motion_data.magnitude.shape
        cell_h, cell_w = h // grid_size, w // grid_size
        
        cell_magnitudes = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                cell_mag = np.mean(motion_data.magnitude[y1:y2, x1:x2])
                cell_magnitudes.append(cell_mag)
        
        cell_magnitudes = np.array(cell_magnitudes)
        
        # Calculate coefficient of variation (lower = more uniform)
        if np.mean(cell_magnitudes) > 0:
            cv = np.std(cell_magnitudes) / np.mean(cell_magnitudes)
            uniformity = 1 / (1 + cv)  # Transform to 0-1 range
        else:
            uniformity = 1.0
        
        return float(uniformity)
    
    def reset(self) -> None:
        """Reset detector state."""
        self.prev_frame = None

