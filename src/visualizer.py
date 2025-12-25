"""
Visualization Module for Baby Kick Visualization.
Creates heat maps, contour overlays, and composite visualizations.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from .kick_detector import KickEvent
from .motion_detector import MotionData


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""
    # Color map thresholds (in displacement units)
    low_threshold: float = 0.5   # Blue/Green boundary
    mid_threshold: float = 2.0   # Yellow/Orange boundary
    high_threshold: float = 5.0  # Red/Magenta boundary
    
    # Overlay settings
    default_opacity: float = 0.5
    contour_levels: int = 8
    contour_thickness: int = 1
    
    # Kick highlight settings
    highlight_duration_frames: int = 15
    highlight_color: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    pulse_amplitude: float = 0.3  # Opacity pulse range
    
    # Display settings
    show_contours: bool = True
    show_kick_markers: bool = True
    show_motion_vectors: bool = False
    vector_scale: float = 3.0
    vector_step: int = 15


class KickVisualizer:
    """
    Creates visual overlays for motion and kick detection.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        
        # Create custom colormap for heat visualization
        self.heatmap_cmap = self._create_kick_colormap()
        
        # Active highlights for detected kicks
        self.active_highlights: List[Tuple[KickEvent, int]] = []  # (event, remaining_frames)
    
    def _create_kick_colormap(self) -> LinearSegmentedColormap:
        """Create custom colormap for kick visualization."""
        colors = [
            (0.0, 0.0, 0.3),    # Dark blue (no motion)
            (0.0, 0.5, 0.5),    # Cyan (minimal)
            (0.0, 0.8, 0.0),    # Green (low)
            (1.0, 1.0, 0.0),    # Yellow (moderate)
            (1.0, 0.5, 0.0),    # Orange (medium)
            (1.0, 0.0, 0.0),    # Red (high)
            (1.0, 0.0, 1.0),    # Magenta (very high)
        ]
        
        return LinearSegmentedColormap.from_list('kick_heat', colors, N=256)
    
    def create_heatmap(
        self,
        magnitude: np.ndarray,
        normalize: bool = True,
        max_value: Optional[float] = None
    ) -> np.ndarray:
        """
        Create heat map from motion magnitude.
        
        Args:
            magnitude: Motion magnitude array
            normalize: Whether to normalize values
            max_value: Optional max value for normalization
            
        Returns:
            BGR heat map image
        """
        if normalize:
            if max_value is None:
                max_value = self.config.high_threshold
            
            # Normalize to 0-1 range
            normalized = np.clip(magnitude / max_value, 0, 1)
        else:
            normalized = magnitude
        
        # Apply colormap
        colored = self.heatmap_cmap(normalized)
        
        # Convert to BGR (OpenCV format)
        bgr = (colored[:, :, :3] * 255).astype(np.uint8)
        bgr = cv2.cvtColor(bgr, cv2.COLOR_RGB2BGR)
        
        return bgr
    
    def create_contour_overlay(
        self,
        magnitude: np.ndarray,
        base_image: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Create topographical contour lines overlay.
        
        Args:
            magnitude: Motion magnitude array
            base_image: Optional base image for overlay
            
        Returns:
            Contour overlay image (BGR)
        """
        h, w = magnitude.shape
        
        if base_image is not None:
            overlay = base_image.copy()
        else:
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Smooth magnitude for cleaner contours
        smoothed = cv2.GaussianBlur(magnitude, (7, 7), 0)
        
        # Calculate contour levels
        max_mag = max(smoothed.max(), self.config.high_threshold)
        levels = np.linspace(
            self.config.low_threshold,
            max_mag,
            self.config.contour_levels
        )
        
        # Draw contours for each level
        for i, level in enumerate(levels):
            # Create binary mask at this level
            mask = (smoothed >= level).astype(np.uint8) * 255
            
            # Find contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Color based on level (interpolate through colormap)
            t = i / max(len(levels) - 1, 1)
            color = self.heatmap_cmap(t)[:3]
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            
            # Draw contours
            cv2.drawContours(
                overlay,
                contours,
                -1,
                color_bgr,
                self.config.contour_thickness
            )
        
        return overlay
    
    def create_motion_vector_overlay(
        self,
        motion_data: MotionData,
        base_image: np.ndarray
    ) -> np.ndarray:
        """Create motion vector arrows overlay."""
        overlay = base_image.copy()
        h, w = motion_data.magnitude.shape
        step = self.config.vector_step
        scale = self.config.vector_scale
        
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx = motion_data.flow_x[y, x] * scale
                fy = motion_data.flow_y[y, x] * scale
                
                mag = np.sqrt(fx**2 + fy**2)
                if mag > 0.5:  # Only draw significant vectors
                    # Color based on magnitude
                    t = min(mag / (self.config.high_threshold * scale), 1.0)
                    color = self.heatmap_cmap(t)[:3]
                    color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
                    
                    end_x = int(x + fx)
                    end_y = int(y + fy)
                    
                    cv2.arrowedLine(
                        overlay,
                        (x, y),
                        (end_x, end_y),
                        color_bgr,
                        1,
                        tipLength=0.3
                    )
        
        return overlay
    
    def highlight_kick(
        self,
        image: np.ndarray,
        kick_event: KickEvent,
        frame_in_highlight: int
    ) -> np.ndarray:
        """
        Add pulsing highlight for detected kick.
        
        Args:
            image: Base image to overlay on
            kick_event: Kick event to highlight
            frame_in_highlight: Current frame within highlight duration
            
        Returns:
            Image with kick highlight
        """
        overlay = image.copy()
        
        # Calculate pulse effect
        progress = frame_in_highlight / self.config.highlight_duration_frames
        pulse = np.sin(progress * np.pi * 2) * self.config.pulse_amplitude + 0.5
        
        x, y, w, h = kick_event.bounding_box
        
        # Draw pulsing rectangle
        color = self.config.highlight_color
        thickness = max(2, int(3 * pulse))
        
        cv2.rectangle(
            overlay,
            (x, y),
            (x + w, y + h),
            color,
            thickness
        )
        
        # Draw center marker
        cx, cy = kick_event.center
        radius = max(5, int(10 * pulse))
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        
        # Draw kick label
        label = f"KICK {kick_event.intensity:.1f}"
        font_scale = 0.5 + 0.2 * pulse
        cv2.putText(
            overlay,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            2
        )
        
        return overlay
    
    def add_kick_highlight(self, kick_event: KickEvent) -> None:
        """Add new kick to active highlights."""
        self.active_highlights.append(
            (kick_event, self.config.highlight_duration_frames)
        )
    
    def update_highlights(self) -> None:
        """Update highlight timers, removing expired ones."""
        self.active_highlights = [
            (event, remaining - 1)
            for event, remaining in self.active_highlights
            if remaining > 1
        ]
    
    def create_composite_frame(
        self,
        original_frame: np.ndarray,
        motion_data: Optional[MotionData],
        roi: Optional[Tuple[int, int, int, int]] = None,
        opacity: float = 0.5,
        detected_kicks: Optional[List[KickEvent]] = None
    ) -> np.ndarray:
        """
        Create complete visualization composite.
        
        Args:
            original_frame: Original video frame
            motion_data: Motion analysis data
            roi: Region of interest (x, y, w, h)
            opacity: Overlay opacity
            detected_kicks: List of detected kicks to highlight
            
        Returns:
            Composited visualization frame
        """
        result = original_frame.copy()
        
        if motion_data is None:
            return result
        
        # Create heat map
        heatmap = self.create_heatmap(motion_data.magnitude)
        
        # Add contours if enabled
        if self.config.show_contours:
            heatmap = self.create_contour_overlay(motion_data.magnitude, heatmap)
        
        # Apply to ROI or full frame
        if roi is not None:
            x, y, w, h = roi
            # Resize heatmap to match ROI
            if heatmap.shape[:2] != (h, w):
                heatmap = cv2.resize(heatmap, (w, h))
            
            # Blend in ROI
            roi_region = result[y:y+h, x:x+w]
            blended = cv2.addWeighted(roi_region, 1 - opacity, heatmap, opacity, 0)
            result[y:y+h, x:x+w] = blended
            
            # Draw ROI boundary
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            if heatmap.shape[:2] != result.shape[:2]:
                heatmap = cv2.resize(heatmap, (result.shape[1], result.shape[0]))
            result = cv2.addWeighted(result, 1 - opacity, heatmap, opacity, 0)
        
        # Add motion vectors if enabled
        if self.config.show_motion_vectors:
            result = self.create_motion_vector_overlay(motion_data, result)
        
        # Apply kick highlights
        for kick_event, remaining in self.active_highlights:
            frame_in_highlight = self.config.highlight_duration_frames - remaining
            
            # Adjust coordinates if ROI is specified
            if roi is not None:
                # Offset kick coordinates by ROI position
                adjusted_event = KickEvent(
                    frame_number=kick_event.frame_number,
                    timestamp=kick_event.timestamp,
                    center=(kick_event.center[0] + roi[0], kick_event.center[1] + roi[1]),
                    bounding_box=(
                        kick_event.bounding_box[0] + roi[0],
                        kick_event.bounding_box[1] + roi[1],
                        kick_event.bounding_box[2],
                        kick_event.bounding_box[3]
                    ),
                    intensity=kick_event.intensity,
                    duration_frames=kick_event.duration_frames,
                    confidence=kick_event.confidence
                )
                result = self.highlight_kick(result, adjusted_event, frame_in_highlight)
            else:
                result = self.highlight_kick(result, kick_event, frame_in_highlight)
        
        self.update_highlights()
        
        return result
    
    def create_timeline_graph(
        self,
        kick_events: List[KickEvent],
        magnitude_history: List[float],
        fps: float,
        width: int = 800,
        height: int = 150
    ) -> np.ndarray:
        """
        Create timeline graph showing motion intensity and kick events.
        
        Returns:
            BGR image of timeline graph
        """
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        
        # Plot magnitude history
        if magnitude_history:
            times = np.arange(len(magnitude_history)) / fps
            ax.plot(times, magnitude_history, 'b-', alpha=0.7, linewidth=1)
            ax.fill_between(times, magnitude_history, alpha=0.3)
        
        # Mark kick events
        for kick in kick_events:
            ax.axvline(
                kick.timestamp,
                color='red',
                linestyle='--',
                alpha=0.8,
                linewidth=2
            )
            ax.plot(
                kick.timestamp,
                kick.intensity,
                'ro',
                markersize=8
            )
        
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Motion Intensity', fontsize=8)
        ax.set_title('Motion Timeline', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Convert to image
        fig.tight_layout()
        fig.canvas.draw()
        
        # Get image from figure
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        return img
    
    def create_side_by_side(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        label_original: str = "Original",
        label_processed: str = "Processed"
    ) -> np.ndarray:
        """Create side-by-side comparison of original and processed frames."""
        # Ensure same size
        if original.shape != processed.shape:
            processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
        
        # Add labels
        original_labeled = original.copy()
        processed_labeled = processed.copy()
        
        cv2.putText(
            original_labeled,
            label_original,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            processed_labeled,
            label_processed,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
        
        # Combine horizontally
        combined = np.hstack([original_labeled, processed_labeled])
        
        return combined
    
    def create_statistics_overlay(
        self,
        image: np.ndarray,
        stats: dict,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """Add statistics text overlay to image."""
        result = image.copy()
        
        x, y = position
        line_height = 25
        
        lines = [
            f"Kicks Detected: {stats.get('total_kicks', 0)}",
            f"Avg Intensity: {stats.get('avg_intensity', 0):.2f}",
            f"Confidence: {stats.get('avg_confidence', 0):.1%}",
            f"Kicks/min: {stats.get('kicks_per_minute', 0):.1f}"
        ]
        
        # Draw background
        max_width = max(len(line) * 12 for line in lines)
        cv2.rectangle(
            result,
            (x - 5, y - 20),
            (x + max_width, y + len(lines) * line_height),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            result,
            (x - 5, y - 20),
            (x + max_width, y + len(lines) * line_height),
            (0, 255, 0),
            1
        )
        
        for i, line in enumerate(lines):
            cv2.putText(
                result,
                line,
                (x, y + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
        
        return result
    
    def reset(self) -> None:
        """Reset visualizer state."""
        self.active_highlights = []

