"""
Kick Detection Module for Baby Kick Visualization.
Differentiates baby kicks from breathing motions using temporal and spatial analysis.
"""

import numpy as np
from scipy import signal
from scipy.ndimage import label
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque

from .motion_detector import MotionData


@dataclass
class KickEvent:
    """Container for detected kick event."""
    frame_number: int
    timestamp: float
    center: Tuple[int, int]  # (x, y) center of kick
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    intensity: float  # Peak displacement magnitude
    duration_frames: int  # Duration in frames
    confidence: float  # Detection confidence (0-1)


@dataclass
class KickDetectorConfig:
    """Configuration for kick detection."""
    # Frequency filtering
    breathing_freq_min: float = 0.2  # Hz (12 breaths/min)
    breathing_freq_max: float = 0.4  # Hz (24 breaths/min)
    kick_freq_min: float = 0.5  # Hz
    
    # Thresholds
    magnitude_threshold: float = 2.0  # Minimum displacement for kick
    uniformity_threshold: float = 0.6  # Below this = localized (kick-like)
    spike_threshold: float = 2.5  # Std devs above mean for spike detection
    
    # Temporal
    min_kick_duration_frames: int = 3  # Minimum frames for valid kick
    max_kick_duration_frames: int = 30  # Maximum frames for single kick
    cooldown_frames: int = 10  # Frames between separate kicks
    
    # Spatial
    min_kick_area: int = 100  # Minimum pixels for kick region
    max_kick_area_ratio: float = 0.3  # Max ratio of ROI area
    
    # Buffer
    temporal_buffer_size: int = 90  # ~3 seconds at 30fps


class KickDetector:
    """
    Detects and classifies baby kicks vs breathing motions.
    Uses temporal frequency analysis and spatial localization.
    """
    
    def __init__(self, config: Optional[KickDetectorConfig] = None, fps: float = 30.0):
        self.config = config or KickDetectorConfig()
        self.fps = fps
        
        # Temporal buffers for signal analysis
        self.magnitude_buffer = deque(maxlen=self.config.temporal_buffer_size)
        self.uniformity_buffer = deque(maxlen=self.config.temporal_buffer_size)
        self.motion_buffer: deque = deque(maxlen=self.config.temporal_buffer_size)
        
        # Detection state
        self.frame_count = 0
        self.detected_kicks: List[KickEvent] = []
        self.last_kick_frame = -self.config.cooldown_frames
        self.in_kick_event = False
        self.current_kick_start = 0
        self.current_kick_magnitudes: List[Tuple[int, float, np.ndarray]] = []
        
        # Design bandpass filter for kick frequencies
        self._design_filters()
    
    def _design_filters(self) -> None:
        """Design digital filters for frequency separation."""
        nyquist = self.fps / 2
        
        # High-pass filter to remove breathing (keep > 0.5 Hz)
        if self.config.kick_freq_min < nyquist:
            self.hp_b, self.hp_a = signal.butter(
                2,
                self.config.kick_freq_min / nyquist,
                btype='high'
            )
        else:
            self.hp_b, self.hp_a = [1], [1]
        
        # Low-pass filter for breathing detection
        if self.config.breathing_freq_max < nyquist:
            self.lp_b, self.lp_a = signal.butter(
                2,
                self.config.breathing_freq_max / nyquist,
                btype='low'
            )
        else:
            self.lp_b, self.lp_a = [1], [1]
    
    def process_frame(
        self,
        motion_data: MotionData,
        uniformity: float,
        frame_number: int
    ) -> Optional[KickEvent]:
        """
        Process motion data for a single frame.
        
        Args:
            motion_data: Motion analysis from MotionDetector
            uniformity: Motion uniformity score
            frame_number: Current frame number
            
        Returns:
            KickEvent if kick detected and completed, None otherwise
        """
        self.frame_count = frame_number
        
        # Add to buffers
        self.magnitude_buffer.append(motion_data.mean_magnitude)
        self.uniformity_buffer.append(uniformity)
        self.motion_buffer.append(motion_data)
        
        # Check for kick conditions
        is_kick_frame = self._is_kick_frame(motion_data, uniformity)
        
        # State machine for kick detection
        completed_kick = None
        
        if is_kick_frame:
            if not self.in_kick_event:
                # Start new kick event
                if frame_number - self.last_kick_frame >= self.config.cooldown_frames:
                    self.in_kick_event = True
                    self.current_kick_start = frame_number
                    self.current_kick_magnitudes = []
            
            if self.in_kick_event:
                self.current_kick_magnitudes.append(
                    (frame_number, motion_data.max_magnitude, motion_data.magnitude)
                )
        else:
            if self.in_kick_event:
                # End kick event
                duration = frame_number - self.current_kick_start
                
                if (self.config.min_kick_duration_frames <= duration <= 
                    self.config.max_kick_duration_frames):
                    # Valid kick - create event
                    completed_kick = self._create_kick_event()
                    self.detected_kicks.append(completed_kick)
                    self.last_kick_frame = frame_number
                
                self.in_kick_event = False
                self.current_kick_magnitudes = []
        
        return completed_kick
    
    def _is_kick_frame(self, motion_data: MotionData, uniformity: float) -> bool:
        """Determine if current frame shows kick-like motion."""
        # Check magnitude threshold
        if motion_data.max_magnitude < self.config.magnitude_threshold:
            return False
        
        # Check uniformity (kicks are localized, low uniformity)
        if uniformity > self.config.uniformity_threshold:
            return False
        
        # Check for temporal spike
        if len(self.magnitude_buffer) >= 10:
            recent_mags = list(self.magnitude_buffer)[-30:]
            mean_mag = np.mean(recent_mags[:-1]) if len(recent_mags) > 1 else 0
            std_mag = np.std(recent_mags[:-1]) if len(recent_mags) > 1 else 1
            
            if std_mag > 0:
                z_score = (motion_data.mean_magnitude - mean_mag) / std_mag
                if z_score < self.config.spike_threshold * 0.5:  # Relaxed threshold
                    # Not a significant spike, but might still be part of kick
                    pass
        
        return True
    
    def _create_kick_event(self) -> KickEvent:
        """Create KickEvent from accumulated kick data."""
        if not self.current_kick_magnitudes:
            raise ValueError("No kick data to create event")
        
        # Find peak frame
        peak_idx = max(range(len(self.current_kick_magnitudes)),
                      key=lambda i: self.current_kick_magnitudes[i][1])
        peak_frame, peak_mag, peak_magnitude_map = self.current_kick_magnitudes[peak_idx]
        
        # Find kick center and bounding box from peak magnitude map
        center, bbox = self._find_kick_location(peak_magnitude_map)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_confidence()
        
        return KickEvent(
            frame_number=peak_frame,
            timestamp=peak_frame / self.fps,
            center=center,
            bounding_box=bbox,
            intensity=float(peak_mag),
            duration_frames=len(self.current_kick_magnitudes),
            confidence=confidence
        )
    
    def _find_kick_location(
        self,
        magnitude_map: np.ndarray
    ) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
        """Find the location and bounding box of kick in magnitude map."""
        # Threshold to find kick region
        threshold = np.percentile(magnitude_map, 90)
        kick_mask = magnitude_map > threshold
        
        # Find connected components
        labeled, num_features = label(kick_mask)
        
        if num_features == 0:
            # Fallback to max location
            max_loc = np.unravel_index(np.argmax(magnitude_map), magnitude_map.shape)
            center = (int(max_loc[1]), int(max_loc[0]))  # (x, y)
            bbox = (center[0] - 20, center[1] - 20, 40, 40)
            return center, bbox
        
        # Find largest component
        component_sizes = []
        for i in range(1, num_features + 1):
            component_sizes.append(np.sum(labeled == i))
        
        largest_component = np.argmax(component_sizes) + 1
        component_mask = labeled == largest_component
        
        # Find bounding box
        rows = np.any(component_mask, axis=1)
        cols = np.any(component_mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Calculate center
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        
        bbox = (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        center = (int(center_x), int(center_y))
        
        return center, bbox
    
    def _calculate_confidence(self) -> float:
        """Calculate detection confidence score."""
        if not self.current_kick_magnitudes or len(self.uniformity_buffer) < 5:
            return 0.5
        
        # Factor 1: Duration appropriateness
        duration = len(self.current_kick_magnitudes)
        ideal_duration = 10  # frames
        duration_score = 1 - abs(duration - ideal_duration) / max(duration, ideal_duration)
        
        # Factor 2: Magnitude distinctiveness
        magnitudes = [m[1] for m in self.current_kick_magnitudes]
        peak_mag = max(magnitudes)
        
        buffer_mags = list(self.magnitude_buffer)
        if len(buffer_mags) > len(magnitudes):
            baseline = np.mean(buffer_mags[:-len(magnitudes)])
            mag_ratio = peak_mag / (baseline + 0.01)
            magnitude_score = min(1.0, mag_ratio / 5)
        else:
            magnitude_score = 0.5
        
        # Factor 3: Localization (low uniformity)
        recent_uniformity = list(self.uniformity_buffer)[-duration:]
        uniformity_score = 1 - np.mean(recent_uniformity)
        
        # Weighted combination
        confidence = (
            0.3 * duration_score +
            0.4 * magnitude_score +
            0.3 * uniformity_score
        )
        
        return float(np.clip(confidence, 0, 1))
    
    def apply_temporal_filter(
        self,
        signal_data: np.ndarray,
        filter_type: str = 'highpass'
    ) -> np.ndarray:
        """Apply temporal filtering to signal."""
        if len(signal_data) < 10:
            return signal_data
        
        if filter_type == 'highpass':
            filtered = signal.filtfilt(self.hp_b, self.hp_a, signal_data)
        else:
            filtered = signal.filtfilt(self.lp_b, self.lp_a, signal_data)
        
        return filtered
    
    def get_filtered_magnitude_map(self) -> Optional[np.ndarray]:
        """
        Get temporally filtered magnitude map to isolate kick motion.
        
        Returns:
            Filtered magnitude map or None if insufficient data
        """
        if len(self.motion_buffer) < 10:
            return None
        
        # Stack magnitude maps
        mag_stack = np.array([m.magnitude for m in self.motion_buffer])
        
        # Apply high-pass filter along time axis
        filtered_stack = np.zeros_like(mag_stack)
        for i in range(mag_stack.shape[1]):
            for j in range(mag_stack.shape[2]):
                pixel_signal = mag_stack[:, i, j]
                filtered_stack[:, i, j] = self.apply_temporal_filter(pixel_signal)
        
        # Return most recent filtered frame
        return filtered_stack[-1]
    
    def detect_breathing_pattern(self) -> Optional[float]:
        """
        Detect breathing frequency from motion data.
        
        Returns:
            Estimated breathing frequency in Hz, or None if not detected
        """
        if len(self.magnitude_buffer) < 60:  # Need ~2 seconds minimum
            return None
        
        # Get uniformity signal (breathing shows high uniformity)
        uniformity_signal = np.array(list(self.uniformity_buffer))
        
        # Apply FFT
        fft_result = np.fft.rfft(uniformity_signal)
        freqs = np.fft.rfftfreq(len(uniformity_signal), 1/self.fps)
        
        # Find peak in breathing frequency range
        breathing_mask = (freqs >= self.config.breathing_freq_min) & \
                        (freqs <= self.config.breathing_freq_max)
        
        if not np.any(breathing_mask):
            return None
        
        breathing_power = np.abs(fft_result[breathing_mask])
        breathing_freqs = freqs[breathing_mask]
        
        peak_idx = np.argmax(breathing_power)
        breathing_freq = breathing_freqs[peak_idx]
        
        return float(breathing_freq)
    
    def get_statistics(self) -> dict:
        """Get detection statistics."""
        if not self.detected_kicks:
            return {
                'total_kicks': 0,
                'avg_intensity': 0,
                'avg_confidence': 0,
                'kicks_per_minute': 0
            }
        
        intensities = [k.intensity for k in self.detected_kicks]
        confidences = [k.confidence for k in self.detected_kicks]
        
        duration_seconds = self.frame_count / self.fps if self.fps > 0 else 1
        kicks_per_minute = len(self.detected_kicks) / (duration_seconds / 60)
        
        return {
            'total_kicks': len(self.detected_kicks),
            'avg_intensity': float(np.mean(intensities)),
            'avg_confidence': float(np.mean(confidences)),
            'kicks_per_minute': float(kicks_per_minute)
        }
    
    def reset(self) -> None:
        """Reset detector state."""
        self.magnitude_buffer.clear()
        self.uniformity_buffer.clear()
        self.motion_buffer.clear()
        self.frame_count = 0
        self.detected_kicks = []
        self.last_kick_frame = -self.config.cooldown_frames
        self.in_kick_event = False
        self.current_kick_magnitudes = []

