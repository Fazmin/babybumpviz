"""
Video Processing Module for Baby Kick Visualization.
Handles video loading, frame extraction, and preprocessing.
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Optional, List
from dataclasses import dataclass
import tempfile
import os


@dataclass
class VideoMetadata:
    """Container for video metadata."""
    width: int
    height: int
    fps: float
    frame_count: int
    duration: float
    codec: str
    

class VideoProcessor:
    """
    Handles video loading, frame extraction, and preprocessing.
    """
    
    def __init__(self, video_path: Optional[str] = None):
        self.video_path = video_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.metadata: Optional[VideoMetadata] = None
        
        if video_path:
            self.load_video(video_path)
    
    def load_video(self, video_path: str) -> VideoMetadata:
        """
        Load video file and extract metadata.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoMetadata object with video properties
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Extract metadata
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Get codec
        fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        self.metadata = VideoMetadata(
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec
        )
        
        return self.metadata
    
    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Get a specific frame by number."""
        if self.cap is None:
            raise RuntimeError("No video loaded")
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()
        
        return frame if ret else None
    
    def iter_frames(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        Iterate through video frames.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number (None for all)
            step: Frame step size
            
        Yields:
            Tuple of (frame_number, frame)
        """
        if self.cap is None:
            raise RuntimeError("No video loaded")
        
        if end_frame is None:
            end_frame = self.metadata.frame_count
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_num in range(start_frame, end_frame, step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            
            if not ret:
                break
                
            yield frame_num, frame
    
    def preprocess_frame(
        self,
        frame: np.ndarray,
        denoise: bool = True,
        normalize: bool = True,
        blur_kernel: int = 3
    ) -> np.ndarray:
        """
        Preprocess frame for motion detection.
        
        Args:
            frame: Input BGR frame
            denoise: Apply denoising
            normalize: Apply histogram normalization
            blur_kernel: Gaussian blur kernel size
            
        Returns:
            Preprocessed frame
        """
        processed = frame.copy()
        
        # Convert to grayscale for processing
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        else:
            gray = processed
        
        # Apply denoising
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Normalize histogram
        if normalize:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        # Apply Gaussian blur
        if blur_kernel > 0:
            gray = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
        
        return gray
    
    def stabilize_frame(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        max_shift: int = 50
    ) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Stabilize frame against reference using feature matching.
        
        Args:
            frame: Frame to stabilize
            reference_frame: Reference frame
            max_shift: Maximum allowed pixel shift
            
        Returns:
            Tuple of (stabilized_frame, (dx, dy) shift)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_ref = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            gray_ref = reference_frame
        
        # Use phase correlation for shift detection
        shift, _ = cv2.phaseCorrelate(
            np.float32(gray),
            np.float32(gray_ref)
        )
        
        dx, dy = shift
        
        # Limit shift to max allowed
        dx = np.clip(dx, -max_shift, max_shift)
        dy = np.clip(dy, -max_shift, max_shift)
        
        # Apply translation
        M = np.float32([[1, 0, -dx], [0, 1, -dy]])
        stabilized = cv2.warpAffine(
            frame,
            M,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return stabilized, (dx, dy)
    
    def extract_roi_frames(
        self,
        roi: Tuple[int, int, int, int],
        start_frame: int = 0,
        end_frame: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Extract ROI region from all frames.
        
        Args:
            roi: (x, y, width, height) tuple
            start_frame: Starting frame
            end_frame: Ending frame
            
        Returns:
            List of ROI frames
        """
        x, y, w, h = roi
        roi_frames = []
        
        for _, frame in self.iter_frames(start_frame, end_frame):
            roi_frame = frame[y:y+h, x:x+w]
            roi_frames.append(roi_frame)
        
        return roi_frames
    
    def save_processed_video(
        self,
        frames: List[np.ndarray],
        output_path: str,
        fps: Optional[float] = None
    ) -> str:
        """
        Save processed frames as video.
        
        Args:
            frames: List of processed frames
            output_path: Output video path
            fps: Output FPS (uses original if None)
            
        Returns:
            Path to saved video
        """
        if not frames:
            raise ValueError("No frames to save")
        
        if fps is None:
            fps = self.metadata.fps if self.metadata else 30.0
        
        height, width = frames[0].shape[:2]
        
        # Use mp4v codec for compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is in correct format
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        
        out.release()
        return output_path
    
    def reset(self) -> None:
        """Reset video to beginning."""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release(self) -> None:
        """Release video resources."""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def __del__(self):
        self.release()

