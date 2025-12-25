# Baby Kick Visualization - Source Package
"""
Core modules for baby kick detection and visualization.
"""

from .video_processor import VideoProcessor
from .motion_detector import MotionDetector
from .kick_detector import KickDetector
from .visualizer import KickVisualizer
from .utils import *

__version__ = "1.0.0"
__all__ = [
    "VideoProcessor",
    "MotionDetector", 
    "KickDetector",
    "KickVisualizer",
]

