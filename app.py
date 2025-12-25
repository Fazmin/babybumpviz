"""
Baby Kick Visualization App
A Python application to detect and visualize baby kicks in pregnancy videos.

Run with: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List
import time
from collections import deque
import json

from src.video_processor import VideoProcessor
from src.motion_detector import MotionDetector
from src.kick_detector import KickDetector, KickDetectorConfig
from src.visualizer import KickVisualizer, VisualizationConfig


# Page configuration
st.set_page_config(
    page_title="Baby Kick Visualizer",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def load_css():
    css_file = Path(__file__).parent / "assets" / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'video_loaded': False,
        'video_path': None,
        'processing': False,
        'processed_frames': [],
        'kick_events': [],
        'magnitude_history': [],
        'roi': None,
        'current_frame_idx': 0,
        'video_metadata': None,
        'show_analysis': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_landing_page():
    """Render the landing page with intro and features."""
    # Hero section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">👶 Baby Kick Visualizer</div>
        <div class="hero-subtitle">
            Advanced motion detection technology to capture and visualize 
            those precious moments when your baby kicks. Transform your 
            pregnancy videos into stunning heat map visualizations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Detection</div>
            <div class="feature-desc">
                AI-powered algorithm distinguishes baby kicks from 
                breathing motions with high accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌡️</div>
            <div class="feature-title">Heat Map Overlay</div>
            <div class="feature-desc">
                Beautiful topographical color maps highlight 
                movement intensity in real-time.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Detailed Analytics</div>
            <div class="feature-desc">
                Track kick frequency, intensity, and patterns 
                with comprehensive statistics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Call to action
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎬 Get Started")
        st.markdown("Upload a video of your baby bump to begin analysis.")


def render_sidebar():
    """Render sidebar with controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        
        st.markdown("### Detection Parameters")
        
        sensitivity = st.slider(
            "Sensitivity",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Higher values detect smaller movements"
        )
        
        magnitude_threshold = st.slider(
            "Magnitude Threshold",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.25,
            help="Minimum movement required for kick detection"
        )
        
        st.markdown("### Visualization")
        
        overlay_opacity = st.slider(
            "Overlay Opacity",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Transparency of the heat map overlay"
        )
        
        show_contours = st.checkbox("Show Contour Lines", value=True)
        show_vectors = st.checkbox("Show Motion Vectors", value=False)
        
        st.markdown("### Display Options")
        
        display_mode = st.radio(
            "View Mode",
            ["Overlay", "Side by Side", "Heat Map Only"],
            index=0
        )
        
        return {
            'sensitivity': sensitivity,
            'magnitude_threshold': magnitude_threshold,
            'overlay_opacity': overlay_opacity,
            'show_contours': show_contours,
            'show_vectors': show_vectors,
            'display_mode': display_mode
        }


def save_uploaded_video(uploaded_file) -> str:
    """Save uploaded video to temporary file."""
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    return temp_path


def draw_roi_selection(frame: np.ndarray) -> Tuple[int, int, int, int]:
    """Allow user to select ROI on frame."""
    st.markdown("""
    <div class="roi-hint">
        <strong>💡 Tip:</strong> Draw a rectangle around the abdomen area 
        to focus the analysis on the region of interest.
    </div>
    """, unsafe_allow_html=True)
    
    # Use streamlit-drawable-canvas for ROI selection
    try:
        from streamlit_drawable_canvas import st_canvas
        
        # Resize frame for display
        display_height = 400
        scale = display_height / frame.shape[0]
        display_width = int(frame.shape[1] * scale)
        
        display_frame = cv2.resize(frame, (display_width, display_height))
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        from PIL import Image
        bg_image = Image.fromarray(display_frame)
        
        canvas_result = st_canvas(
            fill_color="rgba(102, 126, 234, 0.2)",
            stroke_width=2,
            stroke_color="#667eea",
            background_image=bg_image,
            update_streamlit=True,
            height=display_height,
            width=display_width,
            drawing_mode="rect",
            key="roi_canvas",
        )
        
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data.get("objects", [])
            if objects:
                obj = objects[-1]  # Use last drawn rectangle
                x = int(obj["left"] / scale)
                y = int(obj["top"] / scale)
                w = int(obj["width"] / scale)
                h = int(obj["height"] / scale)
                return (x, y, w, h)
        
        # Default ROI (center 60% of frame)
        h, w = frame.shape[:2]
        return (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))
        
    except ImportError:
        st.warning("Install streamlit-drawable-canvas for interactive ROI selection")
        # Default ROI
        h, w = frame.shape[:2]
        
        col1, col2 = st.columns(2)
        with col1:
            roi_x = st.number_input("ROI X", 0, w, int(w * 0.2))
            roi_w = st.number_input("ROI Width", 10, w, int(w * 0.6))
        with col2:
            roi_y = st.number_input("ROI Y", 0, h, int(h * 0.2))
            roi_h = st.number_input("ROI Height", 10, h, int(h * 0.6))
        
        return (roi_x, roi_y, roi_w, roi_h)


def process_video(
    video_path: str,
    roi: Tuple[int, int, int, int],
    settings: dict,
    progress_callback=None
) -> Tuple[List[np.ndarray], List, List[float]]:
    """Process video and detect kicks."""
    
    # Initialize processors
    video_proc = VideoProcessor(video_path)
    motion_detector = MotionDetector(sensitivity=settings['sensitivity'])
    
    kick_config = KickDetectorConfig(
        magnitude_threshold=settings['magnitude_threshold']
    )
    kick_detector = KickDetector(config=kick_config, fps=video_proc.metadata.fps)
    
    vis_config = VisualizationConfig(
        show_contours=settings['show_contours'],
        show_motion_vectors=settings['show_vectors']
    )
    visualizer = KickVisualizer(config=vis_config)
    
    processed_frames = []
    kick_events = []
    magnitude_history = []
    
    total_frames = video_proc.metadata.frame_count
    
    for frame_num, frame in video_proc.iter_frames():
        # Extract ROI
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        
        # Preprocess
        gray_roi = video_proc.preprocess_frame(roi_frame, denoise=False)
        
        # Detect motion
        motion_data = motion_detector.analyze_motion(gray_roi)
        
        # Remove global motion (camera shake)
        motion_data = motion_detector.remove_global_motion(motion_data)
        
        # Calculate uniformity
        uniformity = motion_detector.calculate_motion_uniformity(motion_data)
        
        # Detect kicks
        kick_event = kick_detector.process_frame(motion_data, uniformity, frame_num)
        
        if kick_event:
            kick_events.append(kick_event)
            visualizer.add_kick_highlight(kick_event)
        
        # Record magnitude
        magnitude_history.append(motion_data.mean_magnitude)
        
        # Create visualization
        if settings['display_mode'] == "Side by Side":
            # Create heat map for ROI
            heatmap = visualizer.create_heatmap(motion_data.magnitude)
            if settings['show_contours']:
                heatmap = visualizer.create_contour_overlay(motion_data.magnitude, heatmap)
            
            # Resize heatmap to match ROI
            heatmap = cv2.resize(heatmap, (w, h))
            
            # Create side-by-side
            processed = visualizer.create_side_by_side(roi_frame, heatmap)
            
        elif settings['display_mode'] == "Heat Map Only":
            heatmap = visualizer.create_heatmap(motion_data.magnitude)
            if settings['show_contours']:
                heatmap = visualizer.create_contour_overlay(motion_data.magnitude, heatmap)
            processed = cv2.resize(heatmap, (w, h))
            
        else:  # Overlay mode
            processed = visualizer.create_composite_frame(
                frame,
                motion_data,
                roi=roi,
                opacity=settings['overlay_opacity'],
                detected_kicks=kick_events
            )
        
        processed_frames.append(processed)
        
        # Update progress
        if progress_callback:
            progress_callback((frame_num + 1) / total_frames)
    
    video_proc.release()
    
    return processed_frames, kick_events, magnitude_history


def render_statistics_panel(kick_events: list, magnitude_history: list, fps: float):
    """Render statistics panel."""
    st.markdown("### 📊 Detection Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_kicks = len(kick_events)
    avg_intensity = np.mean([k.intensity for k in kick_events]) if kick_events else 0
    avg_confidence = np.mean([k.confidence for k in kick_events]) if kick_events else 0
    
    duration_sec = len(magnitude_history) / fps if fps > 0 else 1
    kicks_per_min = total_kicks / (duration_sec / 60) if duration_sec > 0 else 0
    
    with col1:
        st.metric("Total Kicks", total_kicks)
    
    with col2:
        st.metric("Avg Intensity", f"{avg_intensity:.2f}")
    
    with col3:
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")
    
    with col4:
        st.metric("Kicks/min", f"{kicks_per_min:.1f}")


def render_kick_events_table(kick_events: list):
    """Render table of detected kick events."""
    if not kick_events:
        st.info("No kicks detected yet.")
        return
    
    st.markdown("### 🦶 Detected Kicks")
    
    data = []
    for i, kick in enumerate(kick_events):
        data.append({
            "#": i + 1,
            "Time": f"{kick.timestamp:.2f}s",
            "Frame": kick.frame_number,
            "Intensity": f"{kick.intensity:.2f}",
            "Duration": f"{kick.duration_frames} frames",
            "Confidence": f"{kick.confidence:.0%}",
            "Location": f"({kick.center[0]}, {kick.center[1]})"
        })
    
    st.dataframe(data, use_container_width=True)


def export_results(
    processed_frames: list,
    kick_events: list,
    magnitude_history: list,
    fps: float,
    video_path: str
):
    """Export processing results."""
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📹 Export Video"):
            with st.spinner("Exporting video..."):
                output_path = video_path.replace('.', '_processed.')
                
                if processed_frames:
                    h, w = processed_frames[0].shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    
                    for frame in processed_frames:
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        out.write(frame)
                    
                    out.release()
                    
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            "⬇️ Download Video",
                            f.read(),
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
    
    with col2:
        if st.button("📊 Export Data (CSV)"):
            if kick_events:
                csv_data = "frame,timestamp,intensity,duration,confidence,center_x,center_y\n"
                for kick in kick_events:
                    csv_data += f"{kick.frame_number},{kick.timestamp:.3f},{kick.intensity:.3f},"
                    csv_data += f"{kick.duration_frames},{kick.confidence:.3f},"
                    csv_data += f"{kick.center[0]},{kick.center[1]}\n"
                
                st.download_button(
                    "⬇️ Download CSV",
                    csv_data,
                    file_name="kick_events.csv",
                    mime="text/csv"
                )
    
    with col3:
        if st.button("📋 Export JSON"):
            if kick_events:
                json_data = {
                    "total_kicks": len(kick_events),
                    "events": [
                        {
                            "frame": k.frame_number,
                            "timestamp": k.timestamp,
                            "intensity": k.intensity,
                            "duration_frames": k.duration_frames,
                            "confidence": k.confidence,
                            "center": list(k.center),
                            "bounding_box": list(k.bounding_box)
                        }
                        for k in kick_events
                    ]
                }
                
                st.download_button(
                    "⬇️ Download JSON",
                    json.dumps(json_data, indent=2),
                    file_name="kick_events.json",
                    mime="application/json"
                )


def main():
    """Main application entry point."""
    init_session_state()
    
    # Get sidebar settings
    settings = render_sidebar()
    
    # Main content area
    if not st.session_state.video_loaded:
        render_landing_page()
    
    # Video upload section
    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📁 Upload Video",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video of your baby bump for analysis"
    )
    
    if uploaded_file:
        # Save and load video
        if not st.session_state.video_loaded or st.session_state.video_path is None:
            with st.spinner("Loading video..."):
                video_path = save_uploaded_video(uploaded_file)
                st.session_state.video_path = video_path
                st.session_state.video_loaded = True
                
                # Get video metadata
                video_proc = VideoProcessor(video_path)
                st.session_state.video_metadata = video_proc.metadata
                video_proc.release()
        
        # Display video info
        metadata = st.session_state.video_metadata
        if metadata:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Resolution", f"{metadata.width}x{metadata.height}")
            with col2:
                st.metric("FPS", f"{metadata.fps:.1f}")
            with col3:
                st.metric("Duration", f"{metadata.duration:.1f}s")
            with col4:
                st.metric("Frames", metadata.frame_count)
        
        # ROI Selection
        st.markdown("### 🎯 Select Region of Interest")
        
        video_proc = VideoProcessor(st.session_state.video_path)
        first_frame = video_proc.get_frame(0)
        video_proc.release()
        
        if first_frame is not None:
            roi = draw_roi_selection(first_frame)
            st.session_state.roi = roi
            
            st.info(f"Selected ROI: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
        
        # Process button
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Start Analysis", use_container_width=True):
                st.session_state.processing = True
                st.session_state.show_analysis = True
        
        # Processing
        if st.session_state.processing:
            st.markdown("### 🔄 Processing Video...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Processing: {progress*100:.1f}%")
            
            with st.spinner("Analyzing motion patterns..."):
                processed_frames, kick_events, magnitude_history = process_video(
                    st.session_state.video_path,
                    st.session_state.roi,
                    settings,
                    progress_callback=update_progress
                )
                
                st.session_state.processed_frames = processed_frames
                st.session_state.kick_events = kick_events
                st.session_state.magnitude_history = magnitude_history
                st.session_state.processing = False
            
            st.success(f"✅ Analysis complete! Detected {len(kick_events)} kicks.")
        
        # Show results
        if st.session_state.show_analysis and st.session_state.processed_frames:
            st.markdown("---")
            st.markdown("## 🎬 Results")
            
            # Video playback
            if st.session_state.processed_frames:
                frame_idx = st.slider(
                    "Frame",
                    0,
                    len(st.session_state.processed_frames) - 1,
                    st.session_state.current_frame_idx
                )
                st.session_state.current_frame_idx = frame_idx
                
                frame = st.session_state.processed_frames[frame_idx]
                
                # Convert BGR to RGB for display
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                
                st.image(frame_rgb, use_column_width=True)
                
                # Playback controls
                col1, col2, col3 = st.columns([1, 2, 1])
                with col1:
                    if st.button("⏮️ Previous"):
                        if st.session_state.current_frame_idx > 0:
                            st.session_state.current_frame_idx -= 1
                            st.rerun()
                
                with col3:
                    if st.button("Next ⏭️"):
                        if st.session_state.current_frame_idx < len(st.session_state.processed_frames) - 1:
                            st.session_state.current_frame_idx += 1
                            st.rerun()
            
            # Statistics
            render_statistics_panel(
                st.session_state.kick_events,
                st.session_state.magnitude_history,
                st.session_state.video_metadata.fps if st.session_state.video_metadata else 30
            )
            
            # Timeline visualization
            if st.session_state.magnitude_history:
                st.markdown("### 📈 Motion Timeline")
                
                import plotly.graph_objects as go
                
                fps = st.session_state.video_metadata.fps if st.session_state.video_metadata else 30
                times = [i / fps for i in range(len(st.session_state.magnitude_history))]
                
                fig = go.Figure()
                
                # Motion magnitude line
                fig.add_trace(go.Scatter(
                    x=times,
                    y=st.session_state.magnitude_history,
                    mode='lines',
                    name='Motion Intensity',
                    line=dict(color='#667eea', width=1),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))
                
                # Kick markers
                for kick in st.session_state.kick_events:
                    fig.add_vline(
                        x=kick.timestamp,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Kick",
                        annotation_position="top"
                    )
                
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Motion Intensity",
                    height=250,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Kick events table
            render_kick_events_table(st.session_state.kick_events)
            
            # Export options
            export_results(
                st.session_state.processed_frames,
                st.session_state.kick_events,
                st.session_state.magnitude_history,
                st.session_state.video_metadata.fps if st.session_state.video_metadata else 30,
                st.session_state.video_path
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>👶 Baby Kick Visualizer | Made with ❤️ for expecting parents</p>
        <p>Using advanced optical flow analysis to capture precious moments</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

