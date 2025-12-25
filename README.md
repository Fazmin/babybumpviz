# 👶 Baby Kick Visualizer

A Python application that processes video files to detect and visualize subtle surface movements on a pregnant person's abdomen, specifically highlighting baby kicks while filtering out breathing motions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-blue.svg)

## ✨ Features

- **🎯 Smart Motion Detection**: Uses optical flow analysis to track pixel displacement with sub-pixel accuracy
- **🧠 Intelligent Filtering**: Differentiates baby kicks from breathing motions using frequency and spatial analysis
- **🌡️ Heat Map Visualization**: Beautiful topographical color maps showing movement intensity
- **📊 Comprehensive Analytics**: Track kick frequency, intensity, and temporal patterns
- **🎬 Video Export**: Export processed videos with overlays and motion data

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/babybumpviz.git
cd babybumpviz
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the App

```bash
# Activate virtual environment
source venv/bin/activate

# Run with Uvicorn
uvicorn main:app --reload --port 8000
```

The app will be available at `http://localhost:8000`

## 📖 Usage Guide

### 1. Upload Video
- Click the upload button or drag and drop your video file
- Supported formats: MP4, AVI, MOV, MKV

### 2. Select Region of Interest (ROI)
- Draw a rectangle around the abdomen area
- This focuses the analysis on the relevant region

### 3. Adjust Settings
Use the sidebar controls to fine-tune detection:
- **Sensitivity**: Higher values detect smaller movements
- **Magnitude Threshold**: Minimum movement for kick detection
- **Overlay Opacity**: Transparency of the heat map

### 4. Start Analysis
- Click "Start Analysis" to process the video
- Watch the real-time progress bar

### 5. Review Results
- Browse through processed frames
- View the motion timeline graph
- Check detected kick events table
- Export results in various formats

## 🛠️ Technical Details

### Motion Detection Algorithm

The app uses **Dense Optical Flow (Farneback method)** to track pixel displacement between consecutive frames:

1. **Preprocessing**: Frames are normalized and denoised
2. **Optical Flow**: Calculates motion vectors for each pixel
3. **Global Motion Removal**: Subtracts camera shake
4. **Temporal Filtering**: High-pass filter isolates sudden movements

### Kick vs. Breathing Differentiation

| Characteristic | Breathing | Kicks |
|---------------|-----------|-------|
| Frequency | 0.2-0.4 Hz | >0.5 Hz |
| Pattern | Uniform, cyclic | Localized, sudden |
| Amplitude | Consistent | Variable spikes |
| Duration | Continuous | 0.5-2 seconds |

### Visualization

- **Color Mapping**:
  - 🔵 Blue/Cyan: Minimal movement (0-0.5mm)
  - 🟢 Green: Low movement (0.5-2mm)
  - 🟡 Yellow/Orange: Moderate movement (2-5mm)
  - 🔴 Red/Magenta: Strong movement (5mm+)

- **Contour Lines**: Topographical overlays showing displacement gradients

## 📁 Project Structure

```
babybumpviz/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── src/
│   ├── __init__.py          # Package initialization
│   ├── video_processor.py   # Video loading & preprocessing
│   ├── motion_detector.py   # Optical flow & motion analysis
│   ├── kick_detector.py     # Kick detection algorithm
│   ├── visualizer.py        # Heat maps & overlays
│   └── utils.py             # Utility functions
├── assets/
│   └── styles.css           # Custom CSS styling
└── temp/                    # Temporary processing files
```

## 🎯 Performance Tips

- **Video Resolution**: 1080p or lower recommended for faster processing
- **ROI Selection**: Smaller ROI = faster processing
- **Lighting**: Consistent, well-lit videos produce better results
- **Clothing**: Tight-fitting clothing on the abdomen improves detection

## 📊 Export Options

- **Video**: Processed video with heat map overlays (.mp4)
- **CSV**: Kick events with timestamps and intensities
- **JSON**: Complete analysis data for further processing

## 🔧 Configuration

### Detection Parameters

Edit `src/kick_detector.py` to adjust:

```python
@dataclass
class KickDetectorConfig:
    breathing_freq_min: float = 0.2  # Hz
    breathing_freq_max: float = 0.4  # Hz
    kick_freq_min: float = 0.5       # Hz
    magnitude_threshold: float = 2.0
    uniformity_threshold: float = 0.6
```

### Visualization Settings

Edit `src/visualizer.py` for colors and thresholds:

```python
@dataclass
class VisualizationConfig:
    low_threshold: float = 0.5
    mid_threshold: float = 2.0
    high_threshold: float = 5.0
    default_opacity: float = 0.5
```

## 🐛 Troubleshooting

### Common Issues

1. **"No module named 'cv2'"**
   ```bash
   pip install opencv-python-headless
   ```

2. **Slow processing**
   - Reduce video resolution
   - Use smaller ROI
   - Disable motion vectors visualization

3. **No kicks detected**
   - Lower the magnitude threshold
   - Increase sensitivity
   - Ensure ROI covers the abdomen

## 📝 License

MIT License - feel free to use and modify for your needs.

## 🙏 Acknowledgments

- OpenCV team for the excellent computer vision library
- Streamlit team for the intuitive web framework
- All expecting parents who inspired this project

---

**Made with ❤️ for expecting parents**

