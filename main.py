"""
Baby Kick Visualization App - FastAPI Backend
Run with: uvicorn main:app --reload --port 8000
"""

import os
import uuid
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import asdict

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
import aiofiles

from src.video_processor import VideoProcessor
from src.motion_detector import MotionDetector
from src.kick_detector import KickDetector, KickDetectorConfig, KickEvent
from src.visualizer import KickVisualizer, VisualizationConfig

# Initialize FastAPI app
app = FastAPI(
    title="Baby Kick Visualizer",
    description="Detect and visualize baby kicks in pregnancy videos",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# In-memory storage for processing jobs
processing_jobs: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main application page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    # Validate file type
    allowed_types = ["video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"]
    if file.content_type not in allowed_types and not file.filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload MP4, AVI, or MOV.")
    
    # Generate unique ID
    job_id = str(uuid.uuid4())
    file_ext = Path(file.filename).suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{file_ext}"
    
    # Save file
    async with aiofiles.open(video_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Get video metadata
    try:
        video_proc = VideoProcessor(str(video_path))
        metadata = video_proc.metadata
        first_frame = video_proc.get_frame(0)
        video_proc.release()
        
        # Save first frame as preview
        preview_path = OUTPUT_DIR / f"{job_id}_preview.jpg"
        cv2.imwrite(str(preview_path), first_frame)
        
        # Store job info
        processing_jobs[job_id] = {
            "status": "uploaded",
            "video_path": str(video_path),
            "preview_path": str(preview_path),
            "metadata": {
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "frame_count": metadata.frame_count,
                "duration": metadata.duration
            },
            "progress": 0,
            "kick_events": [],
            "magnitude_history": [],
            "output_video": None
        }
        
        return {
            "job_id": job_id,
            "preview_url": f"/outputs/{job_id}_preview.jpg",
            "metadata": processing_jobs[job_id]["metadata"]
        }
        
    except Exception as e:
        # Cleanup on error
        if video_path.exists():
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")


@app.post("/api/process/{job_id}")
async def start_processing(
    job_id: str,
    background_tasks: BackgroundTasks,
    roi_x: int = Form(...),
    roi_y: int = Form(...),
    roi_width: int = Form(...),
    roi_height: int = Form(...),
    sensitivity: float = Form(1.0),
    magnitude_threshold: float = Form(2.0),
    overlay_opacity: float = Form(0.5),
    show_contours: bool = Form(True),
    show_vectors: bool = Form(False),
    display_mode: str = Form("overlay")
):
    """Start video processing with given parameters."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] == "processing":
        raise HTTPException(status_code=400, detail="Job already processing")
    
    # Update job settings
    job["status"] = "processing"
    job["settings"] = {
        "roi": (roi_x, roi_y, roi_width, roi_height),
        "sensitivity": sensitivity,
        "magnitude_threshold": magnitude_threshold,
        "overlay_opacity": overlay_opacity,
        "show_contours": show_contours,
        "show_vectors": show_vectors,
        "display_mode": display_mode
    }
    
    # Start background processing
    background_tasks.add_task(process_video_task, job_id)
    
    return {"message": "Processing started", "job_id": job_id}


def process_video_task(job_id: str):
    """Background task to process video."""
    job = processing_jobs[job_id]
    settings = job["settings"]
    
    try:
        # Initialize processors
        video_proc = VideoProcessor(job["video_path"])
        motion_detector = MotionDetector(sensitivity=settings["sensitivity"])
        
        kick_config = KickDetectorConfig(
            magnitude_threshold=settings["magnitude_threshold"]
        )
        kick_detector = KickDetector(config=kick_config, fps=video_proc.metadata.fps)
        
        vis_config = VisualizationConfig(
            show_contours=settings["show_contours"],
            show_motion_vectors=settings["show_vectors"]
        )
        visualizer = KickVisualizer(config=vis_config)
        
        roi = settings["roi"]
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
            
            # Create visualization based on display mode
            if settings["display_mode"] == "side_by_side":
                heatmap = visualizer.create_heatmap(motion_data.magnitude)
                if settings["show_contours"]:
                    heatmap = visualizer.create_contour_overlay(motion_data.magnitude, heatmap)
                heatmap = cv2.resize(heatmap, (w, h))
                processed = visualizer.create_side_by_side(roi_frame, heatmap)
            elif settings["display_mode"] == "heatmap":
                heatmap = visualizer.create_heatmap(motion_data.magnitude)
                if settings["show_contours"]:
                    heatmap = visualizer.create_contour_overlay(motion_data.magnitude, heatmap)
                processed = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
            else:  # overlay
                processed = visualizer.create_composite_frame(
                    frame,
                    motion_data,
                    roi=roi,
                    opacity=settings["overlay_opacity"]
                )
            
            processed_frames.append(processed)
            
            # Update progress
            job["progress"] = int((frame_num + 1) / total_frames * 100)
        
        video_proc.release()
        
        # Save output video
        output_path = OUTPUT_DIR / f"{job_id}_processed.mp4"
        if processed_frames:
            h, w = processed_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, video_proc.metadata.fps, (w, h))
            
            for frame in processed_frames:
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                out.write(frame)
            
            out.release()
        
        # Convert kick events to serializable format
        kick_events_data = []
        for kick in kick_events:
            kick_events_data.append({
                "frame_number": kick.frame_number,
                "timestamp": kick.timestamp,
                "center": list(kick.center),
                "bounding_box": list(kick.bounding_box),
                "intensity": kick.intensity,
                "duration_frames": kick.duration_frames,
                "confidence": kick.confidence
            })
        
        # Update job
        job["status"] = "completed"
        job["progress"] = 100
        job["kick_events"] = kick_events_data
        job["magnitude_history"] = magnitude_history
        job["output_video"] = f"/outputs/{job_id}_processed.mp4"
        job["total_kicks"] = len(kick_events)
        
        # Calculate statistics
        if kick_events:
            job["avg_intensity"] = sum(k.intensity for k in kick_events) / len(kick_events)
            job["avg_confidence"] = sum(k.confidence for k in kick_events) / len(kick_events)
        else:
            job["avg_intensity"] = 0
            job["avg_confidence"] = 0
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Get processing status and results."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    response = {
        "status": job["status"],
        "progress": job["progress"]
    }
    
    if job["status"] == "completed":
        response.update({
            "output_video": job["output_video"],
            "kick_events": job["kick_events"],
            "magnitude_history": job["magnitude_history"],
            "total_kicks": job["total_kicks"],
            "avg_intensity": job["avg_intensity"],
            "avg_confidence": job["avg_confidence"],
            "metadata": job["metadata"]
        })
    elif job["status"] == "error":
        response["error"] = job.get("error", "Unknown error")
    
    return response


@app.get("/api/export/{job_id}/{format}")
async def export_results(job_id: str, format: str):
    """Export results in various formats."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Processing not completed")
    
    if format == "json":
        data = {
            "total_kicks": job["total_kicks"],
            "avg_intensity": job["avg_intensity"],
            "avg_confidence": job["avg_confidence"],
            "kick_events": job["kick_events"],
            "metadata": job["metadata"]
        }
        return JSONResponse(content=data)
    
    elif format == "csv":
        csv_content = "frame,timestamp,intensity,duration,confidence,center_x,center_y\n"
        for kick in job["kick_events"]:
            csv_content += f"{kick['frame_number']},{kick['timestamp']:.3f},{kick['intensity']:.3f},"
            csv_content += f"{kick['duration_frames']},{kick['confidence']:.3f},"
            csv_content += f"{kick['center'][0]},{kick['center'][1]}\n"
        
        # Save to file
        csv_path = OUTPUT_DIR / f"{job_id}_kicks.csv"
        with open(csv_path, 'w') as f:
            f.write(csv_content)
        
        return FileResponse(
            csv_path,
            media_type="text/csv",
            filename="kick_events.csv"
        )
    
    elif format == "video":
        video_path = OUTPUT_DIR / f"{job_id}_processed.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Video not found")
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename="processed_video.mp4"
        )
    
    else:
        raise HTTPException(status_code=400, detail="Invalid format")


@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Clean up job files."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # Delete files
    files_to_delete = [
        job.get("video_path"),
        job.get("preview_path"),
        str(OUTPUT_DIR / f"{job_id}_processed.mp4"),
        str(OUTPUT_DIR / f"{job_id}_kicks.csv")
    ]
    
    for file_path in files_to_delete:
        if file_path and Path(file_path).exists():
            os.remove(file_path)
    
    del processing_jobs[job_id]
    
    return {"message": "Job deleted"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

