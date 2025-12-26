// Baby Kick Visualizer - Frontend Application

class BabyKickApp {
    constructor() {
        this.jobId = null;
        this.metadata = null;
        this.roi = { x: 0, y: 0, width: 0, height: 0 };
        this.isDrawing = false;
        this.startX = 0;
        this.startY = 0;
        this.imageScale = 1;
        this.theme = 'dark';
        this.lastResults = null;
        this.kickMarkers = []; // Store kick marker positions for click detection
        this.graphOptions = {
            showGrid: true,
            showFill: true,
            showKicks: true,
            smooth: true
        };
        
        this.init();
    }
    
    init() {
        this.initTheme();
        this.bindEvents();
        this.setupSliders();
        this.setupResizeHandler();
    }
    
    setupResizeHandler() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                // Redraw timeline on resize if results are shown
                if (this.lastResults) {
                    this.drawTimeline(
                        this.lastResults.magnitude_history,
                        this.lastResults.kick_events,
                        this.lastResults.metadata.fps
                    );
                }
            }, 250);
        });
    }
    
    initTheme() {
        // Check for saved theme preference or system preference
        const savedTheme = localStorage.getItem('babykick-theme');
        const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        if (savedTheme) {
            this.theme = savedTheme;
        } else if (!systemPrefersDark) {
            this.theme = 'light';
        }
        
        this.applyTheme();
        
        // Listen for system theme changes
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem('babykick-theme')) {
                this.theme = e.matches ? 'dark' : 'light';
                this.applyTheme();
            }
        });
    }
    
    applyTheme() {
        document.documentElement.setAttribute('data-theme', this.theme);
    }
    
    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        this.applyTheme();
        localStorage.setItem('babykick-theme', this.theme);
        
        // Redraw timeline if results are shown
        if (this.lastResults) {
            this.drawTimeline(
                this.lastResults.magnitude_history,
                this.lastResults.kick_events,
                this.lastResults.metadata.fps
            );
        }
    }
    
    bindEvents() {
        // Theme toggle
        const themeToggle = document.getElementById('theme-toggle');
        themeToggle.addEventListener('click', () => this.toggleTheme());
        
        // Upload area
        const uploadArea = document.getElementById('upload-area');
        const videoInput = document.getElementById('video-input');
        
        uploadArea.addEventListener('click', () => videoInput.click());
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                this.uploadVideo(e.dataTransfer.files[0]);
            }
        });
        videoInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                this.uploadVideo(e.target.files[0]);
            }
        });
        
        // ROI Canvas
        const canvas = document.getElementById('roi-canvas');
        canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        canvas.addEventListener('mousemove', (e) => this.draw(e));
        canvas.addEventListener('mouseup', () => this.stopDrawing());
        canvas.addEventListener('mouseleave', () => this.stopDrawing());
        
        // Touch events for mobile
        canvas.addEventListener('touchstart', (e) => this.startDrawing(e.touches[0]));
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0]);
        });
        canvas.addEventListener('touchend', () => this.stopDrawing());
        
        // Buttons
        document.getElementById('back-btn').addEventListener('click', () => this.showSection('landing'));
        document.getElementById('analyze-btn').addEventListener('click', () => this.startAnalysis());
        document.getElementById('new-analysis').addEventListener('click', () => this.reset());
        
        // Export buttons
        document.getElementById('export-video').addEventListener('click', () => this.exportResults('video'));
        document.getElementById('export-csv').addEventListener('click', () => this.exportResults('csv'));
        document.getElementById('export-json').addEventListener('click', () => this.exportResults('json'));
        
        // Timeline canvas click for seeking to kicks
        const timelineCanvas = document.getElementById('timeline-canvas');
        timelineCanvas.addEventListener('click', (e) => this.handleTimelineClick(e));
        timelineCanvas.addEventListener('mousemove', (e) => this.handleTimelineHover(e));
        timelineCanvas.addEventListener('mouseleave', () => this.handleTimelineLeave());
        
        // Graph options
        document.getElementById('graph-show-grid')?.addEventListener('change', (e) => {
            this.graphOptions.showGrid = e.target.checked;
            this.redrawTimeline();
        });
        document.getElementById('graph-show-fill')?.addEventListener('change', (e) => {
            this.graphOptions.showFill = e.target.checked;
            this.redrawTimeline();
        });
        document.getElementById('graph-show-kicks')?.addEventListener('change', (e) => {
            this.graphOptions.showKicks = e.target.checked;
            this.redrawTimeline();
        });
        document.getElementById('graph-smooth')?.addEventListener('change', (e) => {
            this.graphOptions.smooth = e.target.checked;
            this.redrawTimeline();
        });
    }
    
    redrawTimeline() {
        if (this.lastResults) {
            this.drawTimeline(
                this.lastResults.magnitude_history,
                this.lastResults.kick_events,
                this.lastResults.metadata.fps
            );
        }
    }
    
    handleTimelineClick(e) {
        if (!this.lastResults) {
            console.log('No results available');
            return;
        }
        
        const canvas = document.getElementById('timeline-canvas');
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        console.log('Timeline clicked at:', x, y, 'Kick markers:', this.kickMarkers.length);
        
        // Check if click is near any kick marker
        if (this.kickMarkers && this.kickMarkers.length > 0) {
            for (const marker of this.kickMarkers) {
                const distance = Math.sqrt(
                    Math.pow(x - marker.x, 2) + Math.pow(y - marker.y, 2)
                );
                
                console.log('Checking marker at:', marker.x, marker.y, 'Distance:', distance);
                
                if (distance <= 20) { // 20px click radius
                    // Seek video to this timestamp
                    const video = document.getElementById('result-video');
                    if (video) {
                        console.log('Seeking to:', marker.timestamp);
                        video.currentTime = marker.timestamp;
                        video.play().catch(err => console.log('Play error:', err));
                        
                        // Visual feedback
                        this.highlightMarker(marker);
                        this.showClickFeedback(marker.timestamp);
                    }
                    return;
                }
            }
        }
        
        // If no kick marker clicked, seek to clicked position in timeline
        const padding = 40;
        const containerWidth = canvas.clientWidth;
        const width = containerWidth - padding * 2;
        
        if (x >= padding && x <= padding + width) {
            const progress = (x - padding) / width;
            const video = document.getElementById('result-video');
            if (video && this.lastResults.metadata) {
                const seekTime = progress * this.lastResults.metadata.duration;
                console.log('Seeking timeline to:', seekTime);
                video.currentTime = seekTime;
            }
        }
    }
    
    showClickFeedback(timestamp) {
        // Create visual feedback
        let feedback = document.getElementById('kick-feedback');
        if (!feedback) {
            feedback = document.createElement('div');
            feedback.id = 'kick-feedback';
            feedback.style.cssText = `
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: linear-gradient(135deg, #ec4899, #f472b6);
                color: white;
                padding: 16px 32px;
                border-radius: 12px;
                font-size: 1.2rem;
                font-weight: 600;
                z-index: 10000;
                pointer-events: none;
                box-shadow: 0 10px 40px rgba(236, 72, 153, 0.5);
                animation: kickFeedback 0.8s ease-out forwards;
            `;
            document.body.appendChild(feedback);
            
            // Add animation styles
            const style = document.createElement('style');
            style.textContent = `
                @keyframes kickFeedback {
                    0% { opacity: 0; transform: translate(-50%, -50%) scale(0.8); }
                    20% { opacity: 1; transform: translate(-50%, -50%) scale(1.1); }
                    100% { opacity: 0; transform: translate(-50%, -50%) scale(1); }
                }
            `;
            document.head.appendChild(style);
        }
        
        feedback.textContent = `⚡ Kick at ${timestamp.toFixed(2)}s`;
        feedback.style.animation = 'none';
        feedback.offsetHeight; // Trigger reflow
        feedback.style.animation = 'kickFeedback 0.8s ease-out forwards';
    }
    
    handleTimelineHover(e) {
        const canvas = document.getElementById('timeline-canvas');
        if (!canvas) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        let isOverMarker = false;
        let hoveredMarker = null;
        
        if (this.kickMarkers && this.kickMarkers.length > 0) {
            for (const marker of this.kickMarkers) {
                const distance = Math.sqrt(
                    Math.pow(x - marker.x, 2) + Math.pow(y - marker.y, 2)
                );
                if (distance <= 20) {
                    isOverMarker = true;
                    hoveredMarker = marker;
                    break;
                }
            }
        }
        
        canvas.style.cursor = isOverMarker ? 'pointer' : 'crosshair';
        
        // Show tooltip on hover
        if (isOverMarker && hoveredMarker) {
            this.showMarkerTooltip(e.clientX, e.clientY, hoveredMarker);
        } else {
            this.hideMarkerTooltip();
        }
    }
    
    showMarkerTooltip(x, y, marker) {
        let tooltip = document.getElementById('marker-tooltip');
        if (!tooltip) {
            tooltip = document.createElement('div');
            tooltip.id = 'marker-tooltip';
            tooltip.style.cssText = `
                position: fixed;
                background: rgba(0, 0, 0, 0.9);
                color: white;
                padding: 8px 14px;
                border-radius: 8px;
                font-size: 0.9rem;
                font-weight: 500;
                z-index: 10000;
                pointer-events: none;
                border: 1px solid #db2777;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            `;
            document.body.appendChild(tooltip);
        }
        
        tooltip.innerHTML = `🦶 Kick at <strong>${marker.timestamp.toFixed(2)}s</strong><br><span style="font-size:0.8rem;opacity:0.7">Click to jump</span>`;
        tooltip.style.left = `${x + 15}px`;
        tooltip.style.top = `${y - 50}px`;
        tooltip.style.display = 'block';
    }
    
    hideMarkerTooltip() {
        const tooltip = document.getElementById('marker-tooltip');
        if (tooltip) {
            tooltip.style.display = 'none';
        }
    }
    
    handleTimelineLeave() {
        const canvas = document.getElementById('timeline-canvas');
        if (canvas) canvas.style.cursor = 'crosshair';
        this.hideMarkerTooltip();
    }
    
    highlightMarker(marker) {
        // Flash effect on the timeline when a kick is clicked
        const canvas = document.getElementById('timeline-canvas');
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        
        // Draw a pulse effect
        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.beginPath();
        ctx.arc(marker.x, marker.y, 20, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(219, 39, 119, 0.5)';
        ctx.fill();
        ctx.restore();
        
        // Redraw after animation
        setTimeout(() => this.redrawTimeline(), 300);
    }
    
    setupSliders() {
        const sliders = [
            { id: 'sensitivity', valueId: 'sensitivity-value' },
            { id: 'magnitude', valueId: 'magnitude-value' },
            { id: 'opacity', valueId: 'opacity-value' }
        ];
        
        sliders.forEach(({ id, valueId }) => {
            const slider = document.getElementById(id);
            const valueDisplay = document.getElementById(valueId);
            slider.addEventListener('input', () => {
                valueDisplay.textContent = slider.value;
            });
        });
    }
    
    showSection(sectionId) {
        document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
        document.getElementById(`${sectionId}-section`).classList.add('active');
    }
    
    async uploadVideo(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            // Show loading state
            const uploadArea = document.getElementById('upload-area');
            uploadArea.innerHTML = `
                <div class="upload-icon">⏳</div>
                <h3>Uploading...</h3>
                <p>Please wait</p>
            `;
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Upload failed');
            }
            
            const data = await response.json();
            this.jobId = data.job_id;
            this.metadata = data.metadata;
            
            // Show ROI section
            this.showROISection(data.preview_url);
            
        } catch (error) {
            alert('Error uploading video: ' + error.message);
            this.resetUploadArea();
        }
    }
    
    resetUploadArea() {
        document.getElementById('upload-area').innerHTML = `
            <div class="upload-icon">📁</div>
            <h3>Upload Your Video</h3>
            <p>Drag & drop or click to browse</p>
            <p class="upload-hint">Supports MP4, AVI, MOV</p>
        `;
    }
    
    showROISection(previewUrl) {
        // Update video info
        const videoInfo = document.getElementById('video-info');
        videoInfo.innerHTML = `
            <span>📐 ${this.metadata.width} × ${this.metadata.height}</span>
            <span>🎬 ${this.metadata.fps.toFixed(1)} FPS</span>
            <span>⏱️ ${this.metadata.duration.toFixed(1)}s</span>
            <span>🖼️ ${this.metadata.frame_count} frames</span>
        `;
        
        // Load preview image
        const previewImg = document.getElementById('preview-image');
        previewImg.onload = () => {
            this.setupCanvas();
            // Set default ROI to center 60%
            const canvas = document.getElementById('roi-canvas');
            this.roi = {
                x: Math.round(this.metadata.width * 0.2),
                y: Math.round(this.metadata.height * 0.2),
                width: Math.round(this.metadata.width * 0.6),
                height: Math.round(this.metadata.height * 0.6)
            };
            this.drawROI();
        };
        previewImg.src = previewUrl;
        
        this.showSection('roi');
    }
    
    setupCanvas() {
        const canvas = document.getElementById('roi-canvas');
        const img = document.getElementById('preview-image');
        
        // Match canvas size to displayed image
        canvas.width = img.clientWidth;
        canvas.height = img.clientHeight;
        
        // Calculate scale
        this.imageScale = img.clientWidth / this.metadata.width;
    }
    
    startDrawing(e) {
        const canvas = document.getElementById('roi-canvas');
        const rect = canvas.getBoundingClientRect();
        
        this.isDrawing = true;
        this.startX = e.clientX - rect.left;
        this.startY = e.clientY - rect.top;
    }
    
    draw(e) {
        if (!this.isDrawing) return;
        
        const canvas = document.getElementById('roi-canvas');
        const rect = canvas.getBoundingClientRect();
        
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        
        // Calculate ROI in original image coordinates
        this.roi = {
            x: Math.round(Math.min(this.startX, currentX) / this.imageScale),
            y: Math.round(Math.min(this.startY, currentY) / this.imageScale),
            width: Math.round(Math.abs(currentX - this.startX) / this.imageScale),
            height: Math.round(Math.abs(currentY - this.startY) / this.imageScale)
        };
        
        this.drawROI();
    }
    
    stopDrawing() {
        this.isDrawing = false;
    }
    
    drawROI() {
        const canvas = document.getElementById('roi-canvas');
        const ctx = canvas.getContext('2d');
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw semi-transparent overlay
        ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Clear ROI area
        const scaledX = this.roi.x * this.imageScale;
        const scaledY = this.roi.y * this.imageScale;
        const scaledW = this.roi.width * this.imageScale;
        const scaledH = this.roi.height * this.imageScale;
        
        ctx.clearRect(scaledX, scaledY, scaledW, scaledH);
        
        // Draw ROI border
        ctx.strokeStyle = '#ec4899';
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        
        // Draw corner handles
        const handleSize = 10;
        ctx.fillStyle = '#ec4899';
        
        // Top-left
        ctx.fillRect(scaledX - handleSize/2, scaledY - handleSize/2, handleSize, handleSize);
        // Top-right
        ctx.fillRect(scaledX + scaledW - handleSize/2, scaledY - handleSize/2, handleSize, handleSize);
        // Bottom-left
        ctx.fillRect(scaledX - handleSize/2, scaledY + scaledH - handleSize/2, handleSize, handleSize);
        // Bottom-right
        ctx.fillRect(scaledX + scaledW - handleSize/2, scaledY + scaledH - handleSize/2, handleSize, handleSize);
        
        // Draw dimensions
        ctx.fillStyle = '#ffffff';
        ctx.font = '14px Outfit';
        ctx.fillText(`${this.roi.width} × ${this.roi.height}`, scaledX + 10, scaledY + 25);
    }
    
    async startAnalysis() {
        if (this.roi.width < 50 || this.roi.height < 50) {
            alert('Please draw a larger region of interest');
            return;
        }
        
        const formData = new FormData();
        formData.append('roi_x', this.roi.x);
        formData.append('roi_y', this.roi.y);
        formData.append('roi_width', this.roi.width);
        formData.append('roi_height', this.roi.height);
        formData.append('sensitivity', document.getElementById('sensitivity').value);
        formData.append('magnitude_threshold', document.getElementById('magnitude').value);
        formData.append('overlay_opacity', document.getElementById('opacity').value);
        formData.append('show_contours', document.getElementById('show-contours').checked);
        formData.append('show_vectors', document.getElementById('show-vectors').checked);
        formData.append('display_mode', document.getElementById('display-mode').value);
        
        try {
            const response = await fetch(`/api/process/${this.jobId}`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Processing failed');
            }
            
            this.showSection('processing');
            this.pollStatus();
            
        } catch (error) {
            alert('Error starting analysis: ' + error.message);
        }
    }
    
    async pollStatus() {
        const checkStatus = async () => {
            try {
                const response = await fetch(`/api/status/${this.jobId}`);
                const data = await response.json();
                
                // Update progress
                document.getElementById('progress-fill').style.width = `${data.progress}%`;
                document.getElementById('progress-text').textContent = `${data.progress}%`;
                
                if (data.status === 'completed') {
                    this.showResults(data);
                } else if (data.status === 'error') {
                    alert('Processing error: ' + data.error);
                    this.showSection('roi');
                } else {
                    setTimeout(checkStatus, 500);
                }
            } catch (error) {
                console.error('Status check error:', error);
                setTimeout(checkStatus, 1000);
            }
        };
        
        checkStatus();
    }
    
    showResults(data) {
        // Update stats
        document.getElementById('stat-kicks').textContent = data.total_kicks;
        document.getElementById('stat-intensity').textContent = data.avg_intensity.toFixed(2);
        document.getElementById('stat-confidence').textContent = `${(data.avg_confidence * 100).toFixed(0)}%`;
        document.getElementById('stat-duration').textContent = `${data.metadata.duration.toFixed(1)}s`;
        
        // Set video source
        const video = document.getElementById('result-video');
        video.src = data.output_video;
        
        // Populate kicks table
        this.populateKicksTable(data.kick_events);
        
        // Show section first, then draw timeline after it's visible
        this.showSection('results');
        
        // Store data for timeline redraw
        this.lastResults = data;
        
        // Initialize graph options from checkboxes
        this.initGraphOptions();
        
        // Draw timeline after DOM update (use requestAnimationFrame to ensure visibility)
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                this.drawTimeline(data.magnitude_history, data.kick_events, data.metadata.fps);
            });
        });
    }
    
    initGraphOptions() {
        const gridCheck = document.getElementById('graph-show-grid');
        const fillCheck = document.getElementById('graph-show-fill');
        const kicksCheck = document.getElementById('graph-show-kicks');
        const smoothCheck = document.getElementById('graph-smooth');
        
        if (gridCheck) this.graphOptions.showGrid = gridCheck.checked;
        if (fillCheck) this.graphOptions.showFill = fillCheck.checked;
        if (kicksCheck) this.graphOptions.showKicks = kicksCheck.checked;
        if (smoothCheck) this.graphOptions.smooth = smoothCheck.checked;
    }
    
    drawTimeline(magnitudeHistory, kickEvents, fps) {
        const canvas = document.getElementById('timeline-canvas');
        const container = canvas.parentElement;
        const ctx = canvas.getContext('2d');
        
        // Clear kick markers array
        this.kickMarkers = [];
        
        // Get actual container dimensions
        const containerWidth = container.clientWidth - 48; // Account for padding
        const containerHeight = 180;
        
        // Set canvas size with device pixel ratio for sharp rendering
        const dpr = window.devicePixelRatio || 1;
        canvas.width = containerWidth * dpr;
        canvas.height = containerHeight * dpr;
        canvas.style.width = containerWidth + 'px';
        canvas.style.height = containerHeight + 'px';
        ctx.scale(dpr, dpr);
        
        const padding = 40;
        const width = containerWidth - padding * 2;
        const height = containerHeight - padding * 2;
        
        // Clear canvas with theme-aware background
        const bgColor = this.theme === 'light' ? '#ffffff' : '#0a0a0f';
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (!magnitudeHistory.length) return;
        
        // Find max magnitude for scaling
        const maxMag = Math.max(...magnitudeHistory, 5);
        
        // Draw grid with theme-aware colors (if enabled)
        if (this.graphOptions.showGrid) {
            ctx.strokeStyle = this.theme === 'light' ? '#e0e0e8' : '#2a2a3e';
            ctx.lineWidth = 1;
            
            // Horizontal grid lines
            for (let i = 0; i <= 4; i++) {
                const y = padding + (height / 4) * i;
                ctx.beginPath();
                ctx.moveTo(padding, y);
                ctx.lineTo(containerWidth - padding, y);
                ctx.stroke();
            }
            
            // Vertical grid lines (time markers)
            const numVertLines = 5;
            for (let i = 0; i <= numVertLines; i++) {
                const x = padding + (i / numVertLines) * width;
                ctx.beginPath();
                ctx.setLineDash([4, 4]);
                ctx.moveTo(x, padding);
                ctx.lineTo(x, padding + height);
                ctx.stroke();
                ctx.setLineDash([]);
            }
        }
        
        // Prepare data points
        let dataPoints = magnitudeHistory.map((mag, i) => ({
            x: padding + (i / magnitudeHistory.length) * width,
            y: padding + height - (mag / maxMag) * height
        }));
        
        // Apply smoothing if enabled
        if (this.graphOptions.smooth && dataPoints.length > 10) {
            dataPoints = this.smoothData(dataPoints, 5);
        }
        
        // Draw the line
        ctx.strokeStyle = '#ec4899';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        dataPoints.forEach((point, i) => {
            if (i === 0) {
                ctx.moveTo(point.x, point.y);
            } else if (this.graphOptions.smooth) {
                // Use bezier curves for smooth lines
                const prev = dataPoints[i - 1];
                const cpx = (prev.x + point.x) / 2;
                ctx.quadraticCurveTo(prev.x, prev.y, cpx, (prev.y + point.y) / 2);
            } else {
                ctx.lineTo(point.x, point.y);
            }
        });
        
        // Complete the last point
        if (this.graphOptions.smooth && dataPoints.length > 1) {
            const last = dataPoints[dataPoints.length - 1];
            ctx.lineTo(last.x, last.y);
        }
        
        ctx.stroke();
        
        // Fill under the line (if enabled)
        if (this.graphOptions.showFill) {
            ctx.lineTo(padding + width, padding + height);
            ctx.lineTo(padding, padding + height);
            ctx.closePath();
            ctx.fillStyle = 'rgba(236, 72, 153, 0.15)';
            ctx.fill();
        }
        
        // Draw kick markers (if enabled)
        if (this.graphOptions.showKicks && kickEvents && kickEvents.length > 0) {
            console.log('Drawing', kickEvents.length, 'kick markers');
            
            kickEvents.forEach((kick, index) => {
                const x = padding + (kick.frame_number / magnitudeHistory.length) * width;
                const y = padding + height - (kick.intensity / maxMag) * height;
                
                console.log(`Kick ${index}: x=${x.toFixed(1)}, y=${y.toFixed(1)}, ts=${kick.timestamp}`);
                
                // Store marker position for click detection
                this.kickMarkers.push({
                    x: x,
                    y: y,
                    timestamp: kick.timestamp,
                    frame: kick.frame_number,
                    intensity: kick.intensity
                });
                
                // Draw pulsing outer ring
                ctx.beginPath();
                ctx.arc(x, y, 16, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(219, 39, 119, 0.15)';
                ctx.fill();
                
                // Draw outer glow
                ctx.beginPath();
                ctx.arc(x, y, 12, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(219, 39, 119, 0.3)';
                ctx.fill();
                
                // Draw marker body
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.fillStyle = '#db2777';
                ctx.fill();
                
                // Draw white border
                ctx.beginPath();
                ctx.arc(x, y, 8, 0, Math.PI * 2);
                ctx.strokeStyle = '#ffffff';
                ctx.lineWidth = 2;
                ctx.stroke();
                
                // Draw inner highlight
                ctx.beginPath();
                ctx.arc(x - 2, y - 2, 2.5, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
                ctx.fill();
            });
            
            console.log('Total kick markers stored:', this.kickMarkers.length);
        }
        
        // Draw axes labels with theme-aware colors
        ctx.fillStyle = this.theme === 'light' ? '#5a5a6e' : '#a0a0b0';
        ctx.font = '12px Outfit';
        
        // Time labels
        const duration = magnitudeHistory.length / fps;
        ctx.fillText('0s', padding - 5, padding + height + 20);
        ctx.fillText(`${duration.toFixed(1)}s`, containerWidth - padding - 25, padding + height + 20);
        ctx.fillText(`${(duration / 2).toFixed(1)}s`, padding + width/2 - 10, padding + height + 20);
        
        // Y-axis label
        ctx.save();
        ctx.translate(15, padding + height/2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillText('Intensity', 0, 0);
        ctx.restore();
        
        // Draw legend if kicks are shown
        if (this.graphOptions.showKicks && kickEvents.length > 0) {
            const legendX = containerWidth - 120;
            const legendY = padding + 10;
            
            ctx.fillStyle = '#db2777';
            ctx.beginPath();
            ctx.arc(legendX, legendY, 5, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = this.theme === 'light' ? '#5a5a6e' : '#a0a0b0';
            ctx.font = '11px Outfit';
            ctx.fillText(`${kickEvents.length} kicks detected`, legendX + 12, legendY + 4);
        }
    }
    
    smoothData(data, windowSize) {
        // Simple moving average smoothing
        const smoothed = [];
        for (let i = 0; i < data.length; i++) {
            let sumX = 0, sumY = 0, count = 0;
            for (let j = Math.max(0, i - windowSize); j <= Math.min(data.length - 1, i + windowSize); j++) {
                sumX += data[j].x;
                sumY += data[j].y;
                count++;
            }
            smoothed.push({
                x: data[i].x, // Keep original x
                y: sumY / count // Average y
            });
        }
        return smoothed;
    }
    
    populateKicksTable(kickEvents) {
        const tbody = document.getElementById('kicks-tbody');
        tbody.innerHTML = '';
        
        if (!kickEvents.length) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No kicks detected</td></tr>';
            return;
        }
        
        kickEvents.forEach((kick, index) => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${index + 1}</td>
                <td>${kick.timestamp.toFixed(2)}s</td>
                <td>${kick.intensity.toFixed(2)}</td>
                <td>${kick.duration_frames} frames</td>
                <td>${(kick.confidence * 100).toFixed(0)}%</td>
            `;
            tbody.appendChild(row);
        });
    }
    
    async exportResults(format) {
        if (!this.jobId) return;
        
        try {
            if (format === 'json') {
                const response = await fetch(`/api/export/${this.jobId}/json`);
                const data = await response.json();
                
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                this.downloadBlob(blob, 'kick_events.json');
            } else {
                // For video and CSV, trigger download
                const link = document.createElement('a');
                link.href = `/api/export/${this.jobId}/${format}`;
                link.download = format === 'video' ? 'processed_video.mp4' : 'kick_events.csv';
                link.click();
            }
        } catch (error) {
            alert('Export error: ' + error.message);
        }
    }
    
    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
    }
    
    reset() {
        // Clean up job
        if (this.jobId) {
            fetch(`/api/job/${this.jobId}`, { method: 'DELETE' }).catch(() => {});
        }
        
        this.jobId = null;
        this.metadata = null;
        this.roi = { x: 0, y: 0, width: 0, height: 0 };
        
        this.resetUploadArea();
        this.showSection('landing');
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new BabyKickApp();
});

