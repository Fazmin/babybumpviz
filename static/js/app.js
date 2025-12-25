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
        
        this.init();
    }
    
    init() {
        this.initTheme();
        this.bindEvents();
        this.setupSliders();
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
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 3;
        ctx.strokeRect(scaledX, scaledY, scaledW, scaledH);
        
        // Draw corner handles
        const handleSize = 10;
        ctx.fillStyle = '#7c3aed';
        
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
        
        // Draw timeline
        this.drawTimeline(data.magnitude_history, data.kick_events, data.metadata.fps);
        
        // Populate kicks table
        this.populateKicksTable(data.kick_events);
        
        this.showSection('results');
    }
    
    drawTimeline(magnitudeHistory, kickEvents, fps) {
        const canvas = document.getElementById('timeline-canvas');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;
        
        const padding = 40;
        const width = canvas.width - padding * 2;
        const height = canvas.height - padding * 2;
        
        // Clear canvas with theme-aware background
        const bgColor = this.theme === 'light' ? '#ffffff' : '#0a0a0f';
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        if (!magnitudeHistory.length) return;
        
        // Find max magnitude for scaling
        const maxMag = Math.max(...magnitudeHistory, 5);
        
        // Draw grid with theme-aware colors
        ctx.strokeStyle = this.theme === 'light' ? '#e0e0e8' : '#2a2a3e';
        ctx.lineWidth = 1;
        
        for (let i = 0; i <= 4; i++) {
            const y = padding + (height / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(canvas.width - padding, y);
            ctx.stroke();
        }
        
        // Draw magnitude line
        ctx.strokeStyle = '#7c3aed';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        magnitudeHistory.forEach((mag, i) => {
            const x = padding + (i / magnitudeHistory.length) * width;
            const y = padding + height - (mag / maxMag) * height;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Fill under the line
        ctx.lineTo(padding + width, padding + height);
        ctx.lineTo(padding, padding + height);
        ctx.closePath();
        ctx.fillStyle = 'rgba(124, 58, 237, 0.2)';
        ctx.fill();
        
        // Draw kick markers
        ctx.fillStyle = '#ef4444';
        kickEvents.forEach(kick => {
            const x = padding + (kick.frame_number / magnitudeHistory.length) * width;
            const y = padding + height - (kick.intensity / maxMag) * height;
            
            ctx.beginPath();
            ctx.arc(x, y, 6, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // Draw axes labels with theme-aware colors
        ctx.fillStyle = this.theme === 'light' ? '#5a5a6e' : '#a0a0b0';
        ctx.font = '12px Outfit';
        ctx.fillText('0s', padding, canvas.height - 10);
        ctx.fillText(`${(magnitudeHistory.length / fps).toFixed(1)}s`, canvas.width - padding - 30, canvas.height - 10);
        ctx.fillText('Intensity', 10, padding - 10);
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

