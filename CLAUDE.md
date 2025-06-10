# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
# Main application (stable version with YOLO object detection)
python app.py

# Windows batch launcher
start.bat
```

### Dependencies Management
```bash
# Install required packages
pip install -r requirements.txt

# For CUDA support (recommended for GPU acceleration)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Testing and Development
```bash
# Check CUDA availability and GPU info
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Architecture Overview

### Core System
This is a real-time computer vision surveillance system built around:
- **YOLO Models**: YOLOv11/YOLOv12 for object detection (models stored in `/models/`)
- **Object Tracking**: Custom `BoundingBox` class for persistent object tracking with unique IDs
- **Crowd Detection**: DBSCAN clustering algorithm for identifying person clusters
- **Stream Processing**: Multi-threaded architecture with separate capture and processing threads

### Key Components

#### Main Processing Pipeline (`app.py:552-750`)
1. **Frame Capture**: RTSP/webcam stream reading in separate thread
2. **YOLO Inference**: Real-time object detection using CUDA-accelerated PyTorch
3. **Object Tracking**: Associates detections with existing tracked objects using 16px tolerance
4. **Crowd Analysis**: Uses DBSCAN clustering to identify groups of people
5. **Visual Rendering**: Overlays detection results, tracking info, and UI elements

#### Object Tracking System (`BoundingBox` class, `app.py:245-310`)
- Maintains persistent object IDs across frames
- Tracks object lifecycle: created, seen, idle time, detection count
- Uses spatial containment (16px tolerance) and temporal constraints (2.5s timeout)
- Stores cropped object images for potential AI description

#### Configuration System
- **Stream Source**: `rtsp_stream` variable supports RTSP, webcam (0), or video files
- **Detection Classes**: `classlist` defines which COCO classes to detect
- **Confidence Thresholds**: `class_confidence` dict sets per-class minimum confidence
- **Performance Tuning**: Frame skipping, buffer management, idle detection

### Data Flow
```
RTSP Stream → Queue → YOLO Detection → Object Association → Crowd Analysis → Display
     ↓                                         ↓
Threading System              Persistent Object Tracking
```

### File Structure
- `app.py`: Main application with YOLO detection and tracking
- `db/coco.names`: COCO dataset class labels (80 classes)
- `models/`: Pre-trained YOLO model files (.pt format)
- `snapshots/`: Screenshot captures (press 'S')
- `recordings/`: Video recordings (press 'R')
- `elements/`: Extracted object images for analysis
- `bat/`: Windows batch scripts for various operations

### Interactive Controls
- `Q`: Quit application
- `R`: Start/Stop recording
- `S`: Take snapshot
- `Space`: Frame skip/clear queue
- Mouse wheel: Zoom (1x-6x)
- Right-click drag: Pan view
- Left-click drag: Draw selection rectangle

### Performance Characteristics
- Processes at 640x480 resolution regardless of input stream size
- Targets 15-30 FPS depending on hardware
- Uses automatic frame skipping during idle periods (3+ seconds)
- **YOLO Optimization**: Runs inference only every 3rd frame, caches results for intermediate frames (reduces GPU usage by ~67%)
- **Interactive Pause**: YOLO inference pauses during zoom/pan operations and resumes 500ms after completion
- **Cache Management**: Inference cache is cleared at pan start/end and during zoom operations to ensure fresh detections
- Requires CUDA-compatible GPU for optimal performance

### Integration Points
- **Ollama Integration**: Optional AI descriptions via local LLM (`ollama` endpoint)
- **Stream Flexibility**: Supports RTSP cameras, USB webcams, and video files  
- **Recording System**: MP4 video output with configurable quality
- **Snapshot System**: JPEG image capture with timestamps