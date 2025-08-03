# Machina - Advanced Computer Vision Surveillance System

A real-time object detection and tracking system combining YOLO models with SAM 2.1 segmentation for intelligent video surveillance and analysis.

![Demo Screenshot](demo.png)

## 🎯 Overview

Machina is a sophisticated computer vision application that performs real-time object detection, tracking, and segmentation on video streams. It combines state-of-the-art machine learning models to provide comprehensive scene understanding and object analysis capabilities.

### Key Features

- **Real-time Object Detection**: Uses YOLOv11/YOLOv12 models for fast and accurate object detection
- **Advanced Object Tracking**: Persistent object tracking with unique ID assignment and state management
- **Semantic Segmentation**: Integration with SAM 2.1 (Segment Anything Model) for detailed scene segmentation
- **Crowd Detection**: DBSCAN clustering algorithm for identifying and analyzing crowd formations
- **Multi-stream Support**: Compatible with RTSP streams, webcams, and video files
- **Interactive Controls**: Zoom, pan, and drawing capabilities with mouse controls
- **Recording & Snapshots**: Built-in video recording and snapshot capture functionality
- **AI-powered Descriptions**: Optional integration with Ollama for natural language object descriptions

## 🏗️ Technical Architecture

### Core Components

#### 1. Object Detection Engine (`app.py` / `app2.py`)
- **Model**: YOLOv11n/YOLOv12s for real-time inference
- **Device**: CUDA-accelerated processing with fallback to CPU
- **Classes**: Configurable object classes with confidence thresholds
- **Performance**: Optimized for real-time processing with frame skipping logic

#### 2. Object Tracking System
The `BoundingBox` class implements sophisticated object tracking:
- **Persistent IDs**: Unique object identification across frames
- **State Management**: Tracks object lifecycle (created, seen, idle)
- **Motion Analysis**: Calculates distance, bearing, and movement patterns
- **Feature Extraction**: Deep learning features for improved tracking accuracy

#### 3. Segmentation Integration (app2.py)
- **SAM 2.1 Model**: Automatic mask generation for detailed scene understanding
- **Overlay System**: Visual segmentation overlay with color-coded regions
- **Interactive Toggle**: Real-time enable/disable of segmentation visualization

#### 4. Crowd Analysis
- **DBSCAN Clustering**: Identifies person clusters for crowd detection
- **Density Analysis**: Calculates crowd density and size metrics
- **Visual Indicators**: Real-time crowd boundary visualization

### Data Flow Architecture

```
Video Stream → Frame Processing → YOLO Detection → Object Tracking
     ↓                                                      ↓
SAM Segmentation ← Visual Overlay ← Crowd Analysis ← Feature Extraction
     ↓
Display Output
```

### Technical Implementation Details

#### Object Tracking Algorithm
1. **Detection Phase**: YOLO identifies objects in current frame
2. **Association Phase**: Match detections to existing tracks using:
   - Euclidean distance within 16px tolerance
   - Size similarity comparison
   - Feature similarity (when available)
3. **Update Phase**: Update track states and positions
4. **Cleanup Phase**: Remove stale tracks based on 2.5-second timeout

#### Frame Processing Pipeline
```python
# Core processing loop (app.py:552-750)
def process(photo):
    # 1. Resample HD streams to processing resolution
    if hdstream == True:
        img = resample(photo)
    
    # 2. Convert to tensor and run YOLO inference
    img_tensor = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    results = model(img_tensor, conf=min_confidence, classes=classlist)
    
    # 3. Process detections and update tracking
    for box in boxes:
        point = center(xmin, ymin, xmax, ymax)
        obj = getObject(point, class_name)
        
    # 4. Crowd detection using DBSCAN
    crowds = get_clusters(np.array(rawcrowd), eps=50, min_samples=2)
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or RTSP stream source
- Visual C++ Redistributables (Windows)

### Quick Start
```bash
git clone https://github.com/PsyChip/machina
cd machina
pip install -r requirements.txt

# For CUDA support (recommended)
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Setup Ollama (optional for AI descriptions)
# Install ollama from https://ollama.com/
ollama run llava

python app.py
```

### Dependencies
```
opencv-python==4.10.0.84  # Computer vision operations
ultralytics>=8.3.1        # YOLO model implementation
torch                      # Deep learning framework
scikit-learn              # DBSCAN clustering
numpy                     # Numerical computations
dill>=0.3.9               # Object serialization
pickleshare>=0.7.5        # Caching utilities
```

## 🚀 Usage

### Configuration

#### Stream Sources
```python
# RTSP Camera (app.py:19)
rtsp_stream = "rtsp://<your stream here>"

# Webcam
rtsp_stream = 0

# Video File
rtsp_stream = "path/to/video.mp4"
```

#### Detection Classes (app.py:55-70)
```python
classlist = [
    "person", "car", "motorbike", "bicycle", 
    "truck", "traffic light", "stop sign", 
    "bench", "bird", "cat", "dog", "backpack", 
    "suitcase", "handbag"
]
```

#### Confidence Thresholds (app.py:35-53)
```python
class_confidence = {
    "truck": 0.35,
    "car": 0.15,
    "boat": 0.85,
    "bus": 0.5,
    "person": 0.25,  # Default min_confidence
}
```

### Interactive Controls

| Control | Action |
|---------|--------|
| `Q` | Quit application |
| `R` | Start/Stop recording |
| `S` | Take snapshot |
| `F` | Reset window to stream resolution |
| `M` | Toggle military mode (high-performance processing) |
| `Space` | Frame skip/clear queue |
| `Backspace` | Enter/Exit replay mode |
| **Mouse Wheel** | Zoom in/out (max 6x) |
| **Right Click + Drag** | Pan view |
| **Left Click + Drag** | Draw selection rectangle |

### Command Line Usage
```bash
# Basic execution
python app.py

# Enhanced version with SAM 2.1
python app2.py

# Download SAM models
python sam.py
```

## 📊 Performance Metrics

### Real-world Performance
- **Processing Time**: ~20ms per frame (YOLOv11s on GTX 1060)
- **FPS**: 15-30 FPS depending on hardware
- **Resolution**: Input resampled to 640x480 for processing
- **Memory Usage**: ~2GB RAM + 2-4GB VRAM

### Benchmarks by Hardware
| GPU | Model | FPS | Latency |
|-----|-------|-----|---------|
| GTX 1060 | YOLOv11s | 20-25 | 20ms |
| RTX 3060 | YOLOv11s | 25-30 | 15ms |
| RTX 3060 | YOLOv12s | 20-25 | 25ms |

### Network Streaming
- **Stream Delay**: 1-2 seconds every ~10 minutes
- **Frame Skip**: Automatic during 3+ second idle periods
- **Buffer Size**: 512 frame queue (configurable)

## 🔧 Advanced Features

### AI-Powered Object Descriptions
Integration with Ollama for natural language descriptions (app.py:72-75):
```python
prompts = {
    "person": "get gender and age of this person in 5 words or less",
    "car": "get body type and color of this car in 5 words or less",
}
```

### Object Persistence System
- **Unique IDs**: CRC32-based object identification (app2.py:510)
- **State Tracking**: Created, timestamp, detections, idle time
- **Serialization**: Pickle-based object saving/loading (app2.py:357-372)
- **Image Cropping**: Automatic object image extraction with padding

### Crowd Detection Algorithm
```python
def get_clusters(detected_points, eps=30, min_samples=2):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(detected_points)
    # Returns bounding boxes for person clusters
```

### Zoom and Pan System
- **HD Stream Support**: Automatic detection and resampling (app.py:444-448)
- **Zoom Range**: 1.0x to 6.0x magnification
- **Pan Controls**: Right-click drag with coordinate tracking
- **Coordinate Transform**: Accurate scaling between display and processing resolutions

## 📁 Project Structure

```
machina/
├── app.py              # Main application (stable version)
├── app2.py             # Enhanced version with SAM 2.1 integration
├── sam.py              # SAM 2.1 model downloader utility
├── test.py             # Testing framework (placeholder)
├── start.bat           # Windows launcher script
├── requirements.txt    # Python dependencies
├── demo.png           # Demo screenshot
├── db/
│   └── coco.names     # COCO dataset class labels (80 classes)
├── models/            # YOLO model storage (.pt files)
├── snapshots/         # Captured screenshot images
├── recordings/        # MP4 video recordings
└── elements/          # Extracted object images for training
```

## 🔬 Algorithm Deep Dive

### Object Tracking Implementation (app.py:245-310)
```python
class BoundingBox:
    def __init__(self, name, points, size, image):
        self.x, self.y = points           # Center coordinates
        self.created = millis()           # Creation timestamp
        self.size = size                  # Bounding box size
        self.detections = 0               # Detection count
        self.idle = 0                     # Idle time in seconds
        
    def contains(self, x, y, time):
        # 16-pixel tolerance check
        return (self.min_x <= x <= self.max_x and 
                self.min_y <= y <= self.max_y and 
                (time - self.seen < point_timeout))
```

### Segmentation Pipeline (app2.py:51-134)
1. **Initial Snapshot**: Capture reference frame on first detection
2. **SAM Processing**: Generate masks using SAM2AutomaticMaskGenerator
3. **Color Assignment**: Random vibrant colors for each segment
4. **Overlay Blending**: 70% original + 30% segmentation overlay

### Performance Optimizations
- **Frame Skipping**: Intelligent queue clearing during idle periods (app.py:496-506)
- **GPU Memory Management**: Tensor operations with CUDA optimization
- **Threading**: Separate stream capture and AI processing threads
- **Confidence Filtering**: Class-specific confidence thresholds

## 🛡️ Security & Privacy

### Network Security
- **Local Processing**: All AI inference runs locally
- **RTSP Authentication**: Supports username/password authentication
- **No Cloud Dependencies**: Completely offline operation

### Data Handling
- **Local Storage**: All recordings and snapshots stored locally
- **Temporary Processing**: In-memory frame processing only
- **Configurable Retention**: User-controlled data retention policies

## 🤝 Contributing

### Development Areas
- **Model Integration**: Additional YOLO versions and custom models
- **UI Enhancements**: Web interface and remote monitoring
- **Event Detection**: Advanced behavior analysis and alerts
- **Performance**: Multi-GPU support and optimization

### Customization Examples
```python
# Custom detection classes for specific use cases
# Security monitoring
classlist = ["person", "vehicle", "bag", "weapon"]

# Traffic analysis  
classlist = ["car", "truck", "bus", "motorcycle", "bicycle"]

# Wildlife monitoring
classlist = ["bird", "deer", "bear", "wolf"]
```

## 📄 License & Acknowledgments

### Open Source Components
- **Ultralytics YOLO**: Apache 2.0 License
- **Meta SAM 2.1**: Apache 2.0 License  
- **OpenCV**: Apache 2.0 License
- **PyTorch**: BSD-style License
- **scikit-learn**: BSD License

### Creator
Created by **PsyChip** (root@psychip.net)

### Support
- [Ko-fi Donations](https://ko-fi.com/psychip)
- BTC: `bc1qlq067vldngs37l5a4yjc4wvhyt89wv3u68dsuv`

---

**Note**: This system is designed for educational and research purposes. Ensure compliance with local privacy laws and regulations when deploying in production surveillance environments.