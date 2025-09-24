print(f"Initializing..")
# Essential imports for basic video streaming
import time
import os
import cv2
import queue
import threading
import sys
import json
import math
from datetime import datetime
import argparse

# Heavy imports will be loaded in background

# ============================================
# USER CONFIGURABLE VARIABLES
# ============================================

# YOLO Model and Stream Settings
model = "yolo12m"  # YOLO model to use (yolo11n, yolo12n, yolo12s, etc.)

# Parse command line arguments
parser = argparse.ArgumentParser(description="MACHINA - Video Stream Processor")
parser.add_argument(
    "--stream",
    type=str,
    default="0",
    help="RTSP stream URL or 0 for webcam (default: 0)",
)
args = parser.parse_args()

# Convert '0' string to integer 0 for webcam
if args.stream == "0":
    rtsp_stream = 0
else:
    rtsp_stream = args.stream

# Processing and Performance Settings
yolo_skip_frames = 2  # Process every Nth frame (2 = every 2nd frame)
buffer = 512  # Frame buffer size
min_confidence = 0.15  # Minimum confidence threshold for detections
min_size = 20  # Minimum size for car detections

# Object Tracking Settings
point_timeout = 2500  # Time before objects are considered lost (ms)
stationary_val = 16  # Movement threshold for stationary objects
idle_reset = 3000  # Time before frame skip reset (ms)
obj_max = 16  # Maximum number of tracked objects
padding = 6  # Padding around detected objects for cropping

# Display Settings
opsize = (640, 480)  # Default processing/display resolution
yolo_input_size = 640  # Square size divisible by 32 for YOLO input
snapshot_directory = "snapshots"  # Directory for snapshots
_font = cv2.FONT_HERSHEY_SIMPLEX  # Font for text overlay

# Replay and UI Settings
replay_buffer_max_size = 300  # ~10 seconds at 30fps
resolution_display_duration = 2000  # Resolution display time (ms)

# Object Detection Confidence Thresholds
class_confidence = {
    "truck": 0.35,
    "car": 0.15,
    "boat": 0.85,
    "bus": 0.5,
    "aeroplane": 0.85,
    "frisbee": 0.85,
    "pottedplant": 0.55,
    "train": 0.85,
    "chair": 0.5,
    "parking meter": 0.9,
    "fire hydrant": 0.65,
    "traffic light": 0.65,
    "backpack": 0.65,
    "bicycle": 0.55,
    "bench": 0.75,
    "zebra": 0.90,
    "tvmonitor": 0.80,
}

classlist = [
    "person",
    "car",
    "motorbike",
    "bicycle",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "bird",
    "cat",
    "dog",
    "backpack",
    "suitcase",
    "handbag",
]

# ============================================
# SYSTEM VARIABLES (DO NOT MODIFY)
# ============================================

# Stream and processing state
frames = 0
prev_frames = 0
last_frame = 0
fps = 0
recording = False
out = None
streamsize = (0, 0)
original_opsize = None  # Will store the original stream size for Ctrl+4

# Zoom and pan state
zoom_factor = 1.0
pan_x = 0
pan_y = 0
zoom_mode_active = False
stored_bounding_boxes = []

# UI state variables
hdstream = False
drawing = False
dragging = False
drag_start_x = 0
drag_start_y = 0
draw_start_x = 0
draw_start_y = 0
draw_end_x = 0
draw_end_y = 0
military_mode = False
show_info_overlay = False
show_help_text = False
help_text_start_time = 0
show_frame_skip_display = False
frame_skip_display_start_time = 0
yolo_first_processing_started = False
is_first_run = False
webcam_max_resolution = None  # Store webcam's maximum supported resolution


def transform(xmin, ymin, xmax, ymax, pad):
    x_scale = streamsize[0] / opsize[0]
    y_scale = streamsize[1] / opsize[1]

    new_xmin = int(xmin * x_scale) - pad
    new_ymin = int(ymin * y_scale) - pad
    new_xmax = int(xmax * x_scale) + pad
    new_ymax = int(ymax * y_scale) + pad
    return (new_xmin, new_ymin, new_xmax, new_ymax)


def resample(frame):
    global zoom_factor, pan_x, pan_y, streamsize, opsize

    zoomed_width = int(streamsize[0] / zoom_factor)
    zoomed_height = int(streamsize[1] / zoom_factor)

    center_x = streamsize[0] // 2 + pan_x
    center_y = streamsize[1] // 2 + pan_y

    start_x = max(0, min(streamsize[0] - zoomed_width, center_x - zoomed_width // 2))
    start_y = max(0, min(streamsize[1] - zoomed_height, center_y - zoomed_height // 2))

    zoomed_frame = frame[
        start_y : start_y + zoomed_height, start_x : start_x + zoomed_width
    ]

    return cv2.resize(zoomed_frame, opsize, interpolation=cv2.INTER_LINEAR_EXACT)


def rest(url, payload):
    headers = {"Content-Type": "application/json"}
    r = False
    try:
        data = json.dumps(payload)
        response = requests.post(url, data, headers=headers)
        if response.status_code == 200:
            r = json.loads(response.text)
        else:
            print(response.text)
            return False
    except Exception as e:
        print(f"-- error {e}")
    finally:
        return r


def millis():
    return round(time.perf_counter() * 1000)


def format_duration(seconds):
    """Convert seconds to human readable format like 1h20m5s, 10m, 5s"""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        if remaining_seconds == 0:
            return f"{minutes}m"
        else:
            return f"{minutes}m{remaining_seconds}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        if remaining_minutes == 0 and remaining_seconds == 0:
            return f"{hours}h"
        elif remaining_seconds == 0:
            return f"{hours}h{remaining_minutes}m"
        else:
            return f"{hours}h{remaining_minutes}m{remaining_seconds}s"


def toggle_fullscreen():
    """Toggle between fullscreen and windowed mode"""
    global fullscreen, window

    if fullscreen:
        # Exit fullscreen
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, original_window_size[0], original_window_size[1])
        fullscreen = False
        print("Exited fullscreen")
    else:
        # Enter fullscreen
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        fullscreen = True
        print("Entered fullscreen")


def window_resize_callback(val):
    """Callback for window resize events to maintain aspect ratio"""
    global window_aspect_ratio, original_window_size

    # Get current window size
    try:
        # This is a workaround since OpenCV doesn't provide direct window size callbacks
        # We'll handle aspect ratio preservation in the main loop
        pass
    except:
        pass


def reset_window_to_stream_resolution():
    """Reset window size to match stream resolution"""
    global window, streamsize, original_window_size
    reset_size = opsize

    cv2.resizeWindow(window, reset_size[0], reset_size[1])
    original_window_size = reset_size
    print(f"Reset window to {reset_size[0]}x{reset_size[1]}")


def resize_stream_dimensions(new_size):
    """Resize stream processing dimensions and OpenCV window"""
    global opsize, window, original_window_size, yolo_input_size, fullscreen, cap, rtsp_stream
    global resolution_display_active, resolution_display_text, resolution_display_start_time

    opsize = new_size

    # If using webcam, check against maximum supported resolution
    if rtsp_stream == 0 and cap and webcam_max_resolution:
        if (
            opsize[0] <= webcam_max_resolution[0]
            and opsize[1] <= webcam_max_resolution[1]
        ):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, opsize[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opsize[1])
            print(f"Set webcam resolution to {opsize[0]}x{opsize[1]}")

            # Verify what resolution was actually set
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if (actual_width, actual_height) != opsize:
                print(
                    f"Warning: Webcam set to {actual_width}x{actual_height}, not {opsize[0]}x{opsize[1]}"
                )
        else:
            print(
                f"Cannot set {opsize[0]}x{opsize[1]} - webcam max is {webcam_max_resolution[0]}x{webcam_max_resolution[1]}"
            )
            # Don't change the resolution if it exceeds webcam capability
            return

    # Adjust YOLO input size to be the larger dimension, rounded up to nearest 32
    max_dim = max(opsize[0], opsize[1])
    yolo_input_size = ((max_dim + 31) // 32) * 32  # Round up to nearest multiple of 32

    # If currently in fullscreen, maintain fullscreen after resize
    if fullscreen:
        cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.resizeWindow(window, opsize[0], opsize[1])

    original_window_size = opsize

    # Activate resolution display
    resolution_display_active = True
    resolution_display_text = f"{opsize[0]}x{opsize[1]}"
    resolution_display_start_time = millis()

    print(
        f"Resized stream to {opsize[0]}x{opsize[1]}, YOLO input: {yolo_input_size}x{yolo_input_size}"
    )


def timestamp():
    return int(time.time())


def detect_webcam_max_resolution(cap):
    """Detect the maximum resolution supported by the webcam with fingerprinting"""
    global webcam_max_resolution

    print("DEBUG: Starting webcam resolution detection with fingerprinting...")

    # Get webcam fingerprint
    webcam_id, fingerprint = get_webcam_fingerprint(cap)

    # Load existing webcam configurations
    webcam_configs = load_webcam_config()

    # Check if we already know this webcam
    if webcam_id in webcam_configs:
        print(f"DEBUG: Found existing config for webcam: {webcam_id}")
        saved_config = webcam_configs[webcam_id]
        max_width = saved_config["max_width"]
        max_height = saved_config["max_height"]

        print(f"DEBUG: Using saved resolution: {max_width}x{max_height}")

        # Set the webcam to the known good resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)

        webcam_max_resolution = (max_width, max_height)
        print(f"Webcam resolution loaded from cache: {max_width}x{max_height}")
        return webcam_max_resolution

    # New webcam - need to test resolutions
    print(f"DEBUG: New webcam detected: {webcam_id}")
    print("DEBUG: Testing resolutions for the first time...")

    # Test resolutions from highest to lowest - stop at first working one
    test_resolutions = [
        (1920, 1080),  # 1080p
        (1280, 720),  # 720p
        (640, 480),  # 480p
    ]

    print(
        f"DEBUG: Testing resolutions in order (will stop at first working): {[f'{w}x{h}' for w, h in test_resolutions]}"
    )

    max_width = 0
    max_height = 0

    for i, (width, height) in enumerate(test_resolutions):
        print(
            f"DEBUG: Testing resolution {i+1}/{len(test_resolutions)}: {width}x{height}"
        )

        # Try to set resolution
        print(f"DEBUG: Setting width to {width}...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        print(f"DEBUG: Setting height to {height}...")
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Check what was actually set
        print("DEBUG: Getting actual width...")
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print("DEBUG: Getting actual height...")
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"DEBUG: Actual resolution set: {actual_width}x{actual_height}")

        # If we got the requested resolution, this is supported
        if (
            actual_width >= width * 0.9 and actual_height >= height * 0.9
        ):  # Allow 10% tolerance
            print(f"DEBUG: Resolution {width}x{height} works! Using this resolution.")
            max_width = actual_width
            max_height = actual_height
            print(f"DEBUG: Selected resolution: {max_width}x{max_height}")
            break  # Stop testing - use the first working resolution
        else:
            print(f"DEBUG: Resolution {width}x{height} not supported, trying next...")

    print("DEBUG: Resolution detection complete")
    webcam_max_resolution = (max_width, max_height)

    # Save the results for this webcam
    webcam_configs[webcam_id] = {
        "fingerprint": fingerprint,
        "max_width": max_width,
        "max_height": max_height,
        "tested_date": datetime.now().isoformat(),
        "resolutions_tested": test_resolutions,
    }

    if save_webcam_config(webcam_configs):
        print(f"DEBUG: Saved webcam config for future use: {webcam_id}")

    print(f"Webcam maximum resolution detected and saved: {max_width}x{max_height}")
    return webcam_max_resolution


def get_available_resolutions():
    """Get available resolutions based on current stream type"""
    global webcam_max_resolution, rtsp_stream

    # Define resolution presets
    all_resolutions = [(640, 480), (800, 600), (1024, 768), (1280, 800), (1920, 1080)]

    if rtsp_stream == 0 and webcam_max_resolution:
        # Filter resolutions that fit within webcam capability
        available = []
        for res in all_resolutions:
            if (
                res[0] <= webcam_max_resolution[0]
                and res[1] <= webcam_max_resolution[1]
            ):
                available.append(res)
        return available if available else [webcam_max_resolution]
    else:
        # For RTSP streams, all resolutions are available (will be scaled)
        return all_resolutions


def cycle_resolution(direction):
    """Cycle through available resolution presets with + and - keys"""
    global opsize

    available_resolutions = get_available_resolutions()

    # Find current resolution in available list
    try:
        current_index = available_resolutions.index(opsize)
    except ValueError:
        # Current resolution not in list, start from first
        current_index = 0

    if direction > 0:  # + key - increase resolution
        current_index = (current_index + 1) % len(available_resolutions)
    else:  # - key - decrease resolution
        current_index = (current_index - 1) % len(available_resolutions)

    new_resolution = available_resolutions[current_index]

    # Show available resolutions for debugging
    if rtsp_stream == 0:
        print(f"Available webcam resolutions: {available_resolutions}")

    resize_stream_dimensions(new_resolution)
    print(f"Resolution cycled to: {new_resolution[0]}x{new_resolution[1]}")


def load_config():
    """Load configuration from config.json, create if doesn't exist"""
    global yolo_skip_frames, opsize, show_info_overlay, is_first_run

    config_file = "config.json"
    default_config = {
        "processing_nth_frame": 2,
        "screen_resolution": [640, 480],
        "window_position": [100, 100],
        "first_run": True,
    }

    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                yolo_skip_frames = config.get("processing_nth_frame", 2)
                opsize = tuple(config.get("screen_resolution", [640, 480]))
                window_pos = config.get("window_position", [100, 100])
                is_first_run = config.get("first_run", False)
                print(
                    f"Config loaded: skip={yolo_skip_frames}, resolution={opsize}, pos={window_pos}"
                )
                return window_pos
        else:
            # First run - create config file
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=2)
            is_first_run = True
            print("First run detected - created config.json")
            return default_config["window_position"]
    except Exception as e:
        print(f"Error loading config: {e}")
        is_first_run = True
        return default_config["window_position"]


def save_config():
    """Save current configuration to config.json"""
    global yolo_skip_frames, opsize, window

    config_file = "config.json"

    try:
        # Get current window position (if possible)
        window_pos = [100, 100]  # Default fallback
        try:
            # OpenCV doesn't provide direct way to get window position
            # We'll use the stored values or defaults
            pass
        except:
            pass

        config = {
            "processing_nth_frame": yolo_skip_frames,
            "screen_resolution": list(opsize),
            "window_position": window_pos,
            "first_run": False,
        }

        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Configuration saved: skip={yolo_skip_frames}, resolution={opsize}")
    except Exception as e:
        print(f"Error saving config: {e}")


def get_gpu_info():
    """Get GPU name and VRAM info"""
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_vram_gb = round(gpu_memory / 1024**3, 1)
            return gpu_name, f"{gpu_vram_gb}GB"
        else:
            return "No CUDA GPU", "N/A"
    except:
        return "Unknown GPU", "N/A"


def get_webcam_fingerprint(cap):
    """Get webcam identifying information for fingerprinting"""
    try:
        # Try to get webcam properties that might identify it
        backend_name = cap.getBackendName()

        # Get various properties that might be unique to this webcam
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        contrast = cap.get(cv2.CAP_PROP_CONTRAST)
        saturation = cap.get(cv2.CAP_PROP_SATURATION)
        hue = cap.get(cv2.CAP_PROP_HUE)

        # Create a fingerprint from available properties
        fingerprint = {
            "backend": backend_name,
            "default_width": width,
            "default_height": height,
            "fps": fps,
            "brightness": brightness,
            "contrast": contrast,
            "saturation": saturation,
            "hue": hue,
        }

        # Create a unique ID by combining key properties
        unique_id = f"{backend_name}_{width}x{height}_{int(fps)}fps"

        print(f"DEBUG: Webcam fingerprint created: {unique_id}")
        print(f"DEBUG: Fingerprint details: {fingerprint}")

        return unique_id, fingerprint

    except Exception as e:
        print(f"DEBUG: Error getting webcam fingerprint: {e}")
        # Fallback fingerprint
        return "unknown_webcam", {"backend": "unknown"}


def load_webcam_config():
    """Load webcam configuration from webcam.json"""
    config_file = "webcam.json"

    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                print(f"DEBUG: Loaded webcam config with {len(config)} entries")
                return config
        else:
            print("DEBUG: No webcam.json found, will create new one")
            return {}
    except Exception as e:
        print(f"DEBUG: Error loading webcam config: {e}")
        return {}


def save_webcam_config(webcam_configs):
    """Save webcam configuration to webcam.json"""
    config_file = "webcam.json"

    try:
        with open(config_file, "w") as f:
            json.dump(webcam_configs, f, indent=2)
        print(f"DEBUG: Saved webcam config with {len(webcam_configs)} entries")
        return True
    except Exception as e:
        print(f"DEBUG: Error saving webcam config: {e}")
        return False


def get_app_modified_time():
    """Get last modified time of the running script with relative time"""
    try:
        # Get the actual script filename that's being executed
        script_path = os.path.abspath(sys.argv[0])
        script_name = os.path.basename(script_path)

        modified_time = os.path.getmtime(script_path)
        modified_datetime = datetime.fromtimestamp(modified_time)
        current_datetime = datetime.now()

        # Calculate days difference
        time_diff = current_datetime - modified_datetime
        days_ago = time_diff.days

        # Format the date (handle both Linux and Windows formats)
        try:
            formatted_date = modified_datetime.strftime("%-d %b %Y")  # Linux format
        except:
            formatted_date = modified_datetime.strftime("%#d %b %Y")  # Windows format

        if days_ago == 0:
            relative_time = "today"
        elif days_ago == 1:
            relative_time = "1 day ago"
        else:
            relative_time = f"{days_ago} days ago"

        return f"{script_name}: {formatted_date} ({relative_time})"
    except:
        return "Script: Unknown"


def draw_info_overlay(img):
    """Draw the MACHINA info overlay with reduced brightness"""
    overlay = img.copy()

    # Reduce brightness by 50%
    overlay = cv2.convertScaleAbs(overlay, alpha=0.5, beta=0)

    height, width = img.shape[:2]

    # Draw MACHINA title at top
    title = "MACHINA"
    title_font_scale = 2.0
    title_thickness = 3
    title_size = cv2.getTextSize(title, _font, title_font_scale, title_thickness)[0]
    title_x = (width - title_size[0]) // 2
    title_y = 60

    # Draw title with shadow
    cv2.putText(
        overlay,
        title,
        (title_x + 2, title_y + 2),
        _font,
        title_font_scale,
        (0, 0, 0),
        title_thickness + 2,
    )
    cv2.putText(
        overlay,
        title,
        (title_x, title_y),
        _font,
        title_font_scale,
        (255, 255, 255),
        title_thickness,
    )

    # Draw separator line
    line_y = title_y + 20
    line_start_x = width // 4
    line_end_x = 3 * width // 4
    cv2.line(overlay, (line_start_x, line_y), (line_end_x, line_y), (255, 255, 255), 2)

    # Keyboard commands list
    commands = [
        "SPACE - Frame skip",
        "Q - Quit",
        "R - Toggle recording",
        "S - Take snapshot",
        "F - Reset window size",
        "M - Toggle military mode",
        "ENTER - Toggle fullscreen",
        "ESC - Exit fullscreen",
        "TAB - Toggle this info",
        "BACKSPACE - Toggle replay mode",
        "1-6 - Change resolution",
        "+ / - - Adjust frame processing frequency",
        "Mouse wheel - Zoom",
        "Right click drag - Pan",
        "Left click drag - Save selection to elements folder",
    ]

    # Draw commands
    cmd_y = line_y + 40
    cmd_font_scale = 0.6
    cmd_thickness = 1
    line_height = 25

    for i, cmd in enumerate(commands):
        y_pos = cmd_y + (i * line_height)
        if y_pos > height - 100:  # Don't go too far down
            break
        cv2.putText(
            overlay,
            cmd,
            (50, y_pos),
            _font,
            cmd_font_scale,
            (255, 255, 255),
            cmd_thickness,
        )

    # Get system info
    gpu_name, gpu_vram = get_gpu_info()
    app_modified = get_app_modified_time()

    # Draw system info at bottom
    info_y = height - 60
    cv2.putText(
        overlay, f"GPU: {gpu_name}", (20, info_y), _font, 0.5, (0, 255, 255), 1
    )  # Yellow
    cv2.putText(
        overlay, f"VRAM: {gpu_vram}", (20, info_y + 20), _font, 0.5, (0, 0, 255), 1
    )  # Red
    cv2.putText(
        overlay, app_modified, (20, info_y + 40), _font, 0.4, (255, 255, 255), 1
    )

    return overlay


labels = open("db/coco.names").read().strip().split("\n")
classlist = [labels.index(x) for x in classlist]

object_count = 0
old_count = 0
obj_break = millis()
obj_idle = 0
obj_list = []
obj_max = 16
fskip = False
last_fskip = timestamp()
obj_score = labels

bounding_boxes = []
obj_number = 1

# YOLO processing state
yolo_frame_count = 0
cached_yolo_results = None
zoom_pan_active = False
zoom_pan_pause_time = 0
last_yolo_processing_duration = 0  # Store last YOLO processing time

# Person detection cache for clustering
cached_person_detections = []
last_person_update_frame = 0

# Replay system variables
replay_buffer = []
replay_mode = False
replay_index = 0
replay_last_flash_time = 0

# Window management variables
fullscreen = False
window_aspect_ratio = 4 / 3  # Default aspect ratio
original_window_size = (640, 480)

# Resolution display variables
resolution_display_active = False
resolution_display_text = ""
resolution_display_start_time = 0

# Selection and clipboard variables
selection_complete = False
clipboard_message_active = False
clipboard_message_start_time = 0
clipboard_message_duration = 2000  # Show message for 2 seconds
selection_start_x = 0
selection_start_y = 0
selection_end_x = 0
selection_end_y = 0
clean_processed_img = None  # Store clean image before UI overlays


def center(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    return (center_x, center_y)


def _size(x1, y1, x2, y2):
    return abs(x1 - y2)


def mouse_callback(event, x, y, flags, param):
    global drawing, draw_start_x, draw_start_y, draw_end_x, draw_end_y, dragging, drag_start_x, drag_start_y, zoom_factor, pan_x, pan_y, zoom_pan_active, zoom_pan_pause_time, cached_yolo_results, zoom_mode_active, stored_bounding_boxes, bounding_boxes, fullscreen, window, selection_complete, clipboard_message_active, clipboard_message_start_time, selection_start_x, selection_start_y, selection_end_x, selection_end_y
    if event == cv2.EVENT_RBUTTONDOWN:
        dragging = True
        zoom_pan_active = True
        zoom_pan_pause_time = millis()
        cached_yolo_results = None  # Clear cache when pan starts
        drag_start_x = x
        drag_start_y = y

    if event == cv2.EVENT_RBUTTONUP:
        dragging = False
        zoom_pan_active = False
        cached_yolo_results = None  # Clear cache when pan ends

    if event == cv2.EVENT_LBUTTONUP:
        if drawing and draw_start_x > 0 and draw_end_x > 0:
            # Store selection coordinates before reset
            selection_start_x = draw_start_x
            selection_start_y = draw_start_y
            selection_end_x = draw_end_x
            selection_end_y = draw_end_y
            # Mark selection as complete for clipboard copying
            selection_complete = True
        drawing = False
        draw_end_x = 0
        draw_end_y = 0
        draw_start_x = 0
        draw_start_y = 0

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start drawing
        drawing = True
        draw_end_x = 0
        draw_end_y = 0
        draw_start_x = x
        draw_start_y = y

    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_pan_active = True
        zoom_pan_pause_time = millis()
        cached_yolo_results = None  # Clear cache on zoom in/out

        # Store objects when entering zoom mode
        if zoom_factor == 1.0 and not zoom_mode_active:
            pan_x = 0
            pan_y = 0

        old_zoom_factor = zoom_factor

        if flags > 0:
            zoom_factor = min(6.0, zoom_factor * 1.1)
        else:
            zoom_factor = max(1.0, zoom_factor / 1.1)

        # Entering zoom mode - store current tracking state
        if old_zoom_factor == 1.0 and zoom_factor > 1.0 and not zoom_mode_active:
            stored_bounding_boxes = copy.deepcopy(bounding_boxes)
            zoom_mode_active = True
            print(f"Entering zoom mode - stored {len(stored_bounding_boxes)} objects")

        # Exiting zoom mode - restore tracking state only when returning to 1.0x
        elif zoom_factor == 1.0 and zoom_mode_active:
            zoom_mode_active = False
            bounding_boxes = copy.deepcopy(stored_bounding_boxes)
            stored_bounding_boxes = []
            print(f"Exiting zoom mode - restored {len(bounding_boxes)} objects")

    if event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            draw_end_x = x
            draw_end_y = y

        if dragging:
            dx = x - drag_start_x
            dy = y - drag_start_y

            pan_x -= int(dx * zoom_factor)
            pan_y -= int(dy * zoom_factor)

            drag_start_x = x
            drag_start_y = y


class BoundingBox:
    def __init__(self, name, points, size, image):
        global obj_number
        self.nr = obj_number
        obj_number += 1
        self.x, self.y = points
        self.created = millis()
        self.timestamp = self.created
        self.size = size
        self.name = name
        self.checkin = True
        self.detections = 0
        self.idle = 0
        self.image = image
        self.desc = False
        self.state = 0
        self.seen = self.created
        self.disappeared_cycles = 0  # Track consecutive disappearance cycles

        self.init()
        print(
            "New object: " + self.name + "#" + str(self.nr) + " size:" + str(self.size)
        )

    def see(self):
        self.seen = millis()

    def ping(self):
        self.timestamp = millis()
        idle = self.timestamp - self.created
        if idle >= 1000:
            self.idle = idle // 1000
        else:
            self.idle = 0
        return self.idle

    def export(self):
        _, buffer = cv2.imencode(".png", self.image)
        base64_image = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return base64_image

    def init(self):
        self.min_x = self.x - stationary_val
        self.max_x = self.x + stationary_val
        self.min_y = self.y - (stationary_val)
        self.max_y = self.y + (stationary_val)

    def contains(self, x, y, time):
        return (
            ((self.checkin == False) and self.min_x <= x <= self.max_x)
            and (self.min_y <= y <= self.max_y)
            and (time - self.seen < point_timeout)
        )

    def update(self, time, new_x, new_y):
        self.checkin = True
        self.timestamp = time
        idle = self.timestamp - self.created
        if idle >= 1000:
            self.idle = idle // 1000
        else:
            self.idle = 0
        self.x = new_x
        self.y = new_y
        self.detections += 1
        self.disappeared_cycles = (
            0  # Reset disappearance counter when object is detected
        )
        self.init()


def resetIteration():
    global bounding_boxes
    # Reset checkin status and increment disappeared_cycles for objects not seen
    for item in bounding_boxes:
        if item.checkin:
            item.disappeared_cycles = 0  # Reset counter if object was seen
        else:
            item.disappeared_cycles += 1  # Increment if object not detected
        item.checkin = False


def save_selection_to_elements(frame, start_x, start_y, end_x, end_y):
    """Save the selected rectangular region to elements folder as PNG"""
    global clipboard_message_active, clipboard_message_start_time

    try:
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(start_x, end_x))
        y1 = max(0, min(start_y, end_y))
        x2 = min(w, max(start_x, end_x))
        y2 = min(h, max(start_y, end_y))

        # Extract the selected region
        selected_region = frame[y1:y2, x1:x2]

        if selected_region.size > 0:
            # Create elements directory if it doesn't exist
            elements_dir = "elements"
            if not os.path.exists(elements_dir):
                os.makedirs(elements_dir)

            # Generate filename with current date-time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"element_{timestamp}.png"
            filepath = os.path.join(elements_dir, filename)

            # Save the selected region as PNG
            success = cv2.imwrite(filepath, selected_region)

            if success:
                # Show success message
                clipboard_message_active = True
                clipboard_message_start_time = millis()
                print(f"Element saved: {filepath} ({x2-x1}x{y2-y1})")
                return True
            else:
                print("Failed to save element")
                return False

    except Exception as e:
        print(f"Error saving element: {e}")
        return False

    return False


def take_snapshot(frame):
    global snapshot_directory

    if not os.path.exists(snapshot_directory):
        os.makedirs(snapshot_directory)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"snapshot_{timestamp}.jpg"
    filepath = os.path.join(snapshot_directory, filename)

    cv2.imwrite(filepath, frame)
    print(f"Snapshot saved: {filepath}")


def start_recording(cap):
    global recording, out
    if not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recordings/recording_{timestamp}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(filename, fourcc, 20, opsize)

        recording = True
        print(f"Started recording: {filename}")


def stop_recording():
    global recording, out
    if recording:
        out.release()
        recording = False
        print("Stopped recording")


def add(num):
    if len(obj_list) >= obj_max:
        obj_list.pop(0)
    obj_list.append(num)


def average():
    l = len(obj_list)
    if l <= 0:
        return 0
    return round(sum(obj_list) / l)


def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, dash_length=8):
    def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length):

        dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = [
                int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes),
            ]
            end = [
                int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
                int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes),
            ]
            cv2.line(img, tuple(start), tuple(end), color, thickness)

    draw_dashed_line(img, pt1, (pt2[0], pt1[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt2[0], pt1[1]), pt2, color, thickness, dash_length)
    draw_dashed_line(img, pt2, (pt1[0], pt2[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt1[0], pt2[1]), pt1, color, thickness, dash_length)
    return img


def generate_color_shades(num_classes):
    # This function will be called after numpy is imported
    colors = np.zeros((num_classes, 3), dtype=np.uint8)

    green = [0, 200, 0]
    orange = [0, 165, 255]
    yellow = [0, 200, 255]
    red = [0, 0, 255]

    base_colors = [green, orange, yellow, red]
    num_base_colors = len(base_colors)

    for i in range(num_classes):
        base_color_index = i % num_base_colors
        base_color = np.array(base_colors[base_color_index])
        shade_factor = (i // num_base_colors) / (num_classes // num_base_colors + 1)
        shade = (
            base_color * (1 - shade_factor) + np.array([128, 128, 128]) * shade_factor
        )
        colors[i] = shade.astype(np.uint8)
    return colors


# start showing the stream in this line

# Global variables for background loading
yolo_model_loaded = False
yolo_model = None
yolo_loading_complete = False
basic_stream_mode = True
colors = None
device = None

print(f"Starting up..")

# Load configuration
window_position = load_config()
print(f"First run status: {is_first_run}")

# Initialize stream immediately
print("DEBUG: Initializing video capture...")
loop = True
print(f"DEBUG: Creating VideoCapture with rtsp_stream = {rtsp_stream}")
cap = cv2.VideoCapture(rtsp_stream)

print(f"DEBUG: VideoCapture created, isOpened() = {cap.isOpened()}")

if rtsp_stream == 0:
    print("DEBUG: Webcam mode detected, starting resolution detection...")
    # Detect webcam maximum supported resolution
    max_res = detect_webcam_max_resolution(cap)

    print(f"DEBUG: Setting webcam to max resolution {max_res[0]}x{max_res[1]}...")
    # Set to maximum supported resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_res[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_res[1])

    print("DEBUG: Verifying final webcam resolution...")
    # Verify final setting
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at {actual_width}x{actual_height}")

    # Update opsize to match webcam capability if needed
    if max_res[0] < opsize[0] or max_res[1] < opsize[1]:
        print(f"Adjusting display size to webcam capability: {max_res[0]}x{max_res[1]}")
        opsize = max_res

print("DEBUG: Getting FPS property...")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"DEBUG: FPS = {fps}")

print("DEBUG: Attempting to read first frame...")
ret, img = cap.read()
print(
    f"DEBUG: First frame read result: ret={ret}, img shape={img.shape if img is not None else 'None'}"
)

print("DEBUG: Setting up window and getting stream size...")
window = str(rtsp_stream)
print("DEBUG: Getting stream width...")
stream_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("DEBUG: Getting stream height...")
stream_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
streamsize = (stream_width, stream_height)
print(f"DEBUG: Stream size = {streamsize}")

# Store original stream size for Ctrl+4 functionality
if streamsize[0] > 0 and streamsize[1] > 0:
    original_opsize = streamsize  # Store original stream dimensions
    window_aspect_ratio = streamsize[0] / streamsize[1]
    original_window_size = streamsize
else:
    original_opsize = opsize  # Fallback to default size
    window_aspect_ratio = opsize[0] / opsize[1]
    original_window_size = opsize

if streamsize[0] > opsize[0] or streamsize[1] > opsize[1]:
    hdstream = True
    print("HD stream mode enabled")
elif rtsp_stream == 0:
    hdstream = True
    print("Webcam mode - HD stream enabled for resampling")
else:
    print("Stream resolution matches to window size")

cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, opsize[0], opsize[1])
cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
cv2.setMouseCallback(window, mouse_callback)
q = queue.Queue(maxsize=buffer)


# Background loading function
def load_yolo_and_components():
    """Load YOLO model and other components in background"""
    global yolo_model_loaded, yolo_model, yolo_loading_complete, model, device, colors, bounding_boxes

    print(f"Loading heavy imports and YOLO model in background...")

    # Import heavy libraries in background
    import numpy as np
    import math
    import torch
    import base64
    import requests
    import json
    import copy
    from sklearn.cluster import DBSCAN
    from ultralytics import YOLO

    # Make imports available globally
    globals()["np"] = np
    globals()["math"] = math
    globals()["torch"] = torch
    globals()["base64"] = base64
    globals()["requests"] = requests
    globals()["json"] = json
    globals()["copy"] = copy
    globals()["DBSCAN"] = DBSCAN
    globals()["YOLO"] = YOLO

    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    print(torch.version.cuda)

    colors = generate_color_shades(len(labels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing model..")
    yolo_model = YOLO("models/" + model + ".pt")
    print(f"Loading model to {device}")
    yolo_model.to(device)

    print("loading objects")
    bounding_boxes = []

    # Update global model reference
    model = yolo_model
    yolo_model_loaded = True
    yolo_loading_complete = True

    # Check if first run and show help immediately after YOLO loads
    global is_first_run, show_info_overlay
    if is_first_run:
        show_info_overlay = True
        print("First run detected - help overlay will be shown")

    print("YOLO model loaded successfully! Switching to full processing mode.")


# Start background loading thread
yolo_thread = threading.Thread(target=load_yolo_and_components)
yolo_thread.daemon = True
yolo_thread.start()


def get_clusters(detected_points, eps=30, min_samples=2):
    if not isinstance(detected_points, np.ndarray) or detected_points.size == 0:
        return {}

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(detected_points)
    labels = db.labels_

    clusters = {}
    for label in set(labels):
        if label == -1:
            continue
        cluster_points = detected_points[labels == label]

        if cluster_points.size == 0:
            continue

        min_x, min_y = np.min(cluster_points, axis=0)
        max_x, max_y = np.max(cluster_points, axis=0)
        count = len(cluster_points)
        clusters[label] = (min_x, min_y, max_x, max_y, count)

    return clusters


def is_point_in_cluster(x, y, clusters):
    """Checks if (x, y) is inside any cluster bounding box."""
    for cluster_id, (min_x, min_y, max_x, max_y, count) in clusters.items():
        if min_x <= x <= max_x and min_y <= y <= max_y:
            return True
    return False


def add_to_replay_buffer(frame):
    """Add frame to circular replay buffer, maintaining max size"""
    global replay_buffer, replay_buffer_max_size

    if len(replay_buffer) >= replay_buffer_max_size:
        replay_buffer.pop(0)  # Remove oldest frame

    replay_buffer.append(frame.copy())


def stream():
    global cap, obj_idle, last_fskip
    if cap.isOpened():
        ret, frame = cap.read()
        while loop:
            ret, frame = cap.read()
            if ret:
                # Add frame to replay buffer when not in replay mode
                if not replay_mode:
                    add_to_replay_buffer(frame)

                if (
                    (obj_idle > 0)
                    and obj_idle >= idle_reset
                    and (timestamp() - last_fskip >= 30)
                ):
                    last_fskip = timestamp()
                    q.queue.clear()
                    obj_idle = 0
                    # print(f"Frame skip")
                else:
                    q.put(frame)
            else:
                print("Can't receive frame, restarting video...")
                cap.release()
                cap = cv2.VideoCapture(rtsp_stream)


def find_closest_point(points, point):
    closest_point = None
    min_distance = float("inf")

    for x, y in points:
        distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x, y)

    return closest_point, min_distance


def getObject(point, cname):
    global bounding_boxes
    x, y = point
    time = millis()

    for i, box in enumerate(bounding_boxes):
        if cname != box.name:
            continue

        if box.contains(x, y, time):
            box.update(time, x, y)
            bounding_boxes[i].checkin = True
            return bounding_boxes[i]

        # Remove objects that have been missing for 3+ consecutive cycles
        if box.disappeared_cycles >= 3:
            del bounding_boxes[i]
    return False


def is_point_inside(x, y, list):
    for item in list:
        if item[0] <= x <= item[2] and item[1] <= y <= item[3]:
            return True
    return False


def process_basic_stream(photo):
    """Basic stream processing without YOLO - just display the video"""
    # Simple resize if needed, avoiding resample which might use numpy
    if hdstream == True:
        img = cv2.resize(photo, opsize, interpolation=cv2.INTER_LINEAR)
    else:
        img = photo.copy()

    # Apply military mode processing
    if military_mode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Add loading indicator
    if not yolo_loading_complete:
        cv2.putText(img, "Loading YOLO model...", (16, 30), _font, 0.5, (0, 0, 0), 2)
        cv2.putText(
            img, "Loading YOLO model...", (16, 30), _font, 0.5, (0, 255, 255), 1
        )
    else:
        cv2.putText(
            img,
            "YOLO model ready! Switching automatically...",
            (16, 30),
            _font,
            0.5,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            img,
            "YOLO model ready! Switching automatically...",
            (16, 30),
            _font,
            0.5,
            (0, 255, 0),
            1,
        )

    return img


def process(photo):
    if hdstream == True:
        img = resample(photo)
    else:
        img = photo.copy()

    # Apply military mode processing
    if military_mode:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    global obj_score, bounding_boxes, yolo_frame_count, cached_yolo_results, zoom_pan_active, zoom_pan_pause_time, dragging, zoom_mode_active, cached_person_detections, last_person_update_frame

    # Check if zoom/pan operation finished recently (wait 500ms after last activity)
    current_time = millis()
    if zoom_pan_active and not dragging and (current_time - zoom_pan_pause_time > 500):
        zoom_pan_active = False
        cached_yolo_results = None  # Clear cache when zoom/pan timeout ends
        cached_person_detections = []  # Clear person cache when zoom/pan ends

    # Skip YOLO inference during zoom/pan operations or when zoomed in
    if zoom_pan_active or dragging or zoom_factor > 1.0:
        # Use last cached results during zoom/pan or when zoomed
        results = cached_yolo_results
    else:
        # Run YOLO inference based on skip setting (0 = all frames, >0 = every nth frame)
        yolo_processing_happened = (
            yolo_skip_frames == 0 or yolo_frame_count % (yolo_skip_frames + 1) == 0
        )
        if yolo_processing_happened:
            # Trigger help text on first YOLO processing
            global yolo_first_processing_started, show_help_text, help_text_start_time, is_first_run, show_info_overlay
            if not yolo_first_processing_started:
                yolo_first_processing_started = True

                if is_first_run:
                    # First run - show help screen immediately
                    show_info_overlay = True
                    print("First run detected - showing help overlay")
                else:
                    # Regular run - show help text 2 seconds after first YOLO processing
                    import threading

                    def delayed_help_text():
                        import time

                        time.sleep(2)
                        global show_help_text, help_text_start_time
                        show_help_text = True
                        help_text_start_time = millis()

                    help_thread = threading.Thread(target=delayed_help_text)
                    help_thread.daemon = True
                    help_thread.start()

            # Start timing YOLO processing
            global last_yolo_processing_duration
            yolo_start = time.perf_counter_ns()

            # Create square input for YOLO (required to be divisible by 32)
            yolo_img = cv2.resize(
                img, (yolo_input_size, yolo_input_size), interpolation=cv2.INTER_LINEAR
            )

            img_tensor = (
                torch.from_numpy(cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB))
                .to(device)
                .float()
                / 255.0
            )
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                results = model(
                    img_tensor,
                    verbose=False,
                    iou=0.45,
                    half=False,
                    max_det=32,
                    conf=min_confidence,
                    classes=classlist,
                )

            # Store YOLO processing duration
            last_yolo_processing_duration = time.perf_counter_ns() - yolo_start

            # Cache the results for next 2 frames
            cached_yolo_results = results
        else:
            # Use cached results from previous YOLO inference
            results = cached_yolo_results

        yolo_frame_count += 1

    obj_score = [0 for _ in range(len(obj_score))]
    c = 0
    points = []
    boxes = [box for r in results for box in r.boxes] if results else []
    now = millis()

    # Only reset iteration when not in zoom mode
    if not zoom_mode_active:
        resetIteration()

    rawcrowd = []

    # Cache person detections after YOLO processing for clustering continuity
    if "yolo_processing_happened" in locals() and yolo_processing_happened:
        # Will update cached_person_detections after processing boxes
        pass

    for i, box in enumerate(boxes):
        class_id = int(box.cls)
        class_name = labels[class_id]
        confidence = float(box.conf)
        c = c + 1
        if (class_name in class_confidence) and (
            confidence <= class_confidence[class_name]
        ):
            continue

        if confidence <= min_confidence:
            continue

        xmin, ymin, xmax, ymax = box.xyxy[0]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

        # Transform coordinates from YOLO square input back to display dimensions
        scale_x = opsize[0] / yolo_input_size
        scale_y = opsize[1] / yolo_input_size
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)
        width = xmax - xmin
        height = ymax - ymin

        if (
            xmin == 0
            or ymin == 0
            or xmax == 0
            or ymax == 0
            or xmax == img.shape[1]
            or ymax == img.shape[0]
        ):
            continue

        if class_name == "car" and (
            (width > height and (width / height) >= 2)
            or (width < min_size or height < min_size)
        ):
            continue

        if zoom_factor > 1.0:
            color = colors[class_id].tolist()
            alpha = 0.35
            color_with_alpha = color + [alpha]

            text = f"{class_name}" + " " + str(round(confidence, 6))
            text_offset_x = xmin
            text_offset_y = ymin - 5

            overlay = img[ymin : ymax + 1, xmin : xmax + 1].copy()
            cv2.rectangle(
                overlay,
                (0, 0),
                (xmax - xmin, ymax - ymin),
                color_with_alpha,
                thickness=-1,
            )
            cv2.addWeighted(
                overlay,
                alpha,
                img[ymin : ymax + 1, xmin : xmax + 1],
                1 - alpha,
                0,
                img[ymin : ymax + 1, xmin : xmax + 1],
            )
            draw_dashed_rectangle(img, (xmin, ymin), (xmax, ymax), color, 1, 8)
            continue

        idx = class_id
        obj_score[idx] = obj_score[idx] + 1

        point = center(xmin, ymin, xmax, ymax)
        if class_name == "person":
            rawcrowd.append(point)

        size = _size(xmin, ymin, xmax, ymax)
        closest, distance = find_closest_point(points, point)
        obj = getObject(point, class_name)
        if obj != False:
            if distance < 6.0:
                continue

            points.append(point)
            obj.see()
            if obj.desc != False:
                sid = obj.desc
            else:
                sid = obj.name + "#" + str(obj.nr)

            color = colors[class_id].tolist()

            if military_mode:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

                # Add entity label with white text on crimson background
                label_text = class_name.upper()
                font_scale = 0.5
                font_thickness = 1

                # Get text dimensions
                text_size = cv2.getTextSize(
                    label_text, _font, font_scale, font_thickness
                )[0]
                text_width = text_size[0]
                text_height = text_size[1]

                # Position label at top-left of bounding box
                label_x = xmin
                label_y = ymin - 5

                # Ensure label doesn't go off-screen
                if label_y - text_height < 0:
                    label_y = ymin + text_height + 5

                # Draw crimson background rectangle
                cv2.rectangle(
                    img,
                    (label_x, label_y - text_height - 2),
                    (label_x + text_width + 4, label_y + 2),
                    (60, 20, 220),  # Crimson in BGR
                    -1,
                )

                # Draw white text
                cv2.putText(
                    img,
                    label_text,
                    (label_x + 2, label_y),
                    _font,
                    font_scale,
                    (255, 255, 255),  # White
                    font_thickness,
                )
            else:
                cv2.circle(img, point, 1, (0, 0, 255), 2)

            idle = format_duration(obj.idle)

            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (200, 200, 200), 1)

        else:
            text = f"{class_name}" + " " + str(round(confidence, 6))
            text_offset_x = xmin
            text_offset_y = ymin - 5

            if military_mode:
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)

                # Add entity label with white text on crimson background
                label_text = class_name.upper()
                font_scale = 0.5
                font_thickness = 1

                # Get text dimensions
                text_size = cv2.getTextSize(
                    label_text, _font, font_scale, font_thickness
                )[0]
                text_width = text_size[0]
                text_height = text_size[1]

                # Position label at top-left of bounding box
                label_x = xmin
                label_y = ymin - 5

                # Ensure label doesn't go off-screen
                if label_y - text_height < 0:
                    label_y = ymin + text_height + 5

                # Draw crimson background rectangle
                cv2.rectangle(
                    img,
                    (label_x, label_y - text_height - 2),
                    (label_x + text_width + 4, label_y + 2),
                    (60, 20, 220),  # Crimson in BGR
                    -1,
                )

                # Draw white text
                cv2.putText(
                    img,
                    label_text,
                    (label_x + 2, label_y),
                    _font,
                    font_scale,
                    (255, 255, 255),  # White
                    font_thickness,
                )
            else:
                cv2.circle(img, point, 1, (255, 255, 0), 2)
            cv2.putText(
                img,
                text,
                (text_offset_x, text_offset_y),
                _font,
                0.35,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                img,
                text,
                (text_offset_x, text_offset_y),
                _font,
                0.35,
                (255, 255, 255),
                1,
            )
            qxmin, qymin, qxmax, qymax = transform(xmin, ymin, xmax, ymax, padding)
            snap = photo[qymin:qymax, qxmin:qxmax]
            item = BoundingBox(class_name, point, size, snap)
            bounding_boxes.append(item)

    # Update cached person detections if YOLO processing happened
    if "yolo_processing_happened" in locals() and yolo_processing_happened:
        cached_person_detections = rawcrowd.copy()
        last_person_update_frame = yolo_frame_count

    if zoom_factor > 1.0:
        add(c)
        return img

    # Skip object tracking operations when in zoom mode
    if zoom_mode_active:
        add(c)
        return img

    # Use cached person detections if YOLO didn't run this frame
    if (
        "yolo_processing_happened" in locals()
        and not yolo_processing_happened
        and cached_person_detections
    ):
        rawcrowd = cached_person_detections.copy()

    cenable = False
    crowds = get_clusters(np.array(rawcrowd), 50, 2)
    rect = []
    if len(crowds) > 0:
        cenable = True
        for crowd in crowds:
            min_x, min_y, max_x, max_y, count = crowds[crowd]
            rect.append((min_x, min_y, max_x, max_y, count))
            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 255, 0), 1)
            cid = str(count) + " people"
            cv2.putText(img, cid, (min_x, min_y - 18), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, cid, (min_x, min_y - 18), _font, 0.35, (255, 255, 255), 1)

    for obj in bounding_boxes:

        if (
            obj.checkin == False
            and obj.detections >= 3
            and obj.idle > 1
            and obj.idle < 8
        ):
            _, distance = find_closest_point(points, (obj.x, obj.y))
            if distance < 6.0:
                continue

            obj.ping()
            if now - obj.seen > point_timeout:
                continue

            if cenable == True and is_point_inside(obj.x, obj.y, rect) == True:
                continue

            if obj.desc != False:
                sid = obj.desc
            else:
                sid = obj.name + "#" + str(obj.nr)

            idle = format_duration(obj.idle)

            cv2.circle(img, (obj.x, obj.y), 1, (0, 255, 255), 2)

    add(c)
    return img


sthread = threading.Thread(target=stream)
print(f"Starting..")

sthread.start()

while loop:
    img = None
    original_img = None  # Store original frame for clipboard copying

    # Handle replay mode
    if replay_mode and len(replay_buffer) > 0:
        if replay_index < len(replay_buffer):
            img = replay_buffer[replay_index]
            original_img = img.copy()  # Store original for clipboard
            replay_index += 1
        else:
            # Replay finished, exit replay mode
            replay_mode = False
            replay_index = 0
            print("Replay completed")
            continue
    elif (q.empty() != True) and (fskip != True):
        img = q.get_nowait()
        original_img = img.copy()  # Store original for clipboard

    if img is not None:

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("frame skip")
            q.queue.clear()

        if key == ord("q"):
            print("Saving configuration...")
            save_config()
            loop = False
        elif key == 8:
            if not replay_mode and len(replay_buffer) > 0:
                replay_mode = True
                replay_index = 0
                q.queue.clear()
                print(f"Starting replay of {len(replay_buffer)} frames")
            elif replay_mode:
                replay_mode = False
                q.queue.clear()
                print("Exiting replay mode")
        elif key == ord("r"):
            if not recording:
                start_recording(cap)
                q.queue.clear()
            else:
                stop_recording()
                q.queue.clear()

        elif key == ord("s"):
            take_snapshot(img)
            q.queue.clear()
        elif key == ord("f"):
            reset_window_to_stream_resolution()
            q.queue.clear()
        elif key == ord("m"):
            military_mode = not military_mode
            print(f"Military mode: {'ON' if military_mode else 'OFF'}")
            yolo_skip_frames = 1 if military_mode else 2
            q.queue.clear()

        # Handle Ctrl+number combinations for resizing
        elif key == ord("1") or key == 49:  # Ctrl+1: 640x480
            resize_stream_dimensions((640, 480))
            q.queue.clear()
        elif key == ord("2") or key == 50:  # Ctrl+2: 800x600
            resize_stream_dimensions((800, 600))
            q.queue.clear()
        elif key == ord("3") or key == 51:  # Ctrl+3: 1024x768
            resize_stream_dimensions((1024, 768))
            q.queue.clear()
        elif key == ord("4") or key == 52:  # Ctrl+4: 1280x800
            resize_stream_dimensions((1280, 800))
            q.queue.clear()
        elif key == ord("5") or key == 53:  # Ctrl+5: 1920x1080
            resize_stream_dimensions((1920, 1080))
            q.queue.clear()
        elif key == ord("6") or key == 54:  # Ctrl+6: original stream size
            if original_opsize:
                resize_stream_dimensions(original_opsize)
                q.queue.clear()
        elif key == 13:  # Enter key - toggle fullscreen
            toggle_fullscreen()
            q.queue.clear()
        elif key == 27:  # Esc key - exit fullscreen only
            if fullscreen:
                toggle_fullscreen()  # Exit fullscreen
            q.queue.clear()
        elif key == ord("+") or key == ord(
            "="
        ):  # + key (decrease frame skip - more processing)
            yolo_skip_frames = max(0, yolo_skip_frames - 1)
            show_frame_skip_display = True
            frame_skip_display_start_time = millis()
            print(
                f"Frame processing: {yolo_skip_frames + 1 if yolo_skip_frames > 0 else 'ALL'} frames"
            )
            q.queue.clear()
        elif key == ord("-"):  # - key (increase frame skip - less processing)
            yolo_skip_frames = min(10, yolo_skip_frames + 1)
            show_frame_skip_display = True
            frame_skip_display_start_time = millis()
            print(f"Frame processing: Every {yolo_skip_frames + 1} frames")
            q.queue.clear()

        elif key == 9:  # Tab key - toggle info overlay
            # On first run, any tab press marks first run as complete
            if is_first_run:
                is_first_run = False
                print("First run completed - normal tab behavior enabled")

            show_info_overlay = not show_info_overlay
            print(f"Info overlay: {'ON' if show_info_overlay else 'OFF'}")

        if key < 255 and key > 0:
            print(f"Key pressed: {key}")

        # Skip YOLO processing during replay mode
        if replay_mode:
            # Simple display processing for replay frames
            if hdstream == True:
                img = cv2.resize(img, opsize, interpolation=cv2.INTER_LINEAR)

            # Apply military mode processing if enabled
            if military_mode:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            # Use basic stream processing until YOLO is loaded, then auto-switch
            if basic_stream_mode:
                img = process_basic_stream(img)
                # Automatically switch to YOLO processing when model is loaded
                if yolo_loading_complete:
                    basic_stream_mode = False
                    print("Switching to YOLO processing mode!")
            else:
                img = process(img)  # YOLO timing is handled inside this function

        # Store clean processed image before UI overlays for selection saving
        # This ensures we capture the image with zoom/pan applied but without UI elements
        clean_processed_img = img.copy()

        # Skip object count and FPS calculations during replay mode
        if not replay_mode:
            object_count = average()
            if object_count != old_count:
                obj_break = millis()
                obj_idle = 0
            else:
                obj_idle = millis() - obj_break

            old_count = object_count
            frames += 1
            if millis() - last_frame >= 1000:
                fps = (frames - prev_frames) * 1
                prev_frames = frames
                last_frame = millis()

        if replay_mode:
            _fps = "REPLAY MODE - LAG: 0ms"
        else:
            _fps = (
                "FPS: "
                + str(fps)
                + " - LAG: "
                + str(last_yolo_processing_duration // 1000000)
                + "ms"
            )

        text_y = img.shape[0] - 5
        cv2.putText(img, _fps, (16, text_y), _font, 0.4, (0, 0, 0), 2)
        if replay_mode:
            cv2.putText(
                img, _fps, (16, text_y), _font, 0.4, (255, 128, 0), 1
            )  # Orange for replay
        else:
            cv2.putText(img, _fps, (16, text_y), _font, 0.4, (0, 255, 0), 1)

        if recording:
            out.write(img)
            cv2.putText(img, "REC", (16, img.shape[0] - 38), _font, 0.5, (0, 0, 0), 2)
            cv2.putText(img, "REC", (16, img.shape[0] - 38), _font, 0.5, (0, 0, 255), 1)

        bb = str(len(bounding_boxes))

        cv2.putText(
            img, "Tracking: " + bb, (16, img.shape[0] - 26), _font, 0.4, (0, 0, 0), 2
        )
        cv2.putText(
            img,
            "Tracking: " + bb,
            (16, img.shape[0] - 26),
            _font,
            0.4,
            (64, 255, 255),
            1,
        )

        # Show flashing REPLAY text instead of clock during replay mode
        if replay_mode:
            # Flash every 500ms
            current_time = millis()
            if current_time - replay_last_flash_time > 500:
                replay_last_flash_time = current_time

            show_replay_text = (current_time - replay_last_flash_time) < 250

            if show_replay_text:
                replay_text = "REPLAY"
                text_size = cv2.getTextSize(replay_text, _font, 0.6, 2)[0]
                text_x = img.shape[1] - text_size[0] - 10
                text_y = img.shape[0] - 8
                cv2.putText(
                    img, replay_text, (text_x, text_y), _font, 0.6, (0, 0, 0), 3
                )
                cv2.putText(
                    img, replay_text, (text_x, text_y), _font, 0.6, (0, 0, 255), 2
                )
        else:
            clock = datetime.now().strftime("%H:%M:%S")
            text_size = cv2.getTextSize(clock, _font, 0.5, 1)[0]
            text_x = img.shape[1] - text_size[0] - 10
            text_y = img.shape[0] - 8
            cv2.putText(img, clock, (text_x, text_y), _font, 0.4, (0, 0, 0), 2)
            cv2.putText(img, clock, (text_x, text_y), _font, 0.4, (255, 255, 255), 1)

        # Handle selection completion for element saving
        if selection_complete and clean_processed_img is not None:
            if selection_start_x > 0 and selection_end_x > 0:
                # Use the clean processed image which has zoom/pan applied but no UI overlays
                # This ensures we save exactly what's visible under the rectangle without UI elements
                save_selection_to_elements(
                    clean_processed_img,
                    selection_start_x,
                    selection_start_y,
                    selection_end_x,
                    selection_end_y,
                )

            selection_complete = False

        if drawing and draw_start_x > 0 and draw_end_x > 0:
            # Create semi-transparent green rectangle overlay
            overlay = img.copy()
            cv2.rectangle(
                overlay,
                (draw_start_x, draw_start_y),
                (draw_end_x, draw_end_y),
                (0, 255, 0),
                thickness=-1,  # Fill the rectangle
            )
            # Apply transparency
            alpha = 0.3  # 30% transparency
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

            # Draw thick border
            cv2.rectangle(
                img,
                (draw_start_x, draw_start_y),
                (draw_end_x, draw_end_y),
                (0, 255, 0),
                thickness=2,
            )

        # Display resolution temporarily when changed
        if resolution_display_active:
            current_time = millis()
            if (
                current_time - resolution_display_start_time
                < resolution_display_duration
            ):
                # Draw resolution text in big yellow font at top left
                font_scale = 0.85
                thickness = 2
                color = (0, 200, 200)  # Yellow in BGR
                position = (16, 32)  # Top left corner with some padding

                # Draw black outline for better visibility
                cv2.putText(
                    img,
                    resolution_display_text,
                    position,
                    _font,
                    font_scale,
                    (16, 16, 16),
                    thickness + 3,
                )
                # Draw yellow text
                cv2.putText(
                    img,
                    resolution_display_text,
                    position,
                    _font,
                    font_scale,
                    color,
                    thickness,
                )
            else:
                # Timer expired, disable display
                resolution_display_active = False

        # Show help text for 2 seconds after YOLO processing starts
        if show_help_text:
            current_time = millis()
            if current_time - help_text_start_time < 2000:  # Show for 2 seconds
                help_text = "press tab for help"
                cv2.putText(
                    img, help_text, (16, 50), _font, 0.4, (0, 255, 255), 1
                )  # Small yellow text, no outline
            else:
                show_help_text = False

        # Show frame skip display for 2 seconds when changed
        if show_frame_skip_display:
            current_time = millis()
            if (
                current_time - frame_skip_display_start_time < 2000
            ):  # Show for 2 seconds
                if yolo_skip_frames == 0:
                    skip_text = "Processing: ALL FRAMES"
                else:
                    skip_text = f"Processing: Every {yolo_skip_frames + 1} frames"

                # Position at top center
                text_size = cv2.getTextSize(skip_text, _font, 0.7, 2)[0]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = 40

                cv2.putText(
                    img, skip_text, (text_x, text_y), _font, 0.7, (0, 0, 0), 4
                )  # Black outline
                cv2.putText(
                    img, skip_text, (text_x, text_y), _font, 0.7, (0, 255, 255), 2
                )  # Yellow text
            else:
                show_frame_skip_display = False

        # Show element save success message at bottom center
        if clipboard_message_active:
            current_time = millis()
            if current_time - clipboard_message_start_time < clipboard_message_duration:
                message_text = "Element saved to elements folder"
                font_scale = 0.6
                thickness = 2

                # Get text dimensions for centering
                text_size = cv2.getTextSize(message_text, _font, font_scale, thickness)[
                    0
                ]
                text_x = (img.shape[1] - text_size[0]) // 2
                text_y = img.shape[0] - 15  # Position at bottom with some margin

                # Draw black outline for better visibility
                cv2.putText(
                    img,
                    message_text,
                    (text_x, text_y),
                    _font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                )
                # Draw green text
                cv2.putText(
                    img,
                    message_text,
                    (text_x, text_y),
                    _font,
                    font_scale,
                    (0, 255, 0),
                    thickness,
                )
            else:
                # Timer expired, disable message
                clipboard_message_active = False

        # Apply info overlay if enabled
        if show_info_overlay:
            img = draw_info_overlay(img)

        cv2.imshow(str(rtsp_stream), img)
    else:
        fskip = False
        time.sleep(0.01)

print("closing cv window..")
cv2.destroyAllWindows()
print("terminating..")
loop = False
sys.exit(0)
