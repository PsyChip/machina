print(f"Initializing..")
# Essential imports for basic video streaming
import time
import os
import cv2
import queue
import threading
import sys
from datetime import datetime

# Heavy imports will be loaded in background

model = "yolo11n"
rtsp_stream = (
    "rtsp://psychip:neuromancer1@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
)

_font = cv2.FONT_HERSHEY_SIMPLEX
point_timeout = 2500
stationary_val = 16

buffer = 512
idle_reset = 3000
min_confidence = 0.15
min_size = 20
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

snapshot_directory = "snapshots"

frames = 0
prev_frames = 0
last_frame = 0
fps = 0
recording = False
out = None

opsize = (640, 480)
streamsize = (0, 0)

zoom_factor = 1.0
pan_x = 0
pan_y = 0
zoom_mode_active = False
stored_bounding_boxes = []

hdstream = False
drawing = False
dragging = False
drag_start_x = 0
drag_start_y = 0
military_mode = False

draw_start_x = 0
draw_start_y = 0
draw_end_x = 0
draw_end_y = 0

padding = 6


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


def timestamp():
    return int(time.time())


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

yolo_frame_count = 0
yolo_skip_frames = 3
cached_yolo_results = None
zoom_pan_active = False
zoom_pan_pause_time = 0

# Replay system variables
replay_buffer = []
replay_buffer_max_size = 300  # ~10 seconds at 30fps
replay_mode = False
replay_index = 0
replay_last_flash_time = 0

# Window management variables
fullscreen = False
window_aspect_ratio = 4 / 3  # Default aspect ratio
original_window_size = (640, 480)
last_click_time = 0
double_click_threshold = 300  # milliseconds


def center(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    return (center_x, center_y)


def _size(x1, y1, x2, y2):
    return abs(x1 - y2)


def mouse_callback(event, x, y, flags, param):
    global drawing, draw_start_x, draw_start_y, draw_end_x, draw_end_y, dragging, drag_start_x, drag_start_y, zoom_factor, pan_x, pan_y, zoom_pan_active, zoom_pan_pause_time, cached_yolo_results, zoom_mode_active, stored_bounding_boxes, bounding_boxes, fullscreen, last_click_time, double_click_threshold, window
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
        drawing = False
        draw_end_x = 0
        draw_end_y = 0
        draw_start_x = 0
        draw_start_y = 0

    if event == cv2.EVENT_LBUTTONDOWN:
        # Check for double-click
        current_time = millis()
        if current_time - last_click_time < double_click_threshold:
            # Double-click detected - toggle fullscreen
            toggle_fullscreen()
        else:
            # Single click - start drawing
            drawing = True
            draw_end_x = 0
            draw_end_y = 0
            draw_start_x = x
            draw_start_y = y
        last_click_time = current_time

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
        out = cv2.VideoWriter(filename, fourcc, 20, (640, 480))

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

# Initialize stream immediately
loop = True
cap = cv2.VideoCapture(rtsp_stream)

if rtsp_stream == 0:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fps = cap.get(cv2.CAP_PROP_FPS)
ret, img = cap.read()

window = str(rtsp_stream)
streamsize = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

# Calculate aspect ratio from stream
if streamsize[0] > 0 and streamsize[1] > 0:
    window_aspect_ratio = streamsize[0] / streamsize[1]
    original_window_size = streamsize
else:
    window_aspect_ratio = opsize[0] / opsize[1]
    original_window_size = opsize

if streamsize[0] > opsize[0] or streamsize[1] > opsize[1]:
    hdstream = True
    print("HD stream mode enabled")
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
                    print(f"Frame skip")
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

    global obj_score, bounding_boxes, yolo_frame_count, cached_yolo_results, zoom_pan_active, zoom_pan_pause_time, dragging, zoom_mode_active

    # Check if zoom/pan operation finished recently (wait 500ms after last activity)
    current_time = millis()
    if zoom_pan_active and not dragging and (current_time - zoom_pan_pause_time > 500):
        zoom_pan_active = False
        cached_yolo_results = None  # Clear cache when zoom/pan timeout ends

    # Skip YOLO inference during zoom/pan operations
    if zoom_pan_active or dragging:
        # Use last cached results during zoom/pan
        results = cached_yolo_results
    else:
        # Run YOLO inference only every 3rd frame when not zooming/panning
        if yolo_frame_count % yolo_skip_frames == 0:
            img_tensor = (
                torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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

            if military_mode and (class_name == "person" or class_name == "car"):
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            else:
                cv2.circle(img, point, 1, (0, 0, 255), 2)

            idle = format_duration(obj.idle)

            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (200, 200, 200), 1)

        else:
            text = f"{class_name}" + " " + str(round(confidence, 6))
            text_offset_x = xmin
            text_offset_y = ymin - 5

            if military_mode and (class_name == "person" or class_name == "car"):
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
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

    if zoom_factor > 1.0:
        add(c)
        return img

    # Skip object tracking operations when in zoom mode
    if zoom_mode_active:
        add(c)
        return img

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

    # Handle replay mode
    if replay_mode and len(replay_buffer) > 0:
        if replay_index < len(replay_buffer):
            img = replay_buffer[replay_index]
            replay_index += 1
        else:
            # Replay finished, exit replay mode
            replay_mode = False
            replay_index = 0
            print("Replay completed")
            continue
    elif (q.empty() != True) and (fskip != True):
        img = q.get_nowait()

    if img is not None:

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("frame skip")
            q.queue.clear()

        if key == ord("q"):
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
            yolo_skip_frames = 1 if military_mode else 3
            q.queue.clear()
        if key < 255 and key > 0:
            print(f"Key pressed: {key}")

        start = time.perf_counter_ns()

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
                img = process(img)

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

        duration = time.perf_counter_ns() - start

        if replay_mode:
            _fps = "REPLAY MODE - LAG: " + str(duration // 1000000) + "ms"
        else:
            _fps = "FPS: " + str(fps) + " - LAG: " + str(duration // 1000000) + "ms"

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

        if drawing and draw_start_x > 0 and draw_end_x > 0:
            cv2.rectangle(
                img,
                (draw_start_x, draw_start_y),
                (draw_end_x, draw_end_y),
                (0, 255, 0),
                thickness=2,
            )

        cv2.imshow(str(rtsp_stream), img)
    else:
        fskip = False
        time.sleep(0.01)

print("closing cv window..")
cv2.destroyAllWindows()
print("terminating..")
loop = False
sys.exit(0)
