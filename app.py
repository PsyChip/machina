print(f"Initializing..")
import time
import os
import cv2
import queue
import threading
import numpy as np
import math
import torch
import base64
import requests
import json
import sys
from sklearn.cluster import DBSCAN
from datetime import datetime
from ultralytics import YOLO

model = "yolo11n"
rtsp_stream = (
    "rtsp://<your stream here>"
)

ollama = "http://127.0.0.1:11434/api/generate"
ollama_model = "llava:latest"

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

prompts = {
    "person": "get gender and age of this person in 5 words or less",
    "car": "get body type and color of this car in 5 words or less",
}

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

hdstream = False
drawing = False
dragging = False
drag_start_x = 0
drag_start_y = 0

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

# YOLO frame skipping optimization
yolo_frame_count = 0
yolo_skip_frames = 6
cached_yolo_results = None
zoom_pan_active = False
zoom_pan_pause_time = 0

def genprompt(t):
    if t in prompts:
        return prompts[t]
    return "describe this image in 5 words or less"

def center(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    return (center_x, center_y)

def _size(x1, y1, x2, y2):
    return abs(x1 - y2)


def mouse_callback(event, x, y, flags, param):
    global drawing, draw_start_x, draw_start_y, draw_end_x, draw_end_y, dragging, drag_start_x, drag_start_y, zoom_factor, pan_x, pan_y, zoom_pan_active, zoom_pan_pause_time, cached_yolo_results
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
        drawing = True
        draw_end_x = 0
        draw_end_y = 0
        draw_start_x = x
        draw_start_y = y

    if event == cv2.EVENT_MOUSEWHEEL:
        zoom_pan_active = True
        zoom_pan_pause_time = millis()
        cached_yolo_results = None  # Clear cache on zoom in/out

        if zoom_factor == 1.0:
            pan_x = 0
            pan_y = 0

        if flags > 0:
            zoom_factor = min(6.0, zoom_factor * 1.1)
        else:
            zoom_factor = max(1.0, zoom_factor / 1.1)

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
        self.init()


def resetIteration():
    global bounding_boxes
    [setattr(item, "checkin", False) for item in bounding_boxes]


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
        dist = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
        dashes = int(dist / dash_length)
        for i in range(dashes):
            start = np.array(
                [
                    int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                    int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes),
                ]
            )
            end = np.array(
                [
                    int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
                    int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes),
                ]
            )
            cv2.line(img, tuple(start), tuple(end), color, thickness)

    draw_dashed_line(img, pt1, (pt2[0], pt1[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt2[0], pt1[1]), pt2, color, thickness, dash_length)
    draw_dashed_line(img, pt2, (pt1[0], pt2[1]), color, thickness, dash_length)
    draw_dashed_line(img, (pt1[0], pt2[1]), pt1, color, thickness, dash_length)
    return img


def generate_color_shades(num_classes):
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


print(f"Starting up..")
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.version.cuda)

colors = generate_color_shades(len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing model..")
model = YOLO("models/" + model + ".pt")
print(f"Loading model to {device}")
model.to(device)

print("loading objects")
bounding_boxes = []

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


def stream():
    global cap, obj_idle, last_fskip
    if cap.isOpened():
        ret, frame = cap.read()
        while loop:
            ret, frame = cap.read()
            if ret:
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

        if (time - box.seen) >= point_timeout:
            del bounding_boxes[i]
    return False


def is_point_inside(x, y, list):
    for item in list:
        if item[0] <= x <= item[2] and item[1] <= y <= item[3]:
            return True
    return False


def process(photo):
    if hdstream == True:
        img = resample(photo)

    global obj_score, bounding_boxes, yolo_frame_count, cached_yolo_results, zoom_pan_active, zoom_pan_pause_time, dragging

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

            cv2.circle(img, point, 1, (0, 0, 255), 2)

            idle = str(obj.idle) + "s"

            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (200, 200, 200), 1)

        else:
            text = f"{class_name}" + " " + str(round(confidence, 6))
            text_offset_x = xmin
            text_offset_y = ymin - 5

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

            idle = str(obj.idle) + "s"

            cv2.circle(img, (obj.x, obj.y), 1, (0, 255, 255), 2)

    add(c)
    return img


sthread = threading.Thread(target=stream)
print(f"Starting..")

sthread.start()

while loop:
    if (q.empty() != True) and (fskip != True):

        img = q.get_nowait()

        key = cv2.waitKey(1) & 0xFF
        if key == 32:
            print("frame skip")
            q.queue.clear()

        if key == ord("q"):
            loop = False
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

        start = time.perf_counter_ns()
        img = process(img)

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

        _fps = "FPS: " + str(fps) + " - LAG: " + str(duration // 1000000) + "ms"
        text_y = img.shape[0] - 5
        cv2.putText(img, _fps, (16, text_y), _font, 0.4, (0, 0, 0), 2)
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
