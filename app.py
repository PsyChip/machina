print(f"Initializing..")
import time
import os
import cv2
import queue
import threading
import numpy as np
import math
from datetime import datetime
from ultralytics import YOLO
import torch
import torchvision
import base64
import requests
import json
import pickle
import sys

import clip
import torch.nn as nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# from vdb import VectorDatabase

model = "yolo11m"
rtsp_stream = (
    "rtsp://psychip:neuromancer1@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"
)
ollama = "http://127.0.0.1:11434/api/generate"
ollama_model = "llava:latest"

# vsize = 512 # yolo11n:256 yolo11s:512

_font = cv2.FONT_HERSHEY_SIMPLEX
_gray = cv2.COLOR_BGR2GRAY

cf = 0
cl = 0

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

# change per scenerio, outdoor, indoor, jungle, mountain etc
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

last_event = ""
events = ["cars parked on the side of a road at night"]

if os.path.exists("db/events.pkl"):
    with open("db/events.pkl", "rb") as file:
        events = pickle.load(file)

snapshot_directory = "snapshots"

frames = 0
prev_frames = 0
last_frame = 0
fps = 0
WINDOW_WIDTH = 0
WINDOW_HEIGHT = 0
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

uispace = 0  # 300
padding = 6  # px padding on each element

print("Loading ViT..")
cdevice = "cuda" if torch.cuda.is_available() else "cpu"
cmodel, preprocess = clip.load("ViT-B/32", cdevice)
last_event = ""

print("Loading BLIP..")
bprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
bmodel = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(cdevice)

print("Generating text features of " + str(len(events)))

text_inputs = clip.tokenize(events).to(cdevice)
text_features = cmodel.encode_text(text_inputs)

def match_caption(image):
    mstart = millis()
    image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = cmodel.encode_image(image)

    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    similarity = image_features @ text_features.T
    similarity_score = similarity.max().item()
    best_caption_idx = similarity.argmax().item()
    mend = millis() - mstart
    print(f"Matched caption in {mend}ms")

    return (events[best_caption_idx], similarity_score)

def resolve(file_path):
    if os.path.isabs(file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        return file_path
    else:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        return os.path.abspath(file_path)


def extract_features(img_tensor, model, boxes):
    features = []
    feature_extractor = nn.Sequential(*list(model.model.model[:10])).to(device)

    with torch.no_grad():
        feature_maps = feature_extractor(img_tensor)

    for box in boxes:
        if hasattr(box, "xyxy") and isinstance(box.xyxy, torch.Tensor):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        elif hasattr(box, "xyxy") and isinstance(box.xyxy, list):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        else:
            print(f"Unexpected box format: {type(box)}")
            continue

        stride = img_tensor.shape[2] / feature_maps.shape[2]
        fm_x1, fm_y1 = int(x1 / stride), int(y1 / stride)
        fm_x2, fm_y2 = int(x2 / stride), int(y2 / stride)

        box_features = feature_maps[:, :, fm_y1:fm_y2, fm_x1:fm_x2]
        box_features = nn.functional.adaptive_avg_pool2d(box_features, (1, 1))
        features.append(box_features.flatten().cpu().numpy())

    return features

def preinit():
    for folder in ["elements", "models", "recordings", "snapshots"]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"-- created folder: {folder}")


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
        data = data = json.dumps(payload)
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

labels = open(resolve("db/coco.names")).read().strip().split("\n")
classlist = [labels.index(x) for x in classlist]

object_count = 0
old_count = 0
obj_break = millis()
obj_idle = 0
obj_list = []
obj_max = 16
obj_avg = 0
fskip = False
last_fskip = timestamp()
app_start = timestamp()
obj_score = labels

bounding_boxes = []
obj_number = 1

def save_bounding_boxes(bounding_boxes, filename="db/bounding_boxes.pkl"):
    with open(resolve(filename), "wb") as f:
        pickle.dump(bounding_boxes, f)
    print(f"Saved {len(bounding_boxes)} bounding boxes to {filename}")


def load_bounding_boxes(filename="db/bounding_boxes.pkl"):
    filename = resolve(filename)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            bounding_boxes = pickle.load(f)
        print(f"Loaded {len(bounding_boxes)} bounding boxes from {filename}")
        return bounding_boxes
    else:
        print(f"No saved bounding boxes found at {filename}")
        return []

def crc32(string):
    crc = 0xFFFFFFFF
    for char in string:
        byte = ord(char)
        for _ in range(8):
            if (crc ^ byte) & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
            byte >>= 1
    return crc ^ 0xFFFFFFFF

def genprompt(t):
    if t in prompts:
        return prompts[t]
    return "describe this image in 5 words or less"


def center(xmin, ymin, xmax, ymax):
    center_x = (xmin + xmax) // 2
    center_y = (ymin + ymax) // 2
    return (center_x, center_y)


def distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _size(x1, y1, x2, y2):
    return abs(x1 - y2)

def bearing(x1, y1, x2, y2):
    delta_x = x2 - x1
    delta_y = y2 - y1
    bearing_rad = math.atan2(delta_y, delta_x)
    bearing_deg = math.degrees(bearing_rad)
    return (bearing_deg + 360) % 360


def direction(bearing):
    normalized_bearing = bearing % 360
    directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    index = round(normalized_bearing / 45) % 8
    return directions[index]


def similar(img1, img2):
    hist1 = cv2.calcHist([img1], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist2 = cv2.calcHist([img2], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def match(img1, img2):
    max_val = 0
    try:
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
    except Exception as e:
        max_val = similar(img1, img2)
        # print(f"An error occurred: {e}")
    finally:
        return max_val


def open_app_folder():
    app_folder = os.path.dirname(os.path.abspath(__file__))
    if os.name == "nt":  # Windows
        os.startfile(app_folder)
    elif os.name == "posix":  # macOS and Linux
        subprocess.call(["open" if os.name == "darwin" else "xdg-open", app_folder])


def mouse_callback(event, x, y, flags, param):
    global drawing, draw_start_x, draw_start_y, draw_end_x, draw_end_y, dragging, drag_start_x, drag_start_y, zoom_factor, pan_x, pan_y, zoom_x, zoom_y
    if event == cv2.EVENT_RBUTTONDOWN:
        dragging = True
        drag_start_x = x
        drag_start_y = y

    if event == cv2.EVENT_RBUTTONUP:
        dragging = False

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
        self.px = 0
        self.py = 0
        self.created = millis()
        self.timestamp = self.created
        self.size = size
        self.sid = str(crc32(f"{self.x}-{self.y}-{self.timestamp}-{self.size}"))
        self.name = name
        self.checkin = True
        self.detections = 0
        self.distance = 0
        self.idle = 0
        self.image = image
        self.desc = False
        self.state = 0
        self.seen = self.created
        self.features = None
        self.visible = True

        self.init()
        print(
            "New object: " + self.name + "#" + str(self.nr) + " size:" + str(self.size)
        )
        self.save("elements/" + self.name + "-" + str(self.nr) + ".png")

        """
        vector_db.add_vector(self.features, {
                'class': self.name,
                'sid': self.sid
            })"""

    def hide(self):
        self.visible = False

    def show(self):
        self.visible = True

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["image"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.image = None

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

    def save(self, file):
        cv2.imwrite(file, self.image)

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
        self.px = self.x
        self.py = self.y
        self.x = new_x
        self.y = new_y
        self.detections += 1
        self.init()

    def update_in_array(self, time, new_x, new_y, bounding_boxes):
        for bbox in bounding_boxes:
            if bbox.sid == self.sid:
                bbox.update(time, new_x, new_y)
                return True
            return False


def resetIteration():
    global bounding_boxes
    [setattr(item, "checkin", False) for item in bounding_boxes]


def closest(bounding_boxes, reference_point, class_name, size):
    closest_bbox = False
    min_distance = float("inf")

    for bbox in bounding_boxes:
        if abs(bbox.size - size) > 10:
            continue

        dx = bbox.x - reference_point[0]
        dy = bbox.y - reference_point[1]
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < 1 or distance > 300:
            continue

        if distance < min_distance:
            min_distance = distance
            closest_bbox = bbox

    if closest_bbox != False and distance > 0:
        closest_bbox.distance = distance
        closest_bbox.update(millis(), reference_point[0], reference_point[1])
    return closest_bbox


def blur(image):
    gray = cv2.cvtColor(image, _gray)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    return np.mean(mag)


def find_closest_point(points, point):
    closest_point = None
    min_distance = float("inf")

    for x, y in points:
        distance = math.sqrt((x - point[0]) ** 2 + (y - point[1]) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = (x, y)

    return closest_point, min_distance


def findSimilar(ref):
    closest_bbox = False
    score = 0.85
    for bbox in bounding_boxes:
        s = similar(ref, bbox.image)
        if s > score:
            score = s
            closest_bbox = bbox
    # print("similar:"+str(score))
    return closest_bbox


def findMatch(ref):
    closest_bbox = False
    score = 0.96
    for bbox in bounding_boxes:
        s = match(ref, bbox.image)
        if s > score:
            score = s
            closest_bbox = bbox
    # print("score:"+str(score))
    return closest_bbox


def closestEx(bounding_boxes, reference_point, class_name, size):
    return False

    point = reference_point
    found = []
    for i in range(6):
        c = closest(bounding_boxes, point, class_name, size)
        if c == False and i == 0:
            return False
        if c == False and i > 0:
            return found[-1]
        if i == 1 and found[-1].sid == c.sid:
            return c

        point = (c.x, c.y)
        found.append(c)
        # print("iteration "+str(i))

    return found[-1]

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

        # if (time-box.seen) >= point_timeout:
        #  box.hide()

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

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

colors = generate_color_shades(len(labels))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initializing model..")
model = YOLO("models/" + model + ".pt")
print(f"Loading model to {device}")
model.to(device)

# print("initializing vector store")
# vector_db = VectorDatabase(vector_dimension=vsize)

print("loading objects")
# bounding_boxes = load_bounding_boxes()
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
cv2.resizeWindow(window, opsize[0] + uispace, opsize[1])
cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
cv2.setMouseCallback(window, mouse_callback)

q = queue.Queue(maxsize=buffer)


def stream():
    global cap, obj_idle, last_fskip, idle
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


def fmatch(img1, img2):
    gray1 = cv2.cvtColor(img1, _gray)
    gray2 = cv2.cvtColor(img2, _gray)

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img_matches = cv2.drawMatches(
        img1,
        kp1,
        img2,
        kp2,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = gray2.shape
    aligned_img1 = cv2.warpPerspective(img1, M, (w, h))

    diff = cv2.absdiff(cv2.cvtColor(aligned_img1, _gray), gray2)

    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img1_contours = aligned_img1.copy()
    img2_contours = img2.copy()

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(img1_contours, [contour], 0, (0, 0, 255), 2)
            cv2.drawContours(img2_contours, [contour], 0, (0, 255, 0), 2)


def selectObject(sid, x, y):
    i = 0
    while i < len(bounding_boxes):
        # print(i)
        bbox = bounding_boxes[i]
        if bbox.sid == sid and bbox.checkin == False:
            bbox.update(millis(), x, y)
            bounding_boxes[i].checkin = True
            return bounding_boxes[i]
        else:
            i += 1

    return False


"""
def find_similar_objects(query_vector, class_name, k=5):
    results = vector_db.search_similar(query_vector, k)

    _sid = False
    _c = 25

    for metadata, distance in results:
        if distance >= 1:
            continue

        if distance < _c:
            _c = distance
            _sid = metadata["sid"]

        # print(f"SID: {metadata['sid']}")
        # print(f"Distance: {distance}")
        # print("---")
    return _sid
"""

def process(photo):
    if hdstream == True:
        img = resample(photo)

    global obj_score, bounding_boxes

    img_tensor = (
        torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to(device).float()
        / 255.0
    )
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        results = model(
            img_tensor,
            verbose=False,
            iou=0.45,
            agnostic_nms=True,
            half=False,
            max_det=32,
            conf=min_confidence,
            classes=classlist,
        )

    obj_score = [0 for _ in range(len(obj_score))]
    c = 0
    points = []
    boxes = [box for r in results for box in r.boxes]
    features = extract_features(img_tensor, model, boxes)
    now = millis()
    resetIteration()

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

            cv2.putText(
                img, text, (text_offset_x, text_offset_y), _font, 0.35, (0, 0, 0), 2
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
            continue

        idx = class_id
        obj_score[idx] = obj_score[idx] + 1

        point = center(xmin, ymin, xmax, ymax)
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

            cv2.putText(img, sid, (obj.x, obj.y - 18), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, sid, (obj.x, obj.y - 18), _font, 0.35, (255, 255, 255), 1)
            idle = str(obj.idle) + "s"
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (200, 200, 200), 1)

        else:
            obj = closestEx(bounding_boxes, point, class_name, size)
            if obj != False:
                print(
                    "picked up "
                    + str(obj.nr)
                    + "#"
                    + obj.name
                    + " from "
                    + str(obj.distance)
                )
                cv2.line(img, point, (obj.x, obj.y), (0, 255, 255), 4)
                obj.see()

                if obj.desc != False:
                    sid = obj.desc
                else:
                    sid = obj.name + "#" + str(obj.nr)

                cv2.circle(img, point, 1, (0, 255, 0), 2)

                cv2.putText(img, sid, (obj.x, obj.y - 18), _font, 0.35, (0, 0, 0), 2)
                cv2.putText(
                    img, sid, (obj.x, obj.y - 18), _font, 0.35, (255, 255, 255), 1
                )

                idle = str(obj.idle) + "s"
                cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2)
                cv2.putText(
                    img, idle, (obj.x, obj.y - 6), _font, 0.35, (200, 200, 200), 1
                )
            else:
                if distance < 6.0:
                    continue
                text = f"{class_name}" + " " + str(round(confidence, 6))
                text_offset_x = xmin
                text_offset_y = ymin - 5

                # _sid = find_similar_objects(features[i],class_name)
                _sid = False
                if _sid != False:
                    obj = selectObject(_sid, point[0], point[1])
                    print(str(obj))
                    if obj != False:
                        print(
                            "restored "
                            + str(obj.nr)
                            + "#"
                            + obj.name
                            + "#"
                            + str(obj.desc)
                            + " from vector store"
                        )
                        cv2.line(img, point, (obj.px, obj.py), (0, 255, 0), 4)
                        cv2.circle(img, point, 1, (0, 255, 0), 3)
                        obj.see()
                        sid = (
                            obj.desc
                            if obj.desc != False
                            else obj.name + "#" + str(obj.nr)
                        )
                        cv2.putText(
                            img, sid, (obj.x, obj.y - 18), _font, 0.35, (0, 0, 0), 2
                        )
                        cv2.putText(
                            img,
                            sid,
                            (obj.x, obj.y - 18),
                            _font,
                            0.35,
                            (255, 255, 255),
                            1,
                        )
                        idle = str(obj.idle) + "s"
                        cv2.putText(
                            img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2
                        )
                        cv2.putText(
                            img,
                            idle,
                            (obj.x, obj.y - 6),
                            _font,
                            0.35,
                            (200, 200, 200),
                            1,
                        )
                    else:
                        if distance < 6.0:
                            continue
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
                        qxmin, qymin, qxmax, qymax = transform(
                            xmin, ymin, xmax, ymax, padding
                        )
                        snap = photo[qymin:qymax, qxmin:qxmax]
                        item = BoundingBox(class_name, point, size, snap, features[i])
                        bounding_boxes.append(item)
                else:
                    if distance < 6.0:
                        continue
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
                    qxmin, qymin, qxmax, qymax = transform(
                        xmin, ymin, xmax, ymax, padding
                    )
                    snap = photo[qymin:qymax, qxmin:qxmax]
                    item = BoundingBox(class_name, point, size, snap)
                    bounding_boxes.append(item)

    if zoom_factor > 1.0:
        add(c)
        return img

    for obj in bounding_boxes:
        if (
            obj.checkin == False
            and obj.detections >= 3
            and obj.idle > 1
            and obj.idle < 8
        ):
            closest, distance = find_closest_point(points, (obj.x, obj.y))
            if distance < 6.0:
                continue

            obj.ping()
            if now - obj.seen > point_timeout:
                continue

            if obj.desc != False:
                sid = obj.desc
            else:
                sid = obj.name + "#" + str(obj.nr)

            idle = str(obj.idle) + "s"

            cv2.putText(img, sid, (obj.x, obj.y - 18), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, sid, (obj.x, obj.y - 18), _font, 0.35, (255, 255, 255), 1)
            cv2.circle(img, (obj.x, obj.y), 1, (0, 255, 255), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (0, 0, 0), 2)
            cv2.putText(img, idle, (obj.x, obj.y - 6), _font, 0.35, (200, 200, 200), 1)
    add(c)
    return img


def postreview():
    global bounding_boxes, loop
    while loop:
        for box in bounding_boxes:
            if (box.state == 0) and (box.image is not None):
                res = rest(
                    ollama,
                    {
                        "model": ollama_model,
                        "prompt": genprompt(box.name),
                        "images": [box.export()],
                        "stream": False,
                    },
                )

                if res != False:
                    box.desc = res["response"].strip()
                    box.state = 1
        time.sleep(0.1)


bthread = threading.Thread(target=postreview)
bthread.start()

sthread = threading.Thread(target=stream)
sthread.start()


def uilayer(img):
    height, width = img.shape[:2]

    new_height = height
    new_width = width + uispace

    background_color = [64, 64, 64]
    enlarged_img = np.full((new_height, new_width, 3), background_color, dtype=np.uint8)

    enlarged_img[:height, :width] = img

    cv2.putText(
        enlarged_img,
        "Your text description here",
        (width + 24, 24),
        _font,
        0.5,
        (255, 255, 255),
        1,
        1,
    )
    return enlarged_img


def generate_caption(img):
    inputs = bprocessor(
        img,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        output = bmodel.generate(
            **inputs,
            max_length=50,  # Lower: 10 (short captions, might miss details) | Higher: 60 (longer captions, might add unnecessary info)
            min_length=15,  # Lower: 1 (may generate incomplete captions) | Higher: 10 (forces longer captions, even when unnecessary)
            num_beams=5,  # Lower: 1 (more random, less structured) | Higher: 10 (more accurate but repetitive)
            do_sample=True,  # False (more deterministic captions) | True (allows more diversity)
            top_k=50,  # Lower: 5 (more predictable, limited variation) | Higher: 100 (more diverse, may generate unrelated words)
            top_p=0.8,  # Lower: 0.5 (more strict, reduces diversity) | Higher: 1.0 (more diverse but might add irrelevant details)
            temperature=0.8,  # Lower: 0.1 (more deterministic, repetitive) | Higher: 1.0 (more creative but less reliable)
        )
    return bprocessor.decode(output[0], skip_special_tokens=True)


def take_caption(img):    
    tstart = millis()
    caption = generate_caption(img)
    tend = millis() - tstart
    print(f"Caption generated in {tend}ms")

    if caption in events:
        print("-- " + caption)
        return

    events.append(caption)
    print("New Event >> " + caption)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}.jpg"
    filepath = os.path.join("captions", filename)
    cv2.imwrite(filepath, img)
    filename = filename + ".txt"
    filepath = os.path.join("captions", filename)
    with open(filepath, "w") as file:
        file.write(caption)


print(f"Starting..")

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

        elif key == ord("c"):
            take_caption(img)
            q.queue.clear()

        start = time.perf_counter_ns()

        cf += 1
        if cf >= 10:
            cf = 0
            caption, score = match_caption(img)
            if caption != last_event:
                print("match: " + caption)
                print("score: " + str(score))
                last_event = caption
                q.queue.clear()

        # learning mode, turn off after few weeks
        cl += 1
        if cl >= 30:
            cl = 0
            take_caption(img)
            q.queue.clear()

        img = process(img)

        object_count = average()
        if object_count != old_count:
            obj_break = millis()
            obj_idle = 0
        else:
            obj_idle = millis() - obj_break

        cv2.putText(img, last_event, (16, 16), _font, 0.4, (0, 0, 0), 2)
        cv2.putText(img, last_event, (16, 16), _font, 0.4, (255, 255, 255), 1)

        old_count = object_count
        frames += 1
        if millis() - last_frame >= 1000:
            fps = (frames - prev_frames) * 1
            prev_frames = frames
            last_frame = millis()

        duration = time.perf_counter_ns() - start

        _fps = "FPS: " + str(fps) + " - LAG: " + str(duration // 1000000) + "ms"
        text_size = cv2.getTextSize(_fps, _font, 0.5, 1)[0]
        text_x = 16
        text_y = img.shape[0] - 5
        cv2.putText(img, _fps, (text_x, text_y), _font, 0.4, (0, 0, 0), 2)
        cv2.putText(img, _fps, (text_x, text_y), _font, 0.4, (0, 255, 0), 1)

        """
        line = 16
        _t = line*2
        for i, s in enumerate(obj_score):
         if(s>0):
          _s = labels[i]+": "+str(s)
          cv2.putText(img, _s, (16, _t), _font, 0.4, (0, 0, 0), 2)        
          cv2.putText(img, _s, (16, _t), _font, 0.4, (255, 255, 255), 1)
          _t = _t+line   
        """
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

        # img = uilayer(img)
        cv2.imshow(str(rtsp_stream), img)
    else:
        fskip = False
        time.sleep(0.001)

print("saving events list..")
with open("db/events.pkl", "wb") as file:
    pickle.dump(events, file)

# print("saving vector index..")
# vector_db.save_index()

# print("saving found objects..")
# save_bounding_boxes(bounding_boxes)
print("closing cv window..")
cv2.destroyAllWindows()
print("terminating..")
loop = False
# bthread.join()
# sthread.join()
sys.exit(0)
