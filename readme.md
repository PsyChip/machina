## MACHINA
CCTV viewer with realtime object tagger [WIP]

### Uses
- [LLAVA](https://llava-vl.github.io)
- [YOLO 11](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org)

### How it works
Simply it connects to a high-resolution RTSP stream in a separate thread,
queues the frames into memory as it is and resamples it for processing.

YOLO takes this frame, application gives a specific id based on it's coordinates,
size and timestamp then tries to match the same object on every iteration.

Another thread runs in background, iterates that object array continuously and
makes LLM requests to Ollama server for object tagging

### Object matching
It calculates the center of every detection box, pinpoint on screen and gives 16px
tolerance on all directions. Script tries to find closest object as fallback and
creates a new object in memory in last resort.
You can observe persistent objects in ```/elements``` folder 

### Test Environment
Every input frame resampled to 640x480 for processing, got avg 20ms interference time
with yolo 11 small model (yolo11s.pt) on Geforce GTX 1060 which is almost 7 years old
graphics card. Other models available in "models" directory

Stream delays by 1-2 seconds on every 10~ minutes due to network conditions, script also
have a frame skip mechanism on 3 seconds of detection idle.

### Prerequisites
- Clone the repository
- Install [ollama](https://ollama.com/) server
- Pull the LLAVA model by running ```ollama run llava```
- Open ```app.py``` and set your rtmp stream address at line 18
- Install the dependencies by running ```pip install -r requirements.txt```
- Run the script ```py app.py```

### Shortcuts
S : snapshot, actual image from input stream
R : start/stop recording. it records what you see.
Q : quit app

### Project direction
This is a living project, trying to create a *complete* headless security system by
taking advantage of modern vision, object detection models on my spare time.

Feel free to contribute with code, ideas or even maybe a little bit donation
via ko-fi or bitcoin

[https://ko-fi.com/psychip](https://ko-fi.com/psychip)
BTC: bc1qlq067vldngs37l5a4yjc4wvhyt89wv3u68dsuv

Created by PsyChip
root@psychip.net

.eof