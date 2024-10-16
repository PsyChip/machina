## MACHINA
CCTV viewer with realtime object tagger [WIP]

![partial screenshot](demo.png)

### Uses
- [LLAVA](https://llava-vl.github.io)
- [YOLO 11](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org)
- [FAISS](https://github.com/facebookresearch/faiss)

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
Make sure you have all Visual C++ redistributables if you're running on windows
https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170

### Installation
- Install Python 3.12.x
- Clone the repository
- Install [ollama](https://ollama.com/) server
- Pull the LLAVA model by running ```ollama run llava```
- Install the dependencies by running ```pip install -r requirements.txt```
- Remove pytorch cpu version and install the cuda version
- Open ```app.py``` and set your rtmp stream address at line 18
- Run the script ```py app.py```

```sh
git clone https://github.com/PsyChip/machina
cd machina
pip install -r requirements.txt
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
py app.py
```

### Notes
- It's okay to stick with cpu version of FAISS if you're using yolo nano/small/medium model
- Change ```vsize``` value depending on your chosen yolo model, you need to delete the index when
changing vector size.
- CUDA enabled torch is an absolute necessity for real time interference
- Pretrained models of yolo is not so accurate on low-res streams, it's highly recommended to train
your own model by using object images from your ```/elements``` folder

### Usage
- S : snapshot, actual image from input stream
- R : start/stop recording. it records what you see.
- Q : quit app
- left mouse: select
- middle mouse: zoom
- right mouse: pan

### Project direction
This is a living project, trying to create a *complete* headless security system by
taking advantage of open source object detection models on my spare time.

### TODO
- Additional UI Layer
- RTS style object selection box and detailed information about selected object(s)
- People crowd, car crash, police, ambulance, running human detection [request]
- Webhook callbacks on new object/disappeared object/movement after long stay

Feel free to contribute with code, ideas or even maybe a little bit support
via ko-fi or bitcoin. I'll prioritize the feature requests for every $10 donation 

- [https://ko-fi.com/psychip](https://ko-fi.com/psychip)
- BTC: ```bc1qlq067vldngs37l5a4yjc4wvhyt89wv3u68dsuv```

Created by PsyChip
```root@psychip.net```

.eof