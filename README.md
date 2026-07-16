# AI Blind Assistance System — Real-Time Navigation Aid

A real-time obstacle detection and voice warning system that combines **YOLOv8 object detection** with **MiDaS monocular depth estimation** to help identify nearby obstacles using just a single webcam — no stereo camera or LiDAR needed.

## What It Does

- Detects objects in the camera feed using YOLOv8 (people, chairs, doors, etc.)
- Estimates how close each object is using MiDaS depth estimation on the same frame
- Divides the frame into 3 zones — **left (20%)**, **center (60%)**, **right (20%)**
- Speaks out warnings when something close is detected, e.g.:
  - *"Warning, person ahead"*
  - *"Chair on your left"*
- Uses a **4-second cooldown per zone** so it doesn't spam the same warning every frame — new obstacles in a zone still trigger fresh warnings

## Demo

*(Add a screenshot or GIF of the app running here once you've tested it — shows the bounding boxes, zone lines, and depth values on screen.)*

## How It Works

1. Grab a frame from the webcam
2. Run YOLOv8 on the frame to detect objects and their bounding boxes
3. Run MiDaS on the same frame to generate a depth map (closer = higher value)
4. Sample the depth map at each object's location to judge proximity
5. Determine which zone (left/center/right) the object falls in based on its horizontal position
6. If the object is close **and** that zone's cooldown has expired, speak a warning and reset the cooldown

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [MiDaS](https://github.com/isl-org/MiDaS) (small/fast variant) — monocular depth estimation
- OpenCV — video capture and display
- pyttsx3 — offline text-to-speech
- PyTorch — model backend

## Installation

```bash
pip install -r requirements.txt
```

First run will automatically download the YOLOv8n and MiDaS_small model weights.

## Usage

```bash
python blind_assist.py
```

Press **`q`** in the video window to quit.

## Configuration

Key settings are at the top of `blind_assist.py`:

| Setting | Description | Default |
|---|---|---|
| `ZONE_SPLIT` | Where left/center/right boundaries fall (as % of frame width) | `(0.20, 0.80)` |
| `COOLDOWN_SECONDS` | Minimum seconds between repeat warnings per zone | `4` |
| `CLOSE_DISTANCE_THRESHOLD` | Normalized depth (0=far, 1=near) above which an object is "close" | `0.55` |
| `CONFIDENCE_THRESHOLD` | Minimum YOLO detection confidence to consider | `0.5` |
| `CAMERA_INDEX` | Which webcam to use | `0` |

> **Note:** MiDaS depth values are *relative*, not real-world meters — `CLOSE_DISTANCE_THRESHOLD` needs to be tuned by testing in your actual environment.

## Known Limitations

- MiDaS inference can be slow on CPU-only machines, which may lag the frame rate
- Depth is relative per-frame, not calibrated to real-world distance
- Voice warnings rely on the object staying detected across frames within the cooldown window

## Possible Improvements

- Run MiDaS every 2-3 frames instead of every frame to improve FPS on slower hardware
- Add distance categories (e.g. "very close" vs "approaching") instead of a single threshold
- Support stereo cameras or depth sensors for more accurate real-world distance
- Add adjustable voice language/speed settings for accessibility