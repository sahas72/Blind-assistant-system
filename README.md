# Blind Assistance System (Simple Version)

A lightweight computer vision project that detects nearby obstacles in a video feed and prints a directional warning (left / ahead / right) to the console.

## What it does

- Detects objects in each video frame using **YOLOv8**.
- Estimates how close an object is by how large its bounding box is on screen — bigger box = closer object. No separate depth model needed.
- Splits the frame into three zones (left, center, right) based on where the object is horizontally.
- Prints a warning like `WARNING: chair ahead` when a close object is detected, with a cooldown so the same object/zone combo doesn't spam warnings every frame.

## How it works

1. Each frame is run through YOLOv8 to detect objects and their bounding boxes.
2. For every detected object above the confidence threshold, its bounding box height (relative to the frame) is used as a simple proxy for distance.
3. The object's horizontal position determines its zone: left, center, or right.
4. If the object is "close" (box height ratio passes a threshold) and hasn't triggered a warning for that same zone/object combo recently, a warning is printed.
5. Bounding boxes are drawn on screen in red (close) or green (not close) for visual debugging.

## Requirements

```
opencv-python
ultralytics
```

Install with:
```
pip install -r requirements.txt
```

## Running it

1. Place your input video at `videos/video1.mp4`, or change `VIDEO_SOURCE` in the script to `0` to use your webcam.
2. Run:
```
python simple_blind_assistant.py
```
3. Press `q` in the video window to quit.

## Settings you can tweak

| Setting | What it does |
|---|---|
| `ZONE_SPLIT` | Where the left/center/right boundaries fall (as fractions of frame width) |
| `COOLDOWN_SECONDS` | How long to wait before repeating a warning for the same object/zone |
| `CONFIDENCE_THRESHOLD` | Minimum YOLO confidence to consider a detection valid |
| `CLOSE_SIZE_THRESHOLD` | How large (as a fraction of frame height) a box needs to be before it's considered "close" |
| `YOLO_MODEL_NAME` | Which YOLOv8 model to use (nano is fastest, good for CPU) |

## Notes / Future work

- Currently prints warnings to the console instead of speaking them aloud — text-to-speech was explored but hit a platform-specific audio engine issue that needs further investigation.
- Distance estimation is a simple size-based heuristic rather than true depth estimation, which keeps the project fast and dependency-light.