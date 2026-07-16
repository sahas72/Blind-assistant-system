"""
AI Blind Assistance System — Real-Time Navigation Aid
------------------------------------------------------
Combines YOLOv8 object detection with MiDaS monocular depth estimation
to identify obstacles from a single webcam frame and speak out warnings
like "Warning, person ahead" or "Chair on your left".

HOW IT WORKS (high level):
1. Grab a frame from the webcam.
2. Run YOLOv8 on it to find objects (person, chair, etc.) + bounding boxes.
3. Run MiDaS on the SAME frame to get a depth map (how close/far every pixel is).
4. For each detected object, look at the depth value inside its bounding box
   to estimate "is this close enough to matter".
5. Figure out which zone (left / center / right) the object's center falls in.
6. If it's close AND that zone hasn't warned recently (cooldown), speak it out loud.

INSTALL (run in your PyCharm terminal):
    pip install ultralytics opencv-python torch torchvision pyttsx3 timm

RUN:
    python blind_assist.py

Press 'q' in the video window to quit.
"""

import cv2
import torch
import numpy as np
import time
import threading
import pyttsx3
from ultralytics import YOLO

# ============================================================
# CONFIG — tweak these numbers to tune behaviour
# ============================================================
ZONE_SPLIT = (0.20, 0.80)          # left ends at 20% of width, right starts at 80%
COOLDOWN_SECONDS = 4               # don't repeat a warning for the same zone within this window
CLOSE_DISTANCE_THRESHOLD = 0.55    # normalized depth (0=far, 1=near) above which we consider "close"
CONFIDENCE_THRESHOLD = 0.5         # ignore YOLO detections below this confidence
YOLO_MODEL_NAME = "yolov8n.pt"     # nano model = fastest, good for real-time on CPU
CAMERA_INDEX = 0                   # change if you have multiple webcams


# ============================================================
# VOICE OUTPUT — runs on its own thread so it never freezes the video
# ============================================================
class VoiceAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 170)  # speaking speed
        self.lock = threading.Lock()

    def speak(self, text: str):
        # Run speech in a background thread so the camera loop keeps running smoothly
        threading.Thread(target=self._say, args=(text,), daemon=True).start()

    def _say(self, text: str):
        with self.lock:  # prevents overlapping speech from multiple threads
            self.engine.say(text)
            self.engine.runAndWait()


# ============================================================
# COOLDOWN TRACKER — prevents repeating the same warning every frame
# ============================================================
class CooldownTracker:
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self.last_warning_time = {"left": 0.0, "center": 0.0, "right": 0.0}

    def is_ready(self, zone: str) -> bool:
        return (time.time() - self.last_warning_time[zone]) >= self.cooldown_seconds

    def mark_warned(self, zone: str):
        self.last_warning_time[zone] = time.time()


# ============================================================
# ZONE LOGIC
# ============================================================
def get_zone(box_center_x: float, frame_width: int) -> str:
    """Given an object's horizontal center, return 'left', 'center', or 'right'."""
    ratio = box_center_x / frame_width
    if ratio < ZONE_SPLIT[0]:
        return "left"
    elif ratio > ZONE_SPLIT[1]:
        return "right"
    else:
        return "center"


def zone_phrase(zone: str) -> str:
    return {"left": "on your left", "center": "ahead", "right": "on your right"}[zone]


# ============================================================
# MIDAS DEPTH ESTIMATION SETUP
# ============================================================
def load_midas():
    """Loads the small/fast MiDaS model and its matching preprocessing transform."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.to(device)
    midas.eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    return midas, transform, device


def estimate_depth(midas, transform, device, frame_bgr):
    """Runs MiDaS on a frame and returns a normalized depth map (0=far, 1=near), same size as frame."""
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # Normalize to 0-1 so thresholds are easy to reason about.
    # MiDaS outputs LARGER values for CLOSER objects, so this is already "near=high".
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max - depth_min > 1e-6:
        depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_map = np.zeros_like(depth_map)

    return depth_map


# ============================================================
# MAIN LOOP
# ============================================================
def main():
    print("Loading YOLOv8 model...")
    yolo_model = YOLO(YOLO_MODEL_NAME)

    print("Loading MiDaS depth model (this can take a moment on first run)...")
    midas, midas_transform, device = load_midas()

    voice = VoiceAssistant()
    cooldown = CooldownTracker(COOLDOWN_SECONDS)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check CAMERA_INDEX.")
        return

    print("Running. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame_height, frame_width = frame.shape[:2]

        # --- 1. Object detection ---
        results = yolo_model(frame, verbose=False)[0]

        # --- 2. Depth estimation on the same frame ---
        depth_map = estimate_depth(midas, midas_transform, device, frame)

        # --- 3. Draw zone divider lines (visual reference only) ---
        left_line_x = int(frame_width * ZONE_SPLIT[0])
        right_line_x = int(frame_width * ZONE_SPLIT[1])
        cv2.line(frame, (left_line_x, 0), (left_line_x, frame_height), (255, 255, 0), 1)
        cv2.line(frame, (right_line_x, 0), (right_line_x, frame_height), (255, 255, 0), 1)

        # --- 4. Process each detected object ---
        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2

            # Sample depth in a small region around the box center rather than
            # just one pixel — more stable against noise.
            sample_half = 5
            y_start = max(0, int(box_center_y) - sample_half)
            y_end = min(frame_height, int(box_center_y) + sample_half)
            x_start = max(0, int(box_center_x) - sample_half)
            x_end = min(frame_width, int(box_center_x) + sample_half)
            depth_sample = depth_map[y_start:y_end, x_start:x_end]
            proximity = float(np.mean(depth_sample)) if depth_sample.size > 0 else 0.0

            zone = get_zone(box_center_x, frame_width)
            is_close = proximity >= CLOSE_DISTANCE_THRESHOLD

            # --- Draw box + label for visual debugging ---
            color = (0, 0, 255) if is_close else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {proximity:.2f} [{zone}]"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Voice warning if close + cooldown allows it ---
            if is_close and cooldown.is_ready(zone):
                phrase = f"Warning, {class_name} {zone_phrase(zone)}"
                voice.speak(phrase)
                cooldown.mark_warned(zone)

        cv2.imshow("AI Blind Assistance System", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()