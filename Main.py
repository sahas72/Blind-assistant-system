import cv2
import time
from ultralytics import YOLO

# ---------------- Settings ----------------
ZONE_SPLIT = (0.33, 0.66)           # left ends at 33% of width, right starts at 66%
COOLDOWN_SECONDS = 4                # don't repeat a warning for the same (zone, object) within this window
CONFIDENCE_THRESHOLD = 0.5          # ignore YOLO detections below this confidence
CLOSE_SIZE_THRESHOLD = 0.35         # if the box's height is more than 35% of frame height, treat it as "close"
YOLO_MODEL_NAME = "yolov8n.pt"      # nano model = fastest, good for real-time on CPU
VIDEO_SOURCE = "videos/video1.mp4"  # change to 0 to use your webcam instead


# ---------------- Cooldown tracker ----------------
class CooldownTracker:
    """Keyed by (zone, object type), so different objects in the same zone don't silence each other."""
    def __init__(self, cooldown_seconds: float):
        self.cooldown_seconds = cooldown_seconds
        self.last_warning_time = {}

    def is_ready(self, key) -> bool:
        return (time.time() - self.last_warning_time.get(key, 0.0)) >= self.cooldown_seconds

    def mark_warned(self, key):
        self.last_warning_time[key] = time.time()


def get_zone(box_center_x: float, frame_width: int) -> str:
    """Given an object's horizontal center, return 'left', 'center', or 'right'."""
    ratio = box_center_x / frame_width
    if ratio < ZONE_SPLIT[0]:
        return "left"
    elif ratio > ZONE_SPLIT[1]:
        return "right"
    return "center"


def zone_phrase(zone: str) -> str:
    return {"left": "on your left", "center": "ahead", "right": "on your right"}[zone]


def main():
    print("Loading YOLOv8 model...")
    model = YOLO(YOLO_MODEL_NAME)

    cooldown = CooldownTracker(COOLDOWN_SECONDS)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: could not open video/camera source.")
        return

    print("Running. Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Finished or failed to read frame.")
            break

        frame_height, frame_width = frame.shape[:2]

        # --- Zone divider lines (visual reference only) ---
        left_line_x = int(frame_width * ZONE_SPLIT[0])
        right_line_x = int(frame_width * ZONE_SPLIT[1])
        cv2.line(frame, (left_line_x, 0), (left_line_x, frame_height), (255, 255, 0), 1)
        cv2.line(frame, (right_line_x, 0), (right_line_x, frame_height), (255, 255, 0), 1)

        # --- Object detection ---
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_center_x = (x1 + x2) / 2

            # "Closer" objects take up more of the frame vertically.
            box_height_ratio = (y2 - y1) / frame_height
            is_close = box_height_ratio >= CLOSE_SIZE_THRESHOLD

            zone = get_zone(box_center_x, frame_width)

            # --- Draw box + label for visual debugging ---
            color = (0, 0, 255) if is_close else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {box_height_ratio:.2f} [{zone}]"
            cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # --- Print a warning if close + cooldown allows it ---
            key = (zone, class_name)
            if is_close and cooldown.is_ready(key):
                print(f"WARNING: {class_name} {zone_phrase(zone)}")
                cooldown.mark_warned(key)

        cv2.imshow("Simple Blind Assistance", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()