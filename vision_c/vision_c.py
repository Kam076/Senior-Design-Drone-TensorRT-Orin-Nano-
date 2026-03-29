import numpy as np
import cv2
import time
import os
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO

FRAME_PATH = "/dev/shm/frame.npy"
OUTPUT_PATH = "/dev/shm/C_output.npy"
TOGGLE_PATH = "/dev/shm/active.txt"
MY_ID = "C"
WIDTH = 1280
HEIGHT = 720
CHANNELS = 3

# Check engine
engine_path = "./vision_c/models/yolov8s.engine"
if not os.path.exists(engine_path):
    print(f"[ERROR] Engine file missing: {engine_path}")
    exit(1)

model = YOLO(engine_path, task="detect")
print("Vision C running (TensorRT)")

frame_count = 0

while True:
    # Check toggle
    try:
        with open(TOGGLE_PATH, "r") as f:
            active = f.read().strip()
    except:
        active = "OFF"

    if active != MY_ID:
        time.sleep(0.01)
        continue

    # Load camera frame
    try:
        frame = np.load(FRAME_PATH)
    except:
        time.sleep(0.005)
        continue

    # Run inference
    results = model(frame, conf=0.4, classes=[0], verbose=False)

    # Draw boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Write output
    np.save(OUTPUT_PATH, frame)
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"[INFO] Written {frame_count} frames")
    time.sleep(0.01)
