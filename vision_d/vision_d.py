import numpy as np
import cv2
import time
import os
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO

# Shared memory paths
FRAME_PATH = "/dev/shm/frame.npy"        # from camera
OUTPUT_PATH = "/dev/shm/D_output.npy"    # output for viewer
TOGGLE_PATH = "/dev/shm/active.txt"

MY_ID = "D"

WIDTH = 1280
HEIGHT = 720
CHANNELS = 3

# Load TensorRT YOLO engine
model = YOLO("./models/yolov8s.engine", task="detect")
print("Vision D running (TensorRT)")

while True:
    # Check toggle
    try:
        with open(TOGGLE_PATH) as f:
            active = f.read().strip()
    except:
        active = "OFF"

    if active != MY_ID:
        time.sleep(0.01)
        continue

    # Load frame from camera
    try:
        frame = np.load(FRAME_PATH)
    except:
        time.sleep(0.002)
        continue

    # Run inference (all classes)
    results = model(frame, conf=0.4, verbose=False)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Save output to shared memory
    np.save(OUTPUT_PATH, frame)
