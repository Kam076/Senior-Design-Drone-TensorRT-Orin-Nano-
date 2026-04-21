import numpy as np
import cv2
import time
import os
os.environ["YOLO_AUTOINSTALL"] = "false"

from ultralytics import YOLO

# Shared memory paths
FRAME_PATH = "/dev/shm/frame.npy"
OUTPUT_PATH = "/dev/shm/B_output.npy"
TOGGLE_PATH = "/dev/shm/active.txt"

MY_ID = "B"

model = None  # <-- start unloaded

print("Vision B running (TensorRT)")

# -------------------------
# Helper
# -------------------------
def is_active():
    try:
        with open(TOGGLE_PATH) as f:
            return f.read().strip() == MY_ID
    except:
        return False

# -------------------------
# Main loop
# -------------------------
while True:

    # -------------------------
    # INACTIVE STATE
    # -------------------------
    if not is_active():

        #unload model (frees GPU)
        if model is not None:
            print("[INFO] Unloading model B...")
            del model
            model = None

            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass

        time.sleep(0.1)
        continue

    # -------------------------
    # ACTIVE STATE
    # -------------------------
    # lazy load
    if model is None:
        print("[INFO] Loading model ...")
        model = YOLO("./vision_b/models/drone1n.engine", task="detect")
        names = model.names

    # Load frame
    try:
        frame = np.load(FRAME_PATH)
    except:
        time.sleep(0.002)
        continue

    # Inference
    try:
        results = model(frame, classes=[0], conf=0.4, verbose=False)
    except Exception as e:
        print(f"[ERROR] Inference B: {e}")
        continue

    # Draw boxes
    for r in results:
        if r.boxes is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cls, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            label = f"{names[cls]} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - h - 4), (x1 + w, y1), (255, 50, 50), -1)

            cv2.putText(frame, label, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save output
    try:
        np.save(OUTPUT_PATH, frame)
    except Exception as e:
        print(f"[WARN] Save failed: {e}")
