import cv2
import numpy as np
import time
import os

FRAME_RAW_PATH = "/dev/shm/frame_raw.npy"   # raw feed
FRAME_PATH = "/dev/shm/frame.npy"           # processed/normalized feed if needed

cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Camera started")

while True:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.01)
        continue

    # Save raw frame for viewer
    try:
        np.save(FRAME_RAW_PATH, frame)
    except Exception as e:
        print(f"[ERROR] Failed to save raw frame: {e}")

    # Also save a regular frame if needed for vision containers
    try:
        np.save(FRAME_PATH, frame)
    except Exception as e:
        print(f"[ERROR] Failed to save frame: {e}")

    time.sleep(0.01)
