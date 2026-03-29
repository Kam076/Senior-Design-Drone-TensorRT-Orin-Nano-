import cv2
import numpy as np
import time
import os

# Shared memory paths
TOGGLE_PATH = "/dev/shm/active.txt"
OUTPUT_C = "/dev/shm/C_output.npy"
OUTPUT_D = "/dev/shm/D_output.npy"
OUTPUT_RAW = "/dev/shm/frame_raw.npy"  # unprocessed camera feed

WIDTH = 1280
HEIGHT = 720
CHANNELS = 3

KEY_MAP = {
    ord('0'): 'RAW',   # Raw camera feed
    ord('1'): 'C',     # Vision C (people)
    ord('2'): 'D'      # Vision D (objects)
}

# Ensure toggle file exists
if not os.path.exists(TOGGLE_PATH):
    with open(TOGGLE_PATH, "w") as f:
        f.write("RAW")

print("\nControls:")
print("  0 → Raw Camera")
print("  1 → People Detection (C)")
print("  2 → Object Detection (D)")
print("  q → Quit\n")

def read_frame(path):
    """Read a raw numpy frame from shared memory."""
    try:
        frame = np.load(path)
        return frame
    except:
        return None

while True:
    # --- Handle keyboard ---
    key = cv2.waitKey(1) & 0xFF
    if key in KEY_MAP:
        with open(TOGGLE_PATH, "w") as f:
            f.write(KEY_MAP[key])
        print(f"[CTRL] Active Vision → {KEY_MAP[key]}")

    if key == ord('q'):
        break

    # --- Determine active mode ---
    try:
        with open(TOGGLE_PATH, "r") as f:
            active = f.read().strip()
    except:
        active = "RAW"

    # --- Read the correct frame ---
    if active == "C":
        frame = read_frame(OUTPUT_C)
    elif active == "D":
        frame = read_frame(OUTPUT_D)
    else:  # RAW camera feed
        frame = read_frame(OUTPUT_RAW)

    # --- Display ---
    if frame is not None:
        cv2.putText(
            frame,
            f"Active: {active}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.imshow("Vision Viewer", frame)
    else:
        blank = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        cv2.putText(
            blank,
            f"{active} frame not ready",
            (WIDTH // 4, HEIGHT // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )
        cv2.imshow("Vision Viewer", blank)

    time.sleep(0.01)

cv2.destroyAllWindows()
