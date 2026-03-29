from ultralytics import YOLO

# 1. Load your ONNX model
model = YOLO("./vision_c/models/yolov8s.pt")

# 2. Export the model to TensorRT
# Use half=True for FP16, int8=True for INT8 (requires calibration)
model.export(format="engine", device=0, half=True)  # creates 'your_model.engine'
