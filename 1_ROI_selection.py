#!/usr/bin/env python3
"""
pick_roi.py
-----------
• Shows a 640×480 live preview from Picamera2.
• You drag-select a rectangle, press ENTER/SPACE to accept.
• Saves the rectangle as raw pixel coords (x1, y1, x2, y2) in roi.json.
"""

import cv2
import json
import time
import pathlib
from picamera2 import Picamera2

# 0️⃣  Start the camera in a small preview mode
picam2 = Picamera2()
preview_cfg = picam2.create_video_configuration(
    main={"size": (680, 480), "format": "RGB888"},
    controls={
    "FrameRate": 120,
    "FrameDurationLimits": (8000_000, 8000_000),  # exactly 8 ms per frame → 125 fps
    "ExposureTime":   60,   # 8 ms max
    "AnalogueGain":   1.0,    # low gain to reduce sensor overhead
    },
    buffer_count=8,       # more buffers for smoother pipelining
)
picam2.configure(preview_cfg)
picam2.start()
time.sleep(0.5)  # let AE/G settle

# 1️⃣  Capture one frame and get its true dimensions
frame = picam2.capture_array()  # BGR for OpenCV
height, width = frame.shape[:2]
print(f"Preview frame size: {width}×{height}")
print("Drag a rectangle, ENTER/SPACE = accept, ESC = cancel…")

# 2️⃣  User selects ROI
x1, y1, w, h = map(int,
    cv2.selectROI("Picamera2 preview", frame,
                  fromCenter=False, showCrosshair=True))
cv2.destroyAllWindows()

if w == 0 or h == 0:
    print("No ROI chosen, quitting.")
    picam2.stop()
    exit(0)

# 3️⃣  Compute bottom-right and save to JSON
x2 = x1 + w
y2 = y1 + h
roi_pixels = (x1, y1, x2, y2)
pathlib.Path("roi.json").write_text(json.dumps(roi_pixels))
print(f"Saved ROI corners to roi.json → {roi_pixels}")

picam2.stop()
