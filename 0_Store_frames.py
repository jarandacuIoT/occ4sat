#!/usr/bin/env python3
"""
capture_and_save.py
-------------------
• Captures 10 full‐resolution frames with Picamera2.
• Saves each frame to disk as a PNG.
"""

import time
import cv2
from picamera2 import Picamera2

# Initialize camera
picam2 = Picamera2()

# Get the sensor's native resolution
sensor_w, sensor_h = picam2.sensor_resolution

# Configure for full‐resolution stills
still_cfg = picam2.create_video_configuration(
    main={"size": (680, 480), "format": "RGB888"},
    controls={
    "FrameRate": 120,
    "FrameDurationLimits": (8000_000, 8000_000),  # exactly 8 ms per frame → 125 fps
    "ExposureTime":   60,   # 8 ms max
    "AnalogueGain":   1.0,    # low gain to reduce sensor overhead
    },
    buffer_count=8,       # more buffers for smoother pipelining
)
picam2.configure(still_cfg)

# Start and let exposure settle
picam2.start()
time.sleep(0.5)

# Capture 10 frames
frames = []
for i in range(10):
    frame = picam2.capture_array()  # BGR NumPy array
    frames.append(frame)
print(f"Captured {len(frames)} frames ({sensor_w}×{sensor_h})")

# Save each frame as a PNG
for idx, frame in enumerate(frames):
    fname = f"frame_{idx:02d}.png"
    cv2.imwrite(fname, frame)
    print(f"Saved {fname}")

# Clean up
picam2.stop()
