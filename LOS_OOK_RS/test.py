from picamera2 import Picamera2

picam2 = Picamera2()
# You can find the current sensor mode here: picam2.current_mode
# Then you can list them to find min/max exposure for each mode
print(picam2.sensor_modes)