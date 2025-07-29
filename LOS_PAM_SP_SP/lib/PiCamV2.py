import time
import logging
import os
from picamera2 import Picamera2

logging.basicConfig(level=logging.INFO)


class RpiCamV2:
    def __init__(self):
        self.picam2 = Picamera2()
        print("RpiCamV2 initialized")

    def configure_video(self, Resolution=(1920, 1080), Format='RGB888', FrameRate=30):
        """Configure the camera for capturing a single frame"""
        video_config = self.picam2.create_still_configuration(
            {"size": Resolution, "format": Format},
            controls={"FrameRate": FrameRate}
        )
        self.picam2.configure(video_config)

    def configure_controls(self, exposure_time, analogue_gain=8):
        """Configure camera exposure and gain"""
        # Set exposure time
        min_exp, max_exp, default_exp = self.picam2.camera_controls["ExposureTime"]
        self.picam2.controls.ExposureTime = exposure_time
        logging.info("Exposure time set: %s", exposure_time)

        # Set analogue gain
        min_ag, max_ag, default_ag = self.picam2.camera_controls["AnalogueGain"]
        self.picam2.controls.AnalogueGain = analogue_gain
        logging.info("Analogue Gain set: %s", analogue_gain)

        logging.info("Camera controls: %s", self.picam2.controls)

    def capture_frame(self, folder_name, file_name):
        """Capture and save a single frame as RGB"""
        # Validate folder
        if not os.path.exists(folder_name):
            logging.error("Folder does not exist")
            return

        file_path = os.path.join(folder_name, file_name)
        if os.path.exists(file_path):
            logging.info("Rewriting file: %s", file_path)

        # Start the camera and wait for it to adjust
        self.picam2.start()
        time.sleep(2)  # Allow the camera to adjust before capturing

        # Capture the frame
        frame = self.picam2.capture_array()

        # Save the frame as an RGB file
        with open(file_path, "wb") as f:
            f.write(frame.tobytes())

        self.picam2.stop()
        logging.info("Frame captured and saved: %s", file_path)


class RpiCamV2Config:
    def __init__(self):
        print("RpiCamV2Config initialized")

    def configure(self):
        print("RpiCamV2 configured")


if __name__ == "__main__":
    # Initialize the camera
    cam = RpiCamV2()

    # Configure the camera
    cam.configure_video()
    cam.configure_controls(exposure_time=5000, analogue_gain=8)  # Exposure in microseconds

    # Capture a single frame
    folder_name = "/home/occ4sat/Documents/videos/"
    file_name = "single_frame.rgb"
    cam.capture_frame(folder_name, file_name)
