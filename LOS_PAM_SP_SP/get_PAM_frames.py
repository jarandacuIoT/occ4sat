import time
import os
import numpy as np
import sys
sys.path.insert(0, '..')
from picamera2 import Picamera2

# ----------------------------------------------------------------------
# Global parameters
# ----------------------------------------------------------------------
# ROI rectangle (x_start, y_start) to (x_end, y_end)
roi_x_start, roi_y_start, roi_x_end, roi_y_end,  = (249, 111, 339, 203)

# Transmitter code path
transmitter_folder = '/home/occ4sat/Documents/Multiple_Transmitter/Multiple_transmitter/'
waveform_file      = 'lib/waveform/waveform.h'

# Base output path for ROI dumps
output_base = '/home/occ4sat/Documents/videos/statistics/'

# Experiment settings
VoltageLevels  = np.arange(4094, 1, -50)  # descending by 2
N_frames       = 50                      # frames per voltage level

# PiCamV2 wrapper
picam2 = Picamera2()
video_cfg = picam2.create_video_configuration(
    main={"size": (680, 480), "format": "RGB888"},
    controls={
    "FrameRate": 120,
    "FrameDurationLimits": (8000_000, 8000_000),  # exactly 8 ms per frame → 125 fps
    "ExposureTime":   60,   # 8 ms max
    "AnalogueGain":   1.0,    # low gain to reduce sensor overhead
    },
    buffer_count=8,       # more buffers for smoother pipelining
)
picam2.configure(video_cfg)
picam2.start()

def configure_transmitter(data):
    """Write out the pulse sequence header, build & upload with PlatformIO."""
    header_path = os.path.join(transmitter_folder, waveform_file)
    with open(header_path, "w") as f:
        f.write("#ifndef NPULSE_MOD_H\n#define NPULSE_MOD_H\n")
        f.write("#include <stm32l432xx.h>\n")
        f.write(f"int NDTR_value = {len(data)//2};\n")
        f.write("const uint16_t waveform[] = {")
        f.write(",".join(str(v) for v in data))
        f.write("};\n#endif\n")        
        # Frequency
        # f = open(self.transmitter_folder+self.transmitter_file, "r")
        # lines = f.readlines()
        # lines[77] = "TIM6->PSC = " + str(self.transmitter_PSC) + "; \n"
        # lines[79] = "TIM6->ARR = " + str(self.transmitter_ARR) + "; \n"    # n is the line number you want to edit; subtract 1 as indexing of list starts from 0
        # f.close()   # close the file and reopen in write mode to enable writing to file; you can also open in append mode and use "seek", but you will have some unwanted old data if the new data is shorter in length.
        # f = open(self.transmitter_folder+self.transmitter_file, 'w')
        # f.writelines(lines)
        # # do the remaining operations on the file
        # f.close()
        # time.sleep(1)
        os.chdir(transmitter_folder)        
        print(os.getcwd())
    time.sleep(1)   # ensure file sync
    os.system("/home/occ4sat/.platformio/penv/bin/pio run -d /home/occ4sat/Documents/Multiple_Transmitter/Multiple_transmitter -e nucleo_l432kc -t upload") 


def read_frame(file_path, width, height):
    """Read raw RGB frame from disk into a (H, W, 3) array."""
    expected = width * height * 3
    with open(file_path, "rb") as f:
        raw = f.read()
    if len(raw) != expected:
        raise ValueError(f"Frame size mismatch: got {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))

def save_frame_to_file(frame, output_path):
    """Write a raw RGB888 frame (H×W×3) out to disk."""
    with open(output_path, "wb") as f:
        f.write(frame.astype(np.uint8).tobytes())


if __name__ == "__main__":
    # 1) Prepare the camera once


    # 2) Loop over each voltage level
    for vl in VoltageLevels:
        print(f"\n=== Voltage Level: {vl} ===")
        # configure transmitter at this voltage
        configure_transmitter([vl, vl])

        time.sleep(0.5)  # let sensor settle

        # create subfolder for this voltage level
        vl_folder = os.path.join(output_base, f"VL_{vl}")
        os.makedirs(vl_folder, exist_ok=True)

        # capture N_frames ROI shots
        for idx in range(N_frames):
            frame = picam2.capture_array()
            roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end, :]
            # compute max red in ROI for feedback
            red_max = roi[:, :, 0].max()
            # save ROI into its voltage-specific folder
            fname = f'ROI_ET_{video_cfg["controls"]["ExposureTime"]}_AG_{video_cfg["controls"]["AnalogueGain"]}_VL_{vl}_F{idx:02d}.rgb'
            out_path = os.path.join(vl_folder, fname)
            save_frame_to_file(roi, out_path)

            print(f"[VL {vl}] Frame {idx+1}/{N_frames}: max red = {red_max}")

    print("\nAll voltage levels complete.")
