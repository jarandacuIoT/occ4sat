#!/usr/bin/python3
import numpy as np
from itertools import groupby
from math import gcd
from functools import reduce
import time
import mmap
import multiprocessing as mp
import os
from collections import deque
from concurrent.futures import Future
from ctypes import CDLL, c_int, c_long, get_errno
from threading import Thread
from picamera2 import Picamera2
import sys

class Process(mp.Process):
    def __init__(self, picam2, name='main', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = picam2.camera_configuration()[name]
        self._picam2_pid = os.getpid()
        self._pid_fd = None
        self._send_queue = mp.Queue()
        self._return_queue = mp.Queue()
        self._arrays = {}
        self._return_result = False
        self._syscall = CDLL(None, use_errno=True).syscall
        self._syscall.argtypes = [c_long]
        self.start()
        self._stream = picam2.stream_map[name]
        self._requests_sent = deque()
        self._thread = Thread(target=self._return_thread, daemon=True)
        self._thread.start()

    def _return_thread(self):
        while True:
            result = self._return_queue.get()
            if not self._requests_sent:
                break
            request, future = self._requests_sent.popleft()
            future.set_result(result)
            request.release()

    def send(self, request, *args):
        plane = request.request.buffers[self._stream].planes[0]
        fd = plane.fd
        length = plane.length
        future = Future()
        request.acquire()
        self._requests_sent.append((request, future))
        self._send_queue.put((fd, length, args))
        return future

    def _map_fd(self, picam2_fd):
        if self._pid_fd is None:
            self._pid_fd = os.pidfd_open(self._picam2_pid)
        newfd = self._syscall(438, c_int(self._pid_fd),
                              c_int(picam2_fd), c_int(0))
        if newfd == -1:
            errno = get_errno()
            raise OSError(errno, os.strerror(errno))
        return newfd

    def _format_array(self, mem):
        arr = np.frombuffer(mem, dtype=np.uint8)
        w, h = self.config['size']
        stride = self.config['stride']
        fmt = self.config['format']
        if fmt == 'YUV420':
            return arr.reshape((h + h//2, stride))
        arr = arr.reshape((h, stride))
        if fmt in ('RGB888','BGR888'):
            return arr[:, :w*3].reshape((h, w, 3))
        if fmt in ('XBGR8888','XRGB8888'):
            return arr[:, :w*4].reshape((h, w, 4))
        return arr

    def capture_shared_array(self):
        if self._return_result:
            self._return_queue.put(None)
        self._return_result = True
        msg = self._send_queue.get()
        if msg == "CLOSE":
            self._return_queue.put(None)
            return None
        picam2_fd, length, self.args = msg
        if picam2_fd in self._arrays:
            return self._arrays[picam2_fd]
        fd = self._map_fd(picam2_fd)
        mem = mmap.mmap(fd, length, mmap.MAP_SHARED, mmap.PROT_READ)
        arr = self._format_array(mem)
        self._arrays[picam2_fd] = arr
        return arr

    def set_result(self, result):
        self._return_result = False
        self._return_queue.put(result)

    def run(self):
            # CONFIGURE
            neighbor_weight = 2.0
            kernel = np.array([neighbor_weight, 1.0, neighbor_weight])
            kernel /= kernel.sum()
            # N_SAMPLES = 1000
            # # center of your ROI
            # cx, cy = 290, 180
            # # size of ROI (width, height)
            # roi_w, roi_h = 20, 20

            # # pre‐compute ROI boundaries
            # x1 = cx - roi_w // 2
            # x2 = cx + roi_w // 2
            # y1 = cy - roi_h // 2
            # y2 = cy + roi_h // 2

            # intensities = []       # will hold max intensity per frame
            # max_pixel_coords = []  # will hold (x, y) of that pixel in full image coords

            # for i in range(N_SAMPLES):
            #     frame = self.capture_shared_array()  # assume shape (H, W, C) or (H, W)
            #     if frame is None:
            #         break

            #     # extract ROI
            #     patch = frame[y1:y2, x1:x2]
            #     if patch.ndim == 3:
            #         patch = patch[:, :, 0]  # pick channel 0 (or convert to grayscale first)

            #     # find flat index of max, then row/col in the patch
            #     flat_idx = np.argmax(patch)
            #     max_row, max_col = np.unravel_index(flat_idx, patch.shape)

            #     # get intensity and convert to global image coords
            #     max_intensity = float(patch[max_row, max_col])
            #     global_x = x1 + max_col
            #     global_y = y1 + max_row

            #     intensities.append(max_intensity)
            #     max_pixel_coords.append((global_x, global_y))

            #     print(f"Frame {i}: max intensity={max_intensity:.1f} at ({global_x}, {global_y})")                          # normalize to sum=1

            N_SAMPLES = 1000
            py,px = 111+45, 249+55
            # py, px = 170, 297
            chan   = 0
            threshold = 240
            # FIND NUMBER OF FRAMES REPRESENTING A SYMBOL
            intensities = []
            shortest_run = []
            for _ in range(N_SAMPLES):
                frame = self.capture_shared_array()
                if frame is None:
                    break
                intensities.append(int(frame[py, px, chan]))
                print(intensities)
                N = 7
                box = np.ones(N)/N
                smoothed = np.convolve(intensities, box, mode='same').astype(int)        
                # bits = (smoothed > threshold).astype(np.int8)         
                bits = (np.array(intensities) > threshold).astype(np.int8)                  
                first_transition = next(
                    (i for i in range(1, len(bits)) if bits[i] != bits[i-1]),
                    None
                )
                bits = bits[first_transition:]                    
                changes = np.nonzero(np.diff(bits) != 0)[0] + 1
                boundaries = np.concatenate(([0], changes, [bits.size]))
                run_lengths = np.diff(boundaries)  
                filtered = run_lengths[run_lengths <= 2 * run_lengths.min()]         
                symbol_frame =  np.floor(filtered.min()).astype(int)             
            symbol_frame = 19
            print(symbol_frame)

            # DECODE SYMBOLS
            intensities = []
            post_data = []
            found = False
            min_len = round(7 * symbol_frame)   # how many 1’s in a row to sync
            threshold = 240
            while True:
                frame = self.capture_shared_array()
                if frame is None:
                    break
                val = int(frame[py, px, chan])
                if not found:                    
                    trans_index = 0
                    # 1) still looking for the long run of 1’s
                    intensities.append(val)
                    # smooth + threshold → bits
                    N = 7
                    box = np.ones(N)/N
                    smoothed = np.convolve(intensities, box, mode='same')
                    # bits = (smoothed > threshold).astype(np.int8)
                    bits = (np.array(intensities) > threshold).astype(np.int8)                
                    # print(bits)
                    # locate runs
                    changes = np.nonzero(np.diff(bits) != 0)[0] + 1
                    boundaries = np.concatenate(([0], changes, [bits.size]))
                    run_lengths = np.diff(boundaries)

                    # scan for the first long-enough block of 1’s
                    for i, length in enumerate(run_lengths):
                        start = boundaries[i]
                        end   = boundaries[i+1]
                        if bits[start] == 1 and length >= min_len:
                            print(f"Found 1’s sync block at idx {start}–{end-1} (length {length})")
                            found = True
                            # start collecting *after* this block
                            # initialize post_data with any frames we already read beyond `end`
                            post_data = intensities[end:]
                            break

                else:
                    # print(post_data)
                    # 2) after sync, collect into post_data
                    post_data.append(val)
                    N = 7
                    box = np.ones(N)/N
                    smoothed = np.convolve(post_data, box, mode='same')
                    # bits = (smoothed > threshold).astype(np.int8)
                    bits = (np.array(post_data) > threshold).astype(np.int8)                       
                    results = []
                    if self.find_first_transition(bits)[0]:
                        bits = bits [self.find_first_transition(bits)[1]:]
                        if len(bits)>=2*symbol_frame: #Value depend on order of modulation
                            # print(post_data[-int(4*symbol_frame):])
                            data = np.array(post_data[-int(2*symbol_frame):])
                            print(data)
                            levels      = np.array([247.0, 235.5, 222.5, 208.5, 200.5, 193.5, 182.5, 161.5, 154.5, 145.0, 127.0, 107.5, 77.5, 65.5, 36.0])
                            class_label = (np.arange(15))  # whatever labels you want
                            idx = np.abs(data[:,None] - levels[None,:]).argmin(axis=1)
                            # map back to your class labels
                            labels = class_label[idx]                        
                            for i in range(0, labels.size, symbol_frame):
                                chunk = labels[i : i + symbol_frame]
                                print(chunk)
                                results.append(np.floor(np.mean(chunk) + 0.5).astype(int))
                            print(results)
                            binary_strings = [format(int(x), '04b') for x in results]
                            bit_string = ''.join(binary_strings)
                            print(bit_string)                        
                            ascii_char = chr(int(bit_string, 2))
                            print(ascii_char)
                            N_SAMPLES = 0
                            found = False
                            intensities = []
                            post_data = []

    def close(self):
        self._send_queue.put("CLOSE")
        self._thread.join()
        self.join()
        super().close()

    def find_first_transition(self, lst):
        """
        Returns a tuple (flag, index) where:
        - flag is True if a transition is found, False otherwise
        - index is the 0-based position of the first transition (or None)
        """
        i=-1
        for i in range(1, len(lst)):
            if lst[i] != lst[i-1]:
                return True, i
        return False, i

if __name__ == "__main__":
    # 1️⃣  Start camera with a VIDEO config at 320×240, 60 fps
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

    # 2️⃣  Spawn the zero-copy worker
    worker = Process(picam2, name="main")

    # 3️⃣  Fire frames as fast as you can, without waiting on each one
    while True:
        with picam2.captured_request() as req:
            worker.send(req)
        # no future.result() here → you won’t throttle the camera

    # 4️⃣  Clean up
    worker.close()
    picam2.stop()
    print("Done.")
