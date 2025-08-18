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
import cv2
import time 
import string

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
        # TODO: Compute BER
        # TODO: COMPUTE GMM
        # TODO: COMPUTE SNR.
            output = []
            # CONFIGURE
            neighbor_weight = 2.0
            kernel = np.array([neighbor_weight, 1.0, neighbor_weight])
            kernel /= kernel.sum()              # normalize to sum=1
            N_SAMPLES = 0
            x_start, y_start, x_end, y_end = (340,0,420,479)
            channel = 0
            threshold = 4

            #TODO: CAPTURE MORE FAMES 
            # FIND NUMBER OF FRAMES REPRESENTING A SYMBOL
            intensities = []
            shortest_run = []
            for _ in range(N_SAMPLES):
                frame = self.capture_shared_array()
                if frame is None:
                    break
                sub = frame[y_start:y_end, x_start:x_end, :]
                gray = sub.mean(axis=2)                  
                row_means = gray.mean(axis=1)
                print(row_means.astype(int))
                binary_means = (row_means > threshold).astype(np.uint8) * 255
                print(binary_means)
                cv2.imwrite("asd" + str(_) + ".png", binary_means)                
                symbol_frame = int(round(longest_consecutive_ones(binary_means)/8))
            symbol_frame = 17              
            print(symbol_frame)

            # DECODE SYMBOLS
            min_len = round(4.5 * symbol_frame)   # how many 1’s in a row to sync
            thresh = 10
            # Precompute constants once
            roi_w = x_end - x_start
            roi_h = y_end - y_start
            channels = 3  # RGB888
            thresh_scaled = thresh * roi_w * channels  # compare sums to this (no division)
            out_bytes = bytearray()

            while True:
                frame = self.capture_shared_array()
                if frame is None:
                    break

                # 1) ROI view (no copy) and row sum over channels & columns (uint32 to prevent overflow)
                roi = frame[y_start:y_end, x_start:x_end, :]
                row_sums = roi.sum(axis=(1, 2), dtype=np.uint32)  # shape (roi_h,)

                # 2) Threshold without division
                binary_means = row_sums > thresh_scaled  # bool array length roi_h

                # 3) Find start index of block (first 0 after ≥min_len of 1s)
                idx = first_zero_after_at_least_n_ones(binary_means, min_len)
                if idx is None or idx < 0:
                    continue

                # 4) Slice exactly the bits for one character (8 symbols)
                total_needed = 8 * symbol_frame
                end_idx = idx + total_needed
                if end_idx > binary_means.size:
                    continue  # not enough samples yet

                data = binary_means[idx:end_idx]

                # 5) Majority per symbol (vectorized)
                #    reshape to (8, symbol_frame) → sum along axis=1 → majority
                sym = data.reshape(8, symbol_frame)
                # Using sum is faster than mean; majority threshold at > symbol_frame//2
                bits = (sym.sum(axis=1) > (symbol_frame // 2)).astype(np.uint8)  # shape (8,)

                # 6) Convert 8 bits → one byte (vectorized)
                # Option A (fastest & clean): packbits
                byte_val = np.packbits(bits, bitorder='big')[0]
                # Option B (also fast): dot with bit weights
                # byte_val = int(bits @ BIT_WEIGHTS)

                out_bytes.append(int(byte_val))

                # 7) Optional: batch-print every N chars to reduce I/O overhead
                if len(out_bytes) >= 1000:
                    output = out_bytes.decode('latin1', errors='ignore')  # raw 1:1 bytes→chars
                    print(list(output))
                    print("Total bits" , print(len(output)))
                    report = analyze_alphabet_cycles(output)[1:-1]  # drop last incomplete cycle
                    print(report)
                    total_missing = sum(len(entry["missing"]) for entry in report)
                    print("Total missing:", total_missing)
                    return output,analyze_alphabet_cycles
                    break

            # while True:
            #     frame = self.capture_shared_array()
            #     if frame is None:
            #         break
            #     row_means = frame[y_start:y_end, x_start:x_end, :].mean(axis=(1, 2))        
            #     binary_means = row_means > thresh
            #     index_trans = int(first_zero_after_at_least_n_ones(binary_means, min_len))
            #     data = binary_means[index_trans:index_trans+8*symbol_frame]
            #     results = []

                # for i in range(0, len(data), symbol_frame):
                #     chunk = data[i : i + symbol_frame]   
                #     results.append(int(np.median(chunk)))
                # bit_string = ''.join(str(int(x)) for x in results)
                # ascii_char = chr(int(bit_string, 2))                    
                # output.append(ascii_char)
                # if len(output)==1000:
                #     print(output)
                #     print((analyze_alphabet_cycles(output))[:-1])
                #     break
                    # print(time.time())
                # print(data)


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
        for i in range(1, len(lst)):
            if lst[i] != lst[i-1]:
                return True, i
        return False, i
import numpy as np

def first_zero_after_at_least_n_ones(arr: np.ndarray, n: int) -> int:
    """
    Return the index of the first 0 that comes right after a run of at least
    n consecutive nonzero values in a 1D array, or -1 if no such transition.
    """
    # binarize to 0/1
    a = (arr != 0).astype(int)
    # pad so edge‐runs get detected
    p = np.pad(a, (1,1), mode='constant', constant_values=0)
    # +1 marks run starts, -1 marks run ends (zero after the run)
    d = np.diff(p)
    starts = np.where(d ==  1)[0]
    ends   = np.where(d == -1)[0]
    # lengths of each run
    lengths = ends - starts
    # find the first run >= n
    valid = ends[lengths >= n]
    return int(valid[0]) if valid.size > 0 else -1

def longest_consecutive_ones(arr: np.ndarray) -> int:
    """
    Compute the length of the longest run of nonzero values in a 1D array.
    Works for arrays of 0/1, 0/255, True/False, etc.
    """
    # binarize: any nonzero → 1
    a = (arr != 0).astype(np.int64)

    # pad with 0 at both ends
    p = np.pad(a, (1,1), mode='constant', constant_values=0)

    # diffs: +1 at run starts, -1 at run ends
    d = np.diff(p)
    starts = np.where(d ==  1)[0]
    ends   = np.where(d == -1)[0]

    # trim to equal lengths in pathological cases
    n = min(len(starts), len(ends))
    if n == 0:
        return 0

    # compute run lengths
    lengths = ends[:n] - starts[:n]
    return int(lengths.max())

import string
from typing import List, Dict, Any, Tuple

def analyze_alphabet_cycles(received: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze a stream of characters where the sender repeatedly transmits a..z.

    Returns list of dicts with only:
      - missing: list of missing letters
      - substitutions: list of (expected, got)
      - duplicates: list of duplicated letters
    """
    alphabet = string.ascii_lowercase
    idx = {c: i for i, c in enumerate(alphabet)}

    # Normalize + filter to a..z
    clean: List[str] = []
    for ch in received:
        if not ch:
            continue
        c = str(ch)[0].lower()
        if c in idx:
            clean.append(c)

    out: List[Dict[str, Any]] = []
    if not clean:
        return out

    def next_letter(c: str) -> str:
        j = idx[c]
        return alphabet[j + 1] if j < 25 else None

    # State
    in_cycle = False
    expected = 'a'
    cycle = None
    last_accepted = None

    def start_cycle():
        return {"missing": [], "substitutions": [], "duplicates": []}

    def close_cycle():
        out.append(cycle)

    for c in clean:
        if not in_cycle:
            if c == 'a':
                in_cycle = True
                expected = 'b'
                cycle = start_cycle()
                last_accepted = 'a'
            continue

        if expected is None:
            if c == 'a':
                close_cycle()
                in_cycle = True
                expected = 'b'
                cycle = start_cycle()
                last_accepted = 'a'
            continue

        if c == last_accepted:
            cycle["duplicates"].append(c)
            continue

        if c == expected:
            last_accepted = c
            expected = next_letter(c)
            continue

        if c == 'a':
            cur = expected
            while cur is not None:
                cycle["missing"].append(cur)
                cur = next_letter(cur)
            close_cycle()
            in_cycle = True
            expected = 'b'
            cycle = start_cycle()
            last_accepted = 'a'
            continue

        if idx[c] > idx[expected]:
            for j in range(idx[expected], idx[c]):
                cycle["missing"].append(alphabet[j])
            last_accepted = c
            expected = next_letter(c)
        else:
            cycle["substitutions"].append((expected, c))
            last_accepted = expected
            expected = next_letter(last_accepted)

    if in_cycle and cycle is not None:
        cur = expected
        while cur is not None:
            cycle["missing"].append(cur)
            cur = next_letter(cur)
        close_cycle()

    return out


if __name__ == "__main__":
    # 1️⃣  Start camera with a VIDEO config at 320×240, 60 fps
    picam2 = Picamera2()
    video_cfg = picam2.create_video_configuration(
    main={"size": (680, 480), "format": "RGB888"},
    controls={
    "FrameRate": 300,
    # "FrameDurationLimits": (8000_000, 8000_000),  # exactly 8 ms per frame → 125 fps
    "ExposureTime":   1,   # 8 ms max
    "AnalogueGain":   1.0,    # low gain to reduce sensor overhead
    },
    buffer_count=8,       # more buffers for smoother pipelining
)
    picam2.configure(video_cfg)
    print(video_cfg["controls"])
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
