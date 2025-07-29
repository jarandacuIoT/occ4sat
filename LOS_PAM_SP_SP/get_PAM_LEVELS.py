import sys
import math
import numpy as np
from pathlib import Path

# Configure the folder path and ROI parameters here:
ROOT_DIR = Path("/home/occ4sat/Documents/videos/statistics/")
X_CENTER = 15       # set to None to auto-center horizontally
Y_CENTER = 15       # set to None to auto-center vertically
CHANNELS = 3          # number of channels in .rgb files
SELECT_CHANNEL = 0    # index of the channel to process (0-based). Set to None for all channels.
TARGET_OVERLAP = 3.8e-3 # desired Gaussian overlap threshold
FRAME_COUNT = 1    # number of frames (0 to N) to include in ROI computation, or None for all


def detect_image_shape(file_path: Path, channels: int = CHANNELS):
    size = file_path.stat().st_size
    if size % channels != 0:
        raise ValueError(f"File size {size} not divisible by {channels} channels.")
    total_pixels = size // channels
    factors = []
    max_h = int(math.isqrt(total_pixels))
    for h in range(1, max_h + 1):
        if total_pixels % h == 0:
            w = total_pixels // h
            factors.append((h, w))
    candidates = [(h, w) for h, w in factors if w >= h]
    target_ars = [16/9, 4/3, 3/2, 1]
    def ar_score(pair):
        h, w = pair
        ar = w / h
        return min(abs(ar - t) for t in target_ars)
    height, width = min(candidates, key=ar_score)
    return width, height


def load_frame(file_path: Path, width: int, height: int) -> np.ndarray:
    data = np.fromfile(file_path, dtype=np.uint8)
    expected = width * height * CHANNELS
    if data.size != expected:
        raise ValueError(f"{file_path.name} size mismatch: {data.size} vs {expected}")
    img = data.reshape((height, width, CHANNELS))
    return img[:, :, SELECT_CHANNEL] if SELECT_CHANNEL is not None else img


def compute_folder_roi_stats(folder: Path, width: int, height: int, x: int, y: int, r: int, frame_count=None):
    all_files = sorted(folder.glob('*_F*.rgb'))
    if frame_count is not None:
        selected = []
        for f in all_files:
            idx_str = f.stem.split('F')[-1]
            try:
                idx = int(idx_str)
                if 0 <= idx <= frame_count:
                    selected.append(f)
            except ValueError:
                continue
        files = selected
    else:
        files = all_files
    if not files:
        raise ValueError(f"No frame files selected in {folder}")
    roi_pixels = []
    for rgb_file in files:
        frame = load_frame(rgb_file, width, height)
        H, W = frame.shape[:2]
        Y, X = np.ogrid[:H, :W]
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        roi_pixels.append(frame[mask])
    all_pixels = np.concatenate(roi_pixels)
    return all_pixels.mean(), all_pixels.std(ddof=1)


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma)**2
    return coeff * np.exp(exponent)


def gaussian_overlap(mu1: float, sigma1: float,
                     mu2: float, sigma2: float,
                     num_points: int = 200_000,
                     span: float = 10.0) -> float:
    lo = min(mu1, mu2) - span * max(sigma1, sigma2)
    hi = max(mu1, mu2) + span * max(sigma1, sigma2)
    x = np.linspace(lo, hi, num_points)
    pdf1 = gaussian_pdf(x, mu1, sigma1)
    pdf2 = gaussian_pdf(x, mu2, sigma2)
    return np.trapz(np.minimum(pdf1, pdf2), x)


def find_vl_sequence_for_overlap(root_dir: Path, target: float, RADIUS,x , y):
    # Determine image dimensions and ROI center
    sample = next(root_dir.rglob('*.rgb'), None)
    if sample is None:
        print("No .rgb files found.")
        return
    width, height = detect_image_shape(sample)
    x_c = x if X_CENTER is not None else width // 2
    y_c = y if Y_CENTER is not None else height // 2

    # Gather and sort VL_* folders descending
    vl_dirs = sorted(
        [d for d in root_dir.glob('VL_*') if d.is_dir()],
        key=lambda d: int(d.name.split('_')[1]), reverse=True
    )
    if not vl_dirs:
        print("No VL_* folders found.")
        return

    sequence = []
    ref_idx = 0  # start with highest VL
    ref_mu, ref_sigma = compute_folder_roi_stats(
        vl_dirs[ref_idx], width, height, x_c, y_c, RADIUS, FRAME_COUNT)
    # print(f"Reference {vl_dirs[ref_idx].name}: mu={ref_mu:.3f}, sigma={ref_sigma:.3f}")
    sequence.append(vl_dirs[ref_idx].name)

    # Iteratively find next VL where overlap <= target
    while True:
        found = False
        for idx in range(ref_idx + 1, len(vl_dirs)):
            mu, sigma = compute_folder_roi_stats(
                vl_dirs[idx], width, height, x_c, y_c, RADIUS, FRAME_COUNT)
            ov = gaussian_overlap(ref_mu, ref_sigma, mu, sigma)

            if ov <= target and mu <= ref_mu:
                # print(f"Found next VL: {vl_dirs[idx].name} with overlap={ov:.6f} <= {target}")
                print(f"Selected {vl_dirs[idx].name}: mean intensity = {mu:.3f} (overlap={ov:.6f})")                
                sequence.append(vl_dirs[idx].name)
                ref_idx = idx
                ref_mu, ref_sigma = mu, sigma
                found = True
                break
        if not found:
            break
    print("\nSequence of VLs meeting target overlap:")
    print(len(sequence))
    for name in sequence:
        print(f" - {name}")
        


def main():
    Max_Radius = 1
    for r in range(0,Max_Radius):
        for x in range (30,89):
            for y in range (30,91):
                print("Radius: " + str(r))
                print("x: " + str(x))
                print("y: " + str(y))
                find_vl_sequence_for_overlap(ROOT_DIR, TARGET_OVERLAP, r, x, y)
if __name__ == '__main__':
    main()
