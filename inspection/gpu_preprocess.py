"""GPU implementation of the deployment preprocessing path.

WHY THIS EXISTS
    The supervised segmenter's forward pass is 13-22 ms. The CPU preprocessing
    in front of it is 279 ms. Measured stage costs at 2200x2048 on 20 cores:

        content_bbox     75 ms   full-resolution numpy mean/std reductions
        illum_correct   176 ms   cvtColor + float32 add/clip at full resolution
        pad + resize     24 ms
        upload            3 ms

    Raising cv2.setNumThreads from 1 to 20 changes this by 0.5 ms -- the hot
    stages are numpy reductions and float32 array arithmetic and never enter
    OpenCV's thread pool. The setNumThreads(1) line in enhance.py is therefore
    irrelevant to deployment and still required for the training DataLoader.

    What does work is noticing that the model only ever sees 768x768, while
    every one of those stages runs at 4.5 megapixels. INTER_AREA resize is
    linear, so resize(img + offset) == resize(img) + resize(offset) exactly:
    the illumination correction can be applied after the downsample instead of
    before it, on 7.6x fewer pixels. And the GPU is idle for 90% of the frame.

    Measured: 279 ms -> 22 ms, and the detections are unchanged (see
    scripts/validate_fast_preprocess.py).

EQUIVALENCE, NOT APPROXIMATION
    This reproduces preprocessing/enhance.py bit-for-bit wherever it can, and
    the places it cannot are documented at the point they occur. The three that
    took measurement to find:

    * The offset must be masked to the crop BEFORE the area resize. Masking
      after is wrong on the single row and column where the crop meets the
      black padding -- ~590 of 768^2 pixels. Invisible in a mean deviation,
      but it was the whole p99.9 deviation (31.8 -> 4.5 levels).
    * cv2.cvtColor returns uint8, so grayscale is ROUNDED before the median is
      taken. The median is a global offset added to every pixel in the frame.
    * np.median averages the two central values of an even-length array;
      torch.median returns the lower one. Grayscale is integer-valued, so a
      256-bin histogram reproduces numpy's definition exactly and for free.

    torch.std defaults to ddof=1 and numpy's to ddof=0 -- content_bbox compares
    per-column standard deviations against a threshold of 4.0, so unbiased=False
    is not optional here.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

FIELD_SIDE = 512          # illumination is estimated at this resolution
SIGMA_FRAC = 0.08         # blur sigma as a fraction of that resolution
_KERNEL_CACHE: dict = {}


def _gaussian1d(sigma, device):
    """OpenCV's own kernel, so this is cv2.GaussianBlur rather than something
    that resembles it. For ksize=0 on float32 input OpenCV picks
    ksize = round(sigma*4)*2+1, forced odd."""
    key = (round(float(sigma), 4), str(device))
    if key not in _KERNEL_CACHE:
        ksize = int(round(float(sigma) * 4) * 2 + 1) | 1
        k = cv2.getGaussianKernel(ksize, float(sigma)).astype(np.float32).ravel()
        _KERNEL_CACHE[key] = torch.from_numpy(k).to(device)
    return _KERNEL_CACHE[key]


def _blur(x, sigma):
    """Separable Gaussian with cv2's kernel and cv2's default border handling
    (BORDER_REFLECT_101, which is torch's 'reflect')."""
    k = _gaussian1d(sigma, x.device)
    pad = k.numel() // 2
    x = F.conv2d(F.pad(x, (pad, pad, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return F.conv2d(F.pad(x, (0, 0, pad, pad), mode="reflect"), k.view(1, 1, -1, 1))


def _median_uint8(gray):
    """np.median semantics for integer-valued data, via a 256-bin histogram."""
    cum = torch.bincount(gray.reshape(-1).to(torch.long), minlength=256).cumsum(0)
    n = gray.numel()
    idx = torch.tensor([(n - 1) // 2 + 1, n // 2 + 1], device=cum.device)
    lo, hi = torch.searchsorted(cum, idx).tolist()
    return (lo + hi) / 2.0


def _content_bbox(gray, threshold=15, min_texture_std=4.0):
    """content_bbox, with the four full-resolution reductions done on the GPU.

    Only the reductions move: the trim loops are sequential and operate on
    vectors of a few thousand floats, so they stay on the CPU where a Python
    loop over them costs microseconds. Statistics are exact -- subsampling the
    rows by 4 was tried and moved the crop edge by one column on real frames,
    which shifts all downstream geometry."""
    H, W = gray.shape[-2:]
    g = gray.reshape(H, W)
    col_mean, row_mean = g.mean(0), g.mean(1)
    col_std = g.std(0, unbiased=False)      # numpy's default is ddof=0
    row_std = g.std(1, unbiased=False)
    packed = torch.cat([col_mean, row_mean, col_std, row_std]).cpu().numpy()
    col_mean, row_mean = packed[:W], packed[W:W + H]
    col_std, row_std = packed[W + H:2 * W + H], packed[2 * W + H:]

    cols = np.where(col_mean > threshold)[0]
    rows = np.where(row_mean > threshold)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, W, H
    x1, y1 = int(cols[0]), int(rows[0])
    x2, y2 = int(cols[-1]) + 1, int(rows[-1]) + 1
    while x1 < x2 - 1 and col_std[x1] < min_texture_std:
        x1 += 1
    while x2 > x1 + 1 and col_std[x2 - 1] < min_texture_std:
        x2 -= 1
    while y1 < y2 - 1 and row_std[y1] < min_texture_std:
        y1 += 1
    while y2 > y1 + 1 and row_std[y2 - 1] < min_texture_std:
        y2 -= 1
    return x1, y1, x2, y2


def _to_gray(img):
    """cv2's BGR2GRAY, including the fact that it returns uint8 and therefore
    rounds. Downstream consumers (the median, and the bbox thresholds) see the
    rounded values, so rounding here is part of the definition."""
    return (img[:, 0:1] * 0.114 + img[:, 1:2] * 0.587 + img[:, 2:3] * 0.299).round_()


@torch.no_grad()
def preprocess(frame_bgr, device, input_size=768):
    """Camera frame (HxWx3 uint8 BGR) -> (1,3,S,S) float tensor in [0,1],
    plus the crop box and geometry the caller needs to map results back.

    Returns (tensor, (x1, y1, x2, y2), (height, width, side)).
    """
    img = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(device, non_blocking=True)
    img = img.permute(2, 0, 1).unsqueeze(0).float()
    gray_full = _to_gray(img)

    x1, y1, x2, y2 = _content_bbox(gray_full)
    img = img[:, :, y1:y2, x1:x2]
    gray = gray_full[:, :, y1:y2, x1:x2]
    h, w = img.shape[-2:]
    side = max(h, w)

    target = _median_uint8(gray)
    scale = FIELD_SIDE / side
    fh, fw = max(1, int(h * scale)), max(1, int(w * scale))
    field = _blur(F.interpolate(gray, size=(fh, fw), mode="area"),
                  max(fh, fw) * SIGMA_FRAC)
    field = F.interpolate(field, size=(h, w), mode="bilinear", align_corners=False)

    # Channel 3 carries the offset, already zero outside the crop, so the area
    # resize below computes resize(offset * valid) -- see the module docstring.
    # The -0.5 emulates numpy's astype(uint8), which truncates rather than
    # rounds; the reference does that at full resolution before the resize, and
    # the model was trained on data that went through it.
    canvas = torch.zeros((1, 4, side, side), device=device)
    canvas[:, :3, :h, :w] = img
    canvas[:, 3:4, :h, :w] = target - field - 0.5
    stack = F.interpolate(canvas, size=(input_size, input_size), mode="area")
    out = (stack[:, :3] + stack[:, 3:4]).clamp_(0, 255)
    return out / 255.0, (x1, y1, x2, y2), (h, w, side)
