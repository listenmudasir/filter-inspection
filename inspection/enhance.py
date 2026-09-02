"""
Classical, GPU-free preprocessing that makes defects "pop" before the
backbone ever sees the image. Validated on the real dataset (see
<INSTALL_DIR>(修正)/analysis/): local-contrast gave up
to 3.4x SNR gain on Foreign_Body, the class raw-pixel YOLO detected 0% of.

Both filters operate on the ORIGINAL resolution grayscale image (not the
downsized model input) so thin structures (hair-shaped Foreign_Body strands)
aren't destroyed by resizing before the filter ever runs.
"""
import cv2
import numpy as np

# DataLoader worker processes (num_workers>1) each inherit OpenCV's default
# internal multithreading -- every worker then tries to use ALL CPU cores
# for its own cv2 calls simultaneously, causing severe oversubscription.
# Measured: one training epoch took 27+ CPU-minutes per worker before this,
# on data that should take seconds. Cap each worker to 1 cv2 thread; the
# DataLoader's own multiprocessing already provides the parallelism.
cv2.setNumThreads(1)


def content_bbox(img_bgr: np.ndarray, threshold: int = 15, min_texture_std: float = 4.0):
    """Bounding box of actual content, excluding the black letterbox border
    around the cylindrical filter surface. Confirmed real and consistent on
    this camera's images (~313px / ~7.6% of width, right side) -- not a
    hypothetical. Matters a lot for Branch A specifically: its image-level
    score is the MAX over all patches (standard PatchCore design), so a
    single border patch -- wildly unlike anything in the clean-texture bank
    -- can dominate the whole-image anomaly score regardless of whether a
    real defect is present. This was very likely why the calibration step's
    per-patch score distribution (mean=0.037) and eval.py's per-image max
    score (mean=0.916 on the SAME clean images) were 25x apart, and why an
    early test flagged an entire 4096x3650 frame as one giant "defect".

    Investigated a suspected leftover-border case on
    Image_20260724145046260.jpg (EfficientAD map_st spiked 12.5 vs 0.35 on
    a normal tile there): turned out NOT to be a cropping bug -- the last
    ~40 columns before the detected edge ramp smoothly from 159 to 15, a
    genuine vignetting falloff at the true edge of the material, not a
    residual flat-black strip. threshold=15 is correctly finding where
    material ends. The texture-variance trim below is a legitimate
    defense against an actual flat leftover border (kept since it's a
    real failure mode elsewhere), but the edge-gradient anomaly-score
    spike itself needs fixing at the SCORING stage, not here -- see
    border_margin handling in infer_gated.py, which reuses the same
    border-exclusion pattern already proven for this in the memory_bank
    branch (border_margin=1 patch exclusion, AUROC 0.42->0.78).
    Returns (x1, y1, x2, y2) in pixel coords of the input image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    col_means = gray.mean(axis=0)
    row_means = gray.mean(axis=1)
    cols = np.where(col_means > threshold)[0]
    rows = np.where(row_means > threshold)[0]
    if len(cols) == 0 or len(rows) == 0:
        return 0, 0, gray.shape[1], gray.shape[0]
    x1, y1, x2, y2 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1

    col_stds = gray.std(axis=0)
    row_stds = gray.std(axis=1)
    while x1 < x2 - 1 and col_stds[x1] < min_texture_std:
        x1 += 1
    while x2 > x1 + 1 and col_stds[x2 - 1] < min_texture_std:
        x2 -= 1
    while y1 < y2 - 1 and row_stds[y1] < min_texture_std:
        y1 += 1
    while y2 > y1 + 1 and row_stds[y2 - 1] < min_texture_std:
        y2 -= 1
    return x1, y1, x2, y2


def crop_to_content(img_bgr: np.ndarray, mask: np.ndarray = None, threshold: int = 15):
    """Crop the letterbox border off img (and mask, if given, using the SAME
    box so pixel alignment is preserved). Returns (img, mask, (x1,y1,x2,y2))
    -- the box is always returned so callers can map results back to
    original-image coordinates."""
    x1, y1, x2, y2 = content_bbox(img_bgr, threshold)
    img_c = img_bgr[y1:y2, x1:x2]
    mask_c = mask[y1:y2, x1:x2] if mask is not None else None
    return img_c, mask_c, (x1, y1, x2, y2)


def local_contrast_map(gray: np.ndarray, scales) -> np.ndarray:
    """Multi-scale ring/band-pass contrast: |inner local mean - outer local
    mean|, max over scales. Robust to Stain's variable spread (one fixed
    scale favors either small tight stains or large diffuse ones, not both).
    """
    gray_f = gray.astype(np.float32)
    out = np.zeros_like(gray_f)
    for inner_r, outer_r in scales:
        inner = cv2.blur(gray_f, (inner_r * 2 + 1, inner_r * 2 + 1))
        outer = cv2.blur(gray_f, (outer_r * 2 + 1, outer_r * 2 + 1))
        out = np.maximum(out, np.abs(inner - outer))
    return out


def ridge_strength_map(gray: np.ndarray, sigmas) -> np.ndarray:
    """Multi-scale Hessian ridge/line detector (simplified Frangi). Catches
    thin curvilinear structures (hair-shaped defects) that the isotropic
    local-contrast filter under-serves -- box filters wash out a 1px-wide
    line's thin trailing end (see analysis/images/lcm_previews/Hair_*).
    """
    gray_f = gray.astype(np.float32)
    out = np.zeros_like(gray_f)
    for sigma in sigmas:
        g = cv2.GaussianBlur(gray_f, (0, 0), sigma)
        gxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=5)
        gyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=5)
        gxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=5)
        tmp = np.sqrt(np.clip((gxx - gyy) ** 2 + 4 * gxy ** 2, 0, None))
        lam2 = 0.5 * ((gxx + gyy) + tmp)  # larger-magnitude eigenvalue
        out = np.maximum(out, np.abs(lam2))
    return out


def illumination_correct(img_bgr: np.ndarray, sigma_frac: float = 0.08,
                          estimate_max_side: int = 512) -> np.ndarray:
    """Shading correction: subtract a heavily-blurred version of the image
    (the estimated illumination field) and re-center on the image's median
    brightness, to remove a large-scale illumination gradient. Measured,
    not assumed: brightness ramps smoothly from 34 to 180 across the FULL
    width of every image (only the center 52% is within 10% of peak) --
    this is a global vignetting gradient from the curved cylindrical
    surface under directional lighting, not an edge effect. A border-margin
    exclusion was tried first and only helped partially (AUROC 0.42->0.62,
    still barely above chance) because the gradient extends across the
    whole frame, not just the border. A multiplicative (divide-by-blur)
    version was tried first too and under-corrected the darkest edges
    without a large gain cap, which then risks amplifying sensor noise in
    exactly the lowest-signal pixels -- subtractive correction doesn't have
    that tradeoff and measured better (flatter) besides. Applied per-channel
    on the color image so hue/relative-color information survives, not just
    grayscale brightness.

    The illumination FIELD is estimated on a downsampled copy (max side
    estimate_max_side) then upsampled back -- a huge-sigma Gaussian blur on
    a full ~4000px image is the actual bottleneck that made the tiled SSN
    dataloader spend ~16s/image (measured: one epoch took 27+ CPU-minutes
    per worker before this fix). Illumination varies slowly by definition,
    so estimating it at low resolution changes the result negligibly while
    being ~50x fewer pixels to blur; the image itself is still corrected at
    full native resolution."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    H, W = gray.shape
    scale = min(1.0, estimate_max_side / max(H, W))
    small = cv2.resize(gray, (max(1, int(W * scale)), max(1, int(H * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    sigma = max(small.shape) * sigma_frac
    illum_small = cv2.GaussianBlur(small, (0, 0), sigma)
    illum = cv2.resize(illum_small, (W, H), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else illum_small
    target = float(np.median(gray))
    offset = target - illum  # position-dependent additive correction
    out = img_bgr.astype(np.float32) + offset[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _norm01(x: np.ndarray) -> np.ndarray:
    x = x - x.min()
    m = x.max()
    return x / m if m > 1e-8 else x


def build_input_stack(img_bgr: np.ndarray, cfg: dict) -> np.ndarray:
    """Return an HxWxC float32 [0,1] stack: [R, G, B, LCM?, ridge?] per
    cfg['model']['extra_channels']. Concatenated, not substituted for RGB --
    substitution was tested and measurably hurt Hair-shaped defects (raw RGB
    alone already separates Hair well via DINOv2's pretrained features).

    Illumination-corrected FIRST (see illumination_correct) so the global
    vignetting gradient doesn't dominate the RGB channels, and so LCM/ridge
    are computed from the corrected image too.
    """
    img_bgr = illumination_correct(img_bgr, cfg["preprocessing"].get("illum_sigma_frac", 0.08))
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    channels = [rgb]
    extra = cfg["model"].get("extra_channels", [])
    if "lcm" in extra:
        lcm = local_contrast_map(gray, cfg["preprocessing"]["lcm_scales"])
        channels.append(_norm01(lcm)[..., None])
    if "ridge" in extra:
        ridge = ridge_strength_map(gray, cfg["preprocessing"]["ridge_sigmas"])
        channels.append(_norm01(ridge)[..., None])

    return np.concatenate(channels, axis=-1).astype(np.float32)
