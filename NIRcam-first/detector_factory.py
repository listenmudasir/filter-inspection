"""Detector factory for NIRcam-first: select the inference backend by config.

Makes hybrid EfficientAD a first-class peer of YOLO rather than a bolt-on.
Both backends expose the same call interface, so everything downstream
(SimpleTracker, TwoBandFilter, TrackManager, BlowController) is unchanged.

    config["detector"]["backend"] == "yolo"    -> ultralytics YOLO
    config["detector"]["backend"] == "hybrid"  -> hybrid EfficientAD v4

Config block (add to test_config.json):

    "detector": {
        "backend": "hybrid",
        "yolo": { "weights": "yolov8n.pt" },
        "hybrid": {
            "mode": "region",
            "threshold": 0.94,
            "project_root": "<INSTALL_DIR>",
            "source_resolution": [4096, 3650]
        }
    }

RESOLUTION WARNING
------------------
The hybrid model was trained on MV-CL042-91GC-V2 line-scan frames at
4096x3650. This rig uses MV-CA016 area-scan at 1280x1024 -- a different
camera class and 3.2x lower linear resolution.

Defect features the model relies on are scale-specific: the Stain
discriminant was measured at a 6-11 px band, which maps to ~2-3 px at
1280x1024 and approaches the sampling floor. The model has never been
evaluated on this camera.

`scale_compensation` upsamples incoming frames to the training scale so the
model sees features at the pixel size it learned. This restores scale but
cannot recover detail the sensor never captured -- it is a mitigation, not a
fix. Validate on real MV-CA016 captures before trusting any number.
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

DEFAULT_SOURCE_RESOLUTION = (4096, 3650)   # what the model was trained on


class ScaleCompensatingDetector:
    """Wraps a detector so it sees frames at its training resolution.

    Upsamples the frame to `source_resolution`, runs inference, then maps
    detections back to the original frame's coordinate system so downstream
    tracking and blow geometry stay in native camera pixels.
    """

    def __init__(self, inner, source_resolution=DEFAULT_SOURCE_RESOLUTION):
        self.inner = inner
        self.source_resolution = tuple(source_resolution)
        self.last_latency_ms = 0.0
        self._warned = False

    def __call__(self, img, regions=None, verbose=False, **kwargs):
        h, w = img.shape[:2]
        tw, th = self.source_resolution
        if (w, h) == (tw, th):
            results = self.inner(img, regions=regions, verbose=verbose, **kwargs)
            self.last_latency_ms = getattr(self.inner, "last_latency_ms", 0.0)
            return results

        if not self._warned:
            print(f"[ScaleCompensation] frame {w}x{h} != training {tw}x{th}; "
                  f"upsampling {tw/w:.2f}x. Detail absent from the sensor "
                  f"cannot be recovered -- validate on this camera.")
            self._warned = True

        sx, sy = tw / w, th / h
        scaled = cv2.resize(img, (tw, th), interpolation=cv2.INTER_CUBIC)
        scaled_regions = None
        if regions:
            scaled_regions = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                              for x1, y1, x2, y2 in regions]

        results = self.inner(scaled, regions=scaled_regions, verbose=verbose, **kwargs)
        self.last_latency_ms = getattr(self.inner, "last_latency_ms", 0.0)

        # map boxes back to native camera coordinates
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.numpy().copy()
            xyxy[:, [0, 2]] /= sx
            xyxy[:, [1, 3]] /= sy
            boxes.xyxy._array = xyxy
        return results


def create_detector(config: dict):
    """Return a detector object callable as `detector(frame)`.

    Raises on an unknown backend rather than silently falling back, so a
    typo in config cannot put an unintended model on the line.
    """
    detector_cfg = config.get("detector", {})
    backend = detector_cfg.get("backend", "yolo").lower()

    if backend == "yolo":
        from ultralytics import YOLO
        weights = detector_cfg.get("yolo", {}).get("weights", "yolov8n.pt")
        print(f"[DetectorFactory] backend=yolo weights={weights}")
        return YOLO(weights)

    if backend == "hybrid":
        hybrid_cfg = detector_cfg.get("hybrid", {})

        # Locate the adapter relative to THIS file. It is a sibling in both
        # layouts -- integration/nircam/ in the development tree, nircam/ in
        # the deployment package -- so resolving by sibling rather than by a
        # configured path keeps one code path working in both.
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)

        # project_root supplies the `inspection` package (deployment) or the
        # project root (development). Default to the parent of this file,
        # which is correct for the deployment package.
        project_root = hybrid_cfg.get("project_root") or os.path.dirname(here)
        if not os.path.isdir(project_root):
            raise ValueError(
                "detector.hybrid.project_root must point at the installation "
                f"root (got {project_root!r})")
        for path in (project_root, os.path.join(project_root, "integration")):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)

        try:                                   # deployment package layout
            from adapter import load_model
        except ImportError:                    # development tree layout
            from nircam_adapter import load_model

        detector = load_model(
            mode=hybrid_cfg.get("mode", "region"),
            threshold=hybrid_cfg.get("threshold", 0.94),
            device=hybrid_cfg.get("device", "cuda"),
        )
        if detector is None:
            raise RuntimeError("hybrid detector failed to load")

        source_resolution = tuple(hybrid_cfg.get(
            "source_resolution", DEFAULT_SOURCE_RESOLUTION))
        image_cfg = config.get("image", {})
        native = (image_cfg.get("image_width"), image_cfg.get("image_height"))
        if all(native) and tuple(native) != source_resolution:
            detector = ScaleCompensatingDetector(detector, source_resolution)
        print(f"[DetectorFactory] backend=hybrid mode={hybrid_cfg.get('mode','region')}")
        return detector

    raise ValueError(f"unknown detector backend {backend!r}; expected 'yolo' or 'hybrid'")
