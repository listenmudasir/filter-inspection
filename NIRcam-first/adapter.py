"""Drop-in replacement for NIRcam-first's detect.py, backed by hybrid EfficientAD v4.

DESIGN CONTRACT
---------------
NIRcam-first's SimpleTracker._parse_detections() duck-types the Ultralytics
YOLO Results object:

    detections[0].boxes.xyxy .cpu().numpy()   -> (N, 4) float
    detections[0].boxes.conf .cpu().numpy()   -> (N,)   float
    detections[0].boxes.cls  .cpu().numpy()   -> (N,)   int

This module returns an object with exactly that shape, so simple_tracker.py,
two_band_filter.py, track_manager.py and detect.draw_custom_boxes() all keep
working with no changes. Only the model construction and call site change.

INTEGRATION (integrated_system.py)
----------------------------------
    # from ultralytics import YOLO
    # self.yolo_model = YOLO(model_path)
    from nircam_adapter import load_model
    self.yolo_model = load_model()          # same call signature downstream

`self.yolo_model(frame, verbose=False)` then returns YOLO-shaped results.

TIMING (measured on RTX A5000 against this line's own data)
-----------------------------------------------------------
    capture rate        2.10 FPS   (476 ms/frame, from MVS filename timestamps)
    frames per cartridge 10        over a 4.3 s burst
    conveyor speed      160 px/s

    MODE_REGION  ~203 ms/call  -> 43% of one frame period. Real-time capable.
    MODE_FRAME  ~2241 ms/call  -> exceeds a frame period, but only 52% of the
                                  4.3 s burst, so ONE full-frame analysis per
                                  cartridge fits comfortably.

Prefer MODE_REGION in-line. Use MODE_FRAME for per-cartridge or offline audit.

ACCURACY (locked 317-image test split, threshold 0.94 + track consensus)
    precision 0.603 | recall 0.690 | F1 0.644

CAVEAT: recall is ~0.69, i.e. roughly 1 in 3 labelled defects is not
reported. Ground truth is known to be imperfect in both directions, so the
true figure is likely better -- but do not deploy this as a sole reject
decision without validating against your own acceptance criteria.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT = os.path.dirname(_HERE)
if _PROJECT not in sys.path:
    sys.path.insert(0, _PROJECT)

from inspection.service import InspectionService  # noqa: E402

MODE_FRAME = "frame"
MODE_REGION = "region"

# Our classes, in prototype-codebook order. "Unknown" is the open-set
# rejection outcome: an anomaly the model localised but would not name.
CLASS_NAMES = ["Bug", "Foreign_Body", "Stain", "Unknown"]
_CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


class _Tensorish:
    """numpy array wearing the .cpu().numpy() interface torch tensors have."""

    def __init__(self, array):
        self._array = array

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def __len__(self):
        return len(self._array)

    def __getitem__(self, item):
        return self._array[item]


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Tensorish(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.conf = _Tensorish(np.asarray(conf, dtype=np.float32).reshape(-1))
        self.cls = _Tensorish(np.asarray(cls, dtype=np.int32).reshape(-1))

    def __len__(self):
        return len(self.xyxy)


class _Result:
    """Minimal stand-in for one ultralytics Results element."""

    def __init__(self, detections, names=None):
        xyxy = [d["bbox_xyxy"] for d in detections] or np.empty((0, 4))
        conf = [d["confidence"] for d in detections] or np.empty(0)
        cls = [_CLASS_ID.get(d["defect_class"], _CLASS_ID["Unknown"])
               for d in detections] or np.empty(0, dtype=int)
        self.boxes = _Boxes(xyxy, conf, cls)
        self.names = names or {i: n for i, n in enumerate(CLASS_NAMES)}
        self.raw = detections          # full payload incl. anomaly_score, area_px


class HybridDetector:
    """Callable with the same shape as an ultralytics YOLO model object."""

    def __init__(self, mode=MODE_REGION, threshold=None, device="cuda",
                 min_confidence=0.0, **kwargs):
        self.mode = mode
        self.min_confidence = float(min_confidence)
        service_kwargs = {"device": device}
        if threshold is not None:
            service_kwargs["threshold"] = float(threshold)
        service_kwargs.update({k: v for k, v in kwargs.items()
                               if k in ("config", "checkpoint", "calibration")})
        self.service = InspectionService(**service_kwargs)
        self.last_latency_ms = 0.0

    def __call__(self, img, regions=None, verbose=False, **_ignored):
        """Run inference. Returns [ _Result ] -- a 1-element list, like YOLO.

        regions: optional list of (x1,y1,x2,y2) candidate boxes from a fast
                 upstream detector. When supplied (and mode is MODE_REGION)
                 only those windows are inspected, which is the real-time
                 path. Without them, MODE_REGION degrades to a full frame,
                 so pass regions if you need the 203 ms figure.
        """
        detections = []
        if self.mode == MODE_REGION and regions:
            total = 0.0
            for box in regions:
                out = self.service.infer_region(img, box)
                detections.extend(out["detections"])
                total += out["latency_ms"]
            self.last_latency_ms = total
        else:
            out = self.service.infer_frame(img)
            detections = out["detections"]
            self.last_latency_ms = out["latency_ms"]

        if self.min_confidence > 0:
            detections = [d for d in detections
                          if d["confidence"] >= self.min_confidence]
        if verbose:
            print(f"[HybridDetector] {len(detections)} detections "
                  f"in {self.last_latency_ms:.1f} ms ({self.mode})")
        return [_Result(detections)]


def load_model(weights=None, mode=MODE_REGION, device="cuda", **kwargs):
    """Mirrors detect.load_model(). `weights` accepted and ignored -- paths
    come from the project config so the line cannot silently load a
    checkpoint that disagrees with its calibration."""
    try:
        detector = HybridDetector(mode=mode, device=device, **kwargs)
        print(f"Hybrid EfficientAD v4 loaded (mode={mode}, "
              f"threshold={detector.service.threshold})")
        return detector
    except Exception as exc:                      # match detect.py's behaviour
        import traceback
        print(f"Error loading hybrid model: {exc}")
        traceback.print_exc()
        return None


def detect_objects(model, img, conf_thres=0.25, iou_thres=0.45, imgsz=1280,
                   regions=None):
    """Mirrors detect.detect_objects(). conf_thres is applied; iou_thres and
    imgsz are accepted for signature compatibility and ignored -- this model
    does not use NMS and runs at its own native tile size."""
    if model is None:
        return None
    try:
        model.min_confidence = float(conf_thres)
        return model(img, regions=regions)
    except Exception as exc:
        import traceback
        print(f"Error detecting objects: {exc}")
        traceback.print_exc()
        return None
