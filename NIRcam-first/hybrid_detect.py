"""Drop-in replacement for NIRcam-first's `detect.py`, for the GUI (BasicDemo.py).

GOAL
----
Run the hybrid EfficientAD method inside the existing GUI exactly the way a
YOLO model runs: same button, same file dialog, same call signature, same
downstream chain. No fork of the GUI, no new UI widgets required.

INSTALL
-------
1. Copy this file, `adapter.py` and `detector_factory.py` into the
   NIRcam-first checkout root.
2. Change ONE line in BasicDemo.py:

       line 11:  from detect import load_model
       becomes:  from hybrid_detect import load_model

   That is the entire GUI integration. `load_ai_model()` (line 147),
   `set_ai_model()`, the 載入模型 button (line 986) and every downstream
   consumer keep working unchanged.

3. Optional -- line 34 loads a model at import time from a hardcoded path
   that will not exist on your machine:

       ai_model = load_model(r"C:\\Users\\user1\\Desktop\\...\\best.pt")

   Point it at your deployment checkout instead:

       ai_model = load_model(r"C:\\path\\to\\filter-inspection")

   or leave it -- it fails safely to None and you load via the button.

WHAT THE FILE DIALOG ACCEPTS
---------------------------
The GUI's dialog filters `PyTorch Weights (*.pt)`. Selecting any of these
works, because the package layout is inferred from whatever you pick:

    <checkout>/weights/hybrid_v4.pt     <- the natural choice
    <checkout>                          <- the folder itself
    <checkout>/config/inspection.yaml

To widen the dialog so the folder is selectable, change line 149's filter to
    "Model (*.pt *.yaml);;All Files (*)"

DUAL BACKEND
------------
`load_model()` dispatches on what it is given:
  * a path containing a `weights/hybrid_v4.pt` layout -> hybrid method
  * any other `.pt` file                              -> falls back to YOLO
So the same button still loads YOLO weights if you point it at them, and the
operator does not need to know which backend is active.
"""
from __future__ import annotations

import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _looks_like_hybrid(path: str) -> str | None:
    """Return the package root if `path` points into a deployment checkout."""
    if not path:
        return None
    candidates = []
    if os.path.isdir(path):
        candidates.append(path)
    else:
        candidates.append(os.path.dirname(path))                    # weights/
        candidates.append(os.path.dirname(os.path.dirname(path)))   # root
    for root in candidates:
        if not root:
            continue
        if (os.path.isfile(os.path.join(root, "weights", "hybrid_v4.pt"))
                and os.path.isfile(os.path.join(root, "config", "inspection.yaml"))):
            return root
    return None


def load_model(weights=None, mode="frame", device="auto", threshold=None):
    """Mirrors detect.load_model(). Dispatches supervised / hybrid / YOLO by
    inspecting what `weights` points at, so the GUI's single button serves all
    three."""
    # Supervised segmenter first. It supersedes the hybrid stack on every
    # measured axis except large diffuse staining (F1 0.631 vs 0.559,
    # precision 0.706 vs 0.484, and zero false components on the 51 clean test
    # frames against the hybrid's 9), and it needs neither the 80 MB frozen
    # EfficientAD backbone nor a fitted calibration file.
    try:
        import supervised_detect
        supervised = supervised_detect.load_model(weights, device=device)
        if supervised is not None:
            return supervised
    except Exception as exc:
        # LOUD, not a one-line note. A silent fallback here means the line runs
        # the superseded model while everyone believes otherwise -- 9 false
        # alarms per 51 clean frames instead of 0, with no visible symptom.
        import traceback
        print("=" * 72)
        print("WARNING: the SUPERVISED model could not be loaded.")
        print(f"  reason: {exc}")
        print("  Falling back to the OLD hybrid EfficientAD stack, which is")
        print("  worse on every measured axis except large diffuse staining:")
        print("     F1        0.559  vs  0.631")
        print("     precision 0.484  vs  0.706")
        print("     false alarms on 51 clean frames:  9  vs  0")
        print("  Expected weights: <deploy>/weights/supervised_global.pt")
        print("=" * 72)
        traceback.print_exc()

    root = _looks_like_hybrid(weights) if weights else None

    if root is None and weights:
        # not a hybrid checkout -- preserve the original YOLO behaviour
        try:
            from detect import load_model as yolo_load_model
            print(f"[hybrid_detect] not a hybrid checkout, using YOLO: {weights}")
            return yolo_load_model(weights)
        except Exception as exc:
            print(f"[hybrid_detect] YOLO fallback failed: {exc}")
            traceback.print_exc()
            return None

    if root is None:
        print("[hybrid_detect] no path given and no hybrid checkout found")
        return None

    try:
        for path in (root, os.path.join(root, "nircam")):
            if os.path.isdir(path) and path not in sys.path:
                sys.path.insert(0, path)
        from adapter import HybridDetector

        detector = HybridDetector(
            mode=mode,
            device=device,
            threshold=threshold,
            config=os.path.join(root, "config", "inspection.yaml"),
            checkpoint=os.path.join(root, "weights", "hybrid_v4.pt"),
            calibration=os.path.join(root, "weights", "calibration.json"),
        )
        print(f"[hybrid_detect] hybrid EfficientAD loaded from {root} "
              f"(mode={mode}, device={device})")
        return detector
    except Exception as exc:
        print(f"[hybrid_detect] failed to load hybrid model: {exc}")
        traceback.print_exc()
        return None


def detect_objects(model, img, conf_thres=0.25, iou_thres=0.45, imgsz=1280,
                   regions=None):
    """Mirrors detect.detect_objects(). iou_thres/imgsz are accepted for
    signature compatibility; the hybrid model uses neither (no NMS, fixed
    native tile size)."""
    if model is None:
        return None
    try:
        if hasattr(model, "min_confidence"):
            model.min_confidence = float(conf_thres)
            return model(img, regions=regions)
        return model(img, imgsz=imgsz, conf=conf_thres, iou=iou_thres)  # YOLO
    except Exception as exc:
        print(f"[hybrid_detect] detection error: {exc}")
        traceback.print_exc()
        return None
