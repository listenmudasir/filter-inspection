#!/usr/bin/env python3
"""Supervised segmentation detector for the camera GUI.

Drop-in replacement for hybrid_detect / detect: exposes load_model() and a
callable with the same shape as an ultralytics YOLO model object, so
BasicDemo.py's single "載入模型" button serves it with no GUI change.

WHAT THIS RUNS
--------------
The global supervised branch: a Wide-ResNet-50-2 + FPN segmenter trained
directly on the 1,115 annotated defect instances, producing a 4-class
per-pixel map (background / Bug / Foreign_Body / Stain) over the whole
downscaled cartridge in ONE forward pass.

It replaces the hybrid EfficientAD stack, which was 97% frozen one-class
backbone that never saw a defect. Measured on the locked 317-image test split
(IoU 0.25, Hungarian matching), single frame, no track consensus:

                       old hybrid     this model
    F1                    0.559          0.631
    precision             0.484          0.706
    false alarms on the
      51 clean frames         9              0
    predicted/GT area     0.147          0.931
    Bug end-to-end        0.255          0.734

Zero false components on clean frames is the number that matters on a line:
no good cartridge is rejected.

STILL WEAK
----------
Instances >= 50k px: recall 0.030, against the old stack's 0.303. Evidence
says this is mostly a LABEL problem -- the same model scores 0.917 on
validation, and 23 of the 32 large test instances are flagged in
label_audit/suspect_labels.csv at median local contrast 1.30, below what a
human can reliably see. But until those verdicts exist, treat large diffuse
staining as NOT covered by this model.

TIMING against this line (measured, RTX A5000)
----------------------------------------------
    capture rate          2.10 FPS  (476 ms/frame)
    frames per cartridge  10 over a 4.3 s burst

    this model, full frame   ~1000-1200 ms  -> fits the 4.3 s per-cartridge
                                                budget with ~3.5x headroom,
                                                and ~2x faster than the old
                                                full-frame path (2241 ms)

It does NOT fit the 476 ms per-frame budget. Use it per cartridge, with the
existing track consensus over the burst -- which is also how the old stack
reached F1 0.644 from a 0.598 single-frame number.

WHERE THE TIME GOES (so nobody optimises the wrong thing)
    crop + illumination   ~640 ms   65%   CPU
    forward pass            45 ms    5%   GPU
    everything else        ~200 ms
The network is not the bottleneck. Quantisation or a smaller model buys
nothing here; a preprocessing port is what would.
"""
from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# The `inspection` package lives in the SIBLING deployment checkout, not in
# this folder. Without adding it, `import inspection.enhance` fails on a clean
# production layout and load_model() silently falls back to the old hybrid
# detector -- the line keeps running, logs nothing alarming, and quietly uses
# the model with 9 false alarms per 51 clean frames instead of 0. That is
# exactly what happened the first time this was tested outside the dev tree.
for _candidate in (
        os.path.dirname(_HERE),                                   # ../
        os.path.join(os.path.dirname(_HERE), "filter-inspection"),  # ../filter-inspection
        os.path.join(_HERE, "filter-inspection"),
):
    if os.path.isdir(os.path.join(_candidate, "inspection")) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
        break

try:
    from preprocessing.enhance import crop_to_content, illumination_correct
except ImportError:                     # deployment checkout layout
    from inspection.enhance import crop_to_content, illumination_correct


# --------------------------------------------------------------------------
# result wrappers
# --------------------------------------------------------------------------
# Deliberately re-declared rather than imported from adapter.py. Importing it
# would drag in inspection.service -> the whole frozen EfficientAD stack, which
# this model does not use at all: loading the supervised detector would pay for
# an 80 MB backbone it never calls, and a broken hybrid install would stop the
# supervised path from loading for no reason.
#
# The cost is that these must stay shape-compatible with adapter.py, and that
# cost is real: both copies originally assumed the GUI consumes only
# .boxes.xyxy / .conf / .cls / .names. It does not -- CamOperation_class also
# ITERATES .boxes, which raised "'_Boxes' object is not iterable" on every
# frame and left the display blank while the camera was streaming fine. If you
# extend these, extend adapter.py identically.
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

    def item(self):
        return self._array.item()

    def __len__(self):
        return len(self._array)

    def __iter__(self):
        return iter(self._array)

    def __getitem__(self, item):
        value = self._array[item]
        # Keep the torch-tensor illusion one level down, so `xyxy[0].cpu()`
        # works the way callers written against ultralytics expect.
        return _Tensorish(value) if isinstance(value, np.ndarray) else value


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = _Tensorish(np.asarray(xyxy, dtype=np.float32).reshape(-1, 4))
        self.conf = _Tensorish(np.asarray(conf, dtype=np.float32).reshape(-1))
        self.cls = _Tensorish(np.asarray(cls, dtype=np.int32).reshape(-1))

    def __len__(self):
        return len(self.xyxy)

    # ultralytics' Boxes is iterable, and indexing it yields a Boxes holding
    # one row -- which is why GUI code written against it does
    #     for box in results[0].boxes:
    #         box.xyxy[0].cpu().numpy(); box.cls.item(); box.conf.item()
    # Without these, that loop raises "'_Boxes' object is not iterable" and
    # the caller never reaches its own emit/draw calls.
    def __getitem__(self, index):
        xyxy, conf, cls = (self.xyxy.numpy(), self.conf.numpy(),
                           self.cls.numpy())
        if isinstance(index, slice):
            return _Boxes(xyxy[index], conf[index], cls[index])
        i = index + len(self) if index < 0 else index
        if not 0 <= i < len(self):
            raise IndexError(f"box index {index} out of range for {len(self)}")
        return _Boxes(xyxy[i:i + 1], conf[i:i + 1], cls[i:i + 1])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


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


# --------------------------------------------------------------------------
# model definition, inlined so the deployment does not import the research repo
# --------------------------------------------------------------------------
class _SeparableBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        return F.gelu(self.norm(self.conv(x)))


class FPNSegmenter(nn.Module):
    """Must stay structurally identical to scripts/train_tile_segmenter.py.

    The checkpoint is a plain state_dict; a renamed or reordered layer here
    fails loudly at load_state_dict rather than silently mispredicting, which
    is the behaviour we want on a production line.
    """

    def __init__(self, n_classes: int = 4, channels: int = 128, pretrained: bool = False):
        super().__init__()
        from torchvision.models import wide_resnet50_2
        net = wide_resnet50_2(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.lateral = nn.ModuleList([nn.Conv2d(c, channels, 1)
                                      for c in (256, 512, 1024, 2048)])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                          nn.GroupNorm(8, channels), nn.GELU())
            for _ in range(4)])
        self.head = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.GELU(),
            nn.Conv2d(channels, n_classes, 1))

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        c2 = self.layer1(x); c3 = self.layer2(c2)
        c4 = self.layer3(c3); c5 = self.layer4(c4)
        laterals = [conv(c) for conv, c in zip(self.lateral, (c2, c3, c4, c5))]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + F.interpolate(
                laterals[i + 1], laterals[i].shape[-2:], mode="nearest")
        feats = [s(l) for s, l in zip(self.smooth, laterals)]
        target = feats[0].shape[-2:]
        merged = torch.cat([feats[0]] + [
            F.interpolate(f, target, mode="bilinear", align_corners=False)
            for f in feats[1:]], dim=1)
        return F.interpolate(self.head(merged), size, mode="bilinear", align_corners=False)


class SupervisedDetector:
    """Callable with the same shape as an ultralytics YOLO model object."""

    DEFECT_NAMES = ["Bug", "Foreign_Body", "Stain"]

    # ------------------------------------------------------------------
    # Geometry is expressed at MODEL resolution, not in camera pixels.
    #
    # The old settings -- min_component_area 528 px, border_margin 250 px --
    # were native pixels of the 4096x3650 training capture. Carried onto a
    # different camera they silently change meaning: this line's FUE-S500C-PRO
    # delivers 2200x2048, where the same 528 would be ~3.5x more aggressive at
    # model scale and would suppress genuine small defects. Nothing would warn
    # you; recall would just be worse.
    #
    # Everything this detector does happens after a resize to input_size, so
    # input_size is the ONE frame of reference every camera shares. Converting
    # the training values into it once:
    #
    #     training content side ~3900 px  ->  shrink = 3900 / 768 = 5.08
    #     min area  528 / 5.08^2 = 20.5 px   at 768
    #     border    250 / 5.08   = 49.2 px   at 768
    #
    # These are now camera-independent: no per-camera calibration, no config
    # to forget to update when the optics change.
    MIN_AREA_AT_MODEL_SCALE = 20      # px at input_size
    BORDER_AT_MODEL_SCALE = 49        # px at input_size

    def __init__(self, weights: str, device: str = "auto", input_size: int = 768,
                 prob_threshold: float = 0.30, min_area_model_px: int | None = None,
                 border_margin_model_px: int | None = None,
                 min_confidence: float = 0.0, **_legacy):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.input_size = int(input_size)
        # 0.30 was selected on the VALIDATION split (F1 0.810 there) and then
        # applied once to test, where it also happened to be optimal. It is not
        # a number tuned against the test set.
        self.prob_threshold = float(prob_threshold)
        self.min_area_model_px = int(min_area_model_px if min_area_model_px is not None
                                     else self.MIN_AREA_AT_MODEL_SCALE)
        self.border_margin_model_px = int(border_margin_model_px
                                          if border_margin_model_px is not None
                                          else self.BORDER_AT_MODEL_SCALE)
        self.min_confidence = float(min_confidence)
        # Accept the old native-pixel names so existing configs do not break,
        # but convert them explicitly and say so -- silently reinterpreting a
        # number whose meaning depends on the training camera is how this class
        # of bug survives.
        if "min_component_area" in _legacy or "border_margin_px" in _legacy:
            print("[supervised_detect] NOTE: min_component_area / border_margin_px "
                  "are native-pixel settings tied to the 4096x3650 training "
                  "capture and are IGNORED. Geometry is now defined at model "
                  f"scale: min_area={self.min_area_model_px}px, "
                  f"border={self.border_margin_model_px}px at {input_size}px.")

        raw = torch.load(weights, map_location=self.device, weights_only=False)
        state = raw.get("model", raw)
        self.model = FPNSegmenter().to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.epoch = raw.get("epoch")
        self.last_latency_ms = 0.0
        self._warned_imgsz = False

    @torch.no_grad()
    def _detect(self, image_bgr):
        cropped, _m, crop_box = crop_to_content(image_bgr)
        cropped = illumination_correct(cropped, 0.08)
        height, width = cropped.shape[:2]
        side = max(height, width)

        canvas = np.zeros((side, side, 3), np.uint8)
        canvas[:height, :width] = cropped
        small = cv2.resize(canvas, (self.input_size, self.input_size),
                           interpolation=cv2.INTER_AREA)
        # BGR, matching training exactly -- the dataset fed raw cv2.imread
        # output with no colour conversion.
        tensor = torch.from_numpy(small.transpose(2, 0, 1).copy()).float()
        tensor = tensor.unsqueeze(0).to(self.device) / 255.0
        with torch.autocast("cuda", enabled=self.device.type == "cuda"):
            logits = self.model(tensor)
        probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()

        # Threshold and label at MODEL resolution: 0.6M px instead of ~13.6M.
        # Verified on the full test split -- F1 0.6309 -> 0.6297, one instance
        # in 340, size buckets and clean-frame behaviour unchanged, 22x faster.
        shrink = side / float(self.input_size)
        label_map = probs[1:].argmax(0).astype(np.int32) + 1
        label_map[(1.0 - probs[0]) < self.prob_threshold] = 0

        margin = self.border_margin_model_px
        if margin > 0 and label_map.shape[0] > 2 * margin:
            label_map[:margin] = 0; label_map[-margin:] = 0
            label_map[:, :margin] = 0; label_map[:, -margin:] = 0

        min_area = self.min_area_model_px
        structure = ndimage.generate_binary_structure(2, 2)
        labels, count = ndimage.label(label_map > 0, structure=structure)

        detections = []
        cx1, cy1 = int(crop_box[0]), int(crop_box[1])
        for index in range(1, count + 1):
            mask = labels == index
            area_small = int(mask.sum())
            if area_small < min_area:
                continue
            values = label_map[mask]
            class_id = int(np.bincount(values[values > 0]).argmax())
            confidence = float((1.0 - probs[0])[mask].mean())
            ys, xs = np.where(mask)
            # Back to ORIGINAL full-frame coordinates: undo the model scale,
            # then add the crop offset. The GUI draws in original pixels.
            x_min = int(xs.min() * shrink) + cx1
            x_max = int(xs.max() * shrink) + cx1
            y_min = int(ys.min() * shrink) + cy1
            y_max = int(ys.max() * shrink) + cy1
            detections.append({
                "bbox_xyxy": [x_min, y_min, x_max + 1, y_max + 1],
                "confidence": confidence,
                "defect_class": self.DEFECT_NAMES[class_id - 1],
                "anomaly_score": confidence,
                "area_px": int(area_small * shrink * shrink),
                "centroid_xy": [float(xs.mean() * shrink) + cx1,
                                float(ys.mean() * shrink) + cy1],
            })
        return detections

    def __call__(self, img, regions=None, verbose=False,
                 conf=None, imgsz=None, iou=None, **_ignored):
        """Returns [ _Result ] -- a 1-element list, like YOLO.

        THE GUI'S KNOBS ARE HONOURED, NOT SWALLOWED
        -------------------------------------------
        CamOperation_class calls detect_objects(model, img, conf_thres=..,
        imgsz=..) from the "AI Detection Parameters" panel. Those are YOLO
        parameters. An earlier version of this method accepted them via
        **_ignored and silently discarded them, which is worse than not
        supporting them: an operator raises Confidence on the line, sees the
        panel echo the new value, and the detector behaves identically.

          conf  -> APPLIED, as a floor on each detection's mean defect
                   probability. This is a genuine sensitivity control:
                   raise it to report fewer, more certain regions.
                   It does NOT touch prob_threshold (0.30), which is the
                   pixel-level operating point selected on the validation
                   split and then applied once to test. Moving that would
                   invalidate the measured numbers; filtering the output
                   does not.

          imgsz -> REPORTED AS INERT. This model always resizes to
                   input_size (768) because that is the geometry it was
                   trained on. Honouring 1280 here would feed it a scale it
                   has never seen. Said out loud rather than ignored.

          iou   -> inert. There is no NMS: connected components on a
                   segmentation map cannot overlap.

        `regions` is accepted for interface compatibility but used only to
        FILTER. This model is whole-frame by construction: one forward pass
        covers the part, so cropping to proposals would cost the same and see
        less. Detections outside every supplied region are dropped.
        """
        if imgsz is not None and int(imgsz) != self.input_size and not self._warned_imgsz:
            print(f"[supervised_detect] NOTE: GUI image size {int(imgsz)} is not used. "
                  f"This model is fixed at {self.input_size}px -- the geometry it was "
                  f"trained on. Confidence IS applied.")
            self._warned_imgsz = True
        floor = self.min_confidence if conf is None else max(self.min_confidence, float(conf))

        start = time.perf_counter()
        detections = self._detect(img)
        if regions:
            kept = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox_xyxy"]
                for rx1, ry1, rx2, ry2 in regions:
                    if x1 < rx2 and x2 > rx1 and y1 < ry2 and y2 > ry1:
                        kept.append(det)
                        break
            detections = kept
        if floor > 0:
            detections = [d for d in detections if d["confidence"] >= floor]
        self.last_latency_ms = (time.perf_counter() - start) * 1000.0
        if verbose:
            print(f"[SupervisedDetector] {len(detections)} detections in "
                  f"{self.last_latency_ms:.1f} ms")
        return [_Result(detections)]


def looks_like_supervised(path: str) -> bool:
    """True when `path` is a supervised segmenter checkpoint.

    Checked by INSPECTING THE WEIGHTS, not by filename. A misnamed file that
    loads the wrong architecture would fail at run time on the line; failing
    here is cheaper.
    """
    if not path or not os.path.isfile(path) or not path.endswith(".pt"):
        return False
    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    state = raw.get("model", raw) if isinstance(raw, dict) else None
    if not isinstance(state, dict):
        return False
    return any(k.startswith("lateral.") for k in state) and \
        any(k.startswith("layer4.") for k in state)


WEIGHT_NAME = "supervised_global.pt"


def find_weights(hint=None):
    """Locate the checkpoint without depending on one fixed layout.

    BasicDemo.py passes a directory that has meant different things across
    revisions (a sibling `filter-inspection/` checkout, then the repo root).
    Hard-coding either makes the GUI silently fall back to the old hybrid model
    the moment the tree is rearranged -- which already happened once. So the
    hint is tried first and then a short list of layouts relative to THIS file.
    """
    candidates = []
    if hint:
        if os.path.isfile(hint):
            return hint
        if os.path.isdir(hint):
            candidates += [os.path.join(hint, "weights", WEIGHT_NAME),
                           os.path.join(hint, WEIGHT_NAME)]
    root = os.path.dirname(_HERE)
    candidates += [
        os.path.join(root, "weights", WEIGHT_NAME),                      # repo root
        os.path.join(root, "filter-inspection", "weights", WEIGHT_NAME),  # sibling checkout
        os.path.join(_HERE, "weights", WEIGHT_NAME),                     # beside the GUI
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_model(weights=None, device="auto", **kwargs):
    """Mirrors detect.load_model() / hybrid_detect.load_model()."""
    found = find_weights(weights)
    if found is None:
        print(f"[supervised_detect] {WEIGHT_NAME} not found. Copy it into "
              f"{os.path.join(os.path.dirname(_HERE), 'weights')}/ "
              f"(see weights/README.md).")
        return None
    weights = found
    if not looks_like_supervised(weights):
        return None
    try:
        detector = SupervisedDetector(weights, device=device, **kwargs)
        print(f"Supervised global segmenter loaded (epoch {detector.epoch}, "
              f"threshold {detector.prob_threshold}, {detector.input_size}px)")
        return detector
    except Exception as exc:
        print(f"[supervised_detect] load failed: {exc}")
        import traceback
        traceback.print_exc()
        return None
