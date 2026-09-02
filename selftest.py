#!/usr/bin/env python3
"""Verify a production install BEFORE the line depends on it.

Checks the things that have actually broken in practice, in the order they
break. Every item below cost real debugging time on this project:

  1. numpy < 2        opencv's C extension is built against the 1.x array ABI;
                      2.x raises "_ARRAY_API not found"
  2. headless opencv  the non-headless wheel bundles Qt plugins and sets
                      QT_QPA_PLATFORM_PLUGIN_PATH at import, hijacking PyQt5
  3. Qt environment   /opt/MVS/bin ships its own Qt5 and a stale
                      QT_QPA_PLATFORM_PLUGIN_PATH from another conda env both
                      abort the GUI with "could not load the Qt platform
                      plugin xcb". run_gui.sh sanitises both; this warns if
                      you are launching some other way.
  4. weights present  262 MiB, copied per machine, not in git
  5. THE MODEL THAT LOADS IS THE SUPERVISED ONE -- the failure that motivated
     this file. On a broken layout load_model() silently fell back to the old
     hybrid stack: the line kept running, logged one quiet line, and used a
     model with 9 false alarms per 51 clean frames instead of 0.
  6. inference sanity a blank frame must produce NOTHING

    python selftest.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GUI = os.path.join(HERE, "NIRcam-first")
WEIGHTS = os.path.join(HERE, "weights", "supervised_global.pt")

failures: list[str] = []
warnings: list[str] = []


def check(label, ok, detail="", fatal=True):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        (failures if fatal else warnings).append(label)
    return ok


print("=== filter-inspection self test ===\n")

print("environment")
try:
    import numpy as np
    check("numpy < 2", np.__version__.split(".")[0] == "1", f"found {np.__version__}")
except ImportError:
    check("numpy importable", False)

try:
    import cv2
    qt_dir = os.path.join(os.path.dirname(cv2.__file__), "qt")
    check("opencv headless (no bundled Qt)", not os.path.isdir(qt_dir),
          f"cv2 {cv2.__version__}"
          + ("; found cv2/qt -- install opencv-python-headless" if os.path.isdir(qt_dir) else ""))
except ImportError:
    check("cv2 importable", False)

try:
    import torch
    check("torch importable", True, torch.__version__)
    check("CUDA available", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available()
          else "CPU only -- will NOT meet the per-cartridge budget", fatal=False)
except ImportError:
    check("torch importable", False)

ld = os.environ.get("LD_LIBRARY_PATH", "")
qt_plugin = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH", "")
check("no /opt/MVS/bin on LD_LIBRARY_PATH", "/opt/MVS/bin" not in ld,
      "MVS ships its own Qt5 and will break PyQt5; run_gui.sh strips it",
      fatal=False)
check("Qt plugin path not from another conda env",
      not qt_plugin or "/envs/" not in qt_plugin or "nircam" in qt_plugin,
      qt_plugin or "<unset>", fatal=False)

print("\nfiles")
size = os.path.getsize(WEIGHTS) if os.path.isfile(WEIGHTS) else 0
check("weights present", os.path.isfile(WEIGHTS),
      WEIGHTS if size else "copy supervised_global.pt into weights/ (see weights/README.md)")
check("weights are complete", size > 200_000_000,
      f"{size/2**20:.0f} MiB" if size else "missing")

print("\nmodel")
if not failures:
    sys.path.insert(0, GUI)
    try:
        from hybrid_detect import load_model
        model = load_model(HERE)
        name = type(model).__name__ if model else "None"
        check("a model loaded", model is not None, name)
        # The whole point of this file.
        check("it is the SUPERVISED model (not the old hybrid fallback)",
              name == "SupervisedDetector", f"got {name}")

        if model is not None and name == "SupervisedDetector":
            import numpy as np
            blank = np.full((2048, 2200, 3), 128, np.uint8)   # live camera size
            result = model(blank)[0]
            check("blank frame yields no detections", len(result.boxes) == 0,
                  f"{len(result.boxes)} boxes, {model.last_latency_ms:.0f} ms")
            check("latency fits the per-cartridge budget (4300 ms)",
                  model.last_latency_ms < 4300,
                  f"{model.last_latency_ms:.0f} ms", fatal=False)
    except Exception as exc:
        check("model loads", False, str(exc))
        import traceback
        traceback.print_exc()
else:
    print("  skipped -- fix the environment failures above first")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s) -- {', '.join(failures)}")
    print("DO NOT run the line against this install.")
    sys.exit(1)
if warnings:
    print(f"PASSED with {len(warnings)} warning(s): {', '.join(warnings)}")
    sys.exit(0)
print("PASSED -- install is good.")
