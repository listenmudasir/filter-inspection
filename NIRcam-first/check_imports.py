#!/usr/bin/env python3
"""Verify BasicDemo's dependency chain imports on this Linux machine.

BasicDemo.py itself opens a Qt window, so it cannot be run unattended here.
This exercises everything it imports at module load -- the MVS bindings, the
camera operation class, the TCP server, and our hybrid model hook -- which
is where a platform problem would surface.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

checks = [
    ("PyQt5.QtWidgets",            "GUI toolkit"),
    ("MvImport.MvCameraControl_class", "Hikvision MVS bindings (cross-platform)"),
    ("CamOperation_class",         "camera operation"),
    ("PyUICBasicDemo",             "UI layout"),
    ("tcp_server",                 "TCP server"),
    ("simple_tracker",             "tracker"),
    ("two_band_filter",            "two-band filter"),
    ("hybrid_detect",              "our GUI hook"),
]

failed = 0
for module, label in checks:
    try:
        __import__(module)
        print(f"  OK    {label:42s} ({module})")
    except Exception as exc:
        failed += 1
        print(f"  FAIL  {label:42s} ({module}): {type(exc).__name__}: {exc}")

print()
if failed:
    print(f"{failed} import(s) failed")
    raise SystemExit(1)

# the model hook, as BasicDemo line 34 now calls it
from hybrid_detect import load_model
deploy = os.path.abspath(os.path.join(HERE, "..", "filter-inspection"))
print(f"loading hybrid model from {deploy} ...")
model = load_model(deploy)
print("model loaded:", type(model).__name__ if model else "None (GUI would show 載入失敗)")
print("\nALL IMPORTS OK - BasicDemo.py can start on this machine")
