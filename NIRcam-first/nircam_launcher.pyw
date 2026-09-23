"""Desktop entry point for operators. No terminal.

Launched by the "NIRcam Inspection" desktop shortcut as

    pythonw.exe nircam_launcher.pyw

TWO PROCESSES, deliberately. This file is only the splash: a tkinter window
that spawns BasicDemo.py as a separate process and polls for its window.

The single-process version put the splash up and then loaded the 262 MB
checkpoint on the same thread, which blocked the event loop -- so the ring
could not animate, and the splash could not appear until PyQt5 and torch had
been imported. Here the splash process imports neither: it is on screen in
about a tenth of a second and animates continuously, because the work is
happening somewhere else entirely.

The environment run_gui.bat sets up (Qt plugin path, MVS runtime on PATH,
no user site-packages) is built here and handed to the child, and the child's
stdout/stderr go to a dated log file -- pythonw has no console, and that
output is what makes the system diagnosable.
"""
from __future__ import annotations

import ctypes
import datetime as _dt
import os
import subprocess
import sys
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
ART = os.path.join(HERE, "assets", "splash.png")

# Splash geometry, as fractions of the artwork. The ring centre and radius
# were measured off assets/splash.png (centre 752,512 of 1536x1024); the
# animated arc orbits just OUTSIDE the static arc so the two do not collide.
RING_FX, RING_FY, RING_FR = 0.4898, 0.4996, 0.115
MAX_WIDTH = 620                 # px; the old 921px splash was too dominant
SCREEN_FRACTION = 0.34

try:                            # crisp text on high-DPI line monitors
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

user32 = ctypes.windll.user32
_EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p,
                               ctypes.c_void_p)


def child_window_visible(pid: int) -> bool:
    """True once the child owns a visible, titled top-level window.

    Matching on the process rather than on a window title keeps this working
    when the operator switches the GUI's language, which rewrites the title.
    """
    found = []

    def _cb(hwnd, _lparam):
        owner = ctypes.c_ulong()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if (owner.value == pid and user32.IsWindowVisible(hwnd)
                and user32.GetWindowTextLengthW(hwnd) > 0):
            found.append(hwnd)
        return True

    user32.EnumWindows(_EnumProc(_cb), None)
    return bool(found)


def child_environment() -> dict:
    """What run_gui.bat exports, for the child process."""
    env = dict(os.environ)
    for stale in ("QT_PLUGIN_PATH", "QT_DEBUG_PLUGINS",
                  "QT_QPA_PLATFORM_PLUGIN_PATH"):
        env.pop(stale, None)

    # Point Qt at THIS interpreter's PyQt5 plugins. A path left over from
    # another conda env loads that env's plugins against this env's Qt and
    # aborts the GUI.
    plugins = os.path.join(os.path.dirname(sys.executable), "Lib",
                           "site-packages", "PyQt5", "Qt5", "plugins",
                           "platforms")
    if os.path.isdir(plugins):
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugins

    # MvCameraControl.dll is loaded by name, so its directory must be on PATH.
    common = env.get("CommonProgramFiles(x86)",
                     r"C:\Program Files (x86)\Common Files")
    mvs = os.path.join(common, "MVS", "Runtime", "Win64_x64")
    if os.path.isdir(mvs):
        env["PATH"] = mvs + os.pathsep + env.get("PATH", "")

    env["PYTHONNOUSERSITE"] = "1"
    return env


class Splash:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.angle = 0
        self.proc = None
        self.log_path = ""

        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.configure(bg="#0b1220")

        from PIL import Image, ImageTk
        art = Image.open(ART).convert("RGB")
        width = min(MAX_WIDTH, int(root.winfo_screenwidth() * SCREEN_FRACTION))
        height = round(art.height * width / art.width)
        self.photo = ImageTk.PhotoImage(
            art.resize((width, height), Image.LANCZOS))

        x = (root.winfo_screenwidth() - width) // 2
        y = (root.winfo_screenheight() - height) // 2
        root.geometry(f"{width}x{height}+{x}+{y}")

        self.canvas = tk.Canvas(root, width=width, height=height,
                                highlightthickness=0, bd=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        r = RING_FR * width
        cx, cy = RING_FX * width, RING_FY * height
        self.box = (cx - r, cy - r, cx + r, cy + r)
        self.inner = (cx - r * 0.62, cy - r * 0.62,
                      cx + r * 0.62, cy + r * 0.62)
        self.arc_w = max(3, round(width / 180))

        self.launch()
        self.animate()

    def launch(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.log_path = os.path.join(
            LOG_DIR, f"nircam_{_dt.datetime.now():%Y%m%d}.log")
        log = open(self.log_path, "a", encoding="utf-8", buffering=1)
        log.write(f"\n{'=' * 70}\nlaunched "
                  f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}\n{'=' * 70}\n")
        try:
            # -u: without it Python block-buffers a redirected stdout, so this
            # log sits empty for minutes and loses whatever is still buffered
            # if the process is killed. BasicDemo.py keeps its own log too
            # (session_log.py); this one captures anything that dies before
            # that gets installed.
            self.proc = subprocess.Popen(
                [sys.executable, "-s", "-u", "BasicDemo.py"],
                cwd=HERE, env=child_environment(),
                stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as exc:
            log.write(f"failed to start BasicDemo.py: {exc!r}\n")
            self.fail(f"無法啟動 BasicDemo.py:\n{exc}")

    def fail(self, message: str):
        self.root.destroy()
        ctypes.windll.user32.MessageBoxW(
            None, f"{message}\n\n完整訊息請見記錄檔:\n{self.log_path}",
            "NIRcam Inspection", 0x10)
        sys.exit(1)

    def animate(self):
        self.canvas.delete("spin")
        # Outer arc clockwise, inner counter-clockwise, so it reads as
        # working even when the child is busy for several seconds.
        self.canvas.create_arc(self.box, start=self.angle, extent=78,
                               outline="#22d3ee", width=self.arc_w,
                               style="arc", tags="spin")
        self.canvas.create_arc(self.inner, start=-self.angle * 1.6, extent=112,
                               outline="#6366f1", width=max(2, self.arc_w - 1),
                               style="arc", tags="spin")
        self.angle = (self.angle + 5) % 360

        if self.proc is not None:
            if self.proc.poll() is not None:
                self.fail("啟動失敗，程式已結束。")
            if child_window_visible(self.proc.pid):
                self.root.destroy()
                return

        self.root.after(20, self.animate)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("itri AI detect")
    Splash(root)
    root.mainloop()
