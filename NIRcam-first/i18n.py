# -*- coding: utf-8 -*-
"""Language switching for the NIRcam-first GUI.

DESIGN
------
The Traditional Chinese source string IS the lookup key. `tr("查找設備")`
returns "Find Devices" in English mode and the argument unchanged in Chinese
mode. Two consequences worth knowing:

  * Chinese mode is exactly the original program. `tr()` is the identity
    function, so nothing can regress for the existing operators.
  * A string with no English entry falls through to Chinese rather than
    showing a missing-key placeholder. Partial coverage degrades quietly.

RETRANSLATING LIVE
------------------
Most of the TCP tab builds its labels inline -- `QLabel("主機:")` -- with no
reference kept anywhere, so there is nothing to call setText() on later.
Rather than name ~40 widgets, `register_tree()` walks the widget hierarchy
once at startup and snapshots each widget's original text keyed by id().
`apply_language()` then translates from that snapshot.

The snapshot is what makes repeated switching safe: translation always runs
from the stored Chinese original, never from the currently displayed text,
so zh -> en -> zh -> en cannot drift.

Runtime status strings (frame counts, connection state) are rewritten by
worker code after startup and would revert to Chinese. Those call sites pass
through `tr()` with a format template -- see `TEMPLATES`.
"""
from __future__ import annotations

import json
import os

from PyQt5 import QtWidgets

LANGUAGES = [("繁體中文", "zh_TW"), ("English", "en")]

_LANG = "zh_TW"
_SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".gui_language")

# Static UI chrome. Key = the Traditional Chinese source string.
_EN = {
    # window / tabs
    "主視窗": "Main Window",
    "工業相機 AI 檢測應用 V1.5.3": "Industrial Camera AI Inspection V1.5.3",
    "相機控制": "Camera Control",
    "TCP 控制與辨識結果": "TCP Control & Results",

    # init group -- the IP is deliberately preserved verbatim
    "初始化,(500GCSS)169.254.93.23": "Initialisation, (500GCSS)169.254.93.23",
    "查找設備": "Find Devices",
    "打開設備": "Open Device",
    "關閉設備": "Close Device",
    "載入模型": "Load Model",

    # acquisition group
    "採集": "Acquisition",
    "連續模式": "Continuous",
    "觸發模式": "Trigger",
    "開始採集": "Start Grabbing",
    "停止採集": "Stop Grabbing",
    "軟觸發一次": "Software Trigger",
    "保存影像": "Save Image",

    # parameter group
    "參數": "Parameters",
    "曝光": "Exposure",
    "增益": "Gain",
    "幀率": "Frame Rate",
    "獲取參數": "Get Params",
    "設定參數": "Set Params",

    # displays
    "相機影像將顯示於此": "Camera image will appear here",
    "AI 辨識影像將顯示於此": "AI detection image will appear here",
    "模型辨識結果": "Model Detection Results",
    "辨識結果文字...": "Detection result text...",

    # AI parameters
    "AI 檢測參數設定": "AI Detection Parameters",
    "信心指數:": "Confidence:",
    "影像大小:": "Image size:",
    "更新": "Update",
    "重設": "Reset",

    # boundary filter
    "邊界線過濾設定": "Boundary Line Filter",
    "啟用邊界線過濾": "Enable boundary filter",
    "上線:": "Top:",
    "下線:": "Bottom:",
    "套用": "Apply",
    "上線 25%, 下線 75%": "Top 25%, bottom 75%",

    # TCP
    "TCP 伺服器控制": "TCP Server Control",
    "主機:": "Host:",
    "埠號:": "Port:",
    "啟動": "Start",
    "停止": "Stop",
    "TCP: 未連接": "TCP: not connected",

    # shared memory
    "共享記憶體": "Shared Memory",
    "手動分享": "Share Once",
    "自動分享": "Auto share",
    "未啟動": "Not started",

    # image saving
    "圖片儲存設定": "Image Save Settings",
    "啟用圖片儲存": "Enable image saving",
    "選擇路徑": "Choose Path",
    "未設定路徑": "No path set",
    "儲存: 停用": "Save: off",
    "儲存: 啟用": "Save: on",

    # language selector itself
    "語言": "Language",
}

# Runtime strings built with values. Call as
#   tr_fmt("TCP: LabVIEW已連接 (觸發次數: {n})", n=count)
# so the key stays greppable and the English keeps its placeholders.
TEMPLATES = {
    "目前參數 - 信心指數: {conf}, 影像大小: {size}":
        "Current - confidence: {conf}, image size: {size}",
    "信心: {conf}, 大小: {size}":
        "Conf: {conf}, size: {size}",
    "TCP: LabVIEW已連接 (觸發次數: {count})":
        "TCP: LabVIEW connected ({count} triggers)",
    "TCP: 等待LabVIEW連接...": "TCP: waiting for LabVIEW...",
    "TCP: 伺服器未啟動": "TCP: server not started",
    "共享記憶體: 已啟動 (已傳送 {count} 幀)":
        "Shared memory: running ({count} frames sent)",
    "共享記憶體: 已啟動 (狀態未知)":
        "Shared memory: running (state unknown)",
    "共享記憶體: 未啟動": "Shared memory: not started",
    "目前邊界線: 上線 {top}%, 下線 {bottom}%":
        "Boundaries: top {top}%, bottom {bottom}%",
}


def current_language() -> str:
    return _LANG


def tr(text: str) -> str:
    """Translate a static string. Identity in Chinese mode."""
    if _LANG == "zh_TW":
        return text
    return _EN.get(text, text)


def tr_fmt(template: str, **kwargs) -> str:
    """Translate a format template, then substitute. The Chinese template is
    the key, so call sites stay searchable against the original source."""
    chosen = template if _LANG == "zh_TW" else TEMPLATES.get(template, template)
    try:
        return chosen.format(**kwargs)
    except (KeyError, IndexError):
        # A malformed template must not take the GUI down mid-acquisition.
        return template


# --- live retranslation -----------------------------------------------------

# id(widget) -> (widget, kind, original_text). Holding the widget keeps the
# id stable; without a reference, CPython could recycle the id after GC and
# the snapshot would alias the wrong object.
_registry: dict[int, tuple] = {}


def _accessors(widget):
    """(kind, getter, setter) for widgets whose visible text we translate."""
    if isinstance(widget, QtWidgets.QGroupBox):
        return "title", widget.title, widget.setTitle
    if isinstance(widget, (QtWidgets.QPushButton, QtWidgets.QCheckBox,
                           QtWidgets.QRadioButton, QtWidgets.QLabel)):
        return "text", widget.text, widget.setText
    return None


def register_tree(root) -> None:
    """Snapshot the original text of every translatable widget under `root`.

    Call once, after the UI is fully built and before the first language
    change. Safe to call again; existing entries are left alone so the
    original Chinese is never overwritten by an already-translated value.
    """
    widgets = [root] + root.findChildren(QtWidgets.QWidget)
    for widget in widgets:
        access = _accessors(widget)
        if access is None:
            continue
        kind, getter, _ = access
        if id(widget) in _registry:
            continue
        text = getter()
        if text:
            _registry[id(widget)] = (widget, kind, text)


def register(widget) -> None:
    """Register a single widget created after startup."""
    access = _accessors(widget)
    if access is None:
        return
    _, getter, _ = access
    if id(widget) not in _registry and getter():
        _registry[id(widget)] = (widget, access[0], getter())


def apply_language(main_window=None, tabs=None) -> None:
    """Re-label every registered widget in the current language."""
    for widget, _kind, original in list(_registry.values()):
        access = _accessors(widget)
        if access is None:
            continue
        try:
            access[2](tr(original))
        except RuntimeError:
            # underlying C++ object deleted -- drop it
            _registry.pop(id(widget), None)

    if main_window is not None:
        main_window.setWindowTitle(tr("工業相機 AI 檢測應用 V1.5.3"))
    if tabs is not None:
        for index, label in enumerate(("相機控制", "TCP 控制與辨識結果")):
            if index < tabs.count():
                tabs.setTabText(index, tr(label))


def set_language(lang: str, main_window=None, tabs=None) -> None:
    global _LANG
    if lang not in {code for _, code in LANGUAGES}:
        return
    _LANG = lang
    apply_language(main_window, tabs)
    _save(lang)


# --- persistence ------------------------------------------------------------

def _save(lang: str) -> None:
    try:
        with open(_SETTINGS, "w", encoding="utf-8") as handle:
            json.dump({"language": lang}, handle)
    except OSError:
        pass  # a read-only install must still run


def load_saved() -> str:
    """Last chosen language, defaulting to Traditional Chinese."""
    global _LANG
    try:
        with open(_SETTINGS, encoding="utf-8") as handle:
            lang = json.load(handle).get("language")
        if lang in {code for _, code in LANGUAGES}:
            _LANG = lang
    except (OSError, ValueError):
        pass
    return _LANG
