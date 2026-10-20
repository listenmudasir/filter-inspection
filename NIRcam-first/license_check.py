"""Simple machine-bound license check, shared by nircam_launcher.pyw and
BasicDemo.py so bypassing the splash and launching BasicDemo.exe directly
does not skip the check (see PACKAGING.md section 5).

This is a soft binding meant to stop "copy the install folder to another
machine", not a hardware dongle. The HMAC key that signs and verifies codes
lives in license_secret.py next to this file. That file is gitignored and is
created once with tools/license/new_secret.py; Nuitka compiles it into the
exes at build time. Without it, a public placeholder is used so the dev tree
still runs, and build.ps1 refuses to package a release.
"""
from __future__ import annotations

import ctypes
import hashlib
import hmac
import os
import subprocess
import sys

PLACEHOLDER_SECRET = b"REPLACE_WITH_YOUR_OWN_SECRET_BEFORE_BUILDING_A_RELEASE"

try:
    from license_secret import SECRET_KEY
except ImportError:
    SECRET_KEY = PLACEHOLDER_SECRET

_FINGERPRINT_TIMEOUT_S = 10


def _app_dir() -> str:
    """Directory of the exe/script that was launched -- where license.key lives.

    sys.argv[0], not __file__: in a Nuitka onefile build (nircam_launcher.exe)
    __file__ points into the temp extraction folder, not the install folder.
    Un-frozen, argv[0] is BasicDemo.py / nircam_launcher.pyw, so this is still
    NIRcam-first/.
    """
    return os.path.dirname(os.path.abspath(sys.argv[0]))


def get_machine_fingerprint() -> str | None:
    """SHA-256 of this machine's SMBIOS UUID, or None if it can't be read.

    Deliberately does NOT shell out to wmic.exe: it is deprecated and
    increasingly absent on Windows 11. Get-CimInstance is the supported
    replacement. Any failure here (PowerShell missing, empty/garbled
    output, timeout) is treated as "can't verify" rather than raising --
    callers must not let this crash into a raw traceback in front of the
    vendor.
    """
    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystemProduct).UUID",
            ],
            capture_output=True,
            text=True,
            timeout=_FINGERPRINT_TIMEOUT_S,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except Exception:
        return None

    uuid = (result.stdout or "").strip()
    if result.returncode != 0 or not uuid:
        return None
    return hashlib.sha256(uuid.encode("utf-8")).hexdigest()


def _license_path() -> str:
    return os.path.join(_app_dir(), "license.key")


def verify_license() -> tuple[bool, str]:
    """Check license.key against this machine's fingerprint.

    Returns (ok, message). message is user-facing Traditional Chinese,
    safe to put straight into a message box -- never leaks the fingerprint
    or secret.
    """
    fingerprint = get_machine_fingerprint()
    if fingerprint is None:
        return False, "無法驗證這台機器的授權資訊，請聯絡窗口協助排除。"

    path = _license_path()
    if not os.path.isfile(path):
        return False, "找不到授權檔 license.key，請聯絡窗口取得授權。"

    try:
        with open(path, "r", encoding="utf-8") as f:
            signature = f.read().strip()
    except Exception:
        return False, "授權檔無法讀取，請聯絡窗口協助排除。"

    if not signature:
        return False, "授權檔內容為空，請聯絡窗口重新核發。"

    expected = hmac.new(
        SECRET_KEY, fingerprint.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return False, "授權檔與這台機器不符，請聯絡窗口重新核發。"

    return True, ""


def _show_error(message: str) -> None:
    try:
        ctypes.windll.user32.MessageBoxW(
            None, message, "NIRcam Inspection - 授權驗證", 0x10
        )
    except Exception:
        print(message, file=sys.stderr)


def enforce_license_or_exit() -> None:
    """Verify the license and terminate the process (no traceback) if it
    fails. Call this once from each of the two exes -- see PACKAGING.md
    section 5 for why both need it.
    """
    ok, message = verify_license()
    if not ok:
        _show_error(message)
        sys.exit(1)
