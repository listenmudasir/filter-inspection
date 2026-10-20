"""Ships to the vendor as FingerprintTool.exe (built by build.ps1).

Run once on the target machine, after installing FilterInspection, before
the software has a license.key. Prints the machine's fingerprint string so
the vendor can send it back to you; you feed it into generate_license.py
(kept on your own machine, never shipped) to produce their license.key.

Deliberately does not import anything from the main app besides
license_check, so it can be compiled and shipped standalone without
dragging in torch/PyQt5/the camera SDK.
"""
import os
import sys

# Windows' console defaults to the system ANSI codepage (e.g. cp950 on a
# Traditional Chinese machine), not UTF-8, so printing Chinese text without
# this reconfigure comes out as mojibake in the vendor's terminal. Guarded
# because reconfigure() isn't available on very old Python or when stdout
# has already been replaced with something that doesn't support it.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "NIRcam-first")
)
from license_check import get_machine_fingerprint  # noqa: E402


def main() -> int:
    fingerprint = get_machine_fingerprint()
    if fingerprint is None:
        print("無法讀取這台機器的識別資訊。請確認 PowerShell 可以正常執行，"
              "並將這個錯誤回報給軟體提供者。")
        return 1

    print("=" * 60)
    print("機器授權識別碼（Machine Fingerprint）")
    print("=" * 60)
    print(fingerprint)
    print("=" * 60)
    print("請將上面這一串文字完整回傳給軟體提供者，"
          "以取得對應這台機器的 license.key。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
