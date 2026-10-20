"""Internal tool -- run on YOUR OWN machine with `python`, never shipped to
the vendor and never compiled into an exe (it embeds the same SECRET_KEY
as license_check.py, which must stay private).

Usage:
    python tools/license/generate_license.py <fingerprint-string> [output-path]

<fingerprint-string> is what the vendor read off FingerprintTool.exe and
sent back to you. Writes a signed license.key (defaults to ./license.key
next to this invocation) -- hand that single file back to the vendor to
drop into their install folder.
"""
import hashlib
import hmac
import os
import sys

try:  # avoid mojibake for the Chinese status text on a non-UTF-8 console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "NIRcam-first")
)
from license_check import PLACEHOLDER_SECRET, SECRET_KEY  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    fingerprint = sys.argv[1].strip()
    if len(fingerprint) != 64 or not all(c in "0123456789abcdef" for c in fingerprint.lower()):
        print("這不像是一個有效的機器識別碼（應該是 64 個十六進位字元的 SHA-256 雜湊）。")
        return 1

    if SECRET_KEY == PLACEHOLDER_SECRET:
        print("警告：找不到 NIRcam-first/license_secret.py，正在用公開的預留金鑰簽發。")
        print("  這組授權碼只適用於開發測試版，不能給廠商。先執行 python tools/license/new_secret.py")

    out_path = sys.argv[2] if len(sys.argv) > 2 else "license.key"
    signature = hmac.new(
        SECRET_KEY, fingerprint.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(signature)

    print(f"已產生授權檔：{out_path}")
    print(f"授權碼：{signature}")
    print("首次安裝：把上面的授權碼傳給廠商，貼進安裝精靈的「授權碼」欄位。")
    print("換機器重新啟用：把這個 license.key 檔案傳給廠商，覆蓋安裝目錄裡的舊檔。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
