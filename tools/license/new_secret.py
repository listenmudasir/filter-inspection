"""Create NIRcam-first/license_secret.py with a random HMAC key. Run ONCE.

    python tools/license/new_secret.py

The file is gitignored and is compiled into the exes by build.ps1. Every
license code is derived from it, so:
  - back it up somewhere safe (a password manager). Lost = you can no longer
    issue codes that work with builds already in the field.
  - never overwrite it. A new key invalidates every code issued so far.
  - copy it by hand (not via git) to any other machine that builds releases.
"""
import os
import secrets
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TARGET = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "..", "..", "NIRcam-first", "license_secret.py")


def main() -> int:
    target = os.path.normpath(TARGET)
    if os.path.exists(target):
        print(f"已存在，不覆蓋：{target}")
        print("換金鑰會讓所有已發出的授權碼失效。真的要換，請先手動刪除這個檔案再執行。")
        return 1

    with open(target, "w", encoding="utf-8") as f:
        f.write('"""License signing key. Gitignored -- never commit, back up offline."""\n')
        f.write(f'SECRET_KEY = b"{secrets.token_hex(32)}"\n')

    print(f"已產生授權金鑰檔：{target}")
    print("1. 請立刻把這個檔案備份到安全的地方（例如密碼管理器），遺失就無法再為已出貨的版本產生授權碼。")
    print("2. 這個檔案不會進 git；其他要打包的電腦請手動複製過去。")
    print("3. 之後打包的版本都會使用這把金鑰，舊的測試授權碼要重新產生。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
