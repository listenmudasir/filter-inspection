"""Not for humans -- this is what Inno Setup's install wizard shells out to
(see installer/setup.iss [Code] section) to fold license activation into the
install flow itself, instead of a separate FingerprintTool.exe round trip
after installing. Every result is written to a file the caller passed in,
never to stdout, because Inno Setup's Exec() does not capture stdout.

Commands:
    license_installer_helper.py fingerprint <out_file>
        Writes this machine's fingerprint (or "" on failure) to out_file.

    license_installer_helper.py check <code> <out_file>
        Verifies <code> against this machine's fingerprint. Writes one of:
          OK:<canonical-code>   -- <code> was correct; write THIS string
                                    (not the user's raw input) as license.key,
                                    since it's guaranteed to match what
                                    license_check.verify_license() expects.
          FAIL:NO_FINGERPRINT   -- couldn't read this machine's fingerprint
          FAIL:MISMATCH         -- <code> doesn't match this machine

        Deliberately ASCII-only status codes, not free-text messages: Inno
        Setup's LoadStringFromFile reads into an AnsiString, so anything
        with Traditional Chinese written here would get mangled on the way
        back. The installer script (installer/setup.iss) owns the actual
        user-facing message text for each code.

Shares SECRET_KEY with license_check.py (imported, not duplicated) and with
tools/license/generate_license.py, which is what produces the <code> a
vendor pastes into the installer.
"""
import hashlib
import hmac
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "NIRcam-first")
)
from license_check import SECRET_KEY, get_machine_fingerprint  # noqa: E402


def _write(out_path: str, text: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def cmd_fingerprint(out_path: str) -> int:
    fingerprint = get_machine_fingerprint()
    _write(out_path, fingerprint or "")
    return 0 if fingerprint else 1


def cmd_check(code: str, out_path: str) -> int:
    fingerprint = get_machine_fingerprint()
    if fingerprint is None:
        _write(out_path, "FAIL:NO_FINGERPRINT")
        return 1

    expected = hmac.new(
        SECRET_KEY, fingerprint.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    if hmac.compare_digest(code.strip().lower(), expected):
        _write(out_path, f"OK:{expected}")
        return 0

    _write(out_path, "FAIL:MISMATCH")
    return 1


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    command = sys.argv[1]
    if command == "fingerprint" and len(sys.argv) == 3:
        return cmd_fingerprint(sys.argv[2])
    if command == "check" and len(sys.argv) == 4:
        return cmd_check(sys.argv[2], sys.argv[3])

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
