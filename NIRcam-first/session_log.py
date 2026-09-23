"""Mirror everything the GUI prints into a log file on disk.

WHY
---
The app diagnoses itself through `print()` — frame numbers, pixel formats,
camera error codes, trigger decisions. Where that text ends up depends
entirely on how the app was started, and two of the three ways lose it:

  * `run_gui.bat` / a terminal -- visible live, gone when the window closes.
  * `nircam_launcher.pyw` (pythonw, the desktop shortcut) -- there is no
    console at all. The launcher redirects the child's stdout to a file, but
    Python block-buffers a redirected stream, so that file sits empty for
    minutes at a time and loses whatever was still buffered if the process is
    killed. Exactly when you most want the log, it is not there.

So the capture belongs in the app, not in whichever wrapper started it.
`start_session_log()` tees stdout and stderr into `logs/nircam_YYYYMMDD.log`,
flushing every line, and leaves the original streams working so a terminal
run still shows output live.

The log is SIZE-CAPPED on purpose: the acquisition loop prints per frame, so
a production line running all day at ~23 fps would otherwise write hundreds of
megabytes. Rotation keeps the most recent MAX_BYTES * (BACKUP_COUNT + 1).
"""
import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

LOG_DIR_NAME = "logs"
MAX_BYTES = 5 * 1024 * 1024   # per file
BACKUP_COUNT = 3              # keep ~20 MB of history in total

_started = False


class TeeStream:
    """File-like object that forwards writes to a stream and a line sink.

    Text arrives in whatever chunks `print()` decides to use — typically the
    message and its newline as two separate writes — so partial lines are held
    back until the newline shows up. That keeps one log line per printed line
    instead of splitting a message across timestamps.
    """

    def __init__(self, original, emit_line):
        self._original = original
        self._emit_line = emit_line
        self._pending = ""

    def write(self, text):
        if self._original is not None:
            try:
                self._original.write(text)
                self._original.flush()
            except Exception:
                # A broken console must never take the acquisition loop down.
                pass

        self._pending += text
        while "\n" in self._pending:
            line, self._pending = self._pending.split("\n", 1)
            self._emit_line(line)
        return len(text)

    def flush(self):
        # Emit a trailing partial line so a message that never got its newline
        # (a crash mid-print, say) still reaches the log.
        if self._pending:
            self._emit_line(self._pending)
            self._pending = ""
        if self._original is not None:
            try:
                self._original.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        # Some libraries probe for a real fd; report the original's if there
        # is one, otherwise say "not a real file" the way a pipe-less
        # pythonw stream would.
        if self._original is None:
            raise OSError("no file descriptor")
        return self._original.fileno()


def log_path(base_dir=None):
    """Path of today's log file."""
    base = base_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, LOG_DIR_NAME,
                        "nircam_{:%Y%m%d}.log".format(datetime.datetime.now()))


def start_session_log(base_dir=None):
    """Start teeing stdout/stderr to today's log file. Returns its path.

    Safe to call more than once; only the first call installs anything.
    """
    global _started
    if _started:
        return log_path(base_dir)

    path = log_path(base_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger("nircam.console")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # utf-8-sig, not plain utf-8: the messages are Traditional Chinese, and on
    # a zh-TW Windows box the default-ANSI readers (PowerShell 5.1's
    # Get-Content, `type`, older editors) render BOM-less UTF-8 as mojibake.
    # The BOM is written once per file — Python only emits it at position 0,
    # so appending across sessions does not sprinkle them mid-file.
    handler = RotatingFileHandler(path, maxBytes=MAX_BYTES,
                                  backupCount=BACKUP_COUNT,
                                  encoding="utf-8-sig")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s",
                                           datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)

    sys.stdout = TeeStream(sys.stdout, logger.info)
    sys.stderr = TeeStream(sys.stderr, logger.info)
    _started = True

    print("=" * 70)
    print("NIRcam session started {:%Y-%m-%d %H:%M:%S}".format(
        datetime.datetime.now()))
    print("log file: {}".format(path))
    print("=" * 70)
    return path
