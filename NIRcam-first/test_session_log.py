import io
import os

from session_log import TeeStream, log_path, start_session_log


def _collector():
    lines = []
    return lines, lines.append


def test_tee_forwards_text_to_the_original_stream():
    original = io.StringIO()
    _, emit = _collector()

    tee = TeeStream(original, emit)
    tee.write("hello\n")

    assert original.getvalue() == "hello\n"


def test_tee_emits_one_line_per_printed_line():
    original = io.StringIO()
    lines, emit = _collector()

    tee = TeeStream(original, emit)
    # print() writes the message and its newline separately.
    tee.write("first")
    tee.write("\n")
    tee.write("second\n")

    assert lines == ["first", "second"]


def test_tee_holds_partial_lines_until_the_newline_arrives():
    lines, emit = _collector()
    tee = TeeStream(None, emit)

    tee.write("Frame: 1, ")
    assert lines == []          # nothing emitted mid-line

    tee.write("Size: 2448x2048\n")
    assert lines == ["Frame: 1, Size: 2448x2048"]


def test_tee_splits_a_multi_line_chunk():
    lines, emit = _collector()
    tee = TeeStream(None, emit)

    tee.write("a\nb\nc\n")

    assert lines == ["a", "b", "c"]


def test_tee_flush_emits_a_trailing_partial_line():
    # A crash mid-print must not swallow the last thing the app said.
    lines, emit = _collector()
    tee = TeeStream(None, emit)

    tee.write("dying message without newline")
    tee.flush()

    assert lines == ["dying message without newline"]


def test_tee_works_without_an_original_stream():
    # pythonw gives sys.stdout = None; writing must not raise.
    lines, emit = _collector()
    tee = TeeStream(None, emit)

    tee.write("no console here\n")

    assert lines == ["no console here"]


def test_tee_survives_a_broken_original_stream():
    class Broken:
        def write(self, text):
            raise OSError("console went away")

        def flush(self):
            raise OSError("console went away")

    lines, emit = _collector()
    tee = TeeStream(Broken(), emit)

    tee.write("still logged\n")   # must not raise

    assert lines == ["still logged"]


def test_log_path_is_a_dated_file_under_the_logs_dir(tmp_path):
    path = log_path(str(tmp_path))

    assert os.path.dirname(path) == os.path.join(str(tmp_path), "logs")
    assert os.path.basename(path).startswith("nircam_")
    assert path.endswith(".log")


def test_start_session_log_creates_the_file_and_captures_prints(tmp_path,
                                                                monkeypatch):
    import session_log

    monkeypatch.setattr(session_log, "_started", False)
    real_stdout, real_stderr = session_log.sys.stdout, session_log.sys.stderr
    try:
        path = start_session_log(str(tmp_path))
        print("captured line")
        session_log.sys.stdout.flush()
    finally:
        session_log.sys.stdout, session_log.sys.stderr = real_stdout, real_stderr
        for handler in session_log.logging.getLogger("nircam.console").handlers[:]:
            handler.close()
            session_log.logging.getLogger("nircam.console").removeHandler(handler)

    with open(path, encoding="utf-8") as handle:
        contents = handle.read()

    assert "captured line" in contents
    assert "NIRcam session started" in contents


def test_log_is_readable_by_windows_default_ansi_tools(tmp_path, monkeypatch):
    # The operator messages are Traditional Chinese. Without a BOM, the
    # default-ANSI readers on a zh-TW Windows box show mojibake, which defeats
    # the point of having the log at all.
    import session_log

    monkeypatch.setattr(session_log, "_started", False)
    real_stdout, real_stderr = session_log.sys.stdout, session_log.sys.stderr
    try:
        path = start_session_log(str(tmp_path))
        print("正在執行自動初始化...")
        session_log.sys.stdout.flush()
    finally:
        session_log.sys.stdout, session_log.sys.stderr = real_stdout, real_stderr
        logger = session_log.logging.getLogger("nircam.console")
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    raw = open(path, "rb").read()

    assert raw.startswith(b"\xef\xbb\xbf")     # UTF-8 BOM, exactly once
    assert raw.count(b"\xef\xbb\xbf") == 1
    with open(path, encoding="utf-8-sig") as handle:
        assert "正在執行自動初始化..." in handle.read()
