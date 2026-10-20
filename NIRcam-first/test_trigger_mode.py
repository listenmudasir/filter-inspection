"""The trigger configuration set in MVS must survive opening the camera here.

FakeCamera stands in for MvCamera: it holds GenICam enum nodes by name and
records every write, so the tests can assert what the program changed.
"""
from CamOperation_class import CameraOperation

# TriggerSource enum values as the Hikvision cameras report them.
SOURCES = {0: "Line0", 1: "Line1", 2: "Line2", 7: "Software"}


class FakeCamera:
    def __init__(self, trigger_mode=1, trigger_source=0):
        self.nodes = {"TriggerMode": trigger_mode, "TriggerSource": trigger_source}
        self.writes = []

    def MV_CC_GetEnumValue(self, key, st):
        if key not in self.nodes:
            return 0x80000106  # MV_E_GC_PROPERTY
        st.nCurValue = self.nodes[key]
        return 0

    def MV_CC_GetEnumEntrySymbolic(self, key, entry):
        if key != "TriggerSource" or entry.nValue not in SOURCES:
            return 0x80000106
        entry.chSymbolic = SOURCES[entry.nValue].encode("ascii")
        return 0

    def MV_CC_SetEnumValue(self, key, value):
        self.writes.append((key, value))
        self.nodes[key] = value
        return 0

    def MV_CC_SetEnumValueByString(self, key, value):
        self.writes.append((key, value))
        self.nodes[key] = {v: k for k, v in SOURCES.items()}[value]
        return 0


def _operation(cam):
    op = CameraOperation(cam, None, 0)
    op.b_open_device = True
    return op


def test_reads_line0_trigger_set_in_mvs():
    op = _operation(FakeCamera(trigger_mode=1, trigger_source=0))

    assert op.Get_trigger_mode() == (True, "Line0")


def test_reads_software_trigger():
    op = _operation(FakeCamera(trigger_mode=1, trigger_source=7))

    assert op.Get_trigger_mode() == (True, "Software")


def test_reads_continuous_mode():
    is_trigger, _ = _operation(FakeCamera(trigger_mode=0)).Get_trigger_mode()

    assert is_trigger is False


def test_unreadable_trigger_mode_reports_unknown():
    cam = FakeCamera()
    del cam.nodes["TriggerMode"]

    assert _operation(cam).Get_trigger_mode() == (None, None)


def test_switching_to_trigger_mode_keeps_the_cameras_source():
    cam = FakeCamera(trigger_mode=0, trigger_source=0)  # Line0, currently continuous

    assert _operation(cam).Set_trigger_mode(True) == 0

    assert cam.writes == [("TriggerMode", 1)]
    assert cam.nodes["TriggerSource"] == 0


def test_explicit_source_is_still_applied():
    cam = FakeCamera(trigger_mode=1, trigger_source=0)

    assert _operation(cam).Set_trigger_mode(True, source="Software") == 0

    assert ("TriggerSource", "Software") in cam.writes
