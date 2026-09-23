"""Pure helpers for resolving and decoding the raw camera frame buffer shape.

Kept dependency-free (numpy only) so these can be unit tested without the
camera SDK: CamOperation_class.py's own imports require the vendor DLL to be
loadable (MvImport/MvCameraControl_class.py calls ctypes.cdll.LoadLibrary at
module import time), so it can't be imported in a plain test environment.
"""
import numpy as np


def resolve_capture_shape(frame_height, frame_width, manual_width, manual_height):
    """Return the (height, width) to reshape the raw frame buffer to.

    The manual override wins only when both dimensions are set and positive;
    otherwise the camera-reported frame size is used.
    """
    if manual_width and manual_height and manual_width > 0 and manual_height > 0:
        return manual_height, manual_width
    return frame_height, frame_width


def decode_raw_frame(buffer, height, width):
    """Reshape a single-channel raw frame out of a possibly oversized buffer.

    Uses an explicit count=height*width so only the frame's own bytes are
    read, even when `buffer` is sized to the camera's PayloadSize rather than
    the actual frame length (the mismatch that causes
    "cannot reshape array of size N into shape (H, W)").
    """
    return np.frombuffer(buffer, dtype=np.uint8, count=height * width).reshape(height, width)


def normalize_pixel_type(en_pixel_type):
    """Strip the vendor "custom" flag bit (bit 31 / 0x80000000) some cameras
    OR into an otherwise-standard GenICam PFNC PixelType value.

    Observed in production: a camera configured as BayerRG8 (0x01080009)
    reported frames with enPixelType=2164785161 (0x81080009) — the same
    value with bit 31 set — which meant none of the exact-equality pixel
    format comparisons in CamOperation_class.py recognized it, so every
    frame was silently skipped as "Unsupported pixel format". The extra bit
    does not change the underlying pixel layout, so masking it off before
    comparing against the known PixelType_Gvsp_* constants is safe.
    """
    return en_pixel_type & 0x7FFFFFFF
