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


# Hikvision marks its High Bandwidth (lossless-compressed transport) pixel
# formats by setting bit 31 on the standard PFNC value — PixelType_Gvsp_HB_BayerRG8
# is 0x81080009 against BayerRG8's 0x01080009. The wire payload is compressed,
# so it must go through MV_CC_HB_Decode before it can be read as raw Bayer/mono;
# reshaping it directly produces noise.
_HIGH_BANDWIDTH_FLAG = 0x80000000


def is_high_bandwidth_pixel_type(en_pixel_type):
    """True when the camera reported a High Bandwidth (HB_*) pixel format."""
    return bool(en_pixel_type & _HIGH_BANDWIDTH_FLAG)


def pixel_type_bytes_per_pixel(en_pixel_type):
    """Bytes per pixel for a PFNC pixel type, rounded up for packed depths.

    Bits 23-16 of a PFNC value hold the bit depth (0x08 for BayerRG8, 0x10 for
    Mono10, 0x18 for RGB8_Packed). Works on HB_* values too, since the High
    Bandwidth flag lives in bit 31 and leaves the depth field untouched — that
    is what lets a decode buffer be sized before the decode runs.
    """
    bits_per_pixel = (en_pixel_type >> 16) & 0xFF
    return (bits_per_pixel + 7) // 8
