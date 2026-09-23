import numpy as np
import pytest

from image_shape import (
    resolve_capture_shape,
    decode_raw_frame,
    is_high_bandwidth_pixel_type,
    pixel_type_bytes_per_pixel,
)


def test_resolve_capture_shape_uses_camera_size_when_no_override():
    assert resolve_capture_shape(2048, 2448, None, None) == (2048, 2448)


def test_resolve_capture_shape_uses_camera_size_when_override_partial():
    # Only one of width/height set -> override must not apply.
    assert resolve_capture_shape(2048, 2448, 2448, None) == (2048, 2448)
    assert resolve_capture_shape(2048, 2448, None, 2048) == (2048, 2448)


def test_resolve_capture_shape_uses_camera_size_when_override_zero():
    assert resolve_capture_shape(2048, 2448, 0, 0) == (2048, 2448)


def test_resolve_capture_shape_uses_manual_override_when_both_set():
    assert resolve_capture_shape(2048, 2448, 2200, 1944) == (1944, 2200)


def test_decode_raw_frame_reads_only_needed_bytes_from_oversized_buffer():
    height, width = 4, 3
    frame_bytes = bytes(range(height * width))
    # Buffer padded larger than one frame, like a PayloadSize-sized grab buffer.
    oversized_buffer = frame_bytes + bytes(50)

    result = decode_raw_frame(oversized_buffer, height, width)

    assert result.shape == (height, width)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, np.arange(12, dtype=np.uint8).reshape(4, 3))


def test_decode_raw_frame_raises_when_buffer_too_small():
    with pytest.raises(ValueError):
        decode_raw_frame(bytes(5), 4, 3)  # needs 12 bytes, only has 5


# --- High Bandwidth (Hikvision lossless-compressed transport) -----------------
# HB pixel types are the standard PFNC value with bit 31 set, e.g.
# PixelType_Gvsp_HB_BayerRG8 = 0x81080009 vs BayerRG8 = 0x01080009.
# The payload is compressed and must go through MV_CC_HB_Decode before it can
# be treated as raw Bayer/mono data.

def test_is_high_bandwidth_pixel_type_detects_hb_variants():
    assert is_high_bandwidth_pixel_type(0x81080009) is True   # HB_BayerRG8
    assert is_high_bandwidth_pixel_type(0x81080001) is True   # HB_Mono8
    assert is_high_bandwidth_pixel_type(0x82180014) is True   # HB_RGB8_Packed


def test_is_high_bandwidth_pixel_type_rejects_standard_variants():
    assert is_high_bandwidth_pixel_type(0x01080009) is False  # BayerRG8
    assert is_high_bandwidth_pixel_type(0x01080001) is False  # Mono8
    assert is_high_bandwidth_pixel_type(0x02180014) is False  # RGB8_Packed
    assert is_high_bandwidth_pixel_type(0) is False


def test_pixel_type_bytes_per_pixel_reads_pfnc_bit_depth():
    assert pixel_type_bytes_per_pixel(0x01080009) == 1   # BayerRG8, 8 bpp
    assert pixel_type_bytes_per_pixel(0x01100003) == 2   # Mono10, 16 bpp
    assert pixel_type_bytes_per_pixel(0x02180014) == 3   # RGB8_Packed, 24 bpp


def test_pixel_type_bytes_per_pixel_ignores_high_bandwidth_flag():
    # Sizing the decode buffer has to work straight off the HB value.
    assert pixel_type_bytes_per_pixel(0x81080009) == 1   # HB_BayerRG8
    assert pixel_type_bytes_per_pixel(0x82180014) == 3   # HB_RGB8_Packed


def test_pixel_type_bytes_per_pixel_rounds_sub_byte_depths_up():
    # Packed 10/12-bit formats carry a non-byte-aligned depth; a decode buffer
    # must never be sized short, so round up.
    assert pixel_type_bytes_per_pixel(0x010C0004) == 2   # Mono10_Packed, 12 bpp
