import numpy as np
import pytest

from image_shape import resolve_capture_shape, decode_raw_frame


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
