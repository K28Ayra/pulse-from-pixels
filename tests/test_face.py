import numpy as np
import pytest

pytest.importorskip("mediapipe")

from pulsecam.face import FACE_OVAL, mean_rgb_in_polygon  # noqa: E402


def test_face_oval_is_a_closed_outline():
    assert len(FACE_OVAL) == 36
    assert len(set(FACE_OVAL)) == 36


def test_mean_inside_polygon_ignores_outside_pixels():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[20:60, 20:60] = [200, 100, 50]
    square = np.array([[25, 25], [55, 25], [55, 55], [25, 55]], dtype=float)
    assert np.allclose(mean_rgb_in_polygon(frame, square), [200, 100, 50])


def test_polygon_outside_the_frame_gives_nan():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    far_away = np.array([[80, 80], [90, 80], [90, 90]], dtype=float)
    assert np.all(np.isnan(mean_rgb_in_polygon(frame, far_away)))
