import numpy as np

from pulsecam.skin import ita_degrees, ita_group, srgb_to_lab


def test_white_is_lightness_100():
    lab = srgb_to_lab([255, 255, 255])
    assert abs(lab[0] - 100) < 0.1
    assert abs(lab[1]) < 0.5
    assert abs(lab[2]) < 0.5


def test_lighter_skin_has_a_larger_angle():
    light = ita_degrees([230, 190, 165])
    medium = ita_degrees([190, 140, 110])
    dark = ita_degrees([90, 60, 45])
    assert light > medium > dark
    assert ita_group(light) in ["very light", "light"]
    assert ita_group(dark) in ["brown", "dark"]
