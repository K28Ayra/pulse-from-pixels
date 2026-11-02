"""Individual Typology Angle (ITA) from mean skin colour, a rough skin-tone proxy.

ITA = atan((L* - 50) / b*) in degrees. Groups follow Chardon et al. (1991). Camera
white balance and lighting shift it, so it only ranks people within one dataset.
"""

import numpy as np

GROUPS = [(55, "very light"), (41, "light"), (28, "intermediate"), (10, "tan"), (-30, "brown")]


def srgb_to_lab(rgb):
    """8-bit sRGB (..., 3) to CIE L*a*b* with a D65 white."""
    c = np.asarray(rgb, dtype=float) / 255.0
    linear = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    matrix = np.array([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ])
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    t = xyz / white
    f = np.where(t > (6 / 29) ** 3, np.cbrt(t), t / (3 * (6 / 29) ** 2) + 4 / 29)
    lightness = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def ita_degrees(rgb):
    lab = srgb_to_lab(rgb)
    return np.degrees(np.arctan2(lab[..., 0] - 50, lab[..., 2]))


def ita_group(ita):
    for lowest, name in GROUPS:
        if ita > lowest:
            return name
    return "dark"
