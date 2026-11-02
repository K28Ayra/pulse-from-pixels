"""MediaPipe Face Landmarker wrapper and skin-region colour averages."""

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = Path("models") / "face_landmarker.task"
CROP_BOX = 1.5  # crop side relative to the face-oval box, as in rPPG-Toolbox
# a box that moves every frame adds jitter the deep models react to
BOX_EVERY = 30

_connections = mp.tasks.vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL
# the oval connections form one closed chain, so their start points in order are the outline
FACE_OVAL = [c.start for c in _connections]


class FaceTracker:
    """Landmarks for one face per frame, in pixels. Call close() when done."""

    def __init__(self, model_path=MODEL_PATH):
        base = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self.last_ms = -1

    def find(self, rgb_frame, time_s):
        """(478, 2) array of x, y pixel positions, or None if no face."""
        # VIDEO mode needs strictly increasing timestamps, UBFC repeats some
        ms = max(int(round(time_s * 1000)), self.last_ms + 1)
        self.last_ms = ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.landmarker.detect_for_video(image, ms)
        if not result.face_landmarks:
            return None
        height, width = rgb_frame.shape[:2]
        points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
        return points

    def close(self):
        self.landmarker.close()


def mean_rgb_in_polygon(rgb_frame, polygon):
    """Mean R, G, B of the pixels inside a polygon given as (k, 2) pixel points."""
    height, width = rgb_frame.shape[:2]
    left = max(int(np.floor(polygon[:, 0].min())), 0)
    top = max(int(np.floor(polygon[:, 1].min())), 0)
    right = min(int(np.ceil(polygon[:, 0].max())) + 1, width)
    bottom = min(int(np.ceil(polygon[:, 1].max())) + 1, height)
    if right <= left or bottom <= top:
        return np.full(3, np.nan)

    crop = rgb_frame[top:bottom, left:right]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    shifted = polygon - np.array([left, top])
    cv2.fillPoly(mask, [np.round(shifted).astype(np.int32)], 1)
    if mask.sum() == 0:
        return np.full(3, np.nan)
    means = cv2.mean(crop, mask=mask)
    return np.array(means[:3])


# Skin patches as landmark outlines (MediaPipe face mesh numbers). "left" and
# "right" are as seen in the image. The forehead stops at landmarks 108/151/337
# because hair often covers the top of it.
PATCHES = {
    "forehead": [108, 151, 337, 336, 9, 107],
    "nose": [193, 168, 417, 351, 195, 122],
    "cheek_left": [117, 118, 101, 36, 205, 187, 123],
    "cheek_right": [346, 347, 330, 266, 425, 411, 352],
    "jaw_left": [207, 216, 212, 214, 192, 213],
    "jaw_right": [427, 436, 432, 434, 416, 433],
    "chin": [84, 17, 314, 421, 200, 201],
    "face": FACE_OVAL,
}


def crop_box(points):
    """Square box (left, top, side) around the face oval."""
    outline = points[FACE_OVAL]
    centre = (outline.min(axis=0) + outline.max(axis=0)) / 2
    side = CROP_BOX * (outline.max(axis=0) - outline.min(axis=0)).max()
    left = int(round(centre[0] - side / 2))
    top = int(round(centre[1] - side / 2))
    return left, top, int(round(side))


def face_crop(frame, box, size):
    left, top, side = box
    # pad so a box hanging over the border still gives a square crop
    padded = cv2.copyMakeBorder(frame, side, side, side, side, cv2.BORDER_CONSTANT, value=0)
    crop = padded[top + side:top + 2 * side, left + side:left + 2 * side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
