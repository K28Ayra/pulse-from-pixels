import numpy as np

from pulsecam.datasets import (
    count_frames,
    iter_frames,
    list_ubfc_subjects,
    load_ubfc_subject,
    video_info,
)
from pulsecam.synthetic import fake_ubfc_subject


def test_fake_ubfc_layout_loads(tmp_path):
    fake_ubfc_subject(tmp_path, "subject10")
    fake_ubfc_subject(tmp_path, "subject9")
    fake_ubfc_subject(tmp_path, "subject1")

    names = list_ubfc_subjects(tmp_path)
    assert names == ["subject1", "subject9", "subject10"]

    subject = load_ubfc_subject(tmp_path, "subject9")
    assert len(subject["ppg"]) == len(subject["ppg_times"])
    assert np.allclose(subject["device_hr"], 72)


def test_video_reading(tmp_path):
    n_written = fake_ubfc_subject(tmp_path, "subject1", fps=30, seconds=2)
    path = tmp_path / "DATASET_2" / "subject1" / "vid.avi"

    info = video_info(path)
    assert info["width"] == 64
    assert info["height"] == 48
    assert round(info["fps"]) == 30

    assert count_frames(path) == n_written

    first = next(iter_frames(path))
    assert first.shape == (48, 64, 3)
    # the oval is skin coloured, so red should beat blue in the middle (RGB order)
    middle = first[24, 32]
    assert middle[0] > middle[2]
