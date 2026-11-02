"""Heart rate from the webcam. Nothing is saved and nothing is sent anywhere.

    python scripts/live.py                    # camera from configs/default.yaml
    python scripts/live.py --video vid.avi    # a file instead of the camera
    python scripts/live.py --video vid.avi --no-window   # print readings only

Press q in the window to quit. Needs models/face_landmarker.task and
models/factorizephys.onnx (setup_tools.sh, scripts/export_onnx.py).
"""

import argparse
import time

import cv2
import numpy as np

from pulsecam.config import load_config
from pulsecam.face import FaceTracker
from pulsecam.live import LiveHeartRate
from pulsecam.privacy import block_writes_and_network

GREEN = (80, 200, 80)
GREY = (150, 150, 150)
TRACE_S = 6


def open_source(cfg, video):
    if video:
        return cv2.VideoCapture(video)
    camera = cfg["camera"]
    capture = cv2.VideoCapture(camera["index"])
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, camera["width"])
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, camera["height"])
    capture.set(cv2.CAP_PROP_FPS, camera["fps"])
    return capture


def draw(frame, heart, reading, fps):
    """HR, flag, face box and the last few seconds of pulse, drawn on the frame."""
    height, width = frame.shape[:2]
    if heart.box is not None:
        left, top, side = heart.box
        cv2.rectangle(frame, (left, top), (left + side, top + side), GREY, 1)

    if reading is None or heart.status in ("looking for a face", "starting"):
        cv2.putText(frame, heart.status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREY, 2)
    else:
        colour = GREEN if reading["trusted"] else GREY
        text = str(int(round(reading["hr"]))) + " bpm"
        cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, colour, 3)
        if not reading["trusted"]:
            cv2.putText(frame, "not trusted, keep still", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)

    times, values = heart.pulse()
    recent = times >= times[-1] - TRACE_S if len(times) else times
    if recent.sum() > 2:
        t = times[recent]
        v = values[recent]
        x = (width - 20) * (t - t[0]) / TRACE_S + 10
        y = height - 50 - 30 * v / (np.abs(v).max() + 1e-9)
        points = np.stack([x, y], axis=1).astype(np.int32)
        cv2.polylines(frame, [points], False, GREEN, 2)

    cv2.putText(frame, str(int(fps)) + " fps", (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREY, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="read this file instead of the camera")
    parser.add_argument("--no-window", action="store_true", help="print readings, show nothing")
    args = parser.parse_args()

    cfg = load_config(live_mode=True)
    face = FaceTracker()
    heart = LiveHeartRate(cfg, face.find, background=not args.no_window)
    capture = open_source(cfg, args.video)
    # from here on any file write or network call from Python raises an error
    block_writes_and_network()

    reading = None
    fps = 0.0
    last = time.perf_counter()
    started = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        now = time.perf_counter()
        if args.video:
            time_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        else:
            if started is None:
                started = now
            time_s = now - started
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
        last = now

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new = heart.push(rgb, time_s)
        if new is not None:
            reading = new
            if args.no_window:
                print(round(time_s, 1), "s", round(reading["hr"]), "bpm", "ok" if reading["trusted"] else "not trusted")
        if heart.status == "looking for a face":
            reading = None

        if not args.no_window:
            draw(frame, heart, reading, fps)
            cv2.imshow("pulsecam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    capture.release()
    face.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
