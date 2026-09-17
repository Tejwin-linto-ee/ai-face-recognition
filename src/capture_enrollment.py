"""Caregiver-assisted camera enrollment for an authorized person.

Example:
    python src/capture_enrollment.py --name Tejwin

Press Space to save a good face crop. Capture 12-20 varied images, including
normal appearance, glasses (when worn), and modest left/right head turns.
Then run src/build_embeddings.py again.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

# Ensure src directory is in sys.path for local module resolution
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cv2

from build_embeddings import padded_crop
from recognize import configure_tensorflow_gpu, detect_faces


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture enrollment photos for a caregiver-managed face profile.")
    parser.add_argument("--name", required=True, help="Person name, for example Tejwin")
    parser.add_argument("--faces-dir", type=Path, default=Path("faces"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--target", type=int, default=15, help="Recommended photo count")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9 _-]{1,60}", args.name):
        raise SystemExit("Use letters, numbers, spaces, hyphens, or underscores for --name.")

    destination = args.faces_dir / args.name
    destination.mkdir(parents=True, exist_ok=True)
    configure_tensorflow_gpu(cpu_threads=2)
    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if __import__("os").name == "nt" else cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise SystemExit("Could not open webcam.")

    window_name = "Caregiver Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    saved = 0
    frame_count = 0
    latest_box = None
    message = "One face only. Press SPACE to save a clear photo. Press q to finish."
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame_count += 1
            if frame_count % 5 == 0:
                try:
                    boxes = detect_faces(frame, "yunet", 42, 0.60, 640, 360)
                    latest_box = boxes[0] if len(boxes) == 1 else None
                    if len(boxes) > 1:
                        message = "More than one face visible - ask others to step away."
                    elif not boxes:
                        message = "Look at the camera. Keep your face well lit."
                    else:
                        message = "Good. Press SPACE to save this photo."
                except Exception as exc:
                    message = f"Detector problem: {exc}"
            if latest_box is not None:
                x, y, w, h = latest_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 190, 0), 3)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (24, 24, 24), -1)
            cv2.putText(frame, f"{args.name}: {saved}/{args.target} photos", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, message, (16, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" ") and latest_box is not None:
                x, y, w, h = latest_box
                crop = padded_crop(frame, {"x": x, "y": y, "w": w, "h": h})
                if crop is not None:
                    filename = destination / f"enroll_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(str(filename), crop)
                    saved += 1
                    message = f"Saved {saved}. Change expression or head angle, then save another."
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} photos to {destination}. Run: python build_embeddings.py")


if __name__ == "__main__":
    main()
