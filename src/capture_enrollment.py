"""Biometric User Enrollment Assistant for Multimodal Biometric Smart Hub.

Example:
    python src/capture_enrollment.py --name "Alex"

Press Space to save a high-quality face crop. Capture 12-20 varied samples,
including slight head turns, different expressions, and with/without glasses.
Then run src/build_embeddings.py to update the biometric database.
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
from detector import create_face_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture enrollment face crops for an authorized user profile.")
    parser.add_argument("--name", required=True, help="Authorized user identity, e.g. Alex")
    parser.add_argument("--faces-dir", type=Path, default=Path("faces"), help="Directory to store captured face crops")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--target", type=int, default=15, help="Target sample count (12-20 recommended)")
    parser.add_argument("--detector-onnx", type=str, default=None, help="Optional SCRFD ONNX model path")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9 _-]{1,60}", args.name):
        raise SystemExit("Use alphanumeric characters, spaces, hyphens, or underscores for --name.")

    destination = args.faces_dir / args.name
    destination.mkdir(parents=True, exist_ok=True)

    detector = create_face_detector(model_path=args.detector_onnx, conf_threshold=0.60)

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not camera.isOpened():
        raise SystemExit("[Enrollment Error] Could not connect to camera interface.")

    window_name = "Biometric User Enrollment"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    saved = 0
    frame_count = 0
    latest_box = None
    message = "Single face only. Press SPACE to capture sample. Press 'q' to complete."

    auto_mode = False
    last_auto_capture_time = 0.0

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_count += 1
            now = time.monotonic()

            if frame_count % 4 == 0:
                try:
                    dets = detector.detect(frame)
                    if len(dets) > 1:
                        message = "Multiple faces detected - only one person in view."
                        latest_box = None
                    elif len(dets) == 1:
                        latest_box = dets[0][0]
                        if auto_mode:
                            message = "AUTO-CAPTURE ACTIVE: slowly move head left/right/smile."
                        else:
                            message = "Alignment good. Press SPACE or 'A' (Auto-Capture 50-100 photos)."
                    else:
                        latest_box = None
                        message = "Center face in scanner view with good illumination."
                except Exception as exc:
                    message = f"Detector error: {exc}"

            # Auto-capture logic (every 0.25s while face is visible)
            trigger_save = False
            if auto_mode and latest_box is not None and (now - last_auto_capture_time >= 0.22) and saved < args.target:
                trigger_save = True
                last_auto_capture_time = now

            if latest_box is not None:
                x, y, w, h = latest_box
                box_color = (0, 255, 255) if auto_mode else (0, 220, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

            # Banner overlay
            mode_badge = " [AUTO-BURST ON]" if auto_mode else " [SPACE / 'A' Auto]"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
            cv2.putText(frame, f"User ID: {args.name} | Samples: {saved}/{args.target}{mode_badge}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, message, (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27) or saved >= args.target:
                break
            if key in (ord("a"), ord("A")):
                auto_mode = not auto_mode
            if key == ord(" "):
                trigger_save = True

            if trigger_save and latest_box is not None:
                x, y, w, h = latest_box
                crop = padded_crop(frame, {"x": x, "y": y, "w": w, "h": h})
                if crop is not None:
                    filename = destination / f"enroll_{int(time.time() * 1000)}.jpg"
                    cv2.imwrite(str(filename), crop)
                    saved += 1
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n[Enrollment Complete] Saved {saved} samples to {destination}.")
    print("Run: python src/build_embeddings.py to recompile biometric database.")


if __name__ == "__main__":
    main()
