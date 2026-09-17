"""Fast, face-only quality checks and conservative low-light enhancement."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceQuality:
    score: float
    brightness: float
    sharpness: float
    contrast: float
    reason: str

    @property
    def usable(self) -> bool:
        return self.score >= 0.45


def assess_face_quality(face_bgr: np.ndarray) -> FaceQuality:
    """Rate only the detected face crop; no background is used in the score."""
    if face_bgr is None or face_bgr.size == 0:
        return FaceQuality(0.0, 0.0, 0.0, 0.0, "FACE NOT AVAILABLE")
    height, width = face_bgr.shape[:2]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    size_score = min(1.0, min(width, height) / 100.0)
    light_score = min(1.0, max(0.0, (brightness - 25.0) / 75.0))
    contrast_score = min(1.0, contrast / 38.0)
    sharpness_score = min(1.0, sharpness / 120.0)
    score = 0.30 * size_score + 0.25 * light_score + 0.20 * contrast_score + 0.25 * sharpness_score

    if min(width, height) < 48:
        reason = "MOVE CLOSER TO THE CAMERA"
    elif brightness < 55:
        reason = "LIGHTING IS TOO LOW"
    elif brightness > 225:
        reason = "AVOID BRIGHT GLARE"
    elif sharpness < 35:
        reason = "HOLD STILL - FACE IS BLURRY"
    elif contrast < 18:
        reason = "IMPROVE LIGHTING ON YOUR FACE"
    else:
        reason = "FACE QUALITY GOOD"
    return FaceQuality(float(score), brightness, sharpness, contrast, reason)


def enhance_face_for_embedding(face_bgr: np.ndarray, quality: FaceQuality | None = None) -> np.ndarray:
    """Apply modest CLAHE/gamma correction only when a face is genuinely dim.

    Conservative processing avoids creating artificial details that could harm
    biometric matching. Enrollment and live recognition use the same routine.
    """
    quality = quality or assess_face_quality(face_bgr)
    if quality.brightness >= 90:
        return face_bgr
    ycrcb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(y)
    enhanced = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)
    # Mild gamma lifting only for very dark face crops.
    if quality.brightness < 65:
        gamma = 0.82
        table = np.array([((index / 255.0) ** gamma) * 255 for index in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)
    return enhanced
