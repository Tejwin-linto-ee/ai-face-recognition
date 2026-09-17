"""Enroll images in faces/<identity>/ into a local FaceNet512 embedding database.

Run once after adding or replacing enrollment photographs:
    python src/build_embeddings.py --faces-dir faces --output model/facenet512.npz

Use 12-30 clear images per person. Include normal lighting, glasses (if normally
worn), modest pose changes, and only one face in each image.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

# Ensure src directory is in sys.path for local module resolution
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# DeepFace/TensorFlow must see this before their first import.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from deepface import DeepFace
from face_quality import assess_face_quality, enhance_face_for_embedding

MODEL_NAME = "Facenet512"
SUPPORTED_MODELS = ("Facenet512", "ArcFace", "SFace", "Buffalo_L")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    """Return a unit-length float32 vector suitable for dot-product cosine search."""
    vector = np.asarray(vector, dtype=np.float32)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def image_paths(root: Path) -> Iterable[Path]:
    return (p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def get_largest_face_box(image: np.ndarray, detector_backend: str) -> dict | None:
    """Detect first, so embeddings are made from a face crop rather than a scene."""
    try:
        detections = DeepFace.extract_faces(
            img_path=image,
            detector_backend=detector_backend,
            enforce_detection=False,
            align=True,
        )
    except Exception as exc:
        print(f"  detector error: {exc}")
        return None

    valid = [d for d in detections if d.get("confidence", 0.0) >= 0.70]
    if not valid:
        return None
    return max(valid, key=lambda d: d["facial_area"]["w"] * d["facial_area"]["h"])["facial_area"]


def padded_crop(image: np.ndarray, area: dict, margin: float = 0.18) -> np.ndarray | None:
    """Crop the detected face with a small contextual margin, clamped to the image."""
    x, y, w, h = (int(area[k]) for k in ("x", "y", "w", "h"))
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


def embed_face(face_bgr: np.ndarray, model_name: str = MODEL_NAME) -> np.ndarray:
    """Embed an already-detected crop; skip avoids a second expensive detector pass."""
    result = DeepFace.represent(
        img_path=face_bgr,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
        normalization="base",
    )
    return l2_normalize(np.asarray(result[0]["embedding"], dtype=np.float32))


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "model" / "facenet512.npz"

    parser = argparse.ArgumentParser(description="Build a normalized face embedding database.")
    parser.add_argument("--faces-dir", type=Path, default=Path("faces"))
    parser.add_argument("--output", type=Path, default=default_output, help="Output .npz embeddings file.")
    parser.add_argument("--detector", default="retinaface", help="retinaface, yunet, or mtcnn")
    parser.add_argument("--model", choices=SUPPORTED_MODELS, default=MODEL_NAME, help="Facenet512 is fastest; ArcFace is a stronger optional model.")
    parser.add_argument("--min-images-per-person", type=int, default=5)
    args = parser.parse_args()

    if not args.faces_dir.is_dir():
        raise SystemExit(f"Enrollment directory not found: {args.faces_dir.resolve()}")

    identities = sorted(p for p in args.faces_dir.iterdir() if p.is_dir())
    if not identities:
        raise SystemExit("Add folders such as faces/Tejwin/ and faces/Supriya/ first.")

    # Warm up once: weights are loaded only at enrollment startup.
    print(f"Loading {args.model}; detector={args.detector} …")
    DeepFace.build_model(args.model)

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    per_identity: dict[str, int] = {}

    for identity_dir in identities:
        accepted = 0
        print(f"Enrolling {identity_dir.name}")
        for path in image_paths(identity_dir):
            image = cv2.imread(str(path))
            if image is None:
                print(f"  skipped unreadable: {path.name}")
                continue
            area = get_largest_face_box(image, args.detector)
            if area is None:
                print(f"  skipped no confident face: {path.name}")
                continue
            crop = padded_crop(image, area)
            if crop is None or min(crop.shape[:2]) < 48:
                print(f"  skipped face too small: {path.name}")
                continue
            quality = assess_face_quality(crop)
            if not quality.usable:
                print(f"  skipped poor quality ({quality.reason.lower()}): {path.name}")
                continue
            try:
                embeddings.append(embed_face(enhance_face_for_embedding(crop, quality), args.model))
            except Exception as exc:
                print(f"  skipped embedding error {path.name}: {exc}")
                continue
            labels.append(identity_dir.name)
            sources.append(str(path.relative_to(args.faces_dir)))
            accepted += 1
        per_identity[identity_dir.name] = accepted
        print(f"  accepted {accepted} images")

    too_few = [name for name, count in per_identity.items() if count < args.min_images_per_person]
    if too_few:
        raise SystemExit(
            "Enrollment aborted. Add clearer/diverse photos for: " + ", ".join(too_few)
            + f" (need at least {args.min_images_per_person} accepted photos each)."
        )
    if not embeddings:
        raise SystemExit("No embeddings built.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.vstack(embeddings).astype(np.float32)
    np.savez_compressed(
        args.output, embeddings=matrix, labels=np.asarray(labels), sources=np.asarray(sources), model=np.asarray(args.model)
    )
    metadata = {
        "model": args.model,
        "detector": args.detector,
        "embedding_count": int(len(labels)),
        "identity_counts": per_identity,
        "normalization": "L2; cosine similarity is a dot product",
        "quality_filter": "face size, lighting, contrast, and sharpness; dim faces use conservative CLAHE/gamma enhancement",
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved {len(labels)} normalized embeddings for {len(per_identity)} people to {args.output}")


if __name__ == "__main__":
    main()
