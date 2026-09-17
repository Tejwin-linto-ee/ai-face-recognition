"""Responsive webcam recognition using precomputed FaceNet512 embeddings.

The camera/UI loop never runs DeepFace. Detection and embedding run serially in
one background worker, so an older CPU can still render a fluid webcam feed.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import hmac
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure src directory is in sys.path for local module resolution
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
import tensorflow as tf
from deepface import DeepFace

from build_embeddings import MODEL_NAME, embed_face, padded_crop
from face_quality import FaceQuality, assess_face_quality, enhance_face_for_embedding

try:
    import edge_tts
    import pygame
except ImportError:  # Voice remains optional if package installation failed.
    edge_tts = None
    pygame = None


# Starting points, not calibrated probabilities. Validate before using hardware.
AUTHORIZED_SIMILARITY = 0.70
LOW_CONFIDENCE_SIMILARITY = 0.60
HISTORY_SIZE = 6

# `low` is designed for an i7-5050U-class CPU. It uses fast YuNet and lightweight
# IoU track association, avoiding slow/broken OpenCV KCF/CSRT implementations.
PERFORMANCE_PROFILES = {
    "low": {
        # Show a clear HD view, but run expensive detection on a smaller copy.
        "detector": "yunet", "width": 1280, "height": 720,
        "analysis_width": 480, "analysis_height": 270,
        "detect_every": 8, "min_face": 28, "detection_confidence": 0.55,
        "stable_votes": 2, "track_ttl": 5, "display_scale": 1.0, "cpu_threads": 2,
    },
    "balanced": {
        "detector": "yunet", "width": 1280, "height": 720,
        "analysis_width": 640, "analysis_height": 360,
        "detect_every": 6, "min_face": 40, "detection_confidence": 0.60,
        "stable_votes": 3, "track_ttl": 7, "display_scale": 1.0, "cpu_threads": 3,
    },
    "quality": {
        "detector": "retinaface", "width": 1280, "height": 720,
        "analysis_width": 960, "analysis_height": 540,
        "detect_every": 5, "min_face": 60, "detection_confidence": 0.70,
        "stable_votes": 4, "track_ttl": 8, "display_scale": 1.0, "cpu_threads": 0,
    },
}


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    w, h = min(w, width - x), min(h, height - y)
    return x, y, max(0, w), max(0, h)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union else 0.0


def configure_tensorflow_gpu(cpu_threads: int) -> list[object]:
    """Use an NVIDIA GPU when TensorFlow can see one; otherwise use CPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus and cpu_threads:
        # Leave a CPU core available for webcam capture and window rendering.
        # This must happen before FaceNet is built.
        try:
            tf.config.threading.set_intra_op_parallelism_threads(cpu_threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError as exc:
            print(f"Could not limit TensorFlow CPU threads: {exc}")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Could not set GPU memory growth: {exc}")
    return gpus


class VoiceGuide:
    """Natural online speech prompts on a separate thread, rate limited per message."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and edge_tts is not None and pygame is not None
        self.last_spoken: dict[str, float] = {}
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue(maxsize=10)
        self.cancelled_keys: set[str] = set()
        self.current_key: str | None = None
        self.voice_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        if enabled and not self.enabled:
            print("Voice disabled: install edge-tts and pygame with pip install -r requirements.txt")
        if self.enabled:
            self.thread = threading.Thread(target=self._run, name="voice-guide", daemon=True)
            self.thread.start()

    def say(self, key: str, text: str, every_seconds: float = 8.0) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self.last_spoken.get(key, 0.0) < every_seconds:
            return
        try:
            with self.voice_lock:
                self.cancelled_keys.discard(key)
            self.messages.put_nowait((key, text))
            self.last_spoken[key] = now
        except queue.Full:
            pass

    def cancel(self, key: str) -> None:
        """Stop a prompt when the camera status has already changed."""
        if not self.enabled:
            return
        with self.voice_lock:
            self.cancelled_keys.add(key)
            should_stop = self.current_key == key
        if should_stop:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def _run(self) -> None:
        try:
            pygame.mixer.init()
            voice_name = "en-GB-RyanNeural"
            print(f"Natural voice: {voice_name}")
            while not self.stop_event.is_set():
                try:
                    key, text = self.messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                audio_path: Path | None = None
                try:
                    with self.voice_lock:
                        if key in self.cancelled_keys:
                            continue
                        self.current_key = key
                    file_descriptor, temporary_name = tempfile.mkstemp(suffix=".mp3")
                    os.close(file_descriptor)
                    audio_path = Path(temporary_name)
                    asyncio.run(
                        edge_tts.Communicate(text, voice_name, rate="+15%").save(str(audio_path))
                    )
                    with self.voice_lock:
                        if key in self.cancelled_keys:
                            continue
                    print(f"Speaking: {text}")
                    pygame.mixer.music.load(str(audio_path))
                    pygame.mixer.music.set_volume(1.0)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and not self.stop_event.is_set():
                        time.sleep(0.05)
                    pygame.mixer.music.unload()
                except Exception as exc:
                    print(f"Voice error: {exc}")
                finally:
                    with self.voice_lock:
                        if self.current_key == key:
                            self.current_key = None
                    if audio_path is not None:
                        audio_path.unlink(missing_ok=True)
                    self.messages.task_done()
        except Exception as exc:
            print(f"Voice disabled: {exc}")
            self.enabled = False
        finally:
            pygame.mixer.quit()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


@dataclass
class FaceTrack:
    id: int
    box: tuple[int, int, int, int]
    missed_detections: int = 0
    label_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=HISTORY_SIZE))
    similarity_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=HISTORY_SIZE))
    label: str = "Scanning"
    similarity: float = 0.0
    quality: FaceQuality | None = None
    motion_reference: tuple[int, int, int, int] | None = None
    motion_verified: bool = False
    deep_liveness: str = "not checked"

    def add_match(self, label: str, similarity: float, stable_votes: int) -> None:
        self.label_history.append(label)
        self.similarity_history.append(similarity)
        candidate, votes = collections.Counter(self.label_history).most_common(1)[0]
        self.label = candidate if votes >= stable_votes else "Scanning"
        scores = [score for name, score in zip(self.label_history, self.similarity_history) if name == candidate]
        self.similarity = float(np.median(scores)) if scores else 0.0

    def update_box(self, box: tuple[int, int, int, int]) -> None:
        self.box = box
        if self.motion_reference is None or self.motion_verified:
            return
        old_x, old_y, old_w, old_h = self.motion_reference
        new_x, new_y, new_w, new_h = box
        old_center = (old_x + old_w / 2, old_y + old_h / 2)
        new_center = (new_x + new_w / 2, new_y + new_h / 2)
        center_shift = float(np.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1]))
        scale_change = abs(new_w - old_w) / max(old_w, 1)
        if center_shift >= max(18.0, old_w * 0.10) or scale_change >= 0.12:
            self.motion_verified = True

    def start_motion_challenge(self) -> None:
        if self.motion_reference is None:
            self.motion_reference = self.box


class TrackManager:
    """Very-lightweight multi-face tracking by association across detections."""

    def __init__(self, track_ttl: int) -> None:
        self.tracks: dict[int, FaceTrack] = {}
        self.next_id = 1
        self.track_ttl = track_ttl

    def associate_detections(self, detections: list[tuple[int, int, int, int]]) -> list[int]:
        """Update boxes without optical tracking; return IDs needing an embedding."""
        candidates = sorted(
            ((iou(track.box, box), track_id, box_index)
             for track_id, track in self.tracks.items()
             for box_index, box in enumerate(detections)),
            reverse=True,
        )
        used_tracks: set[int] = set()
        used_boxes: set[int] = set()
        changed: list[int] = []
        for overlap, track_id, box_index in candidates:
            if overlap < 0.18 or track_id in used_tracks or box_index in used_boxes:
                continue
            track = self.tracks[track_id]
            track.update_box(detections[box_index])
            track.missed_detections = 0
            used_tracks.add(track_id)
            used_boxes.add(box_index)
            changed.append(track_id)

        for box_index, box in enumerate(detections):
            if box_index not in used_boxes:
                track = FaceTrack(id=self.next_id, box=box)
                self.tracks[track.id] = track
                changed.append(track.id)
                self.next_id += 1

        for track_id, track in list(self.tracks.items()):
            if track_id not in used_tracks and track_id not in changed:
                track.missed_detections += 1
            if track.missed_detections > self.track_ttl:
                del self.tracks[track_id]
        return changed


class EmbeddingDatabase:
    def __init__(self, path: Path) -> None:
        database = np.load(path, allow_pickle=False)
        self.embeddings = np.asarray(database["embeddings"], dtype=np.float32)
        self.labels = np.asarray(database["labels"]).astype(str)
        # Older databases did not store their model; retain FaceNet512 support.
        self.model = str(database["model"].item()) if "model" in database.files else MODEL_NAME
        self.embeddings /= np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12)

    def match(self, embedding: np.ndarray, quality_score: float) -> tuple[str, float]:
        """Fast cosine search with a per-person top-3 median."""
        similarities = self.embeddings @ embedding
        candidates: list[tuple[float, str]] = []
        for label in np.unique(self.labels):
            best_three = np.sort(similarities[self.labels == label])[-3:]
            candidates.append((float(np.median(best_three)), label))
        score, label = max(candidates)
        # Require a slightly stronger match when the face is dim, blurry, or
        # small. This reduces false accepts without penalizing good captures.
        penalty = max(0.0, min(1.0, 1.0 - quality_score))
        authorized_threshold = AUTHORIZED_SIMILARITY + 0.05 * penalty
        low_confidence_threshold = LOW_CONFIDENCE_SIMILARITY + 0.03 * penalty
        if score >= authorized_threshold:
            return label, score
        if score >= low_confidence_threshold:
            return "Low confidence", score
        return "Unknown", score


def detect_faces(
    frame: np.ndarray,
    detector: str,
    min_face: int,
    detection_confidence: float,
    analysis_width: int,
    analysis_height: int,
) -> list[tuple[int, int, int, int]]:
    """Detect on a small image, then map boxes to the native camera image."""
    source_height, source_width = frame.shape[:2]
    if (source_width, source_height) == (analysis_width, analysis_height):
        analysis_frame = frame
    else:
        analysis_frame = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
    faces = DeepFace.extract_faces(
        img_path=analysis_frame,
        detector_backend=detector,
        enforce_detection=False,
        align=False,
    )
    boxes = []
    scale_x = source_width / analysis_width
    scale_y = source_height / analysis_height
    for face in faces:
        area = face.get("facial_area", {})
        if face.get("confidence", 0.0) < detection_confidence or not all(k in area for k in ("x", "y", "w", "h")):
            continue
        small_box = tuple(int(area[k]) for k in ("x", "y", "w", "h"))
        if min(small_box[2:]) < min_face:
            continue
        box = clip_box(
            (
                round(small_box[0] * scale_x), round(small_box[1] * scale_y),
                round(small_box[2] * scale_x), round(small_box[3] * scale_y),
            ),
            source_width,
            source_height,
        )
        boxes.append(box)
    return boxes


def deepface_antispoof(face_bgr: np.ndarray) -> tuple[bool | None, float, str]:
    """Optional DeepFace anti-spoof check, run only after an identity matches."""
    try:
        result = DeepFace.extract_faces(
            img_path=face_bgr,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            anti_spoofing=True,
        )[0]
    except Exception as exc:
        return None, 0.0, str(exc)
    if "is_real" not in result:
        return None, 0.0, "anti-spoof output was not available"
    return bool(result["is_real"]), float(result.get("antispoof_score", 0.0)), ""


@dataclass
class InferenceResult:
    kind: str
    payload: object
    track_id: int | None = None


class InferenceWorker:
    """One TensorFlow worker. The webcam window never waits for inference."""

    def __init__(self) -> None:
        self.tasks: queue.Queue[tuple[str, object, object, int | None]] = queue.Queue(maxsize=1)
        self.results: queue.Queue[InferenceResult] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="face-inference", daemon=True)
        self.thread.start()

    def submit_detection(
        self,
        frame: np.ndarray,
        detector: str,
        min_face: int,
        detection_confidence: float,
        analysis_width: int,
        analysis_height: int,
    ) -> bool:
        return self._submit(
            "detect", frame.copy(),
            (detector, min_face, detection_confidence, analysis_width, analysis_height), None,
        )

    def submit_embedding(self, track_id: int, face_crop: np.ndarray, model: str) -> bool:
        return self._submit("embed", face_crop.copy(), model, track_id)

    def submit_liveness(self, track_id: int, face_crop: np.ndarray) -> bool:
        return self._submit("liveness", face_crop.copy(), None, track_id)

    def _submit(self, kind: str, image: np.ndarray, settings: object, track_id: int | None) -> bool:
        try:
            self.tasks.put_nowait((kind, image, settings, track_id))
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                kind, image, settings, track_id = self.tasks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if kind == "detect":
                    detector, min_face, detection_confidence, analysis_width, analysis_height = settings
                    boxes = detect_faces(
                        image, detector, min_face, detection_confidence, analysis_width, analysis_height
                    )
                    self.results.put(InferenceResult("detect", boxes))
                elif kind == "embed":
                    self.results.put(InferenceResult("embed", embed_face(image, settings), track_id))
                elif kind == "liveness":
                    self.results.put(InferenceResult("liveness", deepface_antispoof(image), track_id))
            except Exception as exc:
                self.results.put(InferenceResult("error", f"{kind}: {exc}", track_id))
            finally:
                self.tasks.task_done()

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2.0)


def display_label(track: FaceTrack) -> tuple[str, tuple[int, int, int]]:
    if track.label not in ("Unknown", "Low confidence", "Scanning"):
        return f"AUTHORIZED: {track.label} {track.similarity:.2f}", (0, 190, 0)
    if track.label == "Unknown":
        return f"UNKNOWN {track.similarity:.2f}", (0, 0, 230)
    if track.label == "Low confidence":
        return f"LOW CONFIDENCE {track.similarity:.2f}", (0, 165, 255)
    return "SCANNING...", (255, 200, 0)


def lighting_status(frame: np.ndarray, tracks: dict[int, FaceTrack]) -> tuple[str, tuple[int, int, int]]:
    """Assess illumination from the largest detected face, not the bright wall behind it."""
    if tracks:
        track = max(tracks.values(), key=lambda item: item.box[2] * item.box[3])
        x, y, w, h = clip_box(track.box, frame.shape[1], frame.shape[0])
        region = frame[y:y + h, x:x + w]
    else:
        # Before a face is found, estimate light from the centre of the camera view.
        height, width = frame.shape[:2]
        region = frame[height // 4: height * 3 // 4, width // 4: width * 3 // 4]
    if region.size == 0:
        return "LIGHT: CHECK CAMERA", (0, 165, 255)
    brightness = float(cv2.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))[0])
    if brightness < 55:
        return "LIGHT: LOW - move near a lamp", (0, 0, 230)
    if brightness < 85:
        return "LIGHT: DIM - improve lighting", (0, 165, 255)
    if brightness > 220:
        return "LIGHT: TOO BRIGHT - avoid glare", (0, 165, 255)
    return "LIGHT: GOOD", (0, 190, 0)


def distance_status(frame: np.ndarray, track: FaceTrack | None) -> tuple[str, tuple[int, int, int]]:
    """Give simple distance guidance using the height of the face box."""
    if track is None:
        return "LOOK AT THE CAMERA", (255, 200, 0)
    ratio = track.box[3] / max(frame.shape[0], 1)
    if ratio < 0.15:
        return "MOVE CLOSER TO THE CAMERA", (0, 165, 255)
    if ratio > 0.65:
        return "MOVE A LITTLE BACK", (0, 165, 255)
    return "POSITION GOOD", (0, 190, 0)


def draw_status_banner(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    """A large, high-contrast instruction banner for an accessible UI."""
    height, width = frame.shape[:2]
    banner_height = max(58, height // 11)
    cv2.rectangle(frame, (0, 0), (width, banner_height), (24, 24, 24), -1)
    cv2.putText(
        frame, text, (18, int(banner_height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
        max(0.58, min(0.90, width / 1500)), color, 2, cv2.LINE_AA,
    )


def primary_track(tracks: dict[int, FaceTrack]) -> FaceTrack | None:
    return max(tracks.values(), key=lambda item: item.box[2] * item.box[3]) if tracks else None


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_db = repo_root / "model" / "facenet512.npz"
    if not default_db.is_file():
        default_db = Path("model/facenet512.npz")

    parser = argparse.ArgumentParser(description="Responsive FaceNet512 webcam recognition.")
    parser.add_argument("--database", type=Path, default=default_db, help="Path to precomputed embeddings .npz database.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--profile", choices=PERFORMANCE_PROFILES, default="low")
    parser.add_argument("--detector", choices=("retinaface", "yunet", "mtcnn"), help="Override the profile detector.")
    parser.add_argument("--width", type=int, help="Override profile capture width.")
    parser.add_argument("--height", type=int, help="Override profile capture height.")
    parser.add_argument("--display-scale", type=float, help="Override profile window scale only.")
    parser.add_argument("--show-fps", action="store_true", help="Show the FPS diagnostic in the camera view.")
    parser.add_argument("--no-voice", action="store_true", help="Disable spoken guidance.")
    parser.add_argument("--require-motion", action="store_true", help="Require a small head/face movement before authorization.")
    parser.add_argument("--deep-liveness", action="store_true", help="Run DeepFace anti-spoof verification after a match (slower).")
    args = parser.parse_args()

    db_path = args.database
    if not db_path.is_file():
        candidate = repo_root / db_path
        if candidate.is_file():
            db_path = candidate
        else:
            raise SystemExit(f"Embedding DB not found: {args.database}. Run build_embeddings.py first.")

    profile = PERFORMANCE_PROFILES[args.profile]
    detector = args.detector or profile["detector"]
    width = args.width or profile["width"]
    height = args.height or profile["height"]
    display_scale = args.display_scale or profile["display_scale"]
    database = EmbeddingDatabase(db_path)

    gpus = configure_tensorflow_gpu(profile["cpu_threads"])
    print(f"Loading {database.model} once...")
    DeepFace.build_model(database.model)
    print("TensorFlow GPUs:", [gpu.name for gpu in gpus] or "CPU fallback")
    print(
        f"Profile={args.profile} | capture={width}x{height} | "
        f"analysis={profile['analysis_width']}x{profile['analysis_height']} | detector={detector}"
    )

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        raise SystemExit("Could not open webcam.")

    # Some integrated cameras cannot supply HD. Use their actual mode for the
    # window rather than stretching a smaller image into a blurry large one.
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width > 0 and actual_height > 0:
        width, height = actual_width, actual_height

    window_name = "Face AI - Authorization & Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, int(width * display_scale), int(height * display_scale))

    manager = TrackManager(track_ttl=profile["track_ttl"])
    worker = InferenceWorker()
    voice = VoiceGuide(enabled=True)
    
    fallback_pin = os.environ.get("FACE_AI_PIN", os.environ.get("MEDICINE_BOX_PIN", ""))
    detection_pending = False
    embeddings_pending: set[int] = set()
    liveness_pending: set[int] = set()
    frame_number = 0
    current_face_count = 0
    selected_track_id: int | None = None
    pin_mode = False
    pin_entry = ""
    pin_notice = ""
    pin_notice_until = 0.0
    manual_authorized_until = 0.0
    # Do not greet the same person again when face tracking creates a new ID.
    last_announced_label: str | None = None
    no_face_since: float | None = None
    fps_samples: collections.deque[float] = collections.deque(maxlen=30)
    last_time = time.perf_counter()
    print("Press q or Esc to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            # Mirror the webcam preview so left/right movement feels natural.
            frame = cv2.flip(frame, 1)
            frame_number += 1

            # Consume completed inference without blocking the webcam.
            while True:
                try:
                    result = worker.results.get_nowait()
                except queue.Empty:
                    break
                if result.kind == "detect":
                    detection_pending = False
                    current_face_count = len(result.payload)
                    updated_ids = manager.associate_detections(result.payload)
                    if current_face_count == 0:
                        selected_track_id = None
                        continue
                    # When several faces are visible, use the largest box:
                    # it is normally the person closest to the camera.
                    candidate_ids = [track_id for track_id in updated_ids if track_id in manager.tracks]
                    if not candidate_ids:
                        continue
                    selected_track_id = max(
                        candidate_ids,
                        key=lambda track_id: manager.tracks[track_id].box[2] * manager.tracks[track_id].box[3],
                    )
                    for track_id in (selected_track_id,):
                        if track_id in embeddings_pending or track_id not in manager.tracks:
                            continue
                        track = manager.tracks[track_id]
                        x, y, w, h = clip_box(track.box, frame.shape[1], frame.shape[0])
                        crop = padded_crop(frame, {"x": x, "y": y, "w": w, "h": h})
                        if crop is None:
                            continue
                        quality = assess_face_quality(crop)
                        track.quality = quality
                        if not quality.usable:
                            continue
                        crop = enhance_face_for_embedding(crop, quality)
                        if worker.submit_embedding(track_id, crop, database.model):
                            embeddings_pending.add(track_id)
                elif result.kind == "embed":
                    if result.track_id is not None:
                        embeddings_pending.discard(result.track_id)
                        track = manager.tracks.get(result.track_id)
                        if track is not None and result.track_id == selected_track_id and current_face_count > 0:
                            quality_score = track.quality.score if track.quality is not None else 0.45
                            label, score = database.match(result.payload, quality_score)
                            track.add_match(label, score, profile["stable_votes"])
                            if label not in ("Unknown", "Low confidence", "Scanning"):
                                track.start_motion_challenge()
                                if args.deep_liveness and track.deep_liveness == "not checked":
                                    x, y, w, h = clip_box(track.box, frame.shape[1], frame.shape[0])
                                    crop = padded_crop(frame, {"x": x, "y": y, "w": w, "h": h})
                                    if crop is not None and worker.submit_liveness(result.track_id, crop):
                                        track.deep_liveness = "checking"
                                        liveness_pending.add(result.track_id)
                elif result.kind == "liveness":
                    if result.track_id is not None:
                        liveness_pending.discard(result.track_id)
                        track = manager.tracks.get(result.track_id)
                        if track is not None:
                            is_real, _score, error = result.payload
                            if is_real is None:
                                track.deep_liveness = "unavailable"
                                print(f"Deep liveness unavailable: {error}")
                            else:
                                track.deep_liveness = "passed" if is_real else "failed"
                else:
                    if result.track_id is not None:
                        embeddings_pending.discard(result.track_id)
                        liveness_pending.discard(result.track_id)
                    print(f"Inference error: {result.payload}")

            # At most one inference is queued. The display thread continues.
            should_detect = frame_number % profile["detect_every"] == 0 or not manager.tracks
            if should_detect and not detection_pending:
                detection_pending = worker.submit_detection(
                    frame, detector, profile["min_face"], profile["detection_confidence"],
                    profile["analysis_width"], profile["analysis_height"],
                )

            for track in manager.tracks.values():
                x, y, w, h = clip_box(track.box, frame.shape[1], frame.shape[0])
                if current_face_count > 0 and args.require_motion and track.label not in ("Unknown", "Low confidence", "Scanning") and not track.motion_verified:
                    text, color = "VERIFY: TURN HEAD SLIGHTLY", (0, 165, 255)
                elif current_face_count > 0 and args.deep_liveness and track.label not in ("Unknown", "Low confidence", "Scanning") and track.deep_liveness != "passed":
                    if track.deep_liveness == "failed":
                        text, color = "LIVENESS CHECK FAILED", (0, 0, 230)
                    else:
                        text, color = "VERIFYING LIVE FACE", (255, 200, 0)
                else:
                    text, color = display_label(track)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2, cv2.LINE_AA)

            light_text, light_color = lighting_status(frame, manager.tracks)
            active_track = manager.tracks.get(selected_track_id) if selected_track_id is not None else None
            if active_track is None:
                active_track = primary_track(manager.tracks)
            distance_text, distance_color = distance_status(frame, active_track)
            now = time.monotonic()
            # Interrupt outdated guidance immediately instead of making the
            # user listen to a message that is no longer true.
            if current_face_count > 0:
                voice.cancel("no_face")
            if not (light_text.startswith("LIGHT: LOW") or light_text.startswith("LIGHT: DIM")):
                voice.cancel("low_light")
            # A brief missed detection can happen while someone turns their
            # head.  Reset only after the camera has seen no face for 2 s.
            if current_face_count == 0:
                if no_face_since is None:
                    no_face_since = now
                elif now - no_face_since >= 2.0:
                    last_announced_label = None
            else:
                no_face_since = None
            if pin_mode:
                status_text, status_color = f"ENTER CAREGIVER PIN: {'*' * len(pin_entry)}", (255, 200, 0)
            elif now < manual_authorized_until:
                status_text, status_color = "CAREGIVER PIN VERIFIED", (0, 190, 0)
            elif now < pin_notice_until:
                status_text, status_color = pin_notice, (0, 0, 230)
            elif current_face_count == 0:
                status_text, status_color = "PLEASE LOOK AT THE CAMERA", (255, 200, 0)
                voice.say("no_face", "Please look at the camera.", 12)
            elif active_track is not None and active_track.quality is not None and not active_track.quality.usable:
                status_text, status_color = active_track.quality.reason, (0, 165, 255)
                voice.say("quality", active_track.quality.reason.capitalize(), 10)
            elif light_text.startswith("LIGHT: LOW") or light_text.startswith("LIGHT: DIM"):
                status_text, status_color = light_text, light_color
                voice.say("low_light", "Lighting is low. Please move near a lamp.", 12)
            elif distance_text != "POSITION GOOD":
                status_text, status_color = distance_text, distance_color
                voice.say("distance", distance_text.replace("THE CAMERA", "the camera").capitalize(), 10)
            elif active_track is not None and args.require_motion and active_track.label not in ("Unknown", "Low confidence", "Scanning") and not active_track.motion_verified:
                status_text, status_color = "TURN YOUR HEAD SLIGHTLY TO VERIFY", (0, 165, 255)
                voice.say("motion", "Please turn your head slightly to verify.", 10)
            elif active_track is not None and args.deep_liveness and active_track.label not in ("Unknown", "Low confidence", "Scanning") and active_track.deep_liveness == "checking":
                status_text, status_color = "VERIFYING LIVE FACE - PLEASE HOLD STILL", (255, 200, 0)
            elif active_track is not None and args.deep_liveness and active_track.label not in ("Unknown", "Low confidence", "Scanning") and active_track.deep_liveness == "unavailable":
                status_text, status_color = "LIVE VERIFICATION UNAVAILABLE - ASK CAREGIVER", (0, 0, 230)
            elif active_track is not None and args.deep_liveness and active_track.label not in ("Unknown", "Low confidence", "Scanning") and active_track.deep_liveness == "failed":
                status_text, status_color = "LIVENESS CHECK FAILED - ASK CAREGIVER", (0, 0, 230)
                voice.say("liveness_failed", "Live face verification failed. Please ask a caregiver.", 12)
            elif active_track is not None and args.deep_liveness and active_track.label not in ("Unknown", "Low confidence", "Scanning") and active_track.deep_liveness != "passed":
                status_text, status_color = "STARTING LIVE VERIFICATION", (255, 200, 0)
            elif active_track is not None and active_track.label not in ("Unknown", "Low confidence", "Scanning"):
                status_text, status_color = f"WELCOME, {active_track.label.upper()}", (0, 190, 0)
                if active_track.label != last_announced_label:
                    voice.say(
                        "recognized_person",
                        f"{active_track.label}, welcome {active_track.label}.",
                        0,
                    )
                    last_announced_label = active_track.label
            elif active_track is not None and active_track.label == "Unknown":
                status_text, status_color = "FACE NOT RECOGNIZED - ASK CAREGIVER", (0, 0, 230)
                voice.say("unknown", "I could not recognize you. Please look at the camera again.", 12)
            elif active_track is not None and active_track.label == "Low confidence":
                status_text, status_color = "PLEASE HOLD STILL", (0, 165, 255)
                voice.say("low_confidence", "Please hold still and look at the camera.", 10)
            else:
                status_text, status_color = "SCANNING - PLEASE HOLD STILL", (255, 200, 0)

            draw_status_banner(frame, status_text, status_color)
            cv2.putText(frame, light_text, (12, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, light_color, 2, cv2.LINE_AA)
            if current_face_count > 0 and distance_text != "POSITION GOOD":
                cv2.putText(frame, distance_text, (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, distance_color, 2, cv2.LINE_AA)
            elif fallback_pin:
                cv2.putText(frame, "Press P for caregiver PIN", (12, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)

            frame_time = time.perf_counter()
            fps_samples.append(1.0 / max(frame_time - last_time, 1e-6))
            last_time = frame_time
            if args.show_fps:
                cv2.putText(frame, f"FPS {np.mean(fps_samples):.1f}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
            # Window scaling is handled by OpenCV/Windows instead of creating
            # a larger image on the CPU every displayed frame.
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if pin_mode:
                if ord("0") <= key <= ord("9") and len(pin_entry) < 12:
                    pin_entry += chr(key)
                elif key in (8, 127):
                    pin_entry = pin_entry[:-1]
                elif key in (10, 13):
                    pin_mode = False
                    if fallback_pin and hmac.compare_digest(pin_entry, fallback_pin):
                        manual_authorized_until = time.monotonic() + 30
                        voice.say("pin_verified", "Caregiver PIN verified.", 1)
                    else:
                        pin_notice = "INCORRECT PIN - PLEASE ASK CAREGIVER"
                        pin_notice_until = time.monotonic() + 4
                        voice.say("bad_pin", "Incorrect PIN. Please ask a caregiver.", 2)
                    pin_entry = ""
                elif key == 27:
                    pin_mode, pin_entry = False, ""
            elif key in (ord("p"), ord("P")):
                if fallback_pin:
                    pin_mode, pin_entry = True, ""
                    voice.say("pin_prompt", "Please enter the caregiver PIN.", 2)
                else:
                    pin_notice = "PIN FALLBACK NOT CONFIGURED"
                    pin_notice_until = time.monotonic() + 4
    finally:
        worker.close()
        voice.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
