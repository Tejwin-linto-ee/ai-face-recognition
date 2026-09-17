"""Multimodal Biometric Smart Hub - Real-Time Access Terminal.

Dependencies:
    pip install numpy opencv-contrib-python tensorflow deepface onnxruntime edge-tts pygame

Architecture:
    - Main Thread: OpenCV webcam capture (30+ FPS) & ByteTrack bounding box tracking.
    - Child Process: BiometricInferenceProcess running SCRFD/YuNet detection,
      FaceNet512 feature extraction, and MiniFASNet anti-spoofing.
    - Security Logic: L2-normalized cosine search, temporal voting, motion verification,
      secure state unlocking, and system audit logging.
"""
from __future__ import annotations

# Pip Dependencies declaration
# pip install numpy opencv-contrib-python tensorflow deepface onnxruntime edge-tts pygame

import argparse
import asyncio
import collections
import datetime
import hmac
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure local module resolution
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np

from bytetrack import ByteTracker, STrack
from face_quality import FaceQuality, assess_face_quality, enhance_face_for_embedding
from worker import BiometricInferenceProcess, InferenceResponse, InferenceTask

try:
    import edge_tts
    import pygame
except ImportError:
    edge_tts = None
    pygame = None


# Cosine similarity thresholds
AUTHORIZED_SIMILARITY = 0.70
LOW_CONFIDENCE_SIMILARITY = 0.60
HISTORY_SIZE = 6


def system_audit_log(event_type: str, user_id: str | None, confidence: float, details: str) -> None:
    """Structured security audit log with ISO timestamps."""
    now_iso = datetime.datetime.now().isoformat()
    uid = user_id if user_id else "ANONYMOUS"
    print(f"[SECURITY AUDIT] {now_iso} | EVENT={event_type:<18} | USER={uid:<12} | CONF={confidence:.2f} | DETAILS={details}")


def unlock_secure_state(user_id: str, confidence: float) -> None:
    """Execute secure terminal unlock actions (e.g. relay trigger, physical lock, token generation)."""
    system_audit_log("ACCESS_GRANTED", user_id, confidence, "Hardware secure state unlocked")


def clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x, y = max(0, x), max(0, y)
    w, h = min(w, width - x), min(h, height - y)
    return x, y, max(0, w), max(0, h)


def padded_crop(image: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.18) -> np.ndarray | None:
    x, y, w, h = box
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(image.shape[1], x + w + pad_x), min(image.shape[0], y + h + pad_y)
    crop = image[y1:y2, x1:x2]
    return crop if crop.size else None


class VoiceGuide:
    """Natural spoken security guidance on a separate thread."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled and edge_tts is not None and pygame is not None
        self.last_spoken: Dict[str, float] = {}
        self.messages: queue.Queue[Tuple[str, str]] = queue.Queue(maxsize=10)
        self.cancelled_keys: set[str] = set()
        self.current_key: Optional[str] = None
        self.voice_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

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
            voice_name = "en-US-JennyNeural"
            while not self.stop_event.is_set():
                try:
                    key, text = self.messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                audio_path: Optional[Path] = None
                try:
                    with self.voice_lock:
                        if key in self.cancelled_keys:
                            continue
                        self.current_key = key
                    file_descriptor, temporary_name = tempfile.mkstemp(suffix=".mp3")
                    os.close(file_descriptor)
                    audio_path = Path(temporary_name)
                    asyncio.run(
                        edge_tts.Communicate(text, voice_name, rate="+10%").save(str(audio_path))
                    )
                    with self.voice_lock:
                        if key in self.cancelled_keys:
                            continue
                    pygame.mixer.music.load(str(audio_path))
                    pygame.mixer.music.set_volume(1.0)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy() and not self.stop_event.is_set():
                        time.sleep(0.05)
                    pygame.mixer.music.unload()
                except Exception:
                    pass
                finally:
                    with self.voice_lock:
                        if self.current_key == key:
                            self.current_key = None
                    if audio_path is not None:
                        audio_path.unlink(missing_ok=True)
                    self.messages.task_done()
        except Exception:
            self.enabled = False
        finally:
            pygame.mixer.quit()

    def close(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


class EmbeddingDatabase:
    """Precomputed unit-normalized embeddings database with cosine dot-product search."""

    def __init__(self, path: Path) -> None:
        database = np.load(path, allow_pickle=False)
        self.embeddings = np.asarray(database["embeddings"], dtype=np.float32)
        self.labels = np.asarray(database["labels"]).astype(str)
        self.model = str(database["model"].item()) if "model" in database.files else "Facenet512"
        self.embeddings /= np.maximum(np.linalg.norm(self.embeddings, axis=1, keepdims=True), 1e-12)

    def match(self, embedding: np.ndarray, quality_score: float) -> Tuple[str, float]:
        similarities = self.embeddings @ embedding
        candidates: List[Tuple[float, str]] = []
        for label in np.unique(self.labels):
            best_three = np.sort(similarities[self.labels == label])[-3:]
            candidates.append((float(np.median(best_three)), label))
        score, label = max(candidates)

        # Dynamic quality penalty
        penalty = max(0.0, min(1.0, 1.0 - quality_score))
        authorized_threshold = AUTHORIZED_SIMILARITY + 0.05 * penalty
        low_confidence_threshold = LOW_CONFIDENCE_SIMILARITY + 0.03 * penalty

        if score >= authorized_threshold:
            return label, score
        if score >= low_confidence_threshold:
            return "Low confidence", score
        return "Unknown", score


@dataclass
class IdentityTrackState:
    label_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=HISTORY_SIZE))
    similarity_history: collections.deque = field(default_factory=lambda: collections.deque(maxlen=HISTORY_SIZE))
    user_id: str = "Scanning"
    similarity: float = 0.0
    quality: Optional[FaceQuality] = None
    motion_reference: Optional[Tuple[int, int, int, int]] = None
    motion_verified: bool = False
    deep_liveness: str = "not checked"

    def add_match(self, user_id: str, similarity: float, required_votes: int = 2) -> None:
        self.label_history.append(user_id)
        self.similarity_history.append(similarity)
        candidate, votes = collections.Counter(self.label_history).most_common(1)[0]
        self.user_id = candidate if votes >= required_votes else "Scanning"
        scores = [s for u, s in zip(self.label_history, self.similarity_history) if u == candidate]
        self.similarity = float(np.median(scores)) if scores else 0.0

    def check_motion(self, current_box: Tuple[int, int, int, int]) -> None:
        if self.motion_reference is None:
            self.motion_reference = current_box
            return
        if self.motion_verified:
            return

        ox, oy, ow, oh = self.motion_reference
        cx, cy, cw, ch = current_box
        shift = float(np.hypot((cx + cw / 2) - (ox + ow / 2), (cy + ch / 2) - (oy + oh / 2)))
        scale_diff = abs(cw - ow) / max(float(ow), 1.0)
        if shift >= max(18.0, ow * 0.10) or scale_diff >= 0.12:
            self.motion_verified = True


def lighting_status(frame: np.ndarray, active_box: Optional[Tuple[int, int, int, int]]) -> Tuple[str, Tuple[int, int, int]]:
    if active_box is not None:
        x, y, w, h = clip_box(active_box, frame.shape[1], frame.shape[0])
        region = frame[y:y + h, x:x + w]
    else:
        h, w = frame.shape[:2]
        region = frame[h // 4: h * 3 // 4, w // 4: w * 3 // 4]

    if region.size == 0:
        return "LIGHT: CHECK SENSOR", (0, 165, 255)
    brightness = float(cv2.mean(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY))[0])
    if brightness < 55:
        return "LIGHT: LOW - IMPROVE AMBIENT LIGHT", (0, 0, 230)
    if brightness > 220:
        return "LIGHT: TOO BRIGHT - AVOID GLARE", (0, 165, 255)
    return "LIGHT: OPTIMAL", (0, 220, 0)


def distance_status(frame: np.ndarray, active_box: Optional[Tuple[int, int, int, int]]) -> Tuple[str, Tuple[int, int, int]]:
    if active_box is None:
        return "ALIGN FACE WITH TERMINAL SCANNER", (255, 200, 0)
    ratio = active_box[3] / max(float(frame.shape[0]), 1.0)
    if ratio < 0.15:
        return "STEP CLOSER TO SCANNER", (0, 165, 255)
    if ratio > 0.65:
        return "STEP SLIGHTLY BACK", (0, 165, 255)
    return "POSITION: OPTIMAL", (0, 220, 0)


def draw_status_banner(frame: np.ndarray, text: str, color: Tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    banner_height = max(58, height // 11)
    cv2.rectangle(frame, (0, 0), (width, banner_height), (18, 18, 18), -1)
    cv2.putText(
        frame, text, (18, int(banner_height * 0.68)), cv2.FONT_HERSHEY_SIMPLEX,
        max(0.60, min(0.95, width / 1400)), color, 2, cv2.LINE_AA,
    )


def main() -> None:
    # Set start method for Windows / POSIX safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    repo_root = Path(__file__).resolve().parent.parent
    default_db = repo_root / "model" / "facenet512.npz"

    parser = argparse.ArgumentParser(description="Multimodal Biometric Smart Hub - Access Terminal.")
    parser.add_argument("--database", type=Path, default=default_db, help="Path to precomputed embeddings .npz database.")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--detector-onnx", type=str, default=None, help="Path to SCRFD ONNX model (optional).")
    parser.add_argument("--width", type=int, default=1280, help="Webcam capture width.")
    parser.add_argument("--height", type=int, default=720, help="Webcam capture height.")
    parser.add_argument("--show-fps", action="store_true", help="Overlay real-time FPS counter.")
    parser.add_argument("--no-voice", action="store_true", help="Disable synthesized audio guidance.")
    parser.add_argument("--require-motion", action="store_true", help="Require motion challenge verification.")
    parser.add_argument("--deep-liveness", action="store_true", help="Execute DeepFace MiniFASNet anti-spoofing check.")
    args = parser.parse_args()

    db_path = args.database if args.database.is_file() else (repo_root / args.database)
    if not db_path.is_file():
        raise SystemExit(f"Embeddings database not found at {db_path}. Build it first with build_embeddings.py")

    database = EmbeddingDatabase(db_path)
    system_audit_log("SYSTEM_INIT", "SYSTEM", 1.0, f"Loaded database with {len(database.labels)} profiles; model={database.model}")

    # Initialize ByteTracker
    tracker = ByteTracker(track_thresh=0.50, high_thresh=0.60, match_thresh=0.70)
    identity_states: Dict[int, IdentityTrackState] = {}

    # Initialize Multiprocessing IPC
    task_queue: mp.Queue = mp.Queue(maxsize=3)
    result_queue: mp.Queue = mp.Queue()

    # Launch background inference process
    worker = BiometricInferenceProcess(
        task_queue=task_queue,
        result_queue=result_queue,
        model_name=database.model,
        detector_onnx_path=args.detector_onnx,
        cpu_threads=2,
    )
    worker.start()

    voice = VoiceGuide(enabled=not args.no_voice)
    fallback_pin = os.environ.get("SMART_HUB_PIN", os.environ.get("FACE_AI_PIN", ""))

    camera = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(args.camera)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not camera.isOpened():
        worker.stop()
        worker.join(timeout=2.0)
        raise SystemExit("[Terminal Error] Failed to connect to camera interface.")

    window_name = "Multimodal Biometric Smart Hub - Access Terminal"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_number = 0
    selected_track_id: Optional[int] = None
    last_unlocked_user: Optional[str] = None
    terminal_unlocked_until = 0.0
    pin_mode = False
    pin_entry = ""
    pin_notice = ""
    pin_notice_until = 0.0
    pending_embedding_tracks: set[int] = set()
    pending_liveness_tracks: set[int] = set()

    fps_samples = collections.deque(maxlen=30)
    last_time = time.perf_counter()

    print("[Smart Hub] Terminal online. Press 'P' for Security Override PIN, 'q' or 'ESC' to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_number += 1
            now = time.monotonic()

            # 1. Drain background worker responses
            while True:
                try:
                    response: InferenceResponse = result_queue.get_nowait()
                except queue.Empty:
                    break

                if response.task_type == "detect":
                    # Update ByteTrack with fresh detections
                    if response.result is not None:
                        active_tracks = tracker.update(response.result)
                    else:
                        active_tracks = []

                elif response.task_type == "embed":
                    if response.track_id in pending_embedding_tracks:
                        pending_embedding_tracks.discard(response.track_id)
                    if response.track_id in identity_states and response.result is not None:
                        state = identity_states[response.track_id]
                        q_score = state.quality.score if state.quality else 0.50
                        matched_user, match_score = database.match(response.result, q_score)
                        state.add_match(matched_user, match_score, required_votes=2)

                        # Trigger motion challenge or liveness
                        if state.user_id not in ("Unknown", "Low confidence", "Scanning"):
                            if args.deep_liveness and state.deep_liveness == "not checked":
                                state.deep_liveness = "checking"

                elif response.task_type == "liveness":
                    if response.track_id in pending_liveness_tracks:
                        pending_liveness_tracks.discard(response.track_id)
                    if response.track_id in identity_states and response.result is not None:
                        is_real, score = response.result
                        state = identity_states[response.track_id]
                        state.deep_liveness = "passed" if is_real else "failed"
                        if not is_real:
                            system_audit_log("SPOOF_ATTEMPT", state.user_id, score, "MiniFASNet anti-spoof rejection")

            # 2. Asynchronous Sampling: Submit detection every 3rd frame
            if frame_number % 3 == 0:
                try:
                    # Pass small copy for detection efficiency
                    h, w = frame.shape[:2]
                    det_frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                    task_queue.put_nowait(InferenceTask(task_type="detect", image=det_frame, frame_id=frame_number))
                except queue.Full:
                    pass  # Drop frame to keep pipeline synchronous with camera rate

            # 3. Retrieve active tracks from ByteTracker
            active_tracks = [t for t in tracker.tracked_stracks if t.state.name == "TRACKED"]

            # Maintain identity states
            for t in active_tracks:
                if t.track_id not in identity_states:
                    identity_states[t.track_id] = IdentityTrackState()

            # Select primary track (largest face closest to scanner)
            if active_tracks:
                primary = max(active_tracks, key=lambda t: t.bbox_xywh[2] * t.bbox_xywh[3])
                selected_track_id = primary.track_id
            else:
                selected_track_id = None

            # 4. Process feature extraction & liveness for selected track
            if selected_track_id is not None:
                track = next((t for t in active_tracks if t.track_id == selected_track_id), None)
                state = identity_states.get(selected_track_id)
                if track is not None and state is not None:
                    # Update motion tracking
                    state.check_motion(track.bbox_xywh)

                    # Request embedding if not already pending and candidate not finalized
                    if selected_track_id not in pending_embedding_tracks and state.user_id == "Scanning":
                        crop = padded_crop(frame, track.bbox_xywh)
                        if crop is not None:
                            q = assess_face_quality(crop)
                            state.quality = q
                            if q.usable:
                                enhanced = enhance_face_for_embedding(crop, q)
                                try:
                                    task_queue.put_nowait(
                                        InferenceTask(task_type="embed", image=enhanced, track_id=selected_track_id, frame_id=frame_number)
                                    )
                                    pending_embedding_tracks.add(selected_track_id)
                                except queue.Full:
                                    pass

                    # Request liveness if deep anti-spoofing enabled
                    if (
                        args.deep_liveness
                        and state.deep_liveness == "checking"
                        and selected_track_id not in pending_liveness_tracks
                    ):
                        crop = padded_crop(frame, track.bbox_xywh)
                        if crop is not None:
                            try:
                                task_queue.put_nowait(
                                    InferenceTask(task_type="liveness", image=crop, track_id=selected_track_id, frame_id=frame_number)
                                )
                                pending_liveness_tracks.add(selected_track_id)
                            except queue.Full:
                                pass

            # 5. UI Overlay Rendering
            for track in active_tracks:
                x, y, w, h = clip_box(track.bbox_xywh, frame.shape[1], frame.shape[0])
                state = identity_states.get(track.track_id)
                uid = state.user_id if state else "Scanning"
                sim = state.similarity if state else 0.0

                if uid not in ("Unknown", "Low confidence", "Scanning"):
                    if args.require_motion and not state.motion_verified:
                        label_str, box_color = "VERIFY: TURN HEAD SLIGHTLY", (0, 165, 255)
                    elif args.deep_liveness and state.deep_liveness != "passed":
                        if state.deep_liveness == "failed":
                            label_str, box_color = "SPOOF DETECTED - ACCESS DENIED", (0, 0, 230)
                        else:
                            label_str, box_color = "VERIFYING LIVENESS...", (255, 200, 0)
                    else:
                        label_str, box_color = f"AUTHORIZED: {uid} ({sim:.2f})", (0, 220, 0)
                        # Trigger unlock
                        if now > terminal_unlocked_until or last_unlocked_user != uid:
                            terminal_unlocked_until = now + 15.0
                            last_unlocked_user = uid
                            unlock_secure_state(uid, sim)
                            voice.say("welcome", f"Welcome {uid}. Terminal unlocked.", 12)
                elif uid == "Low confidence":
                    label_str, box_color = f"LOW CONFIDENCE ({sim:.2f})", (0, 165, 255)
                elif uid == "Unknown":
                    label_str, box_color = f"UNRECOGNIZED ({sim:.2f})", (0, 0, 230)
                else:
                    label_str, box_color = "SCANNING...", (255, 200, 0)

                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(frame, label_str, (x, max(24, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, box_color, 2, cv2.LINE_AA)

            # Header Status Banner
            primary_box = active_tracks[0].bbox_xywh if active_tracks else None
            light_text, light_color = lighting_status(frame, primary_box)
            dist_text, dist_color = distance_status(frame, primary_box)

            if pin_mode:
                banner_text, banner_color = f"SECURITY OVERRIDE PIN: {'*' * len(pin_entry)}", (255, 200, 0)
            elif now < terminal_unlocked_until:
                banner_text, banner_color = f"ACCESS GRANTED - TERMINAL UNLOCKED ({last_unlocked_user})", (0, 220, 0)
            elif now < pin_notice_until:
                banner_text, banner_color = pin_notice, (0, 0, 230)
            elif not active_tracks:
                banner_text, banner_color = "BIOMETRIC SMART HUB - PRESENT FACE TO TERMINAL", (255, 200, 0)
                voice.say("idle", "Please look at the scanner.", 15)
            else:
                primary_state = identity_states.get(selected_track_id) if selected_track_id else None
                if primary_state and primary_state.user_id not in ("Unknown", "Low confidence", "Scanning"):
                    if args.require_motion and not primary_state.motion_verified:
                        banner_text, banner_color = "CHALLENGE: TURN HEAD SLIGHTLY TO CONFIRM", (0, 165, 255)
                        voice.say("motion", "Please turn head slightly.", 10)
                    elif args.deep_liveness and primary_state.deep_liveness == "failed":
                        banner_text, banner_color = "ACCESS DENIED - LIVENESS CHECK FAILED", (0, 0, 230)
                    else:
                        banner_text, banner_color = f"ACCESS GRANTED - WELCOME {primary_state.user_id.upper()}", (0, 220, 0)
                elif primary_state and primary_state.user_id == "Unknown":
                    banner_text, banner_color = "ACCESS DENIED - UNRECOGNIZED IDENTITY", (0, 0, 230)
                    voice.say("denied", "Access denied. Verification failed.", 10)
                else:
                    banner_text, banner_color = "ANALYZING BIOMETRICS - PLEASE HOLD STILL", (255, 200, 0)

            draw_status_banner(frame, banner_text, banner_color)

            # Secondary Sensor Info
            cv2.putText(frame, light_text, (14, frame.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.52, light_color, 2, cv2.LINE_AA)
            if active_tracks and dist_text != "POSITION: OPTIMAL":
                cv2.putText(frame, dist_text, (14, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, dist_color, 2, cv2.LINE_AA)
            elif fallback_pin:
                cv2.putText(frame, "Press 'P' for Security Override PIN", (14, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1, cv2.LINE_AA)

            # FPS calculation
            frame_time = time.perf_counter()
            fps_samples.append(1.0 / max(frame_time - last_time, 1e-6))
            last_time = frame_time
            if args.show_fps:
                cv2.putText(frame, f"FPS {np.mean(fps_samples):.1f}", (frame.shape[1] - 110, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            # PIN Mode Handling
            if pin_mode:
                if ord("0") <= key <= ord("9") and len(pin_entry) < 12:
                    pin_entry += chr(key)
                elif key in (8, 127):
                    pin_entry = pin_entry[:-1]
                elif key in (10, 13):
                    pin_mode = False
                    if fallback_pin and hmac.compare_digest(pin_entry, fallback_pin):
                        terminal_unlocked_until = now + 30.0
                        last_unlocked_user = "OVERRIDE_ADMIN"
                        system_audit_log("PIN_OVERRIDE_SUCCESS", "ADMIN", 1.0, "Security override PIN accepted")
                        voice.say("pin_ok", "Security override verified. Terminal unlocked.", 2)
                    else:
                        pin_notice = "ACCESS DENIED - INVALID SECURITY PIN"
                        pin_notice_until = now + 4.0
                        system_audit_log("PIN_OVERRIDE_FAILED", "UNKNOWN", 0.0, "Invalid security PIN entered")
                        voice.say("pin_fail", "Incorrect security PIN. Access denied.", 2)
                    pin_entry = ""
                elif key == 27:
                    pin_mode, pin_entry = False, ""
            elif key in (ord("p"), ord("P")):
                if fallback_pin:
                    pin_mode, pin_entry = True, ""
                    voice.say("pin_prompt", "Enter security override PIN.", 2)
                else:
                    pin_notice = "PIN OVERRIDE NOT CONFIGURED"
                    pin_notice_until = now + 4.0

    finally:
        worker.stop()
        worker.join(timeout=2.0)
        voice.close()
        camera.release()
        cv2.destroyAllWindows()
        system_audit_log("SYSTEM_SHUTDOWN", "SYSTEM", 1.0, "Biometric Smart Hub terminated cleanly")


if __name__ == "__main__":
    main()
