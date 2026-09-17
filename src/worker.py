"""Multiprocessing Background Inference Worker for Multimodal Biometric Smart Hub.

Runs SCRFD face detection, FaceNet512 embedding extraction, and MiniFASNet
anti-spoofing in a dedicated background multiprocessing.Process.
Communicates via inter-process multiprocessing.Queue, dropping stale frames
so the main camera loop maintains a fluid 30+ FPS display rate.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple
import cv2
import numpy as np


@dataclass
class InferenceTask:
    task_type: str  # 'detect', 'embed', 'liveness'
    image: np.ndarray
    payload: Any = None
    track_id: int | None = None
    frame_id: int = 0


@dataclass
class InferenceResponse:
    task_type: str
    result: Any
    track_id: int | None = None
    frame_id: int = 0
    error: str | None = None


class BiometricInferenceProcess(mp.Process):
    """Isolated background process hosting ONNX Runtime & TensorFlow models."""

    def __init__(
        self,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        model_name: str = "Facenet512",
        detector_onnx_path: str | None = None,
        cpu_threads: int = 2,
    ) -> None:
        super().__init__(name="BiometricInferenceProcess", daemon=True)
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_name = model_name
        self.detector_onnx_path = detector_onnx_path
        self.cpu_threads = cpu_threads
        self.stop_event = mp.Event()

    def run(self) -> None:
        """Worker main loop running inside the isolated child process."""
        # Suppress TensorFlow C++ log spam
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

        # Lazy imports within the spawned process context
        import tensorflow as tf
        from deepface import DeepFace
        from detector import create_face_detector
        from face_quality import enhance_face_for_embedding

        # Configure GPU or CPU thread allocation
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus and self.cpu_threads > 0:
            try:
                tf.config.threading.set_intra_op_parallelism_threads(self.cpu_threads)
                tf.config.threading.set_inter_op_parallelism_threads(1)
            except RuntimeError:
                pass
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass

        print(f"[WorkerProcess] Initializing {self.model_name} model...")
        DeepFace.build_model(self.model_name)
        detector = create_face_detector(
            model_path=self.detector_onnx_path,
            cpu_threads=self.cpu_threads,
        )
        print("[WorkerProcess] Background AI models online and ready.")

        while not self.stop_event.is_set():
            try:
                task: InferenceTask = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except (KeyboardInterrupt, SystemExit):
                break

            try:
                if task.task_type == "detect":
                    # SCRFD / YuNet face detection
                    detections = detector.detect(task.image)
                    self.result_queue.put(
                        InferenceResponse(
                            task_type="detect",
                            result=(detections, task.payload),
                            frame_id=task.frame_id,
                        )
                    )

                elif task.task_type == "embed":
                    # FaceNet512 / ArcFace embedding extraction
                    face_crop = task.image
                    rep = DeepFace.represent(
                        img_path=face_crop,
                        model_name=self.model_name,
                        detector_backend="skip",
                        enforce_detection=False,
                        align=False,
                        normalization="base",
                    )
                    vec = np.asarray(rep[0]["embedding"], dtype=np.float32)
                    norm = max(float(np.linalg.norm(vec)), 1e-12)
                    normalized_vec = vec / norm

                    self.result_queue.put(
                        InferenceResponse(
                            task_type="embed",
                            result=normalized_vec,
                            track_id=task.track_id,
                            frame_id=task.frame_id,
                        )
                    )

                elif task.task_type == "liveness":
                    # MiniFASNet anti-spoofing check
                    face_crop = task.image
                    spoof_result = DeepFace.extract_faces(
                        img_path=face_crop,
                        detector_backend="skip",
                        enforce_detection=False,
                        align=False,
                        anti_spoofing=True,
                    )[0]
                    is_real = bool(spoof_result.get("is_real", False))
                    score = float(spoof_result.get("antispoof_score", 0.0))
                    self.result_queue.put(
                        InferenceResponse(
                            task_type="liveness",
                            result=(is_real, score),
                            track_id=task.track_id,
                            frame_id=task.frame_id,
                        )
                    )

            except Exception as exc:
                self.result_queue.put(
                    InferenceResponse(
                        task_type=task.task_type,
                        result=None,
                        track_id=task.track_id,
                        frame_id=task.frame_id,
                        error=str(exc),
                    )
                )

    def stop(self) -> None:
        self.stop_event.set()
