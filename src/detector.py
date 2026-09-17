"""High-performance Face Detection Engine using SCRFD & YuNet via ONNX Runtime.

Configures dynamic hardware acceleration via CUDAExecutionProvider when an
NVIDIA GPU is present, falling back to an optimized CPUExecutionProvider
or OpenCV FaceDetectorYN ONNX engine.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def nms_bboxes(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """Non-Maximum Suppression in pure NumPy."""
    if len(bboxes) == 0:
        return []

    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-12)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class BaseFaceDetector:
    def detect(self, image_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        raise NotImplementedError


class SCRFDDetector(BaseFaceDetector):
    """SCRFD (Sample & Computation Redistribution for Face Detection) ONNX Runtime engine."""

    def __init__(
        self,
        model_path: Path | str,
        conf_threshold: float = 0.50,
        nms_threshold: float = 0.45,
        input_size: Tuple[int, int] = (640, 640),
        num_threads: int = 2,
    ) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime is not installed. Install with: pip install onnxruntime")

        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size

        available = ort.get_available_providers()
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        orig_h, orig_w = image_bgr.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / orig_w, target_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(padded, 1.0 / 128.0, (target_w, target_h), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        scores_list = []
        bboxes_list = []

        for idx, stride in enumerate(self._feat_stride_fpn):
            score = outputs[idx]
            bbox = outputs[idx + self.fmc] * stride

            height = target_h // stride
            width = target_w // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(score >= self.conf_threshold)[0]
            if len(pos_inds) == 0:
                continue

            score = score[pos_inds]
            bbox = bbox[pos_inds]
            anchor_centers = anchor_centers[pos_inds]

            x1 = anchor_centers[:, 0] - bbox[:, 0]
            y1 = anchor_centers[:, 1] - bbox[:, 1]
            x2 = anchor_centers[:, 0] + bbox[:, 2]
            y2 = anchor_centers[:, 1] + bbox[:, 3]

            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            scores_list.append(score.flatten())
            bboxes_list.append(boxes)

        if not scores_list:
            return []

        all_scores = np.concatenate(scores_list, axis=0)
        all_bboxes = np.concatenate(bboxes_list, axis=0)

        keep = nms_bboxes(all_bboxes, all_scores, self.nms_threshold)
        results: List[Tuple[Tuple[int, int, int, int], float]] = []

        for k in keep:
            x1, y1, x2, y2 = all_bboxes[k]
            x1 = max(0, int(round(x1 / scale)))
            y1 = max(0, int(round(y1 / scale)))
            x2 = min(orig_w, int(round(x2 / scale)))
            y2 = min(orig_h, int(round(y2 / scale)))
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)
            results.append(((x1, y1, w, h), float(all_scores[k])))

        return results


class YuNetDetector(BaseFaceDetector):
    """Direct OpenCV FaceDetectorYN engine using preloaded ONNX model."""

    def __init__(self, conf_threshold: float = 0.50, nms_threshold: float = 0.40) -> None:
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.model_path = os.path.expanduser("~/.deepface/weights/face_detection_yunet_2023mar.onnx")
        self.active_provider = "CPU (OpenCV FaceDetectorYN)"
        self._detector = None
        self._current_size = (0, 0)

    def detect(self, image_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        h, w = image_bgr.shape[:2]
        if self._detector is None or self._current_size != (w, h):
            if os.path.isfile(self.model_path):
                self._detector = cv2.FaceDetectorYN.create(
                    self.model_path, "", (w, h),
                    score_threshold=self.conf_threshold,
                    nms_threshold=self.nms_threshold,
                    top_k=20,
                )
                self._current_size = (w, h)

        if self._detector is not None:
            _, faces = self._detector.detect(image_bgr)
            if faces is None:
                return []
            results = []
            for f in faces:
                x, y, bw, bh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                score = float(f[14])
                if score >= self.conf_threshold and bw > 15 and bh > 15:
                    results.append(((max(0, x), max(0, y), bw, bh), score))
            return results

        # Fallback to DeepFace extract_faces if ONNX not found
        try:
            from deepface import DeepFace
            faces = DeepFace.extract_faces(
                img_path=image_bgr,
                detector_backend="yunet",
                enforce_detection=False,
                align=False,
            )
            results = []
            for face in faces:
                conf = face.get("confidence", 0.0)
                if conf < self.conf_threshold:
                    continue
                area = face.get("facial_area", {})
                if all(k in area for k in ("x", "y", "w", "h")):
                    results.append(((int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])), float(conf)))
            return results
        except Exception:
            return []


def create_face_detector(
    model_path: Path | str | None = None,
    conf_threshold: float = 0.50,
    nms_threshold: float = 0.40,
    cpu_threads: int = 2,
) -> BaseFaceDetector:
    if model_path and Path(model_path).is_file() and ort is not None:
        try:
            detector = SCRFDDetector(
                model_path=model_path,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
                num_threads=cpu_threads,
            )
            print(f"[Detector] Loaded SCRFD ONNX on provider: {detector.active_provider}")
            return detector
        except Exception as exc:
            print(f"[Detector] Could not load ONNX model ({exc}); falling back.")

    detector = YuNetDetector(conf_threshold=conf_threshold, nms_threshold=nms_threshold)
    print(f"[Detector] Using YuNet engine ({detector.active_provider})")
    return detector
