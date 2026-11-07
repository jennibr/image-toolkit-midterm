"""Feature 4: face detection (Haar cascade + optional MediaPipe)."""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "blaze_face_short_range.tflite"


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float          # 0-1
    raw_score: float | None = None  # Haar stage weight (only for Haar)

    def as_row(self) -> dict:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 3)
        if d["raw_score"] is not None:
            d["raw_score"] = round(d["raw_score"], 3)
        return d


def mediapipe_available() -> bool:
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        return hasattr(vision, "FaceDetector") and hasattr(mp, "Image")
    except Exception:
        return False


def _ensure_model() -> str:
    """Download the BlazeFace .tflite model once into models/."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return str(MODEL_PATH)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_PATH.with_suffix(".part")
    try:
        urllib.request.urlretrieve(MODEL_URL, tmp)
        tmp.replace(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the face model ({exc}). Download it manually from "
            f"{MODEL_URL} and save it as {MODEL_PATH}") from exc
    return str(MODEL_PATH)


def detect_haar(rgb: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 5,
                min_size: int = 30) -> list[Detection]:
    """Haar cascade. Confidence = 1 - exp(-w/2) of OpenCV's level weight (heuristic)."""
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError("This OpenCV build has no CascadeClassifier. "
                           "Install opencv-python-headless>=4.9,<5.")
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    if cascade.empty():
        raise RuntimeError("Could not load the Haar cascade XML.")
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    rects, _, weights = cascade.detectMultiScale3(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
        minSize=(min_size, min_size), outputRejectLevels=True)
    if len(rects) == 0:
        return []
    weights = np.asarray(weights, dtype=float).reshape(-1)
    out = []
    for (x, y, w, h), wt in zip(rects, weights):
        conf = float(1.0 - np.exp(-max(wt, 0.0) / 2.0))
        out.append(Detection(int(x), int(y), int(w), int(h), conf, float(wt)))
    return sorted(out, key=lambda d: d.confidence, reverse=True)


def tile_origins(length: int, side: int, stride: int) -> list[int]:
    """Start positions of windows of size `side` covering [0, length) with given stride."""
    if length <= side:
        return [0]
    xs = list(range(0, length - side + 1, stride))
    if xs[-1] != length - side:
        xs.append(length - side)
    return xs


def detect_mediapipe(rgb: np.ndarray, min_confidence: float = 0.5,
                     tiled: bool = True) -> list[Detection]:
    """MediaPipe BlazeFace (Tasks API, real confidence score).

    BlazeFace short-range works on ~128x128 input, so small faces in large images are
    missed. With tiled=True the image is also scanned in overlapping SQUARE crops
    (side = short edge, and half of it); results are mapped back to full-image
    coordinates. Duplicates should then be merged with non_max_suppression().
    Every model input is square, so no internal letterboxing is needed.
    """
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    H, W = rgb.shape[:2]
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=_ensure_model()),
        min_detection_confidence=min_confidence)
    detector = vision.FaceDetector.create_from_options(options)

    def run(x0: int, y0: int, crop: np.ndarray) -> list[Detection]:
        ch, cw = crop.shape[:2]
        side = max(ch, cw)
        if (ch, cw) != (side, side):  # pad to square (bottom/right) -> offsets stay 0
            padded = np.zeros((side, side, 3), np.uint8)
            padded[:ch, :cw] = crop
            crop = padded
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop))
        found = []
        for det in detector.detect(img).detections:
            bb = det.bounding_box
            x, y = max(0, int(bb.origin_x)), max(0, int(bb.origin_y))
            w, h = min(int(bb.width), cw - x), min(int(bb.height), ch - y)
            if w < 8 or h < 8:
                continue
            score = float(det.categories[0].score) if det.categories else 0.0
            found.append(Detection(x0 + x, y0 + y, w, h, score))
        return found

    try:
        out = run(0, 0, rgb)  # whole image
        if tiled:
            short = min(H, W)
            for side in (short, short // 2):
                if side < 96:
                    continue
                stride = max(1, int(side * 0.6))
                for y0 in tile_origins(H, side, stride):
                    for x0 in tile_origins(W, side, stride):
                        out += run(x0, y0, rgb[y0:y0 + side, x0:x0 + side])
    finally:
        detector.close()
    return sorted(out, key=lambda d: d.confidence, reverse=True)


def iou(a: Detection, b: Detection) -> float:
    """Intersection over Union of two boxes: area(A∩B) / area(A∪B)."""
    ix = max(0, min(a.x + a.w, b.x + b.w) - max(a.x, b.x))
    iy = max(0, min(a.y + a.h, b.y + b.h) - max(a.y, b.y))
    inter = ix * iy
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


def non_max_suppression(dets: list[Detection], iou_threshold: float = 0.3) -> list[Detection]:
    """Greedy NMS: keep the highest-confidence box, drop boxes overlapping it
    with IoU > threshold, repeat. threshold >= 1.0 disables suppression."""
    if iou_threshold >= 1.0:
        return list(dets)
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.confidence, reverse=True):
        if all(iou(d, k) <= iou_threshold for k in kept):
            kept.append(d)
    return kept


def draw_detections(rgb: np.ndarray, dets: list[Detection]) -> np.ndarray:
    out = rgb.copy()
    thick = max(2, int(round(max(out.shape[:2]) / 300)))
    scale = max(0.5, max(out.shape[:2]) / 1200)
    for i, d in enumerate(dets, 1):
        cv2.rectangle(out, (d.x, d.y), (d.x + d.w, d.y + d.h), (0, 255, 0), thick)
        label = f"#{i} {d.confidence:.2f}"
        cv2.putText(out, label, (d.x, max(15, d.y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 0), max(1, thick - 1), cv2.LINE_AA)
    return out