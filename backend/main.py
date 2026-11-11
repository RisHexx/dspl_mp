# -*- coding: utf-8 -*-
"""
FastAPI backend for traffic sign detection.

Two-stage pipeline:
  1. DETECTOR  (YOLO, optional)  – finds every sign in the frame → bounding boxes
  2. CLASSIFIER (your .h5 CNN)   – names each cropped sign (43 GTSRB classes)

If no detector weights are found (``detector.pt`` in the project root), the
server falls back to the old OpenCV auto-crop + single-sign classification so
the app keeps working while the detector is being trained.
"""

import os
import sys
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ── Labels (imported from project root) ──────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from labels import SIGN_NAMES

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 32
MODEL_PATH = os.path.join(ROOT, "traffic_sign_model.h5")
DETECTOR_PATH = os.environ.get("DETECTOR_PATH", os.path.join(ROOT, "detector.pt"))

# Detector settings.
# The detector was trained on GTSDB, where signs are small (<~130 px). A sign
# held close to a webcam is much bigger than anything it has seen, so it gets
# missed. Running the detector on several downscaled copies of the frame fixes
# this: 416 catches small/medium signs, 192 shrinks big close-up signs back
# into the size range the model knows. Boxes from all scales are merged (NMS).
DETECTOR_CONF = float(os.environ.get("DETECTOR_CONF", 0.30))
DETECTOR_SCALES = [int(s) for s in os.environ.get("DETECTOR_SCALES", "416,192").split(",")]
NMS_IOU = 0.5
MAX_DETECTIONS = 10
# If the detector finds nothing, try the old OpenCV auto-crop and keep its
# result only when the classifier is very sure (avoids random boxes).
FALLBACK_WHEN_EMPTY = os.environ.get("FALLBACK_WHEN_EMPTY", "1") == "1"
FALLBACK_MIN_CONF = float(os.environ.get("FALLBACK_MIN_CONF", 0.90))
# GTSRB training images have a ~10% border around the sign, so we pad each
# detector box by the same amount before classifying it.
CROP_PAD = 0.10

classifier = None
detector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier, detector
    print(f"Loading classifier from {MODEL_PATH} ...")
    classifier = load_model(MODEL_PATH)
    print("Classifier loaded.")

    if os.path.exists(DETECTOR_PATH):
        try:
            from ultralytics import YOLO
            detector = YOLO(DETECTOR_PATH)
            print(f"Detector loaded from {DETECTOR_PATH}.")
        except ImportError:
            print("ultralytics is not installed - `pip install ultralytics` to use the detector.")
    else:
        print(f"No detector at {DETECTOR_PATH} - using OpenCV auto-crop fallback.")
    yield


# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Sign Detection API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Stage 1a: YOLO detector ──────────────────────────────────────────────────

def detect_boxes(img):
    """Return a list of (x1, y1, x2, y2, score) boxes for every sign found,
    running the detector at several scales and merging overlapping boxes."""
    candidates = []
    for size in DETECTOR_SCALES:
        res = detector.predict(
            img, conf=DETECTOR_CONF, imgsz=size,
            max_det=MAX_DETECTIONS, verbose=False,
        )[0]
        for (x1, y1, x2, y2), score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
            candidates.append((int(x1), int(y1), int(x2), int(y2), float(score)))

    if len(candidates) <= 1:
        return candidates

    # Merge duplicates of the same sign found at different scales
    rects = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _ in candidates]
    scores = [c[4] for c in candidates]
    keep = cv2.dnn.NMSBoxes(rects, scores, DETECTOR_CONF, NMS_IOU)
    keep = np.array(keep).flatten().tolist()
    return [candidates[i] for i in keep][:MAX_DETECTIONS]


# ── Stage 1b: OpenCV fallback (used when no detector is available) ───────────

def auto_crop_box(img):
    """Guess a single sign region from colour + edges. Returns (x1, y1, x2, y2)."""
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, sat_mask = cv2.threshold(hsv[:, :, 1], 60, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50, 150)

    combined = cv2.bitwise_or(sat_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=3)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 0.05 * h * w:
            x, y, cw, ch = cv2.boundingRect(largest)
            return x, y, x + cw, y + ch

    # Nothing found → centre square with a 15% margin
    margin = int(min(h, w) * 0.15)
    size = min(h, w) - 2 * margin
    x1, y1 = (w - size) // 2, (h - size) // 2
    return x1, y1, x1 + size, y1 + size


# ── Stage 2: classifier ──────────────────────────────────────────────────────

def pad_box(box, w, h, pad=CROP_PAD):
    x1, y1, x2, y2 = box
    px, py = int((x2 - x1) * pad), int((y2 - y1) * pad)
    return max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)


def to_classifier_input(crop_bgr):
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype("float32") / 255.0


def classify_boxes(img, boxes):
    """Crop every box, classify them all in one batch, return result dicts."""
    h, w = img.shape[:2]
    crops, kept = [], []
    for x1, y1, x2, y2, score in boxes:
        px1, py1, px2, py2 = pad_box((x1, y1, x2, y2), w, h)
        if px2 - px1 < 4 or py2 - py1 < 4:
            continue
        crops.append(to_classifier_input(img[py1:py2, px1:px2]))
        kept.append((x1, y1, x2, y2, score))

    if not crops:
        return []

    probs = classifier.predict(np.stack(crops), verbose=0)
    results = []
    for (x1, y1, x2, y2, score), p in zip(kept, probs):
        class_id = int(np.argmax(p))
        results.append({
            "box": [x1, y1, x2, y2],
            "det_score": round(score, 4) if score is not None else None,
            "class_id": class_id,
            "sign_name": SIGN_NAMES.get(class_id, f"Unknown ({class_id})"),
            "confidence": round(float(p[class_id]), 4),
        })
    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


# ── Prediction endpoint ─────────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    h, w = img.shape[:2]

    detections = []
    if detector is not None:
        mode = "detector"
        detections = classify_boxes(img, detect_boxes(img))

    if detector is None or (not detections and FALLBACK_WHEN_EMPTY):
        x1, y1, x2, y2 = auto_crop_box(img)
        fallback = classify_boxes(img, [(x1, y1, x2, y2, None)])
        if detector is None:
            mode, detections = "fallback", fallback
        elif fallback and fallback[0]["confidence"] >= FALLBACK_MIN_CONF:
            mode, detections = "detector+fallback", fallback
    best = detections[0] if detections else None

    return {
        "mode": mode,
        "image_size": [w, h],
        "detections": detections,
        # Top-level fields kept for backward compatibility (best detection)
        "class_id": best["class_id"] if best else None,
        "sign_name": best["sign_name"] if best else None,
        "confidence": best["confidence"] if best else 0.0,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "detector_loaded": detector is not None,
    }
