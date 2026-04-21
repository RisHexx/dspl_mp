# -*- coding: utf-8 -*-
"""
FastAPI backend for traffic sign detection.
Loads the trained .h5 model once at startup and exposes a /predict endpoint.
"""

import os
import sys
import io

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

# ── Labels (imported from project root) ──────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from labels import SIGN_NAMES

# ── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "traffic_sign_model.h5")

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Sign Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ───────────────────────────────────────────────
model = None


@app.on_event("startup")
def load_model_on_startup():
    global model
    print(f"Loading model from {MODEL_PATH} ...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")


# ── Preprocessing helpers (same logic as test_image.py) ──────────────────────

def crop_center_square(img):
    h, w = img.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return img[y_start:y_start + size, x_start:x_start + size]


def auto_crop_sign(img):
    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    _, sat_mask = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    combined = cv2.bitwise_or(sat_mask, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=3)

    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area > 0.05 * h * w:
            x, y, cw, ch = cv2.boundingRect(largest)
            pad_x = int(cw * 0.10)
            pad_y = int(ch * 0.10)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            cw = min(w - x, cw + 2 * pad_x)
            ch = min(h - y, ch + 2 * pad_y)
            return img[y:y + ch, x:x + cw]

    margin = int(min(h, w) * 0.15)
    if margin > 0:
        img = img[margin:h - margin, margin:w - margin]

    return crop_center_square(img)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Read raw bytes → decode → auto-crop → resize → normalise → batch."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("Could not decode image")

    cropped = auto_crop_sign(img)
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalised = resized.astype("float32") / 255.0
    return np.expand_dims(normalised, axis=0)


# ── Prediction endpoint ─────────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    batch = preprocess_image(contents)
    predictions = model.predict(batch, verbose=0)[0]
    class_id = int(np.argmax(predictions))
    confidence = float(predictions[class_id])
    sign_name = SIGN_NAMES.get(class_id, f"Unknown ({class_id})")

    return {
        "class_id": class_id,
        "sign_name": sign_name,
        "confidence": round(confidence, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
