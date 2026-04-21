# -*- coding: utf-8 -*-
"""
Test the trained traffic sign model on a single static image.

Usage:
    python test_image.py path/to/image.jpg

IMPORTANT: The model was trained on GTSRB images where the traffic sign is
           tightly cropped and fills most of the frame. Google images have
           lots of background (sky, road, poles). This script auto-crops to
           the center region so the sign fills the frame like GTSRB.
"""

import sys
import os

# Fix Windows terminal Unicode crash (cp1252 can't print emoji)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from labels import SIGN_NAMES

IMG_SIZE = 32
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "traffic_sign_model.h5")


def crop_center_square(img):
    """
    Crop the largest centered square from the image.
    GTSRB signs are tightly cropped squares — this makes Google images
    look more like the training data by removing side background.
    """
    h, w = img.shape[:2]
    size = min(h, w)
    y_start = (h - size) // 2
    x_start = (w - size) // 2
    return img[y_start:y_start + size, x_start:x_start + size]


def auto_crop_sign(img):
    """
    Try to isolate the traffic sign from a noisy background image.
    Uses color saturation + edge detection to find the sign region.
    Falls back to center-crop if detection fails.
    """
    h, w = img.shape[:2]

    # Convert to HSV to detect saturated colored regions (signs are colorful)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]

    # Threshold on saturation — traffic signs are highly saturated (red, blue, yellow)
    _, sat_mask = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)

    # Also use edges to find the sign boundary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Combine saturation mask + edges
    combined = cv2.bitwise_or(sat_mask, edges)

    # Dilate to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined = cv2.dilate(combined, kernel, iterations=3)

    # Find contours and pick the largest one
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour by area
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Only use contour if it's at least 5% of image area (not noise)
        if area > 0.05 * h * w:
            x, y, cw, ch = cv2.boundingRect(largest)

            # Add 10% padding around the detected region
            pad_x = int(cw * 0.10)
            pad_y = int(ch * 0.10)
            x = max(0, x - pad_x)
            y = max(0, y - pad_y)
            cw = min(w - x, cw + 2 * pad_x)
            ch = min(h - y, ch + 2 * pad_y)

            return img[y:y + ch, x:x + cw]

    # Fallback: center crop with 70% of the image (trim outer edges)
    margin = int(min(h, w) * 0.15)
    if margin > 0:
        img = img[margin:h - margin, margin:w - margin]

    return crop_center_square(img)


def predict(image_path, model):
    """Load an image, preprocess it, and return the prediction."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌  Could not open image: {image_path}")
        sys.exit(1)

    original_h, original_w = img.shape[:2]

    # Auto-crop to isolate the sign (handles Google images with background)
    cropped = auto_crop_sign(img)
    crop_h, crop_w = cropped.shape[:2]

    # OpenCV loads BGR — convert to RGB to match training (PIL / Hugging Face)
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    normalised = resized.astype("float32") / 255.0
    batch = np.expand_dims(normalised, axis=0)

    # Predict
    predictions = model.predict(batch, verbose=0)[0]
    class_id = int(np.argmax(predictions))
    confidence = float(predictions[class_id])
    sign_name = SIGN_NAMES.get(class_id, f"Unknown ({class_id})")

    # Extra info
    print(f"  Original   : {original_w}x{original_h}")
    print(f"  Auto-crop  : {crop_w}x{crop_h}")

    return sign_name, confidence, class_id


def main():
    if len(sys.argv) < 2:
        print("Usage:  python test_image.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"❌  File not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌  Model not found at {MODEL_PATH}")
        print("   Run the notebook first to train and save the model.")
        sys.exit(1)

    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("  ✓ Model loaded.\n")

    sign_name, confidence, class_id = predict(image_path, model)

    print("=" * 50)
    print(f"  Image      : {image_path}")
    print(f"  Prediction : {sign_name}")
    print(f"  Class ID   : {class_id}")
    print(f"  Confidence : {confidence:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    main()
