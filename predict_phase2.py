# ============================================================
# predict_phase2.py — AI Microscope Phase 2
# YOLOv8 Object Detection: Blood Cells + Bacteria
# ============================================================

import os
import cv2
import numpy as np
from PIL import Image

# ── CONFIGURATION ────────────────────────────────────────────
MODEL_PATH = os.path.join("model_phase2", "best_phase2.pt")

# Confidence threshold — detections below this are rejected
# This prevents random objects being classified as cells
CONFIDENCE_THRESHOLD = 0.45

# All 13 class names in correct order
CLASS_NAMES = [
    "Basophil", "Eosinophil", "Erythroblast", "IG", "Lymphocyte",
    "Monocyte", "Neutrophil", "Platelet",
    "Campylobacter", "E.Coli", "Staphylococcus", "Streptococcus", "Yeast"
]

# Class categories for report grouping
BLOOD_CELLS = {
    "Basophil", "Eosinophil", "Erythroblast", "IG",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
}
BACTERIA = {
    "Campylobacter", "E.Coli", "Staphylococcus", "Streptococcus", "Yeast"
}

# Colour per class for bounding boxes (BGR for OpenCV)
CLASS_COLORS = {
    "Basophil":      (255, 100, 100),
    "Eosinophil":    (100, 200, 255),
    "Erythroblast":  (100, 255, 100),
    "IG":            (255, 200, 50),
    "Lymphocyte":    (200, 100, 255),
    "Monocyte":      (50, 200, 200),
    "Neutrophil":    (255, 150, 50),
    "Platelet":      (150, 255, 150),
    "Campylobacter": (50, 50, 255),
    "E.Coli":        (255, 50, 50),
    "Staphylococcus":(180, 50, 255),
    "Streptococcus": (50, 255, 200),
    "Yeast":         (255, 200, 200),
}

# ── LOAD MODEL ───────────────────────────────────────────────
_model = None

def _load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Model not found at '{MODEL_PATH}'.\n"
                "Please ensure best_phase2.pt is in the model_phase2/ folder."
            )
        _model = YOLO(MODEL_PATH)
    return _model


# ── PREPROCESS IMAGE ─────────────────────────────────────────
def load_image(image_input):
    """
    Accepts PIL Image or file path.
    Returns RGB NumPy array.
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image: {image_input}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        return img


# ── MAIN PREDICTION FUNCTION ─────────────────────────────────
def predict_phase2(image_input):
    """
    Runs YOLOv8 detection on the input image.

    Returns dict with:
        - annotated_image: RGB NumPy array with boxes drawn
        - detections: list of dicts with class, confidence, bbox
        - counts: dict of class -> count
        - total_detected: int
        - is_valid_microscope_image: bool
        - summary: dict with blood cell and bacteria breakdown
    """
    model = _load_model()
    img_rgb = load_image(image_input)

    # Run YOLOv8 inference
    results = model(img_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result  = results[0]

    detections = []
    counts = {name: 0 for name in CLASS_NAMES}

    # Parse detections
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            conf       = float(box.conf[0])
            class_id   = int(box.cls[0])
            class_name = CLASS_NAMES[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append({
                "class":      class_name,
                "confidence": round(conf * 100, 1),
                "bbox":       (x1, y1, x2, y2)
            })
            counts[class_name] += 1

    total_detected = sum(counts.values())

    # ── NON-MICROSCOPE IMAGE DETECTION ──────────────────────
    # If nothing is detected above threshold, flag as invalid
    is_valid = total_detected > 0

    # ── DRAW BOUNDING BOXES ──────────────────────────────────
    annotated = img_rgb.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color           = CLASS_COLORS.get(det["class"], (255, 255, 255))
        label           = f"{det['class']} {det['confidence']}%"

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated,
                      (x1, y1 - label_size[1] - 8),
                      (x1 + label_size[0] + 4, y1),
                      color, -1)

        # Draw label text
        cv2.putText(annotated, label,
                    (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 1, cv2.LINE_AA)

    # ── SUMMARY ──────────────────────────────────────────────
    blood_cell_counts   = {k: v for k, v in counts.items() if k in BLOOD_CELLS and v > 0}
    bacteria_counts     = {k: v for k, v in counts.items() if k in BACTERIA and v > 0}
    total_blood_cells   = sum(blood_cell_counts.values())
    total_bacteria      = sum(bacteria_counts.values())

    summary = {
        "blood_cells":        blood_cell_counts,
        "bacteria":           bacteria_counts,
        "total_blood_cells":  total_blood_cells,
        "total_bacteria":     total_bacteria,
    }

    return {
        "annotated_image":           annotated,
        "detections":                detections,
        "counts":                    counts,
        "total_detected":            total_detected,
        "is_valid_microscope_image": is_valid,
        "summary":                   summary,
    }
