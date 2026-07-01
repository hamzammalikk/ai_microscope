
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MODEL_PATH = os.path.join("model_phase2", "best_phase2.pt")
CONFIDENCE_THRESHOLD = 0.45

CLASS_NAMES = [
    "Basophil", "Eosinophil", "Erythroblast", "IG", "Lymphocyte",
    "Monocyte", "Neutrophil", "Platelet",
    "Campylobacter", "E.Coli", "Staphylococcus", "Streptococcus", "Yeast"
]

BLOOD_CELLS = {
    "Basophil", "Eosinophil", "Erythroblast", "IG",
    "Lymphocyte", "Monocyte", "Neutrophil", "Platelet"
}
BACTERIA = {
    "Campylobacter", "E.Coli", "Staphylococcus", "Streptococcus", "Yeast"
}

CLASS_COLORS = {
    "Basophil":      "#FF6464",
    "Eosinophil":    "#64C8FF",
    "Erythroblast":  "#64FF64",
    "IG":            "#FFC832",
    "Lymphocyte":    "#C864FF",
    "Monocyte":      "#32C8C8",
    "Neutrophil":    "#FF9632",
    "Platelet":      "#96FF96",
    "Campylobacter": "#3232FF",
    "E.Coli":        "#FF3232",
    "Staphylococcus":"#B432FF",
    "Streptococcus": "#32FFC8",
    "Yeast":         "#FFC8C8",
}

_model = None

def _load_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}"
            )
        _model = YOLO(MODEL_PATH)
    return _model

def load_image(image_input):
    if isinstance(image_input, str):
        return Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        return Image.fromarray(image_input).convert("RGB")
    else:
        return image_input.convert("RGB")

def predict_phase2(image_input):
    model = _load_model()
    pil_img = load_image(image_input)
    img_array = np.array(pil_img)

    results = model(img_array, conf=CONFIDENCE_THRESHOLD, verbose=False)
    result  = results[0]

    detections = []
    counts = {name: 0 for name in CLASS_NAMES}

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
    is_valid = total_detected > 0

    # Draw boxes using PIL
    annotated = pil_img.copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = CLASS_COLORS.get(det["class"], "#FFFFFF")
        label = f"{det['class']} {det['confidence']}%"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.rectangle([x1, y1 - 18, x1 + len(label) * 7, y1], fill=color)
        draw.text((x1 + 2, y1 - 16), label, fill="black")

    blood_cell_counts = {k: v for k, v in counts.items() if k in BLOOD_CELLS and v > 0}
    bacteria_counts   = {k: v for k, v in counts.items() if k in BACTERIA and v > 0}

    summary = {
        "blood_cells":       blood_cell_counts,
        "bacteria":          bacteria_counts,
        "total_blood_cells": sum(blood_cell_counts.values()),
        "total_bacteria":    sum(bacteria_counts.values()),
    }

    return {
        "annotated_image":           np.array(annotated),
        "detections":                detections,
        "counts":                    counts,
        "total_detected":            total_detected,
        "is_valid_microscope_image": is_valid,
        "summary":                   summary,
    }
