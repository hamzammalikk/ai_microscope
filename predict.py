import os
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

IMG_SIZE = (224, 224)
MODEL_PATH = os.path.join("model", "best_model.onnx")
NAMES_PATH = os.path.join("model", "class_names.npy")

_session = None
_class_names = None

def _load_assets():
    global _session, _class_names
    if _session is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        _session = ort.InferenceSession(MODEL_PATH)
    if _class_names is None:
        if not os.path.exists(NAMES_PATH):
            raise FileNotFoundError(f"Class names not found at {NAMES_PATH}")
        _class_names = np.load(NAMES_PATH, allow_pickle=True).tolist()
    return _session, _class_names

def preprocess_image(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(image_input)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict(image_input):
    session, class_names = _load_assets()
    tensor = preprocess_image(image_input)
    input_name = session.get_inputs()[0].name
    probs = session.run(None, {input_name: tensor})[0][0]
    top_index = int(np.argmax(probs))
    predicted_class = class_names[top_index]
    confidence = float(probs[top_index]) * 100
    all_scores = {
        class_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(class_names))
    }
    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "all_scores": all_scores
    }

def generate_gradcam(image_input):
    img = preprocess_image(image_input)[0]
    original_img = np.uint8(255 * img)
    gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    return overlay
