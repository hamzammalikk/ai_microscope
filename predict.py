# ============================================================
# predict.py — AI Microscope: Level 1
# Loads the trained model and predicts the class of a new image
# Can be used standalone (CLI) or imported by app.py
# ============================================================

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ── CONFIGURATION ────────────────────────────────────────────
IMG_SIZE  = (224, 224)                              # Must match training size
MODEL_PATH = os.path.join("model", "best_model.h5")
NAMES_PATH = os.path.join("model", "class_names.npy")


# ── LOAD MODEL ONCE (cached at module level) ─────────────────
# Loading a model is slow (~1-2 seconds), so we do it once
# and reuse it for every prediction call.

_model       = None
_class_names = None

def _load_assets():
    """Lazily loads model and class names on first call."""
    global _model, _class_names

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"❌ Trained model not found at '{MODEL_PATH}'.\n"
                "Run train.py first to generate the model."
            )
        _model = load_model(MODEL_PATH)

    if _class_names is None:
        if not os.path.exists(NAMES_PATH):
            raise FileNotFoundError(
                f"❌ Class names file not found at '{NAMES_PATH}'.\n"
                "Run train.py first — it saves class_names.npy automatically."
            )
        _class_names = np.load(NAMES_PATH, allow_pickle=True).tolist()

    return _model, _class_names


# ── PREPROCESSING ────────────────────────────────────────────

def preprocess_image(image_input):
    """
    Accepts either:
    - A file path (str)
    - A NumPy array (already loaded image, e.g. from Streamlit upload)

    Returns a normalised 4D tensor ready for model.predict().
    Shape: (1, 224, 224, 3)
    """
    if isinstance(image_input, str):
        # Load from file path
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"❌ Could not read image at path: {image_input}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR → convert to RGB
    else:
        # Assume it's already a NumPy array (from PIL/Streamlit)
        img = np.array(image_input)
        if img.ndim == 2:
            # Grayscale → convert to RGB by stacking channels
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            # RGBA → drop alpha channel
            img = img[:, :, :3]

    # Resize to model input size
    img = cv2.resize(img, IMG_SIZE)

    # Normalise: pixel values 0–255 → 0.0–1.0
    img = img.astype(np.float32) / 255.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)

    return img


# ── PREDICTION FUNCTION ──────────────────────────────────────

def predict(image_input):
    """
    Main prediction function.

    Args:
        image_input: file path (str) or NumPy array / PIL Image

    Returns:
        dict with keys:
            - predicted_class (str)
            - confidence (float, 0–100)
            - all_scores (dict: class → confidence%)
    """
    model, class_names = _load_assets()

    # Preprocess the image
    tensor = preprocess_image(image_input)

    # Run the model — returns array of probabilities, one per class
    probs = model.predict(tensor, verbose=0)[0]  # shape: (num_classes,)

    # Find the top prediction
    top_index      = int(np.argmax(probs))
    predicted_class = class_names[top_index]
    confidence      = float(probs[top_index]) * 100

    # Build a full score dictionary for all classes
    all_scores = {
        class_names[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(class_names))
    }

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2),
        "all_scores": all_scores
    }


# ── GRAD-CAM HEATMAP ─────────────────────────────────────────
# Grad-CAM = Gradient-weighted Class Activation Mapping
# It highlights WHICH PARTS of the image most influenced the prediction.
# Think of it as asking: "What did the AI 'look at' to make this decision?"

def generate_gradcam(image_input, layer_name=None):
    """
    Generates a Grad-CAM heatmap overlay for the prediction.

    Args:
        image_input: file path or NumPy/PIL image
        layer_name: name of the last Conv2D layer (auto-detected if None)

    Returns:
        heatmap_overlay (NumPy array, RGB) — same size as input image
    """
    import tensorflow as tf

    model, class_names = _load_assets()
    tensor = preprocess_image(image_input)  # shape (1, 224, 224, 3)

    # Auto-detect last convolutional layer if not specified
    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        if layer_name is None:
            raise ValueError("❌ No Conv2D layer found in model for Grad-CAM.")

    # Build a sub-model that outputs both the conv layer and final predictions
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    # Record gradients during forward pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor)
        top_class_idx = tf.argmax(predictions[0])
        loss = predictions[:, top_class_idx]  # Score for top predicted class

    # Compute gradients of top class score w.r.t. conv layer feature maps
    grads = tape.gradient(loss, conv_outputs)

    # Pool gradients across spatial dimensions to get per-channel weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its gradient importance
    conv_outputs = conv_outputs[0]                         # Remove batch dim
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis] # Weighted sum
    heatmap = tf.squeeze(heatmap)                          # Flatten to 2D

    # Normalise heatmap to 0–1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to match original image size
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)

    # Convert to colour (blue→green→red heat scale)
    heatmap_colour = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_colour = cv2.cvtColor(heatmap_colour, cv2.COLOR_BGR2RGB)

    # Reconstruct the original image at 224×224 for overlay
    original_img = preprocess_image(image_input)[0]  # (224, 224, 3), 0-1
    original_img = np.uint8(255 * original_img)

    # Blend heatmap over original image (60% original, 40% heatmap)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_colour, 0.4, 0)

    return overlay  # Returns RGB NumPy array


# ── COMMAND-LINE USAGE ───────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"\n🔬 Predicting: {image_path}")

    result = predict(image_path)

    print(f"\n✅ Predicted Class : {result['predicted_class']}")
    print(f"   Confidence      : {result['confidence']:.1f}%")
    print(f"\n📊 All Class Scores:")
    for cls, score in sorted(result["all_scores"].items(),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(score / 5)
        print(f"   {cls:<20} {score:>6.2f}%  {bar}")
