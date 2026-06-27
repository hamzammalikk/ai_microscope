
import os
import io
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")

from predict import predict, generate_gradcam

st.set_page_config(
    page_title="AI Microscope",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.prediction-card {
    background: linear-gradient(135deg, #e8f4fd, #f0f7ff);
    border-left: 5px solid #1a73e8;
    border-radius: 10px;
    padding: 1.5rem;
    margin-top: 1rem;
}
.class-label {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1a73e8;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.markdown("## 🔬 AI Microscope")
    st.markdown("**Level 1 — Image Classifier**")
    st.divider()
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Muhammad Hamza Malik**")
    st.divider()
    st.markdown("### About")
    st.info("Upload a microscope image to classify it using a trained CNN.")
    st.markdown("### How it works")
    st.markdown("""
1. Upload a microscope image
2. AI preprocesses it
3. CNN predicts the class
4. Grad-CAM shows focus area
""")
    st.divider()
    hist_path = os.path.join("outputs", "training_history.png")
    if os.path.exists(hist_path):
        st.image(hist_path, caption="Training Accuracy & Loss")
    cm_path = os.path.join("outputs", "confusion_matrix.png")
    if os.path.exists(cm_path):
        st.image(cm_path, caption="Confusion Matrix")

# MAIN
st.markdown("<h1 style='text-align:center;color:#1a73e8;'>🔬 AI Microscope</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload a microscope image for instant AI classification</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="📂 Upload a microscope image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_image)

    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("### 📷 Uploaded Image")
        st.image(pil_image, caption=f"File: {uploaded_file.name}", use_column_width=True)

    with col_result:
        st.markdown("### 🧠 AI Prediction")
        model_path = os.path.join("model", "best_model.h5")
        if not os.path.exists(model_path):
            st.error("No trained model found. Please run train.py first.")
        else:
            with st.spinner("🔬 Analysing image..."):
                try:
                    result = predict(img_array)
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    st.stop()

            st.markdown(f"""
            <div class="prediction-card">
                <div class="class-label">🏷️ {result["predicted_class"]}</div>
                <p>Confidence: <strong>{result["confidence"]:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Confidence Meter")
            conf = result["confidence"]
            bar_colour = "🟢" if conf >= 80 else "🟡" if conf >= 50 else "🔴"
            st.markdown(f"{bar_colour} **{conf:.1f}%** certainty")
            st.progress(conf / 100)

            st.markdown("#### 🗂️ All Class Scores")
            for cls, score in sorted(result["all_scores"].items(), key=lambda x: x[1], reverse=True):
                cols = st.columns([3, 7])
                cols[0].markdown(f"**{cls}**")
                cols[1].progress(score / 100, text=f"{score:.1f}%")

    st.divider()
    st.markdown("### 🌡️ Grad-CAM — Where Did the AI Look?")
    model_path = os.path.join("model", "best_model.h5")
    if os.path.exists(model_path):
        with st.spinner("🔥 Generating heatmap..."):
            try:
                heatmap_overlay = generate_gradcam(img_array)
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(pil_image, caption="Original Image", use_column_width=True)
                with col_heat:
                    st.image(heatmap_overlay, caption="Grad-CAM Heatmap", use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated: {e}")

else:
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding:3rem; color:#888; border: 2px dashed #ccc; border-radius:12px;'>
        <h2>📤 Upload an Image to Begin</h2>
        <p>Drag & drop or click the upload button above.</p>
    </div>
    """, unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown(
    "<center>🔬 Developed by <b>Muhammad Hamza Malik</b> | AI Microscope v1.0</center>",
    unsafe_allow_html=True
)
