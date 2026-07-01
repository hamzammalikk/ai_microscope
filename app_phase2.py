# ============================================================
# app_phase2.py — AI Microscope Phase 2
# Streamlit Web App: Blood Cell + Bacteria Detection
# Developer: Muhammad Hamza Malik
# Run with: streamlit run app_phase2.py
# ============================================================

import os
import io
import datetime
import numpy as np
import streamlit as st
from PIL import Image

from predict_phase2 import predict_phase2, CLASS_NAMES, BLOOD_CELLS, BACTERIA

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="AI Microscope — Phase 2",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #1a3c6e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .developer {
        text-align: center;
        color: #1a6e3c;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .disclaimer {
        background: #fff8e1;
        border-left: 5px solid #f9a825;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #555;
    }
    .error-card {
        background: #fdecea;
        border-left: 5px solid #e53935;
        border-radius: 8px;
        padding: 1.2rem;
        margin: 1rem 0;
    }
    .count-card {
        background: linear-gradient(135deg, #e8f4fd, #f0f7ff);
        border-left: 5px solid #1a73e8;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .bacteria-card {
        background: linear-gradient(135deg, #fce8e8, #fff0f0);
        border-left: 5px solid #e53935;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microscope.png", width=80)
    st.markdown("## 🔬 AI Microscope")
    st.markdown("**Phase 2 — Detection & Counting**")
    st.divider()

    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Muhammad Hamza Malik**")
    st.divider()

    st.markdown("### 🧬 Detectable Classes")
    st.markdown("**Blood Cells (8):**")
    for cls in sorted(BLOOD_CELLS):
        st.markdown(f"&nbsp;&nbsp;• {cls}")
    st.markdown("**Bacteria (5):**")
    for cls in sorted(BACTERIA):
        st.markdown(f"&nbsp;&nbsp;• {cls}")
    st.divider()

    st.markdown("### ℹ️ About")
    st.info(
        "Phase 2 uses YOLOv8 to detect and count multiple "
        "cells and bacteria simultaneously in a single "
        "microscope image with 89.5% mAP50 accuracy."
    )

    st.markdown("### ⚠️ Disclaimer")
    st.warning(
        "This AI tool is for **research and educational purposes only**. "
        "It is NOT a certified medical device. "
        "Always consult a qualified medical professional for diagnosis."
    )


# ── HEADER ───────────────────────────────────────────────────
st.markdown('<p class="main-title">🔬 AI Microscope — Phase 2</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automatic Detection & Counting of Blood Cells and Bacteria</p>', unsafe_allow_html=True)
st.markdown('<p class="developer">Developed by Muhammad Hamza Malik</p>', unsafe_allow_html=True)

# ── DISCLAIMER BANNER ────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Important Notice:</strong> This AI system is designed for 
<strong>research and educational use only</strong>. While it achieves 89.5% mAP50 accuracy 
on test data, <strong>AI can make mistakes</strong> and should never replace professional 
medical diagnosis. Results must always be verified by a qualified pathologist or clinician.
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    label="📂 Upload a microscope image",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    help="Upload a blood smear or bacterial culture microscope image"
)

# ── PROCESS IMAGE ────────────────────────────────────────────
if uploaded_file is not None:

    pil_image = Image.open(uploaded_file).convert("RGB")

    col_orig, col_result = st.columns([1, 1], gap="large")

    with col_orig:
        st.markdown("### 📷 Original Image")
        st.image(pil_image, caption=f"Uploaded: {uploaded_file.name}",
                 use_column_width=True)

    with col_result:
        st.markdown("### 🧠 AI Detection Result")

        model_path = os.path.join("model_phase2", "best_phase2.pt")
        if not os.path.exists(model_path):
            st.error(
                "⚠️ Model not found at `model_phase2/best_phase2.pt`.\n\n"
                "Please ensure the trained model file is in the correct folder."
            )
            st.stop()

        with st.spinner("🔬 Detecting cells and bacteria..."):
            try:
                result = predict_phase2(pil_image)
            except Exception as e:
                st.error(f"❌ Detection failed: {e}")
                st.stop()

        # ── NON-MICROSCOPE IMAGE ERROR ────────────────────────
        if not result["is_valid_microscope_image"]:
            st.markdown("""
            <div class="error-card">
                <h4>❌ No Cells or Bacteria Detected</h4>
                <p>This image does not appear to be a valid microscope image, 
                or the image quality is too low for reliable detection.</p>
                <p><strong>Please ensure:</strong></p>
                <ul>
                    <li>The image is from a biological microscope</li>
                    <li>The image is in focus and well-lit</li>
                    <li>The image contains blood cells or bacteria</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # ── TOTAL COUNT ───────────────────────────────────
            total = result["total_detected"]
            st.success(f"✅ Detected **{total} organisms** in this image")

            # ── BLOOD CELL COUNTS ─────────────────────────────
            if result["summary"]["blood_cells"]:
                st.markdown(f"""
                <div class="count-card">
                    <h4>🩸 Blood Cells — {result['summary']['total_blood_cells']} detected</h4>
                </div>
                """, unsafe_allow_html=True)

                for cls, count in sorted(
                    result["summary"]["blood_cells"].items(),
                    key=lambda x: x[1], reverse=True
                ):
                    cols = st.columns([3, 7])
                    cols[0].markdown(f"**{cls}**")
                    cols[1].markdown(f"{'🔵' * min(count, 20)} **{count}**")

            # ── BACTERIA COUNTS ───────────────────────────────
            if result["summary"]["bacteria"]:
                st.markdown(f"""
                <div class="bacteria-card">
                    <h4>🦠 Bacteria — {result['summary']['total_bacteria']} detected</h4>
                </div>
                """, unsafe_allow_html=True)

                for cls, count in sorted(
                    result["summary"]["bacteria"].items(),
                    key=lambda x: x[1], reverse=True
                ):
                    cols = st.columns([3, 7])
                    cols[0].markdown(f"**{cls}**")
                    cols[1].markdown(f"{'🔴' * min(count, 20)} **{count}**")

            # ── AI DISCLAIMER ─────────────────────────────────
            st.markdown("""
            <div class="disclaimer">
            🤖 <strong>AI Limitation Notice:</strong> These results are generated 
            by an AI model and <strong>may contain errors</strong>. Detection accuracy 
            varies based on image quality, staining technique, and magnification level. 
            This output should <strong>not</strong> be used for clinical decision-making 
            without professional verification.
            </div>
            """, unsafe_allow_html=True)

    # ── ANNOTATED IMAGE ───────────────────────────────────────
    if result["is_valid_microscope_image"]:
        st.divider()
        st.markdown("### 🎯 Annotated Detection Map")
        st.caption("Each detected organism is highlighted with a coloured bounding box and label.")
        st.image(
            result["annotated_image"],
            caption="AI Detection — Bounding boxes show detected organisms",
            use_column_width=True
        )

        # ── FULL DETECTION TABLE ──────────────────────────────
        st.divider()
        st.markdown("### 📋 Full Detection List")

        if result["detections"]:
            st.caption(f"Showing all {len(result['detections'])} detections above {45}% confidence threshold")

            table_data = []
            for i, det in enumerate(result["detections"], 1):
                category = "🩸 Blood Cell" if det["class"] in BLOOD_CELLS else "🦠 Bacteria"
                table_data.append({
                    "#": i,
                    "Class": det["class"],
                    "Category": category,
                    "Confidence": f"{det['confidence']}%",
                })

            import pandas as pd
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── PDF REPORT ────────────────────────────────────────
        st.divider()
        st.markdown("### 📄 Download Clinical Report")

        if st.button("📥 Generate PDF Report"):
            with st.spinner("Building report..."):
                try:
                    pdf_bytes = _build_pdf_report(
                        uploaded_file.name, result, pil_image
                    )
                    st.download_button(
                        label="⬇️ Download Report",
                        data=pdf_bytes,
                        file_name=f"microscope_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                    st.success("✅ Report ready!")
                except Exception as e:
                    st.warning(f"PDF generation failed: {e}. Install reportlab: pip install reportlab")

# ── EMPTY STATE ───────────────────────────────────────────────
else:
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#888;
                    border: 2px dashed #ccc; border-radius:12px;'>
            <h2>📤 Upload a Microscope Image</h2>
            <p>Supports blood smear slides and bacterial culture images.<br>
            Formats: JPG, PNG, BMP, TIFF</p>
        </div>
        """, unsafe_allow_html=True)


# ── PDF REPORT BUILDER ────────────────────────────────────────
def _build_pdf_report(filename, result, pil_image):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, Image as RLImage
    )
    from reportlab.lib import colors
    import tempfile

    buffer = io.BytesIO()
    doc    = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story  = []

    # Title
    story.append(Paragraph("🔬 AI Microscope Phase 2 — Detection Report", styles["Title"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Developed by: Muhammad Hamza Malik", styles["Normal"]))
    story.append(Paragraph(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"File: {filename}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Disclaimer
    story.append(Paragraph(
        "⚠️ DISCLAIMER: This report is generated by an AI system for research "
        "and educational purposes only. AI can make mistakes. This report must "
        "NOT be used for clinical diagnosis without verification by a qualified "
        "medical professional.",
        styles["Normal"]
    ))
    story.append(Spacer(1, 12))

    # Summary
    story.append(Paragraph("Summary", styles["Heading2"]))
    story.append(Paragraph(f"Total Organisms Detected: {result['total_detected']}", styles["Normal"]))
    story.append(Paragraph(f"Blood Cells: {result['summary']['total_blood_cells']}", styles["Normal"]))
    story.append(Paragraph(f"Bacteria: {result['summary']['total_bacteria']}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Blood cell table
    if result["summary"]["blood_cells"]:
        story.append(Paragraph("Blood Cell Counts", styles["Heading3"]))
        bc_data = [["Cell Type", "Count"]]
        for cls, count in sorted(result["summary"]["blood_cells"].items(),
                                  key=lambda x: x[1], reverse=True):
            bc_data.append([cls, str(count)])
        t = Table(bc_data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3c6e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f0f7ff")])
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Bacteria table
    if result["summary"]["bacteria"]:
        story.append(Paragraph("Bacteria Counts", styles["Heading3"]))
        bact_data = [["Bacteria Type", "Count"]]
        for cls, count in sorted(result["summary"]["bacteria"].items(),
                                  key=lambda x: x[1], reverse=True):
            bact_data.append([cls, str(count)])
        t = Table(bact_data, colWidths=[200, 100])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#b71c1c")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#fff0f0")])
        ]))
        story.append(t)
        story.append(Spacer(1, 12))

    # Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        story.append(Paragraph("Uploaded Image", styles["Heading3"]))
        story.append(RLImage(tmp.name, width=200, height=200))

    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "This report was generated automatically by AI Microscope Phase 2. "
        "For research use only. Muhammad Hamza Malik — AI Microscope Project.",
        styles["Normal"]
    ))

    doc.build(story)
    return buffer.getvalue()


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🔬 AI Microscope Phase 2 | Developed by <b>Muhammad Hamza Malik</b> |
    Model: YOLOv8n | Accuracy: 89.5% mAP50 | 13 Classes |
    ⚠️ For Research & Educational Use Only — AI Can Make Mistakes
</div>
""", unsafe_allow_html=True)
