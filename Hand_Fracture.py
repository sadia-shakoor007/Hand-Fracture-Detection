# ------------------------- #
# ✋ Hand Fracture Detection App
# Developed by Sadia Shakoor
# ------------------------- #

import os
import warnings
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ------------------------- #
# ⚠️ Suppress Deprecation Warnings
# ------------------------- #
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------- #
# ⚙️ Page Configuration
# ------------------------- #
st.set_page_config(page_title="🏥 Hand Fracture Detection", layout="wide")

# ------------------------- #
# 🌈 Custom Styling
# ------------------------- #
st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #e3f2fd 0%, #ede7f6 100%); font-family: 'Inter', sans-serif;}
    h1 {font-size:42px; text-align:center; font-weight:800; color:#1a237e; margin-bottom:5px;}
    h3 {text-align:center; font-weight:400; color:#3949ab; margin-bottom:30px;}
    .stButton>button {background: linear-gradient(90deg, #42a5f5, #1e88e5); color:white; border-radius:10px; font-size:17px; padding:10px 30px;}
    .stButton>button:hover {transform: scale(1.05);}
    footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🧠 Sidebar Info
# ------------------------- #
st.sidebar.title("🤖 About the App")
st.sidebar.info("""
This app uses an **AI-powered YOLOv8 model** to detect **hand fractures** from X-ray images.  
Upload an image and let the model locate and highlight fracture regions instantly.
""")

# ------------------------- #
# 📦 Load Model
# ------------------------- #
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    return model

model = load_model()

# ------------------------- #
# 🖐️ App Title with Hospital Icon
# ------------------------- #
st.markdown("<h1>🏥 Hand Fracture Detection</h1>", unsafe_allow_html=True)

# ------------------------- #
# Heading with X-ray + Hospital Icons
# ------------------------- #
st.markdown("<h3>🩻🩺 Upload a hand X-ray and let AI detect fracture areas.</h3>", unsafe_allow_html=True)

# ------------------------- #
# 📤 Upload Section
# ------------------------- #
uploaded_file = st.file_uploader("Upload Hand X-ray Image (.jpg / .png)", type=["jpg", "jpeg", "png"])

# ------------------------- #
# 🔍 Detection Process
# ------------------------- #
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Run model
    with st.spinner("Detecting fractures..."):
        time.sleep(1)
        results = model(image)
        res_img = results[0].plot()
        result_pil = Image.fromarray(res_img[..., ::-1])
        boxes = results[0].boxes
        time.sleep(0.5)

    # Show both images side by side at same level
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Hand X-ray", use_container_width=True)
    with col2:
        st.image(result_pil, caption="Fracture Detection Result", use_container_width=True)

    # Detection info below images
    if len(boxes) > 0:
        avg_conf = float(np.mean([b.conf for b in boxes]))
        st.progress(int(avg_conf * 100))
        for i, box in enumerate(boxes):
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            st.write(f"{i+1}. {cls_name} — Confidence: {conf*100:.2f}%")
        st.error("⚠️ Possible fractures detected. Please consult an orthopedic specialist.")
    else:
        st.success("✅ No fractures detected in this image. Everything looks good!")

# ------------------------- #
# 🩻 Footer
# ------------------------- #
st.markdown("---")
st.markdown("""
<center>
Developed by <b>Sadia Shakoor</b><br>
Powered by <b>YOLOv8</b> & <b>Streamlit</b>
</center>
""", unsafe_allow_html=True)
