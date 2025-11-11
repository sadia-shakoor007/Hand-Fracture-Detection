# ------------------------- #
# ✋ Hand Fracture Detection App
# Developed by Sadia Shakoor
# ------------------------- #

import os
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time

# ------------------------- #
# ⚙️ Page Configuration
# ------------------------- #
st.set_page_config(page_title="✋ Hand Fracture Detection", page_icon="🦴", layout="wide")

# 🌈 Custom Styling
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #e3f2fd 0%, #ede7f6 100%);
        color: #2c3e50;
        font-family: 'Inter', sans-serif;
    }
    h1 {
        font-size: 42px;
        text-align: center;
        font-weight: 800;
        color: #1a237e;
        margin-bottom: 5px;
    }
    h3 {
        text-align: center;
        font-weight: 400;
        color: #3949ab;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 3px dashed #64b5f6;
        border-radius: 12px;
        background-color: #ffffff;
        padding: 35px;
        text-align: center;
        transition: 0.3s ease;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
    }
    .upload-box:hover {
        border-color: #1976d2;
        background-color: #e3f2fd;
    }
    .stButton>button {
        background: linear-gradient(90deg, #42a5f5, #1e88e5);
        color: white;
        border: none;
        border-radius: 10px;
        font-size: 17px;
        padding: 10px 30px;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: linear-gradient(90deg, #2196f3, #1976d2);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #bbdefb 0%, #e3f2fd 100%);
        color: #1a237e;
    }
    .stProgress > div > div > div > div {
        background-color: #42a5f5 !important;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ------------------------- #
# 🧠 Sidebar Info
# ------------------------- #
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4305/4305656.png", width=80)
st.sidebar.title("🦴 About the App")
st.sidebar.info("""
This app uses an **AI-powered YOLOv8 model** to detect **hand fractures**  
from X-ray images.  

Upload an image and let the model locate and highlight fracture regions instantly.
""")

# ------------------------- #
# 📦 Load Model
# ------------------------- #
@st.cache_resource
def load_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    try:
        model = YOLO(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    return model

model = load_model()

# ------------------------- #
# 🖐️ App Title
# ------------------------- #
st.markdown("<h1>✋ Hand Fracture Detection</h1>", unsafe_allow_html=True)
st.markdown("<h3>Upload a hand X-ray and let AI detect fracture areas.</h3>", unsafe_allow_html=True)

# ------------------------- #
# 📤 Upload Section
# ------------------------- #
st.markdown("""
<div class="upload-box">
📤 <b>Upload a Hand X-ray Image (.jpg / .png)</b>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# ------------------------- #
# 🔍 Detection Process
# ------------------------- #
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="🩻 Uploaded Hand X-ray", use_container_width=True)

    with col2:
        st.write("### 🧠 Analyzing image...")
        with st.spinner("Detecting fractures... please wait"):
            time.sleep(1)
            results = model(image)
            res_img = results[0].plot()
            result_pil = Image.fromarray(res_img[..., ::-1])
            boxes = results[0].boxes
            time.sleep(0.5)

        st.write("### 📊 Detection Results:")
        if len(boxes) > 0:
            avg_conf = float(np.mean([b.conf for b in boxes]))
            st.progress(int(avg_conf * 100))
            st.image(result_pil, caption="✅ Fracture Detection Result", use_container_width=True)
            for i, box in enumerate(boxes):
                cls_name = model.names[int(box.cls)]
                conf = float(box.conf)
                st.write(f"**{i+1}. {cls_name} — Confidence:** {conf*100:.2f}%")
            st.error("⚠️ Possible fractures detected. Please consult an orthopedic specialist.")
        else:
            st.success("✅ No fractures detected in this image.")
            st.markdown("💪 Everything looks good! No visible fracture found.")

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
