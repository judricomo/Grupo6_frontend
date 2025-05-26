import streamlit as st
import tempfile
import os
import subprocess
import sys
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(page_title="HerdNet Animal Detector Demo", layout="centered")

# Title
st.title("🦒 🐃 🐘 HerdNet Animal Detector Demo")

# Find model weights
MODEL_FILES = [f for f in os.listdir('.') if f.endswith('.pth')]
if not MODEL_FILES:
    st.error("No model weights (.pth) found in the repository root.")
    st.stop()
model_choice = st.selectbox("Choose HerdNet model file:", MODEL_FILES)

# Helper to run inference

def run_inference(input_folder: str, weights: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "tools.infer", input_folder, weights, "-device", "cpu"]
    return subprocess.run(cmd, capture_output=True, text=True)

# Tabs
tabs = st.tabs(["Single Image", "Batch Inference", "About"])

# 1. Single Image
with tabs[0]:
    st.header("Single Image Detection")
    uploaded = st.file_uploader("Upload an image:", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        # Downsample large images for faster inference
        MAX_DIM = 2000
        if max(img.width, img.height) > MAX_DIM:
            img = img.copy()
            img.thumbnail((MAX_DIM, MAX_DIM), resample=Image.LANCZOS)
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Detect Animals", key="single_detect"):
            with tempfile.TemporaryDirectory() as tmp:
                inp = os.path.join(tmp, "input")
                os.makedirs(inp, exist_ok=True)
                img.save(os.path.join(inp, uploaded.name))
                with st.spinner("Running inference..."):
                    result = run_inference(inp, model_choice)
                if "No detections found" in result.stdout:
                    st.warning("⚠️ No detections found. Try a higher-resolution image.")
                    st.stop()
                if result.returncode != 0:
                    st.error("Inference failed:")
                    st.code(result.stderr)
                    st.stop()
                res_dir = next(d for d in os.listdir(inp) if d.endswith("_HerdNet_results"))
                root = os.path.join(inp, res_dir)
                csv_file = next(f for f in os.listdir(root) if f.endswith("_detections.csv"))
                df = pd.read_csv(os.path.join(root, csv_file))
                if df.empty:
                    st.warning("⚠️ No detections found. Try a higher-resolution image.")
                    st.stop()
                overlay_path = os.path.join(root, "plots", uploaded.name)
                if os.path.exists(overlay_path):
                    overlay = Image.open(overlay_path)
                    c1, c2 = st.columns(2)
                    c1.image(img, caption="Original", use_container_width=True)
                    c2.image(overlay, caption="Detected", use_container_width=True)
                st.markdown("### Detections CSV")
                st.dataframe(df)
                thumb_dir = os.path.join(root, "thumbnails")
                if os.path.exists(thumb_dir):
                    thumbs = sorted(os.listdir(thumb_dir))
                    if thumbs:
                        st.markdown("### Thumbnails")
                        for i in range(0, len(thumbs), 4):
                            cols = st.columns(4)
                            for j, t in enumerate(thumbs[i:i+4]):
                                cols[j].image(
                                    Image.open(os.path.join(thumb_dir, t)), use_container_width=True
                                )
            st.success("Detection complete!")

# 2. Batch Inference
with tabs[1]:
    st.header("Batch Inference")
    files = st.file_uploader("Upload images:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files and st.button("Detect Batch", key="batch_detect"):
        with tempfile.TemporaryDirectory() as tmp:
            inp = os.path.join(tmp, "input")
            os.makedirs(inp, exist_ok=True)
            for f in files:
                img_f = Image.open(f)
                MAX_DIM = 2000
                if max(img_f.width, img_f.height) > MAX_DIM:
                    img_f = img_f.copy()
                    img_f.thumbnail((MAX_DIM, MAX_DIM), resample=Image.LANCZOS)
                img_f.save(os.path.join(inp, f.name))
            with st.spinner("Running batch inference..."):
                result = run_inference(inp, model_choice)
            if "No detections found" in result.stdout:
                st.warning("⚠️ No detections found in any image. Try higher-resolution images.")
                st.stop()
            if result.returncode != 0:
                st.error("Inference failed:")
                st.code(result.stderr)
                st.stop()
            res_dir = next(d for d in os.listdir(inp) if d.endswith("_HerdNet_results"))
            root = os.path.join(inp, res_dir)
            csv_file = next(f for f in os.listdir(root) if f.endswith("_detections.csv"))
            df = pd.read_csv(os.path.join(root, csv_file))
            if df.empty:
                st.warning("⚠️ No detections found in batch. Try higher-resolution images.")
                st.stop()
            detected = set(df['images'])
            missing = {f.name for f in files} - detected
            if missing:
                st.info(f"Note: {len(missing)} image(s) had no detections and are omitted.")
            st.markdown("### Batch Detections CSV")
            st.dataframe(df)
            st.markdown("---")
            for fname in df['images'].unique():
                st.write(f"#### {fname}")
                orig = Image.open(os.path.join(inp, fname))
                overlay_path = os.path.join(root, "plots", fname)
                if os.path.exists(overlay_path):
                    overlay = Image.open(overlay_path)
                    c1, c2 = st.columns(2)
                    c1.image(orig, caption="Original", use_container_width=True)
                    c2.image(overlay, caption="Detected", use_container_width=True)
                thumb_dir = os.path.join(root, "thumbnails")
                thumbs = [t for t in os.listdir(thumb_dir) if t.startswith(fname[:-4])]
                if thumbs:
                    cols = st.columns(len(thumbs))
                    for i, t in enumerate(thumbs):
                        cols[i].image(
                            Image.open(os.path.join(thumb_dir, t)), use_container_width=True
                        )
        st.success("Batch detection complete!")

# 3. About
with tabs[2]:
    st.header("About")
    st.markdown(
        """
        **Authors**  
        - Alejandro Aristizábal  
        - Alexander Hernández  
        - Juan David Rico  
        - Juan Felipe Jiménez

        **Useful Links**  
        - [HerdNet repo](https://github.com/Alexandre-Delplanque/HerdNet)  
        - [UAV Dataset](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)
        """
    )
