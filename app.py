import streamlit as st
import tempfile
import os
import subprocess
import sys
import pandas as pd
from PIL import Image

# Config
st.set_page_config(page_title="HerdNet Animal Detector Demo", layout="centered")

# App main title
st.title("🦒 🐃 🐘 HerdNet Animal Detector Demo")

# Discover model weights
MODEL_FILES = [f for f in os.listdir('.') if f.endswith('.pth')]
if not MODEL_FILES:
    st.error("No model weights (.pth) found in the repository root.")
    st.stop()
model_choice = st.selectbox("Choose HerdNet model file:", MODEL_FILES)

# Tabs setup
tabs = st.tabs(["Single Image", "Batch Inference", "About"])

# Helper: run inference and capture output

def run_inference(input_folder, weights):
    cmd = [sys.executable, "-m", "tools.infer", input_folder, weights, "-device", "cpu"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

# Single Image Tab
with tabs[0]:
    st.header("Single Image Detection")
    uploaded = st.file_uploader("Upload an image:", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded image", use_container_width=True)
        if st.button("Detect Animals", key="single_detect"):
            with st.spinner("Running inference..."):
                with tempfile.TemporaryDirectory() as tmp:
                    inp = os.path.join(tmp, "input")
                    os.makedirs(inp, exist_ok=True)
                    img.save(os.path.join(inp, uploaded.name))

                    result = run_inference(inp, model_choice)

                    # If infer printed no detections message
                    if "No detections found" in result.stdout:
                        st.warning("⚠️ No detections found. Try using a higher-resolution or higher-definition image for better results.")
                        st.stop()

                    if result.returncode != 0:
                        st.error("Inference failed:")
                        st.code(result.stderr)
                        st.stop()

                    # Read CSV
                    res_dir = next((d for d in os.listdir(inp) if d.endswith("_HerdNet_results")), None)
                    root = os.path.join(inp, res_dir)
                    csv_file = next((f for f in os.listdir(root) if f.endswith("_detections.csv")), None)
                    df = pd.read_csv(os.path.join(root, csv_file))
                    if df.empty:
                        st.warning("⚠️ No detections found. Try using a higher-resolution or higher-definition image for better results.")
                        st.stop()

                    # Display overlay
                    overlay_path = os.path.join(root, "plots", uploaded.name)
                    if os.path.exists(overlay_path):
                        overlay = Image.open(overlay_path)
                        cols = st.columns(2)
                        cols[0].image(img, caption="Original", use_container_width=True)
                        cols[1].image(overlay, caption="Detected Animals", use_container_width=True)

                    # Show CSV
                    st.markdown("### Detections CSV")
                    st.dataframe(df)

                    # Show thumbnails
                    thumb_dir = os.path.join(root, "thumbnails")
                    if os.path.exists(thumb_dir):
                        thumbs = sorted(os.listdir(thumb_dir))
                        if thumbs:
                            st.markdown("### Thumbnails")
                            for i in range(0, len(thumbs), 4):
                                cols = st.columns(4)
                                for j, t in enumerate(thumbs[i:i+4]):
                                    cols[j].image(
                                        Image.open(os.path.join(thumb_dir, t)),
                                        use_container_width=True
                                    )
            st.success("Detection complete!")

# Batch Inference Tab
with tabs[1]:
    st.header("Batch Inference")
    files = st.file_uploader("Upload images:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files and st.button("Detect Batch", key="batch_detect"):
        with st.spinner("Running batch inference..."):
            with tempfile.TemporaryDirectory() as tmp:
                inp = os.path.join(tmp, "input")
                os.makedirs(inp, exist_ok=True)
                for f in files:
                    Image.open(f).save(os.path.join(inp, f.name))

                result = run_inference(inp, model_choice)
                if "No detections found" in result.stdout:
                    st.warning("⚠️ No detections found in any images. Try using higher-resolution images for better results.")
                    st.stop()

                if result.returncode != 0:
                    st.error("Inference failed:")
                    st.code(result.stderr)
                    st.stop()

                res_dir = next((d for d in os.listdir(inp) if d.endswith("_HerdNet_results")), None)
                root = os.path.join(inp, res_dir)
                csv_file = next((f for f in os.listdir(root) if f.endswith("_detections.csv")), None)
                df = pd.read_csv(os.path.join(root, csv_file))
                if df.empty:
                    st.warning("⚠️ No detections found in batch. Try higher-resolution images for better results.")
                    st.stop()

                st.markdown("### Batch Detections CSV")
                st.dataframe(df)
                st.markdown("---")

                for fname in df['images'].unique():
                    st.write(f"#### {fname}")
                    orig = Image.open(os.path.join(inp, fname))
                    overlay_path = os.path.join(root, "plots", fname)
                    if os.path.exists(overlay_path):
                        overlay = Image.open(overlay_path)
                        cols = st.columns(2)
                        cols[0].image(orig, caption="Original", use_container_width=True)
                        cols[1].image(overlay, caption="Detected Animals", use_container_width=True)

        st.success("Batch detection complete!")

# About Tab
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
