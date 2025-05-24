import streamlit as st
import tempfile
import os
import subprocess
import sys
import pandas as pd
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="HerdNet Animal Detector Demo",
    layout="centered"
)

# App main title with icons
st.title("🦒 🐃 🐘 HerdNet Animal Detector Demo 🦒 🐃 🐘")

# Model selection
MODEL_FILES = [f for f in os.listdir('.') if f.endswith('.pth')]
if not MODEL_FILES:
    st.error("No HerdNet model files (.pth) found. Please add your model weights.")
    st.stop()
model_choice = st.selectbox("Choose HerdNet model file:", MODEL_FILES)

# Navigation tabs
tabs = st.tabs(["Single Image", "Batch Inference", "About"])

# Single Image Inference
with tabs[0]:
    st.subheader("🐗 Single Image Detection 🐗")
    uploaded = st.file_uploader("Upload an image:", type=["jpg","jpeg","png"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
        if st.button("Detect Animals", key="single_detect"):
            st.toast("🔍 Running single image inference...", icon="🔍")
            with st.spinner("Processing..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    inp = os.path.join(tmpdir, "input")
                    os.makedirs(inp, exist_ok=True)
                    img.save(os.path.join(inp, uploaded.name))
                    cmd = [sys.executable, "-m", "tools.infer", inp, model_choice, "-device", "cpu"]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode:
                        st.error(res.stderr)
                        st.stop()
                    # load and display results
                    rd = next(d for d in os.listdir(inp) if d.endswith("_HerdNet_results"))
                    root = os.path.join(inp, rd)
                    orig, ann = img, Image.open(os.path.join(root, "plots", uploaded.name))
                    c1, c2 = st.columns(2)
                    c1.image(orig, caption="Original", use_container_width=True)
                    c2.image(ann, caption="Detected Animals", use_container_width=True)
                    df = pd.read_csv(os.path.join(root, next(f for f in os.listdir(root) if f.endswith('_detections.csv'))))
                    st.markdown("### Detection Results")
                    st.dataframe(df)
                    thumbs = sorted(os.listdir(os.path.join(root, "thumbnails")))
                    st.markdown("### Thumbnails")
                    for i in range(0, len(thumbs), 4):
                        cols = st.columns(4)
                        for j, t in enumerate(thumbs[i:i+4]):
                            cols[j].image(
                                Image.open(os.path.join(root, "thumbnails", t)),
                                use_container_width=True
                            )
            st.success("✅ Single image inference complete!")
            st.toast("✅ Done!", icon="✅")

# Batch Inference
with tabs[1]:
    st.subheader("📂 Batch Inference 📂")
    files = st.file_uploader("Upload multiple images:", type=["jpg","jpeg","png"], accept_multiple_files=True)
    if files and st.button("Detect Batch", key="batch_detect"):
        st.toast("🔍 Running batch inference...", icon="🔍")
        with st.spinner("Processing batch..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                inp = os.path.join(tmpdir, "input")
                os.makedirs(inp, exist_ok=True)
                for f in files:
                    Image.open(f).save(os.path.join(inp, f.name))
                cmd = [sys.executable, "-m", "tools.infer", inp, model_choice, "-device", "cpu"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode:
                    st.error(res.stderr)
                    st.stop()
                rd = next(d for d in os.listdir(inp) if d.endswith("_HerdNet_results"))
                root = os.path.join(inp, rd)
                df = pd.read_csv(os.path.join(root, next(f for f in os.listdir(root) if f.endswith("_detections.csv"))))
                st.markdown("### Batch Detection Results")
                st.dataframe(df)
                st.markdown("---")
                for fname in df['images'].unique():
                    st.write(f"#### {fname}")
                    orig = Image.open(os.path.join(inp, fname))
                    ann = Image.open(os.path.join(root, "plots", fname))
                    c1, c2 = st.columns(2)
                    c1.image(orig, use_container_width=True)
                    c2.image(ann, use_container_width=True)
                    thumbs = [t for t in os.listdir(os.path.join(root, "thumbnails")) if t.startswith(fname[:-4])]
                    if thumbs:
                        colset = st.columns(len(thumbs))
                        for idx, t in enumerate(thumbs):
                            colset[idx].image(
                                Image.open(os.path.join(root, "thumbnails", t)),
                                use_container_width=True
                            )
        st.success("✅ Batch inference complete!")
        st.toast("✅ Done!", icon="✅")

# About Tab
with tabs[2]:
    st.subheader("ℹ️ About")
    st.markdown(
        """
        **👨‍💻 Authors**  
        - Alejandro Aristizábal  
        - Alexander Hernández  
        - Juan David Rico  
        - Juan Felipe Jiménez

        **🔗 Useful Links**  
        - 🧠 [HerdNet original repository](https://github.com/Alexandre-Delplanque/HerdNet)  
        - 📊 [UAV Dataset (University of Liège)](https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0)  
        - 📄 [Project Report](https://github.com/judricomo/Grupo5_maia_proyeto/tree/main/Artículo%20Proy.%20Grado)  
        - 🔧 [Albumentations Library](https://albumentations.ai)
        """,
        unsafe_allow_html=True
    )
