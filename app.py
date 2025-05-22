import streamlit as st
import tempfile
import os
import subprocess
import sys
from PIL import Image

# Discover .pth model files in the repo root
MODEL_FILES = [f for f in os.listdir('.') if f.endswith('.pth')]

# Centered layout configuration
st.set_page_config(page_title="HerdNet Animal Detector", layout="centered")
st.title("🐾 HerdNet Animal Detector")

# Ensure at least one model file is available
if not MODEL_FILES:
    st.error("No .pth model files found in the repository root.")
    st.stop()

# Dropdown to select which model to use
model_choice = st.selectbox(
    "Choose HerdNet model file:",
    MODEL_FILES
)

st.markdown(
    "Upload a wildlife image and click **Detect Animals** to run inference."
)

# File uploader widget
uploaded_file = st.file_uploader(
    "Upload an image:",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    original_image = Image.open(uploaded_file)
    st.image(original_image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect Animals"):
        st.info("Running inference, please wait...")
        with st.spinner("Detecting animals..."):
            # Create a temp directory for input/output
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_folder = os.path.join(tmp_dir, "input")
                os.makedirs(input_folder, exist_ok=True)

                # Save uploaded image to input folder
                img_path = os.path.join(input_folder, uploaded_file.name)
                original_image.save(img_path)

                # Invoke the inference script using the same Python interpreter
                cmd = [
                    sys.executable, "-m", "tools.infer",
                    input_folder,
                    model_choice,
                    "-device", "cpu"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)

                # Handle script failure
                if result.returncode != 0:
                    st.error("Inference script failed:")
                    st.code(result.stderr)
                    st.stop()

                # Locate the results directory within the input folder
                results_dirs = [d for d in os.listdir(input_folder) if d.endswith("_HerdNet_results")]
                if not results_dirs:
                    st.error("Inference results folder not found.")
                    st.stop()
                out_root = os.path.join(input_folder, results_dirs[0])

                # Prepare path to annotated output image
                plot_dir = os.path.join(out_root, "plots")
                out_img_path = os.path.join(plot_dir, uploaded_file.name)
                if not os.path.exists(out_img_path):
                    st.error("Annotated image not found in results.")
                    st.stop()

                # Display original and annotated images side by side
                annotated_image = Image.open(out_img_path)
                col1, col2 = st.columns(2)
                col1.image(original_image, caption="Original", use_container_width=True)
                col2.image(annotated_image, caption="Detected Animals", use_container_width=True)

        st.success("Detection complete!")
