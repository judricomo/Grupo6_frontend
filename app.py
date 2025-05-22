import streamlit as st
import tempfile
import os
import subprocess
from PIL import Image

# Discover all .pth files in repo root
MODEL_FILES = [f for f in os.listdir('.') if f.endswith('.pth')]
if not MODEL_FILES:
    st.error("No model .pth files found in repo root.")

st.set_page_config(page_title="HerdNet Animal Detector", layout="wide")
st.title("🐾 HerdNet Animal Detector")

# Let user pick which model weights to use
model_choice = st.selectbox(
    "Choose HerdNet model file", MODEL_FILES,
    help="Select the .pth file to use for inference"
)

st.markdown(
    """
    Upload a wildlife image and click **Detect Animals** to run inference.
    The detection script will run in the background and display the annotated image.
    """
)

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    # Show side-by-side placeholders before inference
    col1, col2 = st.columns(2)
    col1.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect Animals"):
        st.info("Running inference, please wait...")
        with st.spinner("Detecting animals..."):
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Prepare folders
                input_folder = os.path.join(tmp_dir, "input")
                os.makedirs(input_folder, exist_ok=True)
                img_path = os.path.join(input_folder, uploaded_file.name)
                image.save(img_path)

                # Run inference as a module
                result = subprocess.run(
                    [
                        "python", "-m", "tools.infer",
                        input_folder,
                        model_choice,
                        "-device", "cpu"
                    ],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    st.error("Inference failed:")
                    st.code(result.stderr)
                else:
                    # Find output
                    out_root = None
                    for entry in os.listdir(input_folder):
                        if entry.endswith("_HerdNet_results"):
                            out_root = os.path.join(input_folder, entry)
                            break
                    if not out_root:
                        st.error("Could not find results folder.")
                    else:
                        plot_dir = os.path.join(out_root, "plots")
                        out_img_path = os.path.join(plot_dir, uploaded_file.name)
                        if os.path.exists(out_img_path):
                            annotated = Image.open(out_img_path)
                            # Show results side by side
                            col1, col2 = st.columns(2)
                            col1.image(image, caption="Original", use_container_width=True)
                            col2.image(annotated, caption="Detected Animals", use_container_width=True)
                        else:
                            st.error("Annotated image not found in results.")
        st.success("Done!")
