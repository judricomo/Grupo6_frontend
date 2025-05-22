import streamlit as st
import tempfile
import os
import subprocess
from PIL import Image

# Path to your pretrained HerdNet model (include this .pth in your repo)
MODEL_PATH = "herdnet_model_exp_4.pth"

st.set_page_config(page_title="HerdNet Animal Detector", layout="wide")
st.title("🐾 HerdNet Animal Detector")

st.markdown(
    """
    Upload a wildlife image and click **Detect Animals** to run inference.
    The detection script will run in the background and display the annotated image.
    """
)

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"], accept_multiple_files=False
)

if uploaded_file:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Detect Animals"):
        st.info("Running inference, please wait...")
        with st.spinner("Detecting animals..."):
            # Create a temporary folder to hold the input image
            with tempfile.TemporaryDirectory() as tmp_dir:
                input_folder = os.path.join(tmp_dir, "input")
                os.makedirs(input_folder, exist_ok=True)
                img_path = os.path.join(input_folder, uploaded_file.name)
                image.save(img_path)

                # Run the existing infer.py script as a module to ensure imports resolve
                result = subprocess.run(
                    [
                        "python", "-m", "tools.infer",
                        input_folder,
                        MODEL_PATH,
                        "-device", "cpu"
                    ],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    st.error("Inference failed:")
                    st.code(result.stderr)
                else:
                    # Locate the output plots folder
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
                            st.image(annotated, caption="Detected Animals", use_container_width=True)
                        else:
                            st.error("Annotated image not found in results.")
        st.success("Done!")
