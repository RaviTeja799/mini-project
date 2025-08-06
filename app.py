# app.py
import streamlit as st
from PIL import Image
from predictor import predict # Import the prediction function

# --- Page Configuration ---
st.set_page_config(
    page_title="Garbage Detector",
    page_icon="🗑️",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- UI Components ---
st.title("🗑️ Garbage Detection CNN Model")
st.write(
    "Upload an image or use your camera to classify a piece of waste. "
    "The model will predict which of the 30 categories it belongs to."
)

# Create two columns for different input methods
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("📸 Use Your Camera")
    camera_file = st.camera_input("Take a picture")

# --- Prediction Logic ---
# Determine which input has been provided
image_to_process = None
if uploaded_file is not None:
    image_to_process = uploaded_file
elif camera_file is not None:
    image_to_process = camera_file

# If an image is available, process and predict
if image_to_process is not None:
    # Display the image
    image = Image.open(image_to_process)
    st.image(image, caption="Your Image", use_column_width=True)
    
    # Make a prediction
    st.write("Classifying...")
    label, confidence = predict(image)
    
    # Display the result
    st.success(f"**Prediction:** `{label}`")
    st.info(f"**Confidence:** `{confidence*100:.2f}%`")

    # Optional: Display a confidence bar
    st.progress(confidence)
else:
    st.info("Please upload an image or take a picture to get a prediction.")