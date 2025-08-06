# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import os

# # --- Configuration ---
# MODEL_PATH = "models/waste_classifier_v2.tflite"
# CLASS_NAMES = sorted([
#     'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans', 
#     'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds', 
#     'disposable_plastic_cutlery', 'egg_shells', 'food_waste', 
#     'glass_beverage_bottles', 'glass_cosmetic_containers', 
#     'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups', 
#     'plastic_cup_lids', 'plastic_detergent_bottles','plastic_food_containers', 'plastic_shopping_bags', 
#     'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags', 
#     'plastic_water_bottles', 'shoes','steel_food_cans', 'styrofoam_cups','styrofoam_food_containers', 'tea_bags'
# ])

# # --- Prediction function ---
# def predict(input_image):
#     if input_image is None:
#         return None, None
    
#     # Load TFLite model
#     interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
#     interpreter.allocate_tensors()

#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()

#     img_height = input_details[0]['shape'][1]
#     img_width = input_details[0]['shape'][2]

#     # Preprocess
#     image_resized = input_image.resize((img_width, img_height))
#     image_array = np.array(image_resized)
#     image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

#     # Predict
#     interpreter.set_tensor(input_details[0]['index'], image_array)
#     interpreter.invoke()

#     output_data = interpreter.get_tensor(output_details[0]['index'])
#     prediction_probabilities = output_data[0]

#     top_idx = np.argmax(prediction_probabilities)
#     confidence = float(prediction_probabilities[top_idx])
#     return CLASS_NAMES[top_idx], confidence

# # --- Streamlit UI ---
# st.set_page_config(page_title="Garbage Detector", page_icon="🗑", layout="wide")

# st.markdown("<h1 style='text-align:center;'>🗑 Garbage Detection CNN Model</h1>", unsafe_allow_html=True)

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     if st.button("Classify Waste"):
#         label, confidence = predict(image)
#         if label:
#             st.success(f"*Prediction:* {label} ({confidence*100:.2f}%)")
#         else:
#             st.error("No prediction available.")


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