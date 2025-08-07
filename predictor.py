# predictor.py
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

# --- Configuration ---
MODEL_PATH = "models/waste_classifier_v5.tflite"

# IMPORTANT: This list MUST be in the exact same order as the classes the model was trained on.
# I am using the list you provided in your Streamlit code.
CLASS_NAMES = sorted([

'aerosol_cans', 'aluminum_food_cans',

'aluminum_soda_cans',

'cardboard_boxes', 'cardboard_packaging',

'clothing', 'coffee_grounds',

'disposable_plastic_cutlery', 'egg_shells',

'food_waste',

'glass_beverage_bottles',

'glass_cosmetic_containers',

'glass_food_jars', 'magazines', 'newspaper',

'office_paper', 'paper_cups',

'plastic_cup_lids',

'plastic_detergent_bottles','plastic_food_containers',

'plastic_shopping_bags',

'plastic_soda_bottles', 'plastic_straws',

'plastic_trash_bags',

'plastic_water_bottles', 'shoes','steel_food_cans', 'styrofoam_cups','styrofoam_food_containers', 'tea_bags' ])

# --- Model Loading and Caching ---
# Use Streamlit's caching to load the model only once, improving performance.
@st.cache_resource
def load_model():
    """Loads the TFLite model and allocates tensors."""
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Load the model
interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Prediction Function ---
def predict(image: Image.Image):
    """
    Takes a PIL Image, preprocesses it, and returns the predicted class and confidence.
    """
    # Get model input details
    img_height = input_details[0]['shape'][1]
    img_width = input_details[0]['shape'][2]

    # Preprocess the image
    image_resized = image.resize((img_width, img_height))
    image_array = np.array(image_resized)
    
    # Handle transparent images (PNG with 4 channels)
    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    # Perform prediction
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    # Get results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_probabilities = output_data[0]

    # Get top prediction
    top_idx = np.argmax(prediction_probabilities)
    predicted_label = CLASS_NAMES[top_idx]
    confidence = float(prediction_probabilities[top_idx])
    
    return predicted_label, confidence
