# predictor.py
import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
MODEL_PATH = "models/waste_classifier_v5.tflite"

# Class names in alphabetical order (as per your original code)
CLASS_NAMES = sorted([
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
    'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds',
    'disposable_plastic_cutlery', 'egg_shells', 'food_waste',
    'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars',
    'magazines', 'newspaper', 'office_paper', 'paper_cups',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
    'plastic_trash_bags', 'plastic_water_bottles', 'shoes',
    'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
])

# Recycling categories mapping
RECYCLING_INFO = {
    'aluminum_food_cans': {'category': 'Metal Recycling', 'color': '#4CAF50', 'tip': 'Clean and remove labels before recycling'},
    'aluminum_soda_cans': {'category': 'Metal Recycling', 'color': '#4CAF50', 'tip': 'Rinse clean and crush to save space'},
    'cardboard_boxes': {'category': 'Paper Recycling', 'color': '#2196F3', 'tip': 'Remove tape and flatten boxes'},
    'cardboard_packaging': {'category': 'Paper Recycling', 'color': '#2196F3', 'tip': 'Clean and dry before recycling'},
    'glass_beverage_bottles': {'category': 'Glass Recycling', 'color': '#9C27B0', 'tip': 'Remove caps and rinse clean'},
    'glass_food_jars': {'category': 'Glass Recycling', 'color': '#9C27B0', 'tip': 'Remove labels and lids, rinse clean'},
    'magazines': {'category': 'Paper Recycling', 'color': '#2196F3', 'tip': 'Remove plastic covers if any'},
    'newspaper': {'category': 'Paper Recycling', 'color': '#2196F3', 'tip': 'Keep dry and bundle together'},
    'office_paper': {'category': 'Paper Recycling', 'color': '#2196F3', 'tip': 'Remove staples and paper clips'},
    'plastic_soda_bottles': {'category': 'Plastic Recycling', 'color': '#FF9800', 'tip': 'Remove caps, rinse and compress'},
    'plastic_water_bottles': {'category': 'Plastic Recycling', 'color': '#FF9800', 'tip': 'Remove labels if possible, rinse clean'},
    'steel_food_cans': {'category': 'Metal Recycling', 'color': '#4CAF50', 'tip': 'Remove labels, rinse clean'},
}

# --- Model Loading and Caching ---
@st.cache_resource
def load_model():
    """
    Loads the TFLite model and allocates tensors.
    Uses Streamlit's caching for performance optimization.
    """
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            st.info("Please ensure the model file is in the correct location.")
            return None
            
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        logger.info(f"✅ Model loaded successfully from {MODEL_PATH}")
        return interpreter
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"Error loading model: {str(e)}")
        return None

# Load the model
interpreter = load_model()

if interpreter is not None:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Log model details
    logger.info(f"Model input shape: {input_details[0]['shape']}")
    logger.info(f"Model output shape: {output_details[0]['shape']}")
    logger.info(f"Number of classes: {len(CLASS_NAMES)}")

def preprocess_image(image: Image.Image, target_size: tuple = None):
    """
    Preprocess image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (width, height) for resizing
    
    Returns:
        Preprocessed image array
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get target size from model input details if not provided
        if target_size is None and interpreter is not None:
            img_height = input_details[0]['shape'][1]
            img_width = input_details[0]['shape'][2]
            target_size = (img_width, img_height)
        elif target_size is None:
            target_size = (224, 224)  # Default size
        
        # Resize image
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to array and normalize
        image_array = np.array(image_resized, dtype=np.float32)
        
        # Normalize pixel values to [0, 1] if they're in [0, 255]
        if image_array.max() > 1.0:
            image_array = image_array / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        logger.info(f"Image preprocessed to shape: {image_array.shape}")
        return image_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise e

def get_recycling_info(class_name: str):
    """Get recycling information for a given class."""
    return RECYCLING_INFO.get(class_name, {
        'category': 'General Waste', 
        'color': '#757575', 
        'tip': 'Check local recycling guidelines'
    })

def predict(image: Image.Image):
    """
    Takes a PIL Image, preprocesses it, and returns the predicted class and confidence.
    
    Args:
        image: PIL Image object
        
    Returns:
        tuple: (predicted_label, confidence_score)
    """
    if interpreter is None:
        st.error("❌ Model not loaded. Cannot make predictions.")
        return "Error", 0.0
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_probabilities = output_data[0]
        
        # Get top prediction
        top_idx = np.argmax(prediction_probabilities)
        predicted_label = CLASS_NAMES[top_idx]
        confidence = float(prediction_probabilities[top_idx])
        
        logger.info(f"Prediction: {predicted_label} with confidence: {confidence:.4f}")
        
        return predicted_label, confidence
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        logger.error(error_msg)
        st.error(f"❌ {error_msg}")
        return "Error", 0.0

def get_top_predictions(image: Image.Image, top_k: int = 3):
    """
    Get top K predictions for an image.
    
    Args:
        image: PIL Image object
        top_k: Number of top predictions to return
        
    Returns:
        list: List of tuples (label, confidence) sorted by confidence
    """
    if interpreter is None:
        return []
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Set input tensor and run inference
        interpreter.set_tensor(input_details[0]['index'], processed_image)
        interpreter.invoke()
        
        # Get output
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prediction_probabilities = output_data[0]
        
        # Get top K predictions
        top_indices = np.argpartition(prediction_probabilities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(prediction_probabilities[top_indices])[::-1]]
        
        results = []
        for idx in top_indices:
            label = CLASS_NAMES[idx]
            confidence = float(prediction_probabilities[idx])
            results.append((label, confidence))
        
        return results
        
    except Exception as e:
        logger.error(f"Error getting top predictions: {str(e)}")
        return []

def validate_image(image: Image.Image):
    """
    Validate if the image is suitable for prediction.
    
    Args:
        image: PIL Image object
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Check image size
        width, height = image.size
        if width < 32 or height < 32:
            return False, "Image is too small. Please use an image at least 32x32 pixels."
        
        # Check if image is too large (for memory efficiency)
        if width > 2048 or height > 2048:
            return False, "Image is too large. Please use an image smaller than 2048x2048 pixels."
        
        # Check image format
        if image.mode not in ['RGB', 'RGBA', 'L']:
            return False, "Unsupported image format. Please use RGB, RGBA, or grayscale images."
        
        return True, "Image is valid"
        
    except Exception as e:
        return False, f"Error validating image: {str(e)}"

# Model information for debugging
if interpreter is not None:
    MODEL_INFO = {
        'input_shape': input_details[0]['shape'],
        'output_shape': output_details[0]['shape'],
        'input_dtype': input_details[0]['dtype'],
        'output_dtype': output_details[0]['dtype'],
        'num_classes': len(CLASS_NAMES)
    }
else:
    MODEL_INFO = None
