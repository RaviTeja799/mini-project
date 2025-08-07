# app.py
import streamlit as st
from PIL import Image
import os
import random
import base64
from predictor import predict
import glob

# --- Page Configuration ---
st.set_page_config(
    page_title="Garbage Detection Demo",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom CSS for Modern UI ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .demo-description {
        text-align: center;
        padding: 20px;
        background: rgba(103, 126, 234, 0.1);
        border-radius: 10px;
        margin-bottom: 30px;
        border-left: 4px solid #667eea;
    }
    
    .image-gallery {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        justify-content: center;
        padding: 20px;
        background: white;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    .gallery-item {
        flex: 0 0 calc(33.333% - 10px);
        max-width: 150px;
        text-align: center;
        cursor: pointer;
        transition: transform 0.3s ease;
        background: white;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .gallery-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .gallery-item img {
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 8px;
        margin-bottom: 8px;
    }
    
    .gallery-item-name {
        font-size: 11px;
        color: #666;
        font-weight: 500;
        text-transform: capitalize;
    }
    
    .action-buttons {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin: 30px 0;
    }
    
    .action-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border: none;
        border-radius: 25px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .result-container {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        margin-top: 30px;
    }
    
    .confidence-bar {
        background: #f0f0f0;
        border-radius: 10px;
        height: 20px;
        margin: 10px 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: linear-gradient(90deg, #4CAF50, #8BC34A);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .drawer-toggle {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        z-index: 1000;
    }
    
    .random-selection {
        text-align: center;
        margin: 20px 0;
    }
    
    .random-button {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 12px 25px;
        border: none;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .random-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
    }
    
    .upload-section {
        background: white;
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 30px 0;
    }
    
    .section-title {
        font-size: 20px;
        font-weight: 600;
        color: #333;
        margin-bottom: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'selected_image' not in st.session_state:
    st.session_state.selected_image = None
if 'show_gallery' not in st.session_state:
    st.session_state.show_gallery = True
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# --- Helper Functions ---
@st.cache_data
def load_class_images():
    """Load all class images from the images folder"""
    class_names = [
        'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
        'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds',
        'disposable_plastic_cutlery', 'egg_shells', 'food_waste',
        'glass_beverage_bottles', 'glass_cosmetic_containers', 'glass_food_jars',
        'magazines', 'newspaper', 'office_paper', 'paper_cups',
        'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
        'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
        'plastic_trash_bags', 'plastic_water_bottles', 'shoes',
        'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers', 'tea_bags'
    ]
    
    available_images = []
    image_folder = "images/"
    
    if os.path.exists(image_folder):
        for class_name in class_names:
            # Look for images with class name
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                pattern = os.path.join(image_folder, f"{class_name}.{ext.split('.')[-1]}")
                files = glob.glob(pattern)
                if files:
                    available_images.append({
                        'name': class_name,
                        'path': files[0],
                        'display_name': class_name.replace('_', ' ').title()
                    })
                    break
            else:
                # If no specific image found, look for any image with similar name
                for ext in ['jpg', 'jpeg', 'png']:
                    pattern = os.path.join(image_folder, f"*{class_name}*.{ext}")
                    files = glob.glob(pattern)
                    if files:
                        available_images.append({
                            'name': class_name,
                            'path': files[0],
                            'display_name': class_name.replace('_', ' ').title()
                        })
                        break
    
    return available_images

def get_image_as_base64(image_path):
    """Convert image to base64 for HTML display"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except:
        return None

# --- Main UI ---

# Header
st.markdown("""
<div class="main-header">
    <h1>🗑️ Garbage Detection CNN Model</h1>
    <p>Advanced Waste Classification Using Deep Learning</p>
</div>
""", unsafe_allow_html=True)

# Demo Description
st.markdown("""
<div class="demo-description">
    <h3>🎯 Garbage Detection Demo</h3>
    <p>We use our TFLite model to detect and classify waste into 30 different categories. 
    Select an image from the gallery below, upload your own image, or take a photo to get started!</p>
</div>
""", unsafe_allow_html=True)

# Load available images
available_images = load_class_images()

# Image Gallery Section
if st.session_state.show_gallery and available_images:
    st.markdown('<div class="section-title">📸 Select from Sample Images</div>', unsafe_allow_html=True)
    
    # Random selection button
    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        if st.button("🎲 Random Selection", key="random_btn", help="Select a random image"):
            st.session_state.selected_image = random.choice(available_images)
            st.session_state.prediction_made = False
            st.rerun()
    
    # Create image gallery
    cols = st.columns(3)
    for idx, img_data in enumerate(available_images):
        col_idx = idx % 3
        with cols[col_idx]:
            try:
                img = Image.open(img_data['path'])
                if st.button(f"Select", key=f"select_{idx}", help=f"Select {img_data['display_name']}"):
                    st.session_state.selected_image = img_data
                    st.session_state.prediction_made = False
                    st.rerun()
                
                st.image(img, caption=img_data['display_name'], use_column_width=True)
            except Exception as e:
                st.error(f"Error loading {img_data['name']}: {str(e)}")

# Selected Image Processing
if st.session_state.selected_image and not st.session_state.prediction_made:
    st.markdown("---")
    st.markdown('<div class="section-title">🔍 Processing Selected Image</div>', unsafe_allow_html=True)
    
    try:
        selected_img = Image.open(st.session_state.selected_image['path'])
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(selected_img, caption=f"Selected: {st.session_state.selected_image['display_name']}", use_column_width=True)
        
        with col2:
            with st.spinner("🤖 Analyzing image..."):
                label, confidence = predict(selected_img)
                
                st.markdown(f"""
                <div class="result-container">
                    <h3 style="color: #667eea;">🎯 Prediction Results</h3>
                    <p><strong>Predicted Class:</strong> <span style="color: #4CAF50; font-size: 18px;">{label.replace('_', ' ').title()}</span></p>
                    <p><strong>Confidence:</strong> {confidence*100:.2f}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                st.session_state.prediction_made = True
    except Exception as e:
        st.error(f"Error processing selected image: {str(e)}")

# Upload and Camera Section
st.markdown("---")
st.markdown('<div class="section-title">📤 Upload Your Own Image or Take a Photo</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="upload")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📷 Camera Capture")
    camera_file = st.camera_input("Take a picture", key="camera")
    st.markdown('</div>', unsafe_allow_html=True)

# Process uploaded or captured image
image_to_process = uploaded_file or camera_file

if image_to_process is not None:
    st.markdown("---")
    st.markdown('<div class="section-title">🔍 Analysis Results</div>', unsafe_allow_html=True)
    
    try:
        image = Image.open(image_to_process)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Your Image", use_column_width=True)
        
        with col2:
            with st.spinner("🤖 Analyzing your image..."):
                label, confidence = predict(image)
                
                st.markdown(f"""
                <div class="result-container">
                    <h3 style="color: #667eea;">🎯 Detection Results</h3>
                    <p><strong>Detected Item:</strong> <span style="color: #4CAF50; font-size: 18px;">{label.replace('_', ' ').title()}</span></p>
                    <p><strong>Confidence Score:</strong> {confidence*100:.2f}%</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence*100}%"></div>
                    </div>
                    <p style="margin-top: 15px; color: #666; font-size: 14px;">
                        💡 <strong>Tip:</strong> Higher confidence scores indicate more certain predictions.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Recycling information
                if confidence > 0.7:
                    st.success("✅ High confidence prediction! The model is quite certain about this classification.")
                elif confidence > 0.5:
                    st.warning("⚠️ Moderate confidence. You might want to try a clearer image.")
                else:
                    st.info("ℹ️ Low confidence. Consider taking a clearer photo or trying a different angle.")
                    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

# Toggle Gallery Button
if st.button("🎨 Toggle Image Gallery", key="toggle_gallery"):
    st.session_state.show_gallery = not st.session_state.show_gallery
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <p>🌱 <strong>Garbage Detection CNN Model</strong> | Helping classify waste for better recycling</p>
    <p>Powered by TensorFlow Lite | Built with ❤️ using Streamlit</p>
</div>
""", unsafe_allow_html=True)
