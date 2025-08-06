import gradio as gr
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# --- Configuration ---
MODEL_PATH = "models/waste_classifier_v2.tflite"
IMAGE_DIR = "images"
CLASS_NAMES = sorted([
    'aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
    'cardboard_boxes', 'cardboard_packaging', 'clothing', 'coffee_grounds',
    'disposable_cutlery', 'egg_shells', 'food_containers', 'food_waste',
    'fruit_peels', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper', 'paper_cups',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_shopping_bags',
    'plastic_soda_bottles', 'plastic_straws', 'plastic_trash_bags',
    'plastic_water_bottles', 'shoes', 'styrofoam', 'tea_bags', 'vegetable_scraps'
])

# --- TFLite Inference Function ---
def predict(input_image: Image.Image):
    """
    Takes a PIL Image, runs inference with the TFLite model,
    and returns the processed image and prediction results.
    """
    if input_image is None:
        return None, { "No Prediction": 1.0 }

    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensor details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preprocess the image to match the model's input requirements
    img_height = input_details[0]['shape'][1]
    img_width = input_details[0]['shape'][2]
    
    # Resize image using Pillow
    image_resized = input_image.resize((img_height, img_width))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)

    # Set the tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()

    # Get the prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction_probabilities = output_data[0]

    # Create a dictionary of class names and their probabilities
    confidences = {CLASS_NAMES[i]: float(prediction_probabilities[i]) for i in range(len(CLASS_NAMES))}

    return input_image, confidences


# --- Helper to get example images ---
def get_example_images():
    """Finds all images in the example directory."""
    example_paths = []
    if os.path.exists(IMAGE_DIR):
        for f in os.listdir(IMAGE_DIR):
            if f.endswith(('.jpg', '.jpeg', '.png')):
                example_paths.append(os.path.join(IMAGE_DIR, f))
    return sorted(example_paths)


# --- Gradio UI Layout ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple"), css="footer {display: none !important;}") as demo:
    gr.Markdown(
        """
        <div style="text-align: center;">
            <h1>🗑️ Garbage Detection CNN Model</h1>
        </div>
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
                ## Garbage Detection Demo
                Use our TFLite model to detect the type of waste. 
                
                You can either **upload an image**, use your **webcam**, or select one of the **example images** below.
                
                *The model sometimes faces difficulty detecting items, as it is a prototype.*
                """
            )
            # Input Components
            with gr.Tabs() as tabs:
                with gr.TabItem("⬆️ Upload an Image", id=0):
                    input_upload = gr.Image(type="pil", label="Upload Image")
                with gr.TabItem("📸 Take a Photo", id=1):
                    # CORRECTED: Use sources=["webcam"] instead of source="webcam"
                    input_webcam = gr.Image(type="pil", sources=["webcam"], label="Webcam Image")
            
            # Submit Button
            submit_button = gr.Button("Classify Waste", variant="primary")

        with gr.Column(scale=1):
            # Output Components
            output_label = gr.Label(label="Prediction", num_top_classes=3)
            output_image = gr.Image(label="Your Image")

    # Clickable Example Images Gallery (emulates the "app drawer")
    with gr.Accordion("Select an Example Image (Click to Expand)", open=True):
        gr.Examples(
            examples=get_example_images(),
            inputs=input_upload, # Clicking an example will populate the upload input
            outputs=[output_image, output_label],
            fn=predict,
            cache_examples=False, # Re-run prediction each time
            label="Click any image to test"
        )

    # --- CORRECTED & IMPROVED EVENT HANDLING ---
    def process_and_predict(selected_tab, upload_img, webcam_img):
        """
        Decides which image to use based on the active tab, then calls the predict function.
        """
        image_to_process = upload_img if selected_tab == 0 else webcam_img
        return predict(image_to_process)

    submit_button.click(
        fn=process_and_predict,
        inputs=[tabs, input_upload, input_webcam],
        outputs=[output_image, output_label]
    )


if __name__ == "__main__":
    demo.launch(debug=True)