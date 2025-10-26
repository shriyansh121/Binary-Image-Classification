import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from pathlib import Path

# --- 0. STREAMLIT CONFIGURATION (MUST BE FIRST COMMAND) ---
# This must be outside the main() function to run as the first Streamlit command.
st.set_page_config(
    page_title="House and Street Image Classifier (MLOps Demo)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. CONFIGURATION ---

# Define Paths (These must match the output of model_training.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINED_MODEL_PATH = PROJECT_ROOT / "run" / "models" / "trained_model.h5"
MODEL_INPUT_SIZE = (100, 100) # Must match the input_shape used in model_building.py

# Define the class names
CLASS_NAMES = {
    0: "HOUSE (Residential Area)",
    1: "STREET (Infrastructure/Roads)"
}

# --- 2. ROBUST MODEL LOADING (Streamlit Caching) ---

@st.cache_resource
def load_trained_model():
    """
    Loads the trained Keras model once and caches it for faster execution.
    This is critical for Streamlit performance.
    """
    if not TRAINED_MODEL_PATH.exists():
        st.error("MODEL NOT FOUND!")
        st.error(f"Please run 'dvc repro' to train and save the model to: {TRAINED_MODEL_PATH}")
        return None
        
    try:
        model = load_model(TRAINED_MODEL_PATH)
        st.success("Trained model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 3. PREDICTION SERVICE ---

def preprocess_image(uploaded_file):
    """
    Loads, resizes, and normalizes the uploaded image for model input.
    """
    try:
        # Load image using PIL
        img = Image.open(uploaded_file).convert('RGB')
        
        # Resize to the required input dimensions (100x100)
        img_resized = img.resize(MODEL_INPUT_SIZE)
        
        # Convert to numpy array
        img_array = image.img_to_array(img_resized)
        
        # Expand dimensions to create a batch size of 1 (1, 100, 100, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize the image data (0-255 -> 0-1), matching training preprocessing
        return img_array / 255.0
    except Exception as e:
        st.error(f"Error during image preprocessing: {e}")
        return None

def predict_image_class(model, processed_image):
    """
    Generates a prediction from the model.
    """
    prediction = model.predict(processed_image)
    
    # Since the model uses sigmoid activation (binary classification):
    # The output is a single probability value.
    probability = prediction[0][0]
    
    # Threshold the probability (0.5) to get the final class label (0 or 1)
    predicted_class_index = (probability > 0.5).astype(int)
    
    # Get the human-readable result
    predicted_label = CLASS_NAMES[predicted_class_index]
    confidence = probability if predicted_class_index == 1 else (1 - probability)
    
    return predicted_label, confidence, predicted_class_index


# --- 4. STREAMLIT UI ---

def main():
    
    st.title("🛰️ Binary Image Classification")
    st.markdown("---")
    
    # Load the model once
    model = load_trained_model()
    if model is None:
        st.warning("Application halted. Please check the console for model loading errors.")
        return

    st.sidebar.header("Upload Image")
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Create columns for layout: Image on the left, results on the right
        col1, col2 = st.columns([1, 1.5], gap="large")

        # Display Image (Column 1)
        with col1:
            st.subheader("Uploaded Image")
            img_display = Image.open(uploaded_file)
            st.image(img_display, caption='Input Image', use_column_width=True)

        # Process and Predict (Column 2)
        with col2:
            st.subheader("Classification Result")
            with st.spinner('Analyzing image and predicting class...'):
                
                # 1. Preprocess the image
                processed_img = preprocess_image(uploaded_file)
                
                if processed_img is not None:
                    # 2. Get the prediction
                    label, confidence, index = predict_image_class(model, processed_img)
                    
                    # 3. Format the result display
                    confidence_percent = f"{confidence * 100:.2f}%"
                    
                    if index == 1:
                        st.success(f"**Prediction: {label}**")
                        st.balloons()
                    else:
                        st.info(f"**Prediction: {label}**")
                        
                    st.metric(label="Confidence", value=confidence_percent, delta_color="off")

                    st.markdown("""
                        <style>
                            .st-success {color: #17A2B8;}
                            .st-info {color: #5BC0DE;}
                        </style>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.caption(f"Raw Score (P(Street)): {confidence:.6f}")
                else:
                    st.error("Prediction failed due to image processing error.")
    else:
        st.info("Upload an image to get a House/Street classification prediction!")
        
        st.markdown(
            """
            ---
            ### MLOps Pipeline Overview
            This application uses a model trained via a robust, version-controlled pipeline:
            1. **Ingestion:** Data is pulled from **AWS S3** via DVC.
            2. **Preprocessing:** Data is augmented (rotated/flipped) and split.
            3. **Training:** A simple **CNN model** is trained and the resulting `trained_model.h5` artifact is used here.
            """
        )

if __name__ == '__main__':
    main()
