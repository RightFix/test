import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page Configuration
st.set_page_config(
    page_title="Mango Freshness & Disease Classifier",
    page_icon="🥭",
    layout="centered"
)

# Title & Description
st.title("🥭 Mango Quality Classifier")
st.markdown("""
Upload an image of a mango to classify whether it is **Fresh** or **Rotten**. 
This application uses a deep learning convolutional neural network trained on the Mango Disease Dataset.
""")

# Model Path Configuration
MODEL_PATH = "tl_feature_extraction_best.keras"

@st.cache_resource
def load_model():
    """Loads the pre-trained Keras model."""
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.warning(f"Model file '{MODEL_PATH}' not found. Please ensure the model is trained and saved.")
        return None

model = load_model()

# Sidebar Info
st.sidebar.header("About App")
st.sidebar.info("""
* **Model Target:** Mango Fruit Quality
* **Classes:** Fresh, Rotten
* **Input Size:** 224 x 224 pixels
""")

# File Uploader
uploaded_file = st.file_uploader("Choose a mango image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display Uploaded Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Mango Image", use_column_width=True)
    
    # Preprocessing
    st.write("🔍 Analyzing image...")
    
    # Resize and convert to array
    img_resized = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch dimension
    
    # Predict if model is loaded
    if model is not None:
        predictions = model.predict(img_array)
        
        # Binary classification or multi-class check
        if predictions.shape[-1] == 1:
            score = float(predictions[0][0])
            # Assuming 0: Fresh, 1: Rotten (or vice versa based on training)
            if score > 0.5:
                label = "Rotten"
                confidence = score * 100
            else:
                label = "Fresh"
                confidence = (1 - score) * 100
        else:
            class_names = ["Fresh", "Rotten"]
            predicted_class = np.argmax(predictions[0])
            label = class_names[predicted_class]
            confidence = float(predictions[0][predicted_class]) * 100
            
        # Display Results
        if label == "Fresh":
            st.success(f"**Result:** Fresh Mango 🟢")
        else:
            st.error(f"**Result:** Rotten / Diseased Mango 🔴")
            
        st.info(f"**Confidence Score:** {confidence:.2f}%")
    else:
        st.info("Demonstration Mode: Upload successful. Train and save `mango_disease_model.h5` to enable live predictions.")
