import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "tl_feature_extraction_best.keras"
CLASS_NAMES = ["orange", "rottenoranges"]

st.set_page_config(
    page_title="Orange Freshness Classifier",
    layout="centered",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    tensor = preprocess(image)
    probs = model.predict(tensor, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    label = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])
    return label, float(probs[0]), float(probs[1])


# --- UI ---

st.title("Orange Freshness Classifier")
st.caption("Upload an image of an orange. The model will classify it as **Fresh** or **Rotten**.")

st.divider()

uploaded_file = st.file_uploader(
    "Upload an orange image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", width="stretch")

    with col2:
        with st.spinner("Running inference..."):
            try:
                model = load_model()
                label, fresh_prob, rotten_prob = predict(model, image)
            except Exception as e:
                st.error(f"Inference failed: {e}")
                st.stop()

        if label == "orange":
            st.success("### Fresh Orange")
        else:
            st.error("### Rotten Orange")

        st.metric("Confidence", f"{max(fresh_prob, rotten_prob) * 100:.1f}%")

        st.write("**Class probabilities**")
        st.progress(fresh_prob,  text=f"Fresh orange  — {fresh_prob * 100:.1f}%")
        st.progress(rotten_prob, text=f"Rotten orange — {rotten_prob * 100:.1f}%")

        st.divider()
        st.caption(f"Raw output: `[{fresh_prob:.4f}, {rotten_prob:.4f}]`")

else:
    st.info("Upload an orange image above to get a prediction.")

with st.sidebar:
    st.header("About")
    st.write(
        "Classifies orange images into:\n\n"
        "- **Fresh Orange**\n"
        "- **Rotten Orange**\n\n"
        "**Model:** MobileNetV2 Transfer Learning  \n"
        "**Accuracy:** 96.47%  |  **F1:** 96.38%  \n"
        "**Input size:** 224 x 224 px  \n"
        "**Framework:** TensorFlow / Keras"
    )
    st.divider()
    st.code("streamlit run app.py", language="bash")
