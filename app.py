import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "mobilenetv2_transfer.keras"
CLASS_NAMES = ["orange", "rottenoranges"]

st.set_page_config(
    page_title="Orange Freshness Classifier",
    page_icon="",
    layout="centered",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    
    arr = tf.keras.utils.img_to_array(image)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    st.write(f"Raw probabilities: {probs}")
    orange_score = float(probs[0])
    rotten_score = float(probs[1])
    label = CLASS_NAMES[int(np.argmax(probs))]
    return label, orange_score, rotten_score


# --- UI ---

st.title("Orange Freshness Classifier")
st.caption(
    "Upload an image of an orange. "
    "The model returns a freshness score between 0 and 1. "
    "Close to 0 means rotten, close to 1 means fresh."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            try:
                model = load_model()
                label, orange_score, rotten_score = predict(model, image)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

        st.subheader("Result")

        if label == "orange":
            st.success("Fresh Orange")
        else:
            st.error("Rotten Orange")

        st.metric(
            label="Freshness score  |  0 = rotten,  1 = fresh orange",
            value=f"{orange_score:.4f}",
        )

        st.caption("Class probabilities")
        st.progress(orange_score, text=f"Fresh orange: {orange_score:.2%}")
        st.progress(rotten_score, text=f"Rotten orange: {rotten_score:.2%}")

        st.divider()
        st.caption(
            "Model: MobileNetV2 Transfer Learning  |  "
            "Accuracy: 96.47%  |  F1: 96.38%"
        )
