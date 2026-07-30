import numpy as np
import streamlit as st
import tensorflow as tf

IMG_SIZE = (224, 224)
MODEL_PATH = "mobilenetv2_transfer.keras"
CLASS_NAMES = ["orange", "rottenoranges"]

st.set_page_config(
    page_title="Orange Freshness Classifier",
    page_icon="🍊",
    layout="centered",
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(uploaded_file):
    img = tf.keras.utils.load_img(uploaded_file, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    return np.expand_dims(img_array, axis=0)


def predict(model, preprocessed_img):
    prob = float(model.predict(preprocessed_img, verbose=0)[0][0])
    
    # Matching your notebook classification logic
    class_idx = 0 if prob <= 0.5 else 1
    label = CLASS_NAMES[class_idx]

    rotten_score = prob
    orange_score = 1.0 - prob

    return label, orange_score, rotten_score


# --- UI Layout ---

st.title("Orange Freshness Classifier")
st.caption(
    "Upload an image of an orange. "
    "The model returns a freshness score between 0 and 1. "
    "Close to 0 means fresh, close to 1 means rotten."
)

st.divider()

uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(uploaded_file, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            try:
                model = load_model()
                img_array = preprocess(uploaded_file)
                label, orange_score, rotten_score = predict(model, img_array)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

        st.subheader("Result")

        if label == "orange":
            st.success("Fresh Orange")
        else:
            st.error("Rotten Orange")

        st.metric(
            label="Freshness score",
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
