import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
MODEL_PATH = "tl_feature_extraction_best.keras"
CLASS_NAMES = ["orange", "rottenoranges"]

st.set_page_config(
    page_title="Orange Freshness Classifier",
    page_icon="🍊",
    layout="centered",
)

st.markdown("""
    <style>
        .result-box {
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.2rem;
            font-weight: 600;
            text-align: center;
        }
        .fresh {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .rotten {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .prob-label {
            font-size: 0.85rem;
            color: #555;
            margin-bottom: 2px;
        }
        .footer-note {
            font-size: 0.75rem;
            color: #999;
            text-align: center;
            margin-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = tf.keras.utils.img_to_array(image)
    return np.expand_dims(arr, axis=0)


def predict(model, image: Image.Image):
    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]  # softmax output, sums to 1
    predicted_idx = int(np.argmax(probs))
    label = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])
    all_probs = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return label, confidence, all_probs


# --- UI ---

st.title("🍊 Orange Freshness Classifier")
st.caption("Upload a photo of an orange and the model will classify it as fresh or rotten.")
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
                label, confidence, all_probs = predict(model, image)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()

        st.subheader("Result")

        is_fresh = label == "orange"

        if is_fresh:
            st.markdown('<div class="result-box fresh">✅ Fresh Orange</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box rotten">❌ Rotten Orange</div>', unsafe_allow_html=True)

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}",
        )

        st.caption("Class probabilities")
        st.progress(all_probs["orange"], text=f"Fresh orange: {all_probs['orange']:.2%}")
        st.progress(all_probs["rottenoranges"], text=f"Rotten orange: {all_probs['rottenoranges']:.2%}")

        st.markdown(
            '<div class="footer-note">Model: MobileNetV2 Transfer Learning &nbsp;|&nbsp; '
            'Accuracy: 96.47% &nbsp;|&nbsp; F1: 96.38%</div>',
            unsafe_allow_html=True,
        )
