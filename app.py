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

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; }
.footer-caption { font-size: 0.75rem; color: #888; text-align: center; margin-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


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
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Running inference..."):
            try:
                model = load_model()
                tensor = preprocess(image)
                raw = model.predict(tensor, verbose=0)   # shape (1, 1) or (1, 2)

                if raw.shape[-1] == 1:
                    # Sigmoid output
                    score         = float(raw[0][0])
                    fresh_prob    = 1.0 - score
                    rotten_prob   = score
                else:
                    # Softmax output — use raw probabilities directly, do NOT re-apply softmax
                    probs       = raw[0]
                    fresh_prob  = float(probs[CLASS_NAMES.index("orange")])
                    rotten_prob = float(probs[CLASS_NAMES.index("rottenoranges")])

                predicted_class = CLASS_NAMES[int(np.argmax(raw[0]))]
                confidence = fresh_prob if predicted_class == "orange" else rotten_prob

                # Result
                if predicted_class == "orange":
                    st.success("### Fresh Orange")
                else:
                    st.error("### Rotten Orange")

                st.metric("Confidence", f"{confidence * 100:.1f}%")

                st.write("**Class probabilities**")
                st.progress(fresh_prob,  text=f"Fresh orange  — {fresh_prob * 100:.1f}%")
                st.progress(rotten_prob, text=f"Rotten orange — {rotten_prob * 100:.1f}%")

                st.divider()
                st.caption(
                    f"Raw model output: `{raw[0].tolist()}`"
                )

            except Exception as e:
                st.error(f"Inference failed: {e}")

else:
    st.info("Upload an orange image above to get a prediction.")

with st.sidebar:
    st.header("About")
    st.write(
        "This app classifies orange images into two categories:\n\n"
        "- **Fresh Orange** — orange is in good condition\n"
        "- **Rotten Orange** — orange shows signs of decay\n\n"
        "**Model:** `tl_feature_extraction_best.keras`  \n"
        "**Input size:** 224 x 224 px  \n"
        "**Framework:** TensorFlow / Keras  \n"
        "**Accuracy:** 96.47%  |  **F1:** 96.38%"
    )
    st.divider()
    st.write("Place `tl_feature_extraction_best.keras` in the same directory as `app.py`, then run:")
    st.code("streamlit run app.py", language="bash")
