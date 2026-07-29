import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Potato Leaf Classifier",
    layout="centered",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background-color: #0F1F0F;
    color: #F0EDE6;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page header ── */
.page-header {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #2A3F2A;
    margin-bottom: 2rem;
}
.page-header h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.5rem;
    letter-spacing: 0.04em;
    color: #F0EDE6;
    margin: 0 0 0.35rem;
    text-transform: uppercase;
}
.page-header p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #6B8F6B;
    letter-spacing: 0.08em;
    margin: 0;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #6B8F6B;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: block;
}

[data-testid="stFileUploader"] {
    background: #162016;
    border: 1px dashed #2D4A2D;
    border-radius: 4px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4A6741;
}

/* ── Result card ── */
.result-card {
    background: #162016;
    border: 1px solid #2A3F2A;
    border-radius: 4px;
    padding: 1.75rem 2rem;
    margin-top: 1.5rem;
}

.result-verdict {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6B8F6B;
    margin-bottom: 0.4rem;
}

.result-label {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 1.9rem;
    letter-spacing: -0.01em;
    margin: 0 0 1.5rem;
}
.result-label.blight { color: #D4A017; }
.result-label.healthy { color: #5C8A4A; }

/* ── Confidence meter ── */
.meter-row {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}
.meter-item {}
.meter-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.3rem;
}
.meter-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #A0B8A0;
}
.meter-pct {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    font-weight: 600;
}
.meter-pct.blight { color: #D4A017; }
.meter-pct.healthy { color: #5C8A4A; }

.meter-track {
    height: 5px;
    background: #1E301E;
    border-radius: 2px;
    overflow: hidden;
}
.meter-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.6s ease;
}
.meter-fill.blight { background: #D4A017; }
.meter-fill.healthy { background: #5C8A4A; }

/* ── Raw output ── */
.raw-output {
    border-top: 1px solid #2A3F2A;
    padding-top: 1rem;
}
.raw-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4A6741;
    margin-bottom: 0.3rem;
}
.raw-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: #8AAF8A;
}

/* ── Uploaded image caption ── */
[data-testid="stImage"] img {
    border-radius: 4px;
    border: 1px solid #2A3F2A;
}
[data-testid="stImage"] > div > div {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.65rem !important;
    color: #4A6741 !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #5C8A4A !important;
}

/* ── Info / error boxes ── */
.stAlert {
    background: #162016 !important;
    border: 1px solid #2A3F2A !important;
    border-radius: 4px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #A0B8A0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "tl_feature_extraction_best.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Early_Blight", "Healthy"]   # index 0 = Early_Blight, index 1 = Healthy

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)       # model head uses MobileNetV3 preprocess_input internally
    return np.expand_dims(arr, axis=0)

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Potato Leaf Classifier</h1>
    <p>MobileNetV3Small · Transfer Learning · Binary Classification</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
st.markdown('<span class="upload-label">Upload leaf image</span>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    label="upload",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded)
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, caption="input image", use_container_width=True)

    with col_result:
        with st.spinner("Running inference..."):
            try:
                model  = load_model()
                tensor = preprocess(image)
                raw    = model.predict(tensor, verbose=0)[0]   # shape (2,) — softmax output

                blight_prob  = float(raw[0])   # index 0 = Early_Blight
                healthy_prob = float(raw[1])   # index 1 = Healthy

                predicted_idx   = int(np.argmax(raw))
                predicted_class = CLASS_NAMES[predicted_idx]
                is_blight       = predicted_class == "Early_Blight"

                verdict_css  = "blight" if is_blight else "healthy"
                display_name = "Early Blight" if is_blight else "Healthy"

                st.markdown(f"""
<div class="result-card">
    <div class="result-verdict">Prediction</div>
    <div class="result-label {verdict_css}">{display_name}</div>

    <div class="meter-row">
        <div class="meter-item">
            <div class="meter-header">
                <span class="meter-name">Early Blight</span>
                <span class="meter-pct blight">{blight_prob * 100:.1f}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill blight" style="width:{blight_prob * 100:.1f}%"></div>
            </div>
        </div>
        <div class="meter-item">
            <div class="meter-header">
                <span class="meter-name">Healthy</span>
                <span class="meter-pct healthy">{healthy_prob * 100:.1f}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill healthy" style="width:{healthy_prob * 100:.1f}%"></div>
            </div>
        </div>
    </div>

    <div class="raw-output">
        <div class="raw-label">Raw softmax output</div>
        <div class="raw-value">[{raw[0]:.6f}, {raw[1]:.6f}]</div>
    </div>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Inference failed: {e}")

else:
    st.info("Upload a potato leaf image above to run the classifier.")
