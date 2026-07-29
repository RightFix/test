import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Potato Leaf Classifier",
    layout="centered",
)

# ── CSS — only for result card and overrides, no structural HTML ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #0F1F0F; color: #F0EDE6; }

#MainMenu, footer, header { visibility: hidden; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #162016;
    border: 1px dashed #2D4A2D;
    border-radius: 4px;
}
[data-testid="stFileUploader"]:hover { border-color: #4A6741; }

/* Native Streamlit text overrides */
h1, h2, h3 { color: #F0EDE6 !important; }
p, label, .stCaption { color: #A0B8A0 !important; }

/* Image border */
[data-testid="stImage"] img {
    border-radius: 4px;
    border: 1px solid #2A3F2A;
}

/* Info / error */
[data-testid="stNotification"] {
    background: #162016 !important;
    border: 1px solid #2A3F2A !important;
    color: #A0B8A0 !important;
}

/* Result card */
.result-card {
    background: #162016;
    border: 1px solid #2A3F2A;
    border-radius: 4px;
    padding: 1.75rem 2rem;
    margin-top: 0.5rem;
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
.result-label.blight  { color: #D4A017; }
.result-label.healthy { color: #5C8A4A; }

.meter-row {
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
    margin-bottom: 1.5rem;
}
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
.meter-pct.blight  { color: #D4A017; }
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
}
.meter-fill.blight  { background: #D4A017; }
.meter-fill.healthy { background: #5C8A4A; }

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
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "tl_feature_extraction_best.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Early_Blight", "Healthy"]

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ── Page header — native Streamlit, no raw HTML ───────────────────────────────
st.title("Potato Leaf Classifier")
st.caption("MobileNetV3Small · Transfer Learning · Binary Classification")
st.divider()

# ── Input — upload or camera ──────────────────────────────────────────────────
tab_upload, tab_camera = st.tabs(["Upload Image", "Take Photo"])

image = None

with tab_upload:
    uploaded = st.file_uploader(
        label="Upload a potato leaf image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded:
        image = Image.open(uploaded)

with tab_camera:
    captured = st.camera_input("Point camera at a potato leaf")
    if captured:
        image = Image.open(captured)

if image:

    # Run inference before rendering any layout
    result = None
    error  = None
    with st.spinner("Running inference..."):
        try:
            model  = load_model()
            tensor = preprocess(image)
            raw    = model.predict(tensor, verbose=0)[0]   # shape (2,)

            blight_prob  = float(raw[0])
            healthy_prob = float(raw[1])
            is_blight    = int(np.argmax(raw)) == 0

            result = dict(
                raw=raw,
                blight_prob=blight_prob,
                healthy_prob=healthy_prob,
                is_blight=is_blight,
                verdict_css="blight" if is_blight else "healthy",
                display_name="Early Blight" if is_blight else "Healthy",
            )
        except Exception as e:
            error = str(e)

    # Render layout after inference completes
    col_img, col_result = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, use_column_width=True)
        st.caption("Input image")

    with col_result:
        if error:
            st.error(f"Inference failed: {error}")
        elif result:
            r   = result
            bp  = r["blight_prob"] * 100
            hp  = r["healthy_prob"] * 100
            rv0 = r["raw"][0]
            rv1 = r["raw"][1]
            st.markdown(f"""
<div class="result-card">
    <div class="result-verdict">Prediction</div>
    <div class="result-label {r['verdict_css']}">{r['display_name']}</div>
    <div class="meter-row">
        <div class="meter-item">
            <div class="meter-header">
                <span class="meter-name">Early Blight</span>
                <span class="meter-pct blight">{bp:.1f}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill blight" style="width:{bp:.1f}%"></div>
            </div>
        </div>
        <div class="meter-item">
            <div class="meter-header">
                <span class="meter-name">Healthy</span>
                <span class="meter-pct healthy">{hp:.1f}%</span>
            </div>
            <div class="meter-track">
                <div class="meter-fill healthy" style="width:{hp:.1f}%"></div>
            </div>
        </div>
    </div>
    <div class="raw-output">
        <div class="raw-label">Raw softmax output</div>
        <div class="raw-value">[{rv0:.6f}, {rv1:.6f}]</div>
    </div>
</div>
""", unsafe_allow_html=True)

else:
    st.info("Upload a potato leaf image above to run the classifier.")
