import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "tl_feature_extraction_best.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Clean", "Dusty"]   # sigmoid: prob >= 0.5 → Dusty

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Panel Diagnostic",
    page_icon="☀️",
    layout="centered",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0F1117;
    color: #E8EAF0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse at 50% 0%, #1a1f2e 0%, #0F1117 60%);
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hide default elements ── */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    border-bottom: 1px solid #1e2333;
    margin-bottom: 2.5rem;
}
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #F5A623;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #FFFFFF;
    margin: 0 0 0.75rem;
    line-height: 1.15;
}
.hero p {
    font-size: 0.95rem;
    color: #6B7280;
    font-weight: 300;
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #13161f;
    border: 1px dashed #2a2f42;
    border-radius: 10px;
    padding: 0.5rem;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #F5A623;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}
.upload-hint {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: #3a3f52;
    text-align: center;
    letter-spacing: 0.08em;
    margin-top: 0.5rem;
}

/* ── Preview image ── */
[data-testid="stImage"] img {
    border-radius: 8px;
    border: 1px solid #1e2333;
}

/* ── Scanning animation ── */
.scan-wrapper {
    position: relative;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #2a2f42;
}
.scan-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, transparent 0%, #F5A623 50%, transparent 100%);
    animation: scanDown 1.4s ease-in-out infinite;
    box-shadow: 0 0 12px #F5A623aa;
}
@keyframes scanDown {
    0%   { top: 0%; }
    100% { top: 100%; }
}

/* ── Result cards ── */
.result-card {
    border-radius: 10px;
    padding: 2rem 2rem 1.75rem;
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
}
.result-clean {
    background: #0d1f14;
    border: 1px solid #1a3d25;
}
.result-clean::before { background: linear-gradient(90deg, #22c55e, #16a34a); }

.result-dusty {
    background: #1f160a;
    border: 1px solid #3d2a10;
}
.result-dusty::before { background: linear-gradient(90deg, #F5A623, #d97706); }

.result-status {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-clean .result-status { color: #22c55e; }
.result-dusty .result-status { color: #F5A623; }

.result-label {
    font-size: 2rem;
    font-weight: 600;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    line-height: 1;
    margin-bottom: 1.25rem;
}

.metrics-row {
    display: flex;
    gap: 1.5rem;
    flex-wrap: wrap;
}
.metric-block {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: #FFFFFF;
}
.metric-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4B5563;
}

.confidence-bar-track {
    height: 4px;
    background: #1e2333;
    border-radius: 2px;
    margin-top: 1.25rem;
    overflow: hidden;
}
.confidence-bar-fill-clean {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #22c55e, #16a34a);
}
.confidence-bar-fill-dusty {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, #F5A623, #d97706);
}

/* ── Divider & footer ── */
.divider {
    border: none;
    border-top: 1px solid #1e2333;
    margin: 3rem 0 1.5rem;
}
.footer {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #2e3448;
    text-align: center;
    padding-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">☀ Solar Panel Diagnostic</div>
    <h1>Panel Surface<br>Analysis</h1>
    <p>Upload a panel image to detect dust accumulation and assess surface condition.</p>
</div>
""", unsafe_allow_html=True)

# ── Upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload panel image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)
st.markdown(
    '<p class="upload-hint">JPG · PNG · BMP · WEBP</p>',
    unsafe_allow_html=True,
)

image_path = None
if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.getbuffer())
        image_path = tmp.name

# ── Inference ──────────────────────────────────────────────────────────────────
if image_path:
    st.image(image_path, container="stretch", caption="")

    with st.spinner("Running diagnostic…"):
        model = tf.keras.models.load_model(MODEL_PATH)
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prob = model.predict(img_array, verbose=0)[0][0]   # scalar sigmoid output

    # sigmoid: prob = P(Dusty); threshold at 0.5
    label = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if label == "Dusty" else 1 - prob
    bar_pct = int(confidence * 100)

    if label == "Clean":
        st.markdown(f"""
        <div class="result-card result-clean">
            <div class="result-status">● Diagnosis complete</div>
            <div class="result-label">Clean Panel</div>
            <div class="metrics-row">
                <div class="metric-block">
                    <span class="metric-value">{confidence:.4f}</span>
                    <span class="metric-label">Confidence</span>
                </div>
                <div class="metric-block">
                    <span class="metric-value">{bar_pct}%</span>
                    <span class="metric-label">Clean probability</span>
                </div>
            </div>
            <div class="confidence-bar-track">
                <div class="confidence-bar-fill-clean" style="width:{bar_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card result-dusty">
            <div class="result-status">⚠ Dust accumulation detected</div>
            <div class="result-label">Dusty Panel</div>
            <div class="metrics-row">
                <div class="metric-block">
                    <span class="metric-value">{confidence:.4f}</span>
                    <span class="metric-label">Confidence</span>
                </div>
                <div class="metric-block">
                    <span class="metric-value">{bar_pct}%</span>
                    <span class="metric-label">Dusty probability</span>
                </div>
            </div>
            <div class="confidence-bar-track">
                <div class="confidence-bar-fill-dusty" style="width:{bar_pct}%"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown(
    '<div class="footer">Binary CNN · Solar Panel Dataset · Sigmoid Output</div>',
    unsafe_allow_html=True,
)
