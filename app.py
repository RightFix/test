import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH  = "mobilenetv3_transfer.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["Healthy", "Vitiligo"]   # must match training class order

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DermaScan · Vitiligo Detector",
    page_icon="🔬",
    layout="centered",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f14;
    color: #e2e6f0;
}

.stApp {
    background: #0d0f14;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 1.5rem 4rem;
    max-width: 680px;
}

/* ── Header bar ── */
.topbar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 2.5rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid #1e2230;
}
.topbar-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #4f8ef7;
    box-shadow: 0 0 8px #4f8ef755;
}
.topbar-title {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #6b7a99;
}
.topbar-badge {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4f8ef7;
    background: #4f8ef714;
    border: 1px solid #4f8ef730;
    border-radius: 4px;
    padding: 2px 8px;
}

/* ── Hero ── */
.hero-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4f8ef7;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.15;
    color: #f0f4ff;
    margin-bottom: 0.75rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    color: #4f8ef7;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6b7a99;
    line-height: 1.6;
    margin-bottom: 2rem;
    max-width: 480px;
}

/* ── Upload zone ── */
.upload-wrap {
    background: #111420;
    border: 1.5px dashed #2a2f45;
    border-radius: 12px;
    padding: 2rem 1.5rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.2s;
}
.upload-wrap:hover {
    border-color: #4f8ef755;
}
.upload-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #3a4260;
    letter-spacing: 0.08em;
    text-align: center;
    margin-top: 0.4rem;
    margin-bottom: 1.8rem;
}

/* override streamlit file uploader */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
}
[data-testid="stFileUploader"] section {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
}

/* ── Preview image ── */
.img-wrap {
    border-radius: 10px;
    overflow: hidden;
    margin-bottom: 1.5rem;
    border: 1px solid #1e2230;
}

/* ── Scanning animation ── */
.scan-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4f8ef7;
    letter-spacing: 0.1em;
    text-align: center;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.3; }
}

/* ── Result cards ── */
.result-card {
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-top: 1.2rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}

.card-healthy {
    background: #0a1a0f;
    border-color: #1a4028;
}
.card-healthy::before { background: linear-gradient(90deg, #22c55e, #16a34a); }

.card-vitiligo {
    background: #170d0d;
    border-color: #3d1a1a;
}
.card-vitiligo::before { background: linear-gradient(90deg, #ef4444, #dc2626); }

.result-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.eyebrow-healthy { color: #22c55e; }
.eyebrow-vitiligo { color: #ef4444; }

.result-diagnosis {
    font-size: 1.9rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
}
.diag-healthy  { color: #4ade80; }
.diag-vitiligo { color: #f87171; }

.result-meter-wrap {
    background: #ffffff0f;
    border-radius: 99px;
    height: 6px;
    margin-bottom: 0.6rem;
    overflow: hidden;
}
.result-meter-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.fill-healthy  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.fill-vitiligo { background: linear-gradient(90deg, #dc2626, #ef4444); }

.result-score-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}
.score-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
}
.score-healthy  { color: #4ade80; }
.score-vitiligo { color: #f87171; }

.score-label {
    font-size: 0.75rem;
    color: #6b7a99;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #111420;
    border: 1px solid #1e2230;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-top: 1.5rem;
    font-size: 0.78rem;
    color: #6b7a99;
    line-height: 1.6;
}
.disclaimer strong { color: #f59e0b; }

/* ── Footer ── */
.footer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 3rem;
    padding-top: 1.2rem;
    border-top: 1px solid #1e2230;
}
.footer-left {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3a4260;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.footer-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #4f8ef7;
}
</style>
""", unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-dot"></div>
    <div class="topbar-title">DermaScan</div>
    <div class="topbar-badge">MobileNetV3 · 96.9% acc</div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-label">Skin Analysis Tool</div>
<h1 class="hero-title">Detect <span>Vitiligo</span><br>from skin images</h1>
<p class="hero-sub">
    Upload a clear photo of the skin area. The model will classify it as
    healthy or showing signs of vitiligo.
</p>
""", unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
image = None

st.markdown('<div class="upload-wrap">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<p class="upload-hint">JPG · PNG · BMP · WEBP</p>',
    unsafe_allow_html=True,
)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.getbuffer())
        image = tmp.name

# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    st.image(image, use_container_width=True)

    with st.spinner(""):
        st.markdown('<p class="scan-text">SCANNING IMAGE...</p>', unsafe_allow_html=True)
        model = tf.keras.models.load_model(MODEL_PATH)
        img = tf.keras.utils.load_img(image, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        raw = model.predict(img_array, verbose=0)[0][0]

    # ── Prediction logic (unchanged) ──────────────────────────────────────────
    label = CLASS_NAMES[int(raw <= 0.5)]

    if label == "Healthy":
        score = raw
    else:
        score = 1 - raw

    pct = int(score * 100)

    # ── Result card ───────────────────────────────────────────────────────────
    if label == "Healthy":
        st.markdown(f"""
        <div class="result-card card-healthy">
            <div class="result-eyebrow eyebrow-healthy">✓ Classification result</div>
            <div class="result-diagnosis diag-healthy">Healthy Skin</div>
            <div class="result-meter-wrap">
                <div class="result-meter-fill fill-healthy" style="width:{pct}%"></div>
            </div>
            <div class="result-score-row">
                <div class="score-value score-healthy">{score:.4f}</div>
                <div class="score-label">Confidence · {pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card card-vitiligo">
            <div class="result-eyebrow eyebrow-vitiligo">⚠ Classification result</div>
            <div class="result-diagnosis diag-vitiligo">Vitiligo Detected</div>
            <div class="result-meter-wrap">
                <div class="result-meter-fill fill-vitiligo" style="width:{pct}%"></div>
            </div>
            <div class="result-score-row">
                <div class="score-value score-vitiligo">{score:.4f}</div>
                <div class="score-label">Confidence · {pct}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
        <strong>Notice:</strong> This tool is for research and educational purposes only.
        It is not a substitute for professional medical diagnosis.
        Always consult a qualified dermatologist.
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer-row">
    <div class="footer-left">Transfer Learning · MobileNetV3Small</div>
    <div class="footer-right">96.9% accuracy</div>
</div>
""", unsafe_allow_html=True)
