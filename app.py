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

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d0f14;
    color: #e2e6f0;
}
.stApp { background: #0d0f14; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 1.5rem 4rem;
    max-width: 680px;
}

/* ── Top bar ── */
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
.hero-title span { color: #4f8ef7; }
.hero-sub {
    font-size: 0.95rem;
    color: #6b7a99;
    line-height: 1.6;
    margin-bottom: 2rem;
}

/* ── Combined panel ── */
.panel {
    background: #111420;
    border: 1.5px solid #2a2f45;
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 1.5rem;
}

/* Upload state — shown before image */
.panel-upload-area {
    padding: 2.5rem 1.5rem 1rem;
    border-bottom: 1px solid #1e2230;
}
.panel-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3a4260;
    letter-spacing: 0.08em;
    text-align: center;
    padding: 0.6rem 1.5rem 1rem;
}

/* Image preview sits inside the panel */
.panel-img-wrap {
    position: relative;
    width: 100%;
    background: #0d0f14;
}
.panel-img-wrap img {
    width: 100%;
    display: block;
    max-height: 360px;
    object-fit: cover;
}

/* Scan overlay on image */
.scan-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #0d0f14cc;
    backdrop-filter: blur(2px);
}
.scan-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #4f8ef7;
    letter-spacing: 0.12em;
    animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.25; }
}

/* Result strip inside the panel, below the image */
.panel-result {
    padding: 1.2rem 1.5rem;
    border-top: 1px solid;
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.panel-result-healthy  { border-color: #1a4028; background: #0a1a0f; }
.panel-result-vitiligo { border-color: #3d1a1a; background: #170d0d; }

.result-icon {
    font-size: 1.6rem;
    flex-shrink: 0;
}
.result-text { flex: 1; }
.result-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.eyebrow-healthy  { color: #22c55e; }
.eyebrow-vitiligo { color: #ef4444; }

.result-diagnosis {
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 0.55rem;
}
.diag-healthy  { color: #4ade80; }
.diag-vitiligo { color: #f87171; }

.meter-wrap {
    background: #ffffff0f;
    border-radius: 99px;
    height: 5px;
    overflow: hidden;
    margin-bottom: 0.35rem;
}
.meter-fill {
    height: 100%;
    border-radius: 99px;
}
.fill-healthy  { background: linear-gradient(90deg, #22c55e, #4ade80); }
.fill-vitiligo { background: linear-gradient(90deg, #dc2626, #ef4444); }

.score-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}
.score-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 600;
}
.score-healthy  { color: #4ade80; }
.score-vitiligo { color: #f87171; }
.score-sub {
    font-size: 0.7rem;
    color: #6b7a99;
}

/* ── Disclaimer ── */
.disclaimer {
    background: #111420;
    border: 1px solid #1e2230;
    border-left: 3px solid #f59e0b;
    border-radius: 8px;
    padding: 0.85rem 1.1rem;
    margin-top: 0.4rem;
    font-size: 0.76rem;
    color: #6b7a99;
    line-height: 1.6;
}
.disclaimer strong { color: #f59e0b; }

/* ── Footer ── */
.footer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 2.5rem;
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

/* strip default streamlit file uploader border */
[data-testid="stFileUploader"] { background: transparent !important; border: none !important; }
[data-testid="stFileUploader"] section { background: transparent !important; border: none !important; padding: 0 !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; border: none !important; }
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
<p class="hero-sub">Upload a clear photo of the skin area. The model will classify it as healthy or showing signs of vitiligo.</p>
""", unsafe_allow_html=True)

# ── Combined panel ────────────────────────────────────────────────────────────
image = None

st.markdown('<div class="panel">', unsafe_allow_html=True)

# Upload widget always lives inside the panel
st.markdown('<div class="panel-upload-area">', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload skin image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="panel-hint">JPG · PNG · BMP · WEBP</div>', unsafe_allow_html=True)

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.getbuffer())
        image = tmp.name

# Image preview inside the same panel
if image:
    st.image(image, width='stretch')

st.markdown('</div>', unsafe_allow_html=True)  # close .panel

# ── Inference ─────────────────────────────────────────────────────────────────
if image:
    with st.spinner(""):
        st.markdown('<p class="scan-text" style="text-align:center;margin-bottom:1rem;">SCANNING IMAGE...</p>', unsafe_allow_html=True)
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

    # ── Result strip ──────────────────────────────────────────────────────────
    if label == "Healthy":
        st.markdown(f"""
        <div class="panel">
            <div class="panel-result panel-result-healthy">
                <div class="result-icon">✅</div>
                <div class="result-text">
                    <div class="result-eyebrow eyebrow-healthy">✓ Classification result</div>
                    <div class="result-diagnosis diag-healthy">Healthy Skin</div>
                    <div class="meter-wrap">
                        <div class="meter-fill fill-healthy" style="width:{pct}%"></div>
                    </div>
                    <div class="score-row">
                        <div class="score-val score-healthy">{score:.4f}</div>
                        <div class="score-sub">Confidence · {pct}%</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-result panel-result-vitiligo">
                <div class="result-icon">⚠️</div>
                <div class="result-text">
                    <div class="result-eyebrow eyebrow-vitiligo">⚠ Classification result</div>
                    <div class="result-diagnosis diag-vitiligo">Vitiligo Detected</div>
                    <div class="meter-wrap">
                        <div class="meter-fill fill-vitiligo" style="width:{pct}%"></div>
                    </div>
                    <div class="score-row">
                        <div class="score-val score-vitiligo">{score:.4f}</div>
                        <div class="score-sub">Confidence · {pct}%</div>
                    </div>
                </div>
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
