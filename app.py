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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #F7F9FC;
    color: #1A2B4A;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 0 1rem;
    border-bottom: 1.5px solid #E2E8F0;
    margin-bottom: 2.5rem;
}
.topbar-logo {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topbar-logo-icon {
    width: 32px;
    height: 32px;
    background: #0EA5E9;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}
.topbar-brand {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1A2B4A;
    letter-spacing: -0.01em;
}
.topbar-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    background: #EFF6FF;
    color: #0EA5E9;
    border: 1px solid #BFDBFE;
    border-radius: 4px;
    padding: 0.2rem 0.55rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ── Heading block ── */
.heading-block {
    margin-bottom: 2rem;
}
.heading-block h2 {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1A2B4A;
    letter-spacing: -0.025em;
    margin: 0 0 0.35rem;
    line-height: 1.2;
}
.heading-block p {
    font-size: 0.9rem;
    color: #64748B;
    margin: 0;
    font-weight: 400;
    line-height: 1.55;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1.5px dashed #CBD5E1;
    border-radius: 12px;
    padding: 0.25rem;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #0EA5E9;
    box-shadow: 0 0 0 3px #e0f2fe;
}
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }
.upload-hint {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #94A3B8;
    text-align: center;
    letter-spacing: 0.08em;
    margin-top: 0.45rem;
}

/* ── Image preview ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1.5px solid #E2E8F0;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}

/* ── Result card ── */
.result-card {
    background: #FFFFFF;
    border-radius: 14px;
    border: 1.5px solid #E2E8F0;
    padding: 0;
    margin-top: 1.75rem;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.05);
}
.result-header {
    padding: 1.25rem 1.5rem 1.1rem;
    border-bottom: 1px solid #F1F5F9;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-header-clean { background: #F0FDF4; }
.result-header-dusty { background: #FFFBEB; }

.result-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.25rem 0.65rem;
    border-radius: 100px;
    font-weight: 600;
}
.badge-clean {
    background: #DCFCE7;
    color: #15803D;
    border: 1px solid #BBF7D0;
}
.badge-dusty {
    background: #FEF3C7;
    color: #B45309;
    border: 1px solid #FDE68A;
}
.result-timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #94A3B8;
}

/* ── Result body ── */
.result-body {
    padding: 1.5rem;
    display: flex;
    align-items: center;
    gap: 2rem;
}
.result-text { flex: 1; }
.result-label-main {
    font-size: 2.1rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    line-height: 1;
    margin-bottom: 0.35rem;
}
.label-clean { color: #15803D; }
.label-dusty { color: #B45309; }

.result-sub {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1.25rem;
    line-height: 1.5;
}
.stat-row {
    display: flex;
    gap: 1.75rem;
}
.stat-item { display: flex; flex-direction: column; gap: 0.15rem; }
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.15rem;
    font-weight: 600;
    color: #1A2B4A;
}
.stat-key {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #94A3B8;
}

/* ── Gauge ── */
.gauge-wrap {
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
}
.gauge-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #94A3B8;
}

/* ── Footer ── */
.page-footer {
    margin-top: 3rem;
    padding: 1.5rem 0 2rem;
    border-top: 1px solid #E2E8F0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.footer-left {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #CBD5E1;
    letter-spacing: 0.06em;
}
.footer-right {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #CBD5E1;
    letter-spacing: 0.06em;
}
</style>
""", unsafe_allow_html=True)

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="topbar-logo-icon">☀</div>
        <span class="topbar-brand">SolarDx</span>
    </div>
    <span class="topbar-tag">v1.0 · Sigmoid</span>
</div>
""", unsafe_allow_html=True)

# ── Heading ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="heading-block">
    <h2>Panel Surface Inspection</h2>
    <p>Upload a solar panel image to classify surface condition — clean or dust-affected.</p>
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
    st.image(image_path, width='stretch', caption="")

    with st.spinner("Running analysis…"):
        model = tf.keras.models.load_model(MODEL_PATH)
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        prob = model.predict(img_array, verbose=0)[0][0]

    label = CLASS_NAMES[int(prob >= 0.5)]
    confidence = prob if label == "Dusty" else 1 - prob
    bar_pct = int(confidence * 100)

    # SVG gauge — circle that fills to confidence %
    radius = 42
    circumference = 2 * 3.14159 * radius
    stroke_dash = circumference * (bar_pct / 100)
    gauge_color = "#15803D" if label == "Clean" else "#B45309"
    track_color = "#E2E8F0"

    gauge_svg = f"""
    <svg width="110" height="110" viewBox="0 0 110 110">
      <circle cx="55" cy="55" r="{radius}" fill="none" stroke="{track_color}" stroke-width="8"/>
      <circle cx="55" cy="55" r="{radius}" fill="none" stroke="{gauge_color}" stroke-width="8"
              stroke-dasharray="{stroke_dash:.1f} {circumference:.1f}"
              stroke-dashoffset="{circumference * 0.25:.1f}"
              stroke-linecap="round"/>
      <text x="55" y="50" text-anchor="middle" font-family="JetBrains Mono, monospace"
            font-size="14" font-weight="600" fill="{gauge_color}">{bar_pct}%</text>
      <text x="55" y="65" text-anchor="middle" font-family="JetBrains Mono, monospace"
            font-size="7" fill="#94A3B8" letter-spacing="1">CONF</text>
    </svg>
    """

    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S")

    if label == "Clean":
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header result-header-clean">
                <span class="result-badge badge-clean">● Clean surface</span>
                <span class="result-timestamp">{ts}</span>
            </div>
            <div class="result-body">
                <div class="result-text">
                    <div class="result-label-main label-clean">Clean Panel</div>
                    <div class="result-sub">No significant dust accumulation detected.<br>Panel surface appears clear.</div>
                    <div class="stat-row">
                        <div class="stat-item">
                            <span class="stat-value">{confidence:.4f}</span>
                            <span class="stat-key">Raw score</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{bar_pct}%</span>
                            <span class="stat-key">Clean prob</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{1 - confidence:.4f}</span>
                            <span class="stat-key">Dusty prob</span>
                        </div>
                    </div>
                </div>
                <div class="gauge-wrap">
                    {gauge_svg}
                    <span class="gauge-label">Confidence</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-header result-header-dusty">
                <span class="result-badge badge-dusty">⚠ Dust detected</span>
                <span class="result-timestamp">{ts}</span>
            </div>
            <div class="result-body">
                <div class="result-text">
                    <div class="result-label-main label-dusty">Dusty Panel</div>
                    <div class="result-sub">Dust accumulation detected on surface.<br>Cleaning is recommended.</div>
                    <div class="stat-row">
                        <div class="stat-item">
                            <span class="stat-value">{confidence:.4f}</span>
                            <span class="stat-key">Raw score</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{bar_pct}%</span>
                            <span class="stat-key">Dusty prob</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-value">{1 - confidence:.4f}</span>
                            <span class="stat-key">Clean prob</span>
                        </div>
                    </div>
                </div>
                <div class="gauge-wrap">
                    {gauge_svg}
                    <span class="gauge-label">Confidence</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-footer">
    <span class="footer-left">Binary CNN · MobileNetV3 · Sigmoid</span>
    <span class="footer-right">Solar Panel Dataset</span>
</div>
""", unsafe_allow_html=True)
